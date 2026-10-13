//! Batched Zook verifier.
//!
//! Mirror of [`super::prover`] on the receive side. Composes the shared atom
//! ([`verify_whir_round`]) with the batched-specific bookkeeping:
//!   - per-bundle pre-merge loops accumulate into [`BundlePreMergeData`];
//!   - the main loop drains active commitments into [`merge_active_verify`]
//!     (selector verify when `t ≥ 2`, pass-through when `t == 1`) and threads
//!     the result through the atom;
//!   - constraint accumulators are split into a SHARED pile (post-merge rounds)
//!     and per-bundle pre-merge piles, then composed with the scale chains at
//!     final-claim assembly.
//!
//! Bundle scale: `pre_b · round_scale_factors[j] · suffix_scale[j+1]` where
//!   `pre_b           = θ_b · η_j`              (selector weight at bundle b's join round j)
//!   `merge_extra[r]  = θ_carrier_r · η_r`      (or `F::ONE` when no merge re-weights the carrier)
//!   `suffix_scale[r] = Π_{k=r..R-1} round_scale_factors[k] · merge_extra[k]`

use ark_ff::Field;
use ark_std::rand::{distributions::Standard, prelude::Distribution};
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::{
    algebra::{
        embedding::Identity,
        linear_form::{LinearForm, UnivariateEvaluation},
    },
    hash::Hash,
    protocols::{
        code_switch::CovectorUpdateParams,
        irs_commit::Commitment as IrsCommitment,
        params::{
            batched::{BatchedProtocolConfig, MergeSchedule},
            config::RoundConfig,
        },
        zook::{
            batched::{bundle::BundleDescriptor, commit::BundleCommitment},
            block::VerifierBlock,
            round::verify_whir_round,
            verifier::{push_constraints, ImplicitConstraint},
            ClaimGroup, FinalClaim,
        },
    },
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, VerificationResult,
        VerifierState,
    },
    verify,
};

impl<F: Field + Default> BatchedProtocolConfig<Identity<F>> {
    /// Verify all per-poly claims across one or more bundles. Returns a
    /// [`FinalClaim`] that the caller must finish via [`FinalClaim::verify_bundles`].
    #[cfg_attr(feature = "tracing", instrument(skip_all, name = "zook::batched::verify", fields(num_bundles = bundles.len())))]
    #[allow(clippy::too_many_lines)]
    pub fn verify<H>(
        &self,
        vs: &mut VerifierState<H>,
        commitments: Vec<BundleCommitment<F>>,
        bundles: &[&BundleDescriptor<F>],
    ) -> VerificationResult<FinalClaim<F>>
    where
        Standard: Distribution<F>,
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        verify!(commitments.len() == bundles.len());
        verify!(commitments.len() == self.bundle_configs().len());
        verify!(!bundles.is_empty());
        for bundle in bundles {
            bundle.validate()?;
        }

        let n = bundles.len();
        let claim_counts: Vec<usize> = bundles.iter().map(|b| b.total_claims()).collect();
        verify!(self
            .validate_security_target_met_for_claims(&claim_counts)
            .is_ok());

        let mut batching_challenges: Vec<F> = Vec::with_capacity(n);
        let mut initial_sums: Vec<F> = Vec::with_capacity(n);
        for (i, &bundle) in bundles.iter().enumerate() {
            let bcfg = &self.bundle_configs()[i];
            verify!(bundle.num_polys == bcfg.spec.num_polys);
            verify!(bundle.length == bcfg.spec.length);

            let batching_challenge = commitments[i].batching_challenge;
            let initial_sum = bundle.initial_sum(batching_challenge);
            batching_challenges.push(batching_challenge);
            initial_sums.push(initial_sum);
        }

        let mut bundle_pre_scales: Vec<F> = vec![F::ZERO; n];
        let bundle_join_rounds: Vec<Option<usize>> = self
            .bundle_configs()
            .iter()
            .map(|bc| bc.join_round)
            .collect();
        let mut bundle_commitments: Vec<Option<IrsCommitment>> = commitments
            .into_iter()
            .map(|c| Some(c.irs_commitment))
            .collect();

        let mut round_scale_factors: Vec<F> = Vec::new();
        let mut merge_extra_factors: Vec<F> = Vec::new();
        let mut all_round_challenges: Vec<F> = Vec::new();
        let mut challenges_at: Vec<usize> = vec![0];
        let mut constraints: Vec<ImplicitConstraint<F>> = Vec::new();

        let mut bundle_pre_merge: Vec<BundlePreMergeData<F>> =
            (0..n).map(|_| BundlePreMergeData::default()).collect();

        for bundle_idx in 0..n {
            let bcfg = &self.bundle_configs()[bundle_idx];
            if bcfg.pre_merge_rounds.is_empty() {
                continue;
            }
            let current_commitment = bundle_commitments[bundle_idx]
                .take()
                .expect("bundle commitment present before pre-merge");
            let current_sum = initial_sums[bundle_idx];

            let mut block = VerifierBlock::single_source(current_sum, current_commitment);
            for (k, round) in bcfg.pre_merge_rounds.iter().enumerate() {
                let msg_len = round.code_switch().source().message_length();
                let (next, out) = verify_whir_round(round, block, vs)?;
                bundle_pre_merge[bundle_idx]
                    .challenges
                    .extend_from_slice(&out.round_challenges);
                push_pre_merge_constraints(
                    &mut bundle_pre_merge[bundle_idx].constraints,
                    &out.update_params,
                    msg_len,
                    k,
                );
                bundle_pre_merge[bundle_idx]
                    .round_scales
                    .push(out.update_params.original_sl_coeff);
                let challenge_count = bundle_pre_merge[bundle_idx].challenges.len();
                bundle_pre_merge[bundle_idx]
                    .challenges_at
                    .push(challenge_count);
                block = next;
            }
            bundle_commitments[bundle_idx] = Some(block.commitments.remove(0));
            initial_sums[bundle_idx] = block.sum;
        }

        let mut carrier: Option<VerifierBlock<F>> = None;

        let inner_rounds: Vec<&RoundConfig<Identity<F>>> = self
            .inner()
            .first_round()
            .into_iter()
            .chain(self.inner().tail_rounds())
            .collect();
        for (r, round) in inner_rounds.into_iter().enumerate() {
            let carrier_existed = carrier.is_some();
            let mut active_commitments: Vec<IrsCommitment> = Vec::new();
            let mut sums: Vec<F> = Vec::new();
            if let Some(c) = carrier.take() {
                active_commitments.push(c.commitments.into_iter().next().expect("single-source"));
                sums.push(c.sum);
            }
            let joiner_bundle_indices: Vec<usize> = self
                .schedule()
                .join_at(r)
                .map(|j| j.bundle_indices.clone())
                .unwrap_or_default();
            for &idx in &joiner_bundle_indices {
                let cmt = bundle_commitments[idx]
                    .take()
                    .expect("each bundle commitment consumed once at its join round");
                active_commitments.push(cmt);
                sums.push(initial_sums[idx]);
            }
            let num_active_blocks = sums.len();
            assert!(
                num_active_blocks >= 1,
                "round {r}: active list must be non-empty (scheduler invariant)"
            );
            if let Some(join) = self.schedule().join_at(r) {
                debug_assert_eq!(
                    join.t, num_active_blocks,
                    "round {r}: schedule.t mismatches active list size",
                );
            } else {
                debug_assert_eq!(
                    1, num_active_blocks,
                    "round {r}: no join scheduled but active.len != 1",
                );
            }

            let merged = merge_active_verify(
                self.schedule(),
                r,
                active_commitments,
                &sums,
                carrier_existed,
                &joiner_bundle_indices,
                &mut bundle_pre_scales,
                &mut merge_extra_factors,
                vs,
            )?;

            let msg_len = round.code_switch().source().message_length();
            let (next, out) = verify_whir_round(round, merged, vs)?;
            all_round_challenges.extend_from_slice(&out.round_challenges);
            push_constraints(
                &mut constraints,
                &out.update_params,
                msg_len,
                round_scale_factors.len(),
            );
            round_scale_factors.push(out.update_params.original_sl_coeff);
            challenges_at.push(all_round_challenges.len());
            carrier = Some(next);
        }

        assert!(
            self.schedule().basecase_join().is_none(),
            "basecase joins are not yet wired by the orchestrator; \
             the scheduler should have rejected this configuration"
        );
        debug_assert!(
            bundle_commitments.iter().all(Option::is_none),
            "every bundle commitment must be consumed during the round walk"
        );
        debug_assert!(
            bundle_pre_scales.iter().all(|s| *s != F::ZERO),
            "scheduler invariant: every bundle must have its pre_scale assigned during the round walk"
        );

        let final_block = carrier.expect("carrier after R rounds");
        let basecase_opening =
            self.inner()
                .basecase()
                .verify(vs, &final_block.commitments[0], final_block.sum)?;

        let num_completed_rounds = round_scale_factors.len();
        debug_assert_eq!(num_completed_rounds, merge_extra_factors.len());

        // suffix_scale[r] = Π_{k=r..R-1} (round_scale_factors[k] · merge_extra_factors[k])
        let mut suffix_scale = vec![F::ONE; num_completed_rounds + 1];
        for r in (0..num_completed_rounds).rev() {
            suffix_scale[r] = round_scale_factors[r] * merge_extra_factors[r] * suffix_scale[r + 1];
        }

        let full_eval_point: Vec<F> = all_round_challenges
            .iter()
            .chain(basecase_opening.evaluation_points.iter())
            .copied()
            .collect();
        debug_assert_eq!(
            full_eval_point.len(),
            self.inner().tuning().vector_size.trailing_zeros() as usize,
            "full_eval_point length must equal log_2(inner.vector_size)"
        );

        let shared_constraint_sum: F = constraints
            .iter()
            .map(|c| {
                let z_suffix_start = challenges_at[c.added_at_round + 1];
                let z_suffix =
                    &full_eval_point[z_suffix_start..z_suffix_start + c.domain_bits as usize];
                c.batching_weight
                    * suffix_scale[c.added_at_round + 1]
                    * UnivariateEvaluation::new(c.eval_point, 1usize << c.domain_bits)
                        .mle_evaluate(z_suffix)
            })
            .sum();

        let mut pre_merge_constraint_sum = F::ZERO;
        for b in 0..n {
            let pre = &bundle_pre_merge[b];
            if pre.constraints.is_empty() {
                continue;
            }
            let join_round = bundle_join_rounds[b]
                .expect("first-pass: every bundle joins at some round (no basecase joins)");

            let shared_scale_for_bundle = bundle_pre_scales[b]
                * round_scale_factors[join_round]
                * suffix_scale[join_round + 1];

            let bundle_eval_point: Vec<F> = pre
                .challenges
                .iter()
                .chain(all_round_challenges[challenges_at[join_round]..].iter())
                .chain(basecase_opening.evaluation_points.iter())
                .copied()
                .collect();

            for c in &pre.constraints {
                let z_suffix_start = pre.challenges_at[c.added_at_pre_merge_round + 1];
                let z_suffix =
                    &bundle_eval_point[z_suffix_start..z_suffix_start + c.domain_bits as usize];
                let pre_merge_scale_after = pre.scale_suffix_after(c.added_at_pre_merge_round);
                let scale = pre_merge_scale_after * shared_scale_for_bundle;
                pre_merge_constraint_sum += c.batching_weight
                    * scale
                    * UnivariateEvaluation::new(c.eval_point, 1usize << c.domain_bits)
                        .mle_evaluate(z_suffix);
            }
        }

        let constraint_sum = shared_constraint_sum + pre_merge_constraint_sum;
        let linear_forms_contribution = basecase_opening.linear_form_evaluation - constraint_sum;

        let groups: Vec<ClaimGroup<F>> = (0..n)
            .map(|b| {
                let join_round = bundle_join_rounds[b]
                    .expect("first-pass: every bundle joins at some round (no basecase joins)");
                let pre = &bundle_pre_merge[b];

                let initial_claim_scale = pre.total_scale()
                    * bundle_pre_scales[b]
                    * round_scale_factors[join_round]
                    * suffix_scale[join_round + 1];

                let start = challenges_at[join_round];
                let evaluation_point: Vec<F> = pre
                    .challenges
                    .iter()
                    .chain(all_round_challenges[start..].iter())
                    .chain(basecase_opening.evaluation_points.iter())
                    .copied()
                    .collect();
                debug_assert_eq!(
                    evaluation_point.len(),
                    bundles[b].domain_bits(),
                    "bundle eval_point length must match log_2(num_polys · length)"
                );

                ClaimGroup {
                    evaluation_point,
                    initial_claim_scale,
                    batching_challenge: batching_challenges[b],
                }
            })
            .collect();

        Ok(FinalClaim {
            groups,
            linear_forms_contribution,
        })
    }
}

/// Selector-merge the active commitments + per-block sums into one virtual
/// [`VerifierBlock`]. Records the per-bundle pre-scale and per-round merge
/// extra factor into the caller's accumulators. Pass-through when `t == 1`.
// Mirrors the prover-side merge; the accumulators and transcript handle are all
// distinct outputs, so bundling them into a struct would not aid readability.
#[allow(clippy::too_many_arguments)]
fn merge_active_verify<F, H>(
    schedule: &MergeSchedule<F>,
    round_idx: usize,
    active_commitments: Vec<IrsCommitment>,
    sums: &[F],
    carrier_existed: bool,
    joiner_bundle_indices: &[usize],
    bundle_pre_scales: &mut [F],
    merge_extra_factors: &mut Vec<F>,
    vs: &mut VerifierState<H>,
) -> VerificationResult<VerifierBlock<F>>
where
    F: Field + Default + Codec<[H::U]>,
    Standard: Distribution<F>,
    H: DuplexSpongeInterface,
    [u8; 32]: Decoding<[H::U]>,
    U64: Codec<[H::U]>,
{
    debug_assert_eq!(active_commitments.len(), sums.len());
    let t = sums.len();
    if t < 2 {
        merge_extra_factors.push(F::ONE);
        if !carrier_existed {
            bundle_pre_scales[joiner_bundle_indices[0]] = F::ONE;
        }
        return Ok(VerifierBlock {
            sum: sums[0],
            commitments: active_commitments,
            theta: vec![F::ONE],
        });
    }

    let join_cfg = schedule
        .join_at(round_idx)
        .expect("round with t >= 2 must have a schedule entry");
    let opening = join_cfg.selector.verify(vs, sums)?;
    let eta = opening.eta;

    if carrier_existed {
        merge_extra_factors.push(opening.theta[0] * eta);
        for (i, &idx) in joiner_bundle_indices.iter().enumerate() {
            bundle_pre_scales[idx] = opening.theta[i + 1] * eta;
        }
    } else {
        merge_extra_factors.push(F::ONE);
        for (i, &idx) in joiner_bundle_indices.iter().enumerate() {
            bundle_pre_scales[idx] = opening.theta[i] * eta;
        }
    }

    Ok(VerifierBlock {
        sum: opening.sum,
        commitments: active_commitments,
        theta: opening.theta,
    })
}

/// Pre-merge variant of [`push_constraints`] for per-bundle constraint piles.
fn push_pre_merge_constraints<F: Field>(
    constraints: &mut Vec<PreMergeConstraint<F>>,
    update_params: &CovectorUpdateParams<F>,
    msg_len: usize,
    added_at_pre_merge_round: usize,
) {
    let domain_bits = msg_len.trailing_zeros();
    for (bw, ep) in update_params
        .ood_rlc_coeffs
        .iter()
        .zip(&update_params.ood_eval_points)
    {
        constraints.push(PreMergeConstraint {
            eval_point: *ep,
            batching_weight: *bw,
            domain_bits,
            added_at_pre_merge_round,
        });
    }
    for (bw, ep) in update_params
        .in_domain_rlc_coeffs
        .iter()
        .zip(&update_params.in_domain_eval_points)
    {
        constraints.push(PreMergeConstraint {
            eval_point: *ep,
            batching_weight: *bw,
            domain_bits,
            added_at_pre_merge_round,
        });
    }
}

/// Per-bundle pre-merge state. Empty when the bundle has no pre-merge rounds.
///
/// When a bundle has non-empty `pre_merge_rounds`, the verifier runs those rounds
/// against the bundle's own commitment BEFORE it joins the shared active list.
/// This accumulates the round challenges, code-switch scale factors, and emitted
/// constraints during that phase so they can be combined with the shared chain at
/// final-claim assembly.
struct BundlePreMergeData<F: Field> {
    /// Sumcheck challenges across this bundle's pre-merge rounds, in order.
    challenges: Vec<F>,
    /// `challenges_at[k]` = cumulative challenges before pre-merge round `k`;
    /// length = `pre_merge_rounds.len() + 1`.
    challenges_at: Vec<usize>,
    /// `original_sl_coeff` from each pre-merge round's code_switch.
    round_scales: Vec<F>,
    /// Constraints emitted during this bundle's pre-merge round bodies, scaled
    /// by the bundle's pre-merge scale chain (not the shared chain).
    constraints: Vec<PreMergeConstraint<F>>,
}

impl<F: Field> Default for BundlePreMergeData<F> {
    fn default() -> Self {
        Self {
            challenges: Vec::new(),
            challenges_at: vec![0],
            round_scales: Vec::new(),
            constraints: Vec::new(),
        }
    }
}

impl<F: Field> BundlePreMergeData<F> {
    /// Product of all pre-merge round scale factors (`F::ONE` if none).
    fn total_scale(&self) -> F {
        self.round_scales.iter().copied().fold(F::ONE, |a, b| a * b)
    }

    /// Suffix product `Π_{k'>=k+1} round_scales[k']` (`F::ONE` if `k+1 >= round_scales.len()`).
    fn scale_suffix_after(&self, k: usize) -> F {
        self.round_scales
            .iter()
            .skip(k + 1)
            .copied()
            .fold(F::ONE, |a, b| a * b)
    }
}

struct PreMergeConstraint<F: Field> {
    /// OOD alpha or in-domain omega.
    eval_point: F,
    /// RLC coefficient at the time this constraint was added.
    batching_weight: F,
    /// log2 of msg_len when added.
    domain_bits: u32,
    /// Index into the bundle's `pre_merge_rounds` for scale-chain lookup.
    added_at_pre_merge_round: usize,
}
