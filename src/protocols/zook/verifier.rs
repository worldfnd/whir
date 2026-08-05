//! Zook verifier — mirror of [`super::prover`] (Construction 9.7).
//!
//! Per ZK round, the verifier:
//!   1. Receives the **sumcheck-masks tree** commitment (pre-sumcheck — binds
//!      each round poly's mask via Fiat-Shamir).
//!   2. Runs `sumcheck.verify` (orchestrator tracks sumcheck challenges implicitly).
//!      Post-sumcheck `state.sum = δ + γ_sumcheck · dot`.
//!   3. Receives the **cs_mask tree** commitment (post-sumcheck — cs_mask
//!      carries `r_folded` from source IRS randomness) and reads `δ`
//!      cleartext, reconciling `state.sum := (sum − δ) · γ_sumcheck⁻¹`.
//!   4. Runs `code_switch.verify_for_implicit` — accumulates OOD/in-domain
//!      constraints as `ImplicitConstraint` entries instead of updating an
//!      explicit covector.
//!   5. Verifies sumcheck masks at `[1, r_i, …, r_i^{vec_size_A−1}]` (gives
//!      X_i = M_i(r_i)); verifies cs_mask at the post-cs covector mask
//!      region (gives X_cs). Checks `Σ X_i == δ` to bind δ; subtracts X_cs
//!      to project to f-only.
//!
//! After all rounds: receives the basecase IRS commitment, runs
//! `basecase.verify`, constructs `full_eval_point = all_round_challenges ++ evaluation_points`,
//! and checks that the implicit covector evaluates to `basecase.linear_form_evaluation`
//! in O((num_constraints + log N)) operations — eliminating the O(N) per-round
//! covector update bottleneck.

use ark_ff::Field;
use ark_std::rand::{distributions::Standard, prelude::Distribution};
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        geometric_sequence,
        linear_form::{LinearForm, UnivariateEvaluation},
    },
    hash::Hash,
    protocols::{
        code_switch::CovectorUpdateParams,
        mask_proximity,
        params::config::{MaskOracleConfig, ProtocolConfig, RoundConfig},
        sumcheck::SumcheckOpening,
        zook::{
            block::VerifierBlock, commit::Commitment, round::verify_whir_round, ClaimGroup,
            FinalClaim,
        },
    },
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, VerificationResult,
        VerifierMessage, VerifierState,
    },
    verify,
};

impl<M: Embedding + Default> ProtocolConfig<M> {
    /// Verify `f(witness) == evaluations[j]` for every linear_form `f = linear_forms[j]` against
    /// the received commitment.
    ///
    /// The verifier holds no base-field state — all arithmetic is in `M::Target`
    /// — so only round 0's dispatch (over embedding `M`) is embedding-aware.
    #[cfg_attr(feature = "tracing", instrument(skip_all, name = "zook::verify", fields(vector_size = self.tuning().vector_size, num_rounds = self.num_rounds(), num_claims = linear_forms.len())))]
    pub fn verify<H>(
        &self,
        vs: &mut VerifierState<H>,
        commitment: Commitment,
        linear_forms: &[&dyn LinearForm<M::Target>],
        evaluations: &[M::Target],
    ) -> VerificationResult<FinalClaim<M::Target>>
    where
        M::Target: Default + Codec<[H::U]>,
        Standard: Distribution<M::Target>,
        H: DuplexSpongeInterface,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(
            linear_forms.len(),
            evaluations.len(),
            "linear_forms.len() != evaluations.len()"
        );
        assert!(
            !linear_forms.is_empty(),
            "zook requires ≥ 1 (form, value) pair"
        );

        let one = <M::Target as Field>::ONE;

        // RLC challenge binds the form/value set to the commitment.
        let batching_challenge: M::Target = vs.verifier_message();
        let claim_weights = geometric_sequence(one, batching_challenge, linear_forms.len());
        let batched_evaluation: M::Target = evaluations
            .iter()
            .zip(&claim_weights)
            .map(|(v, weight)| *v * weight)
            .sum();

        // Basecase-only path: no rounds, evaluate directly.
        if !self.has_rounds() {
            let opening =
                self.basecase()
                    .verify(vs, &commitment.irs_commitment, batched_evaluation)?;
            // No constraint terms, no round scalings: linear_forms_contribution = opening.linear_form_evaluation,
            // initial_claim_scale = ONE. The caller checks:
            // ONE × Σ_j claim_weight_j × form_j.mle_at(evaluation_point) == linear_forms_contribution
            return Ok(FinalClaim {
                groups: vec![ClaimGroup {
                    evaluation_point: opening.evaluation_points,
                    initial_claim_scale: one,
                    batching_challenge,
                }],
                linear_forms_contribution: opening.linear_form_evaluation,
            });
        }

        let mut block = VerifierBlock::single_source(batched_evaluation, commitment.irs_commitment);
        let mut constraints: Vec<ImplicitConstraint<M::Target>> = Vec::new();
        let mut all_round_challenges: Vec<M::Target> = Vec::new();
        let mut challenges_at: Vec<usize> = vec![0];
        let mut round_scale_factors: Vec<M::Target> = Vec::new();

        // Round 0 verifies over `M` (opens a base source IRS); the tail is ext→ext.
        let first = self
            .first_round()
            .expect("has_rounds() true implies a first round");
        let first_msg_len = first.code_switch().source().message_length();
        let (next, out) = verify_whir_round::<M, H>(first, block, vs)?;
        all_round_challenges.extend_from_slice(&out.round_challenges);
        push_constraints(&mut constraints, &out.update_params, first_msg_len, 0);
        round_scale_factors.push(out.update_params.original_sl_coeff);
        challenges_at.push(all_round_challenges.len());
        block = next;

        for round in self.tail_rounds() {
            let msg_len = round.code_switch().source().message_length();
            let (next, out) = verify_whir_round::<Identity<M::Target>, H>(round, block, vs)?;
            all_round_challenges.extend_from_slice(&out.round_challenges);
            push_constraints(
                &mut constraints,
                &out.update_params,
                msg_len,
                round_scale_factors.len(),
            );
            round_scale_factors.push(out.update_params.original_sl_coeff);
            challenges_at.push(all_round_challenges.len());
            block = next;
        }

        let opening = self
            .basecase()
            .verify(vs, &block.commitments[0], block.sum)?;

        // full_eval_point = all round challenges ++ basecase evaluation points.
        let full_eval_point: Vec<M::Target> = all_round_challenges
            .iter()
            .chain(opening.evaluation_points.iter())
            .copied()
            .collect();
        debug_assert_eq!(
            full_eval_point.len(),
            self.tuning().vector_size.trailing_zeros() as usize,
            "full_eval_point length must equal log2(vector_size)"
        );

        // Compute scale_suffixes[round_idx] = Π_{r'=round_idx..num_completed_rounds-1} round_scale_factors[r'].
        let num_completed_rounds = round_scale_factors.len();
        let mut scale_suffixes = vec![one; num_completed_rounds + 1];
        for round_idx in (0..num_completed_rounds).rev() {
            scale_suffixes[round_idx] =
                round_scale_factors[round_idx] * scale_suffixes[round_idx + 1];
        }

        // constraint_sum = Σ_c c.batching_weight × scale_suffixes[c.round+1] × mle_of_geom(c.eval_point, z_suffix)
        let constraint_sum: M::Target = constraints
            .iter()
            .map(|c| {
                let z_suffix_start = challenges_at[c.added_at_round + 1];
                let z_suffix =
                    &full_eval_point[z_suffix_start..z_suffix_start + c.domain_bits as usize];
                c.batching_weight
                    * scale_suffixes[c.added_at_round + 1]
                    * UnivariateEvaluation::new(c.eval_point, 1usize << c.domain_bits)
                        .mle_evaluate(z_suffix)
            })
            .sum();

        // linear_forms_contribution is what scale_suffixes[0] × initial_forms_mle must equal.
        // The caller verifies: scale_suffixes[0] × Σ_j γ^j × form_j.mle_at(full_eval_point) == linear_forms_contribution
        let linear_forms_contribution = opening.linear_form_evaluation - constraint_sum;

        Ok(FinalClaim {
            groups: vec![ClaimGroup {
                evaluation_point: full_eval_point,
                initial_claim_scale: scale_suffixes[0],
                batching_challenge,
            }],
            linear_forms_contribution,
        })
    }
}

/// Append the round's OOD + in-domain constraints to the accumulator.
pub(crate) fn push_constraints<F: Field>(
    constraints: &mut Vec<ImplicitConstraint<F>>,
    update_params: &CovectorUpdateParams<F>,
    msg_len: usize,
    added_at_round: usize,
) {
    let domain_bits = msg_len.trailing_zeros();
    for (batching_weight, eval_point) in update_params
        .ood_rlc_coeffs
        .iter()
        .zip(&update_params.ood_eval_points)
    {
        constraints.push(ImplicitConstraint {
            eval_point: *eval_point,
            batching_weight: *batching_weight,
            domain_bits,
            added_at_round,
        });
    }
    for (batching_weight, eval_point) in update_params
        .in_domain_rlc_coeffs
        .iter()
        .zip(&update_params.in_domain_eval_points)
    {
        constraints.push(ImplicitConstraint {
            eval_point: *eval_point,
            batching_weight: *batching_weight,
            domain_bits,
            added_at_round,
        });
    }
}

pub(crate) struct ImplicitConstraint<F: Field> {
    /// OOD alpha or in-domain omega.
    pub(crate) eval_point: F,
    /// RLC coefficient at the time this constraint was added.
    pub(crate) batching_weight: F,
    /// log2 of the effective domain size = log2(msg_len when added).
    pub(crate) domain_bits: u32,
    /// Index into `round_scale_factors`: which round added this constraint.
    pub(crate) added_at_round: usize,
}

/// Manages ZK mask verification state for one round.
/// Mirrors `RoundMaskOracle` in `prover.rs` on the receive side.
/// `Disabled` is the Null Object for Standard mode — all methods return Ok(()) / &[].
pub(crate) enum RoundMaskOracleCheck<'a, F: Field> {
    /// Standard mode: no mask oracle.
    Disabled,
    /// ZK — sumcheck-masks commitment received; awaiting cs_mask.
    SumcheckCommitmentReceived {
        mo: &'a MaskOracleConfig<F>,
        sc_commitment: mask_proximity::Commitment,
        /// source IRS randomness length — needed for zk_tail in verify_and_discharge.
        source_mask_len: usize,
    },
    /// ZK — both commitments received, sum reconciled; ready to verify and discharge.
    ReadyForDischarge {
        mo: &'a MaskOracleConfig<F>,
        sc_commitment: mask_proximity::Commitment,
        cs_commitment: mask_proximity::Commitment,
        /// δ = Σ Mᵢ(rᵢ) received from transcript; bound by the sumcheck-mask opening check.
        mask_eval_sum: F,
        source_mask_len: usize,
    },
}

impl<'a, F: Field + Default> RoundMaskOracleCheck<'a, F> {
    /// Receive the sumcheck-masks commitment (ZK) or construct Disabled (Standard).
    pub(crate) fn begin<M, H>(
        round: &'a RoundConfig<M>,
        vs: &mut VerifierState<H>,
    ) -> VerificationResult<Self>
    where
        M: Embedding<Target = F>,
        F: Codec<[H::U]>,
        H: DuplexSpongeInterface,
        Hash: ProverMessage<[H::U]>,
    {
        match round.mask_oracle() {
            None => Ok(Self::Disabled),
            Some(mo) => {
                let sc_commitment = mo.sumcheck_masks().receive_commitment(vs)?;
                let source_mask_len = round.code_switch().source().mask_length();
                Ok(Self::SumcheckCommitmentReceived {
                    mo,
                    sc_commitment,
                    source_mask_len,
                })
            }
        }
    }

    /// Receive the cs_mask commitment + mask_eval_sum (δ) cleartext, then reconcile *sum.
    /// Transitions SumcheckCommitmentReceived → ReadyForDischarge in place.
    /// No-op for Disabled.
    pub(crate) fn receive_cs_mask_and_reconcile<H>(
        &mut self,
        opening: &SumcheckOpening<F>,
        vs: &mut VerifierState<H>,
        sum: &mut F,
    ) -> VerificationResult<()>
    where
        F: Codec<[H::U]>,
        H: DuplexSpongeInterface,
        Hash: ProverMessage<[H::U]>,
    {
        let (mo, sc_commitment, source_mask_len) = match std::mem::replace(self, Self::Disabled) {
            Self::SumcheckCommitmentReceived {
                mo,
                sc_commitment,
                source_mask_len,
            } => (mo, sc_commitment, source_mask_len),
            other => {
                *self = other;
                return Ok(());
            }
        };

        let cs_commitment = mo.cs_mask().receive_commitment(vs)?;
        let mask_eval_sum: F = vs.prover_message()?;

        // Reconcile: sum was (mask_eval_sum + mask_rlc · dot), now (sum − δ)/mask_rlc = dot.
        // mask_rlc is Fiat–Shamir; zero has negligible probability for large fields.
        let mask_rlc_inv = opening
            .mask_rlc
            .inverse()
            .expect("mask_rlc non-zero (negligible probability for large fields)");
        *sum = (*sum - mask_eval_sum) * mask_rlc_inv;

        *self = Self::ReadyForDischarge {
            mo,
            sc_commitment,
            cs_commitment,
            mask_eval_sum,
            source_mask_len,
        };
        Ok(())
    }

    /// Verify both mask trees and subtract the cs_mask contribution from *sum.
    /// No-op for Disabled.
    ///
    /// Soundness: `code_switch.verify_for_implicit` (step 4) checked OOD/in-domain
    /// consistency of the target codeword but did NOT verify the masks are close to
    /// C_zk. Both checks are load-bearing per Theorem 9.10 / Construction 7.2.
    pub(crate) fn verify_and_discharge<H>(
        self,
        round_challenges: &[F],
        msg_len: usize,
        update_params: &CovectorUpdateParams<F>,
        vs: &mut VerifierState<H>,
        sum: &mut F,
    ) -> VerificationResult<()>
    where
        F: Codec<[H::U]>,
        H: DuplexSpongeInterface,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let (mo, sc_commitment, cs_commitment, mask_eval_sum, source_mask_len) = match self {
            Self::ReadyForDischarge {
                mo,
                sc_commitment,
                cs_commitment,
                mask_eval_sum,
                source_mask_len,
            } => (
                mo,
                sc_commitment,
                cs_commitment,
                mask_eval_sum,
                source_mask_len,
            ),
            Self::Disabled => return Ok(()),
            Self::SumcheckCommitmentReceived { .. } => {
                debug_assert!(
                    false,
                    "verify_and_discharge called before receive_cs_mask_and_reconcile"
                );
                return Ok(());
            }
        };

        // --- Sumcheck-masks tree ---
        // Verify each mask at its geometric covector [1, rᵢ, rᵢ², …]; check Σ Xᵢ == mask_eval_sum.
        let sumcheck_vec_size = mo.sumcheck_masks().c_zk_commit().vector_size();
        let sm_covectors: Vec<Vec<F>> = round_challenges
            .iter()
            .map(|&r| geometric_sequence(F::ONE, r, sumcheck_vec_size))
            .collect();
        let sm_refs: Vec<&[F]> = sm_covectors.iter().map(Vec::as_slice).collect();
        let sm_x_values = mo
            .sumcheck_masks()
            .verify(vs, &sc_commitment, Some(&sm_refs))?
            .expect("sumcheck-mask values always returned when covectors passed");
        let sumcheck_x_sum: F = sm_x_values.iter().copied().sum();
        verify!(sumcheck_x_sum == mask_eval_sum);

        // --- cs_mask tree ---
        // Reconstruct zk_tail: the covector region [msg_len .. msg_len + l_zk].
        // OOD points contribute over the full l_zk; in-domain only over source_mask_len.
        let l_zk = mo.l_zk().get();
        let mut zk_tail = vec![F::ZERO; l_zk];

        for (coeff, alpha) in update_params
            .ood_rlc_coeffs
            .iter()
            .zip(&update_params.ood_eval_points)
        {
            let alpha_msg_pow = alpha.pow([msg_len as u64]);
            let mut alpha_l = alpha_msg_pow;
            for entry in &mut zk_tail {
                *entry += *coeff * alpha_l;
                alpha_l *= *alpha;
            }
        }
        for (coeff, omega) in update_params
            .in_domain_rlc_coeffs
            .iter()
            .zip(&update_params.in_domain_eval_points)
        {
            let omega_msg_pow = omega.pow([msg_len as u64]);
            let mut omega_l = omega_msg_pow;
            for entry in &mut zk_tail[..source_mask_len] {
                *entry += *coeff * omega_l;
                omega_l *= *omega;
            }
        }

        let cs_cov: [&[F]; 1] = [&zk_tail];
        let cs_x_values = mo
            .cs_mask()
            .verify(vs, &cs_commitment, Some(&cs_cov))?
            .expect("cs_mask value always returned when covector passed");
        *sum -= cs_x_values[0];

        Ok(())
    }
}
