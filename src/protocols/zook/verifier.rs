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
        embedding::Identity,
        geometric_sequence,
        linear_form::{LinearForm, UnivariateEvaluation},
    },
    hash::Hash,
    protocols::{
        code_switch::CovectorUpdateParams,
        irs_commit::Commitment as IrsCommitment,
        mask_proximity,
        params::protocol_config::{MaskOracleConfig, ProtocolConfig, RoundConfig},
        sumcheck::SumcheckOpening,
        zook::{commit::Commitment, FinalClaim},
    },
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, VerificationResult,
        VerifierMessage, VerifierState,
    },
    verify,
};

impl<F: Field + Default> ProtocolConfig<Identity<F>> {
    /// Verify `f(witness) == evaluations[j]` for every linear_form `f = linear_forms[j]` against
    /// the received commitment.
    #[cfg_attr(feature = "tracing", instrument(skip_all, name = "zook::verify", fields(vector_size = self.tuning().vector_size, num_rounds = self.rounds().len(), num_claims = linear_forms.len())))]
    pub fn verify<H>(
        &self,
        vs: &mut VerifierState<H>,
        commitment: Commitment,
        linear_forms: &[&dyn LinearForm<F>],
        evaluations: &[F],
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
        assert_eq!(
            linear_forms.len(),
            evaluations.len(),
            "linear_forms.len() != evaluations.len()"
        );
        assert!(
            !linear_forms.is_empty(),
            "zook requires ≥ 1 (form, value) pair"
        );

        // RLC challenge binds the form/value set to the commitment.
        let batching_challenge: F = vs.verifier_message();
        let claim_weights = geometric_sequence(F::ONE, batching_challenge, linear_forms.len());
        let batched_evaluation: F = evaluations
            .iter()
            .zip(&claim_weights)
            .map(|(v, weight)| *v * weight)
            .sum();

        // Basecase-only path: no rounds, evaluate directly.
        if self.rounds().is_empty() {
            let opening =
                self.basecase()
                    .verify(vs, &commitment.irs_commitment, batched_evaluation)?;
            // No constraint terms, no round scalings: linear_forms_contribution = opening.linear_form_evaluation,
            // initial_claim_scale = F::ONE. The caller checks:
            // F::ONE × Σ_j claim_weight_j × form_j.mle_at(evaluation_point) == linear_forms_contribution
            return Ok(FinalClaim {
                evaluation_point: opening.evaluation_points,
                initial_claim_scale: F::ONE,
                linear_forms_contribution: opening.linear_form_evaluation,
                rlc_coefficients: claim_weights,
            });
        }

        // Multi-round path: accumulate constraints implicitly, no initial_covector.
        let mut state = VerifierRoundState {
            irs_commitment: commitment.irs_commitment,
            constraints: Vec::new(),
            all_round_challenges: Vec::new(),
            challenges_at: vec![0],
            round_scale_factors: Vec::new(),
            current_msg_len: self.tuning().vector_size,
            sum: batched_evaluation,
        };
        for round in self.rounds() {
            state = verify_round(round, state, vs)?;
        }

        let opening = self
            .basecase()
            .verify(vs, &state.irs_commitment, state.sum)?;

        // full_eval_point = all round challenges ++ basecase evaluation points.
        let full_eval_point: Vec<F> = state
            .all_round_challenges
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
        let num_completed_rounds = state.round_scale_factors.len();
        let mut scale_suffixes = vec![F::ONE; num_completed_rounds + 1];
        for round_idx in (0..num_completed_rounds).rev() {
            scale_suffixes[round_idx] =
                state.round_scale_factors[round_idx] * scale_suffixes[round_idx + 1];
        }

        // Compute constraint contributions (O(num_constraints × log N)).
        // constraint_sum = Σ_c c.batching_weight × scale_suffixes[c.round+1] × mle_of_geom(c.eval_point, z_suffix)
        let constraint_sum: F = state
            .constraints
            .iter()
            .map(|c| {
                let z_suffix_start = state.challenges_at[c.added_at_round + 1];
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
            evaluation_point: full_eval_point,
            initial_claim_scale: scale_suffixes[0],
            linear_forms_contribution,
            rlc_coefficients: claim_weights,
        })
    }
}

struct VerifierRoundState<F: Field> {
    irs_commitment: IrsCommitment,
    /// Deferred constraint terms from code_switch OOD and in-domain constraints.
    constraints: Vec<ImplicitConstraint<F>>,
    /// All sumcheck round challenges seen so far, in order.
    all_round_challenges: Vec<F>,
    /// `challenges_at[r]` = number of cumulative challenges before round r.
    /// Length = number of rounds processed + 1 (initial entry is 0).
    challenges_at: Vec<usize>,
    /// `original_sl_coeff` from each round's code_switch, in order.
    round_scale_factors: Vec<F>,
    /// Current message length (size of post-fold covector this round).
    current_msg_len: usize,
    sum: F,
}

struct ImplicitConstraint<F: Field> {
    /// OOD alpha or in-domain omega.
    eval_point: F,
    /// RLC coefficient at the time this constraint was added.
    batching_weight: F,
    /// log2 of the effective domain size = log2(msg_len when added).
    /// The final contribution is batching_weight * mle_evaluate(eval_point, z_suffix)
    /// where z_suffix has exactly this many elements.
    domain_bits: u32,
    /// Index into `round_scale_factors`: which round added this constraint.
    /// `scale_suffixes[added_at_round + 1]` is the product of all round_scale_factors
    /// from the round after this one to the end.
    added_at_round: usize,
}

/// Manages ZK mask verification state for one round.
/// Mirrors `RoundMaskOracle` in `prover.rs` on the receive side.
/// `Disabled` is the Null Object for Standard mode — all methods return Ok(()) / &[].
enum RoundMaskOracleCheck<'a, F: Field> {
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
    fn begin<H>(
        round: &'a RoundConfig<Identity<F>>,
        vs: &mut VerifierState<H>,
    ) -> VerificationResult<Self>
    where
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
    fn receive_cs_mask_and_reconcile<H>(
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
    fn verify_and_discharge<H>(
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

#[cfg_attr(feature = "tracing", instrument(skip_all, name = "zook::verify_round", fields(msg_len = round.code_switch().source().message_length())))]
fn verify_round<F, H>(
    round: &RoundConfig<Identity<F>>,
    mut state: VerifierRoundState<F>,
    vs: &mut VerifierState<H>,
) -> VerificationResult<VerifierRoundState<F>>
where
    F: Field + Default + Codec<[H::U]>,
    Standard: Distribution<F>,
    H: DuplexSpongeInterface,
    u8: Decoding<[H::U]>,
    [u8; 32]: Decoding<[H::U]>,
    U64: Codec<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    let msg_len = round.code_switch().source().message_length();

    // Receive sumcheck-masks commitment (ZK) or construct Disabled (Standard).
    let mut masker = RoundMaskOracleCheck::begin(round, vs)?;

    // Sumcheck: mutates sum, records round challenges for full_eval_point reconstruction.
    let opening = round.sumcheck().verify(vs, &mut state.sum)?;
    state
        .all_round_challenges
        .extend_from_slice(&opening.round_challenges);
    state.current_msg_len = msg_len;

    // Receive cs_mask commitment + mask_eval_sum (δ), reconcile sum to the unmasked dot.
    masker.receive_cs_mask_and_reconcile(&opening, vs, &mut state.sum)?;

    // Code-switch: accumulate implicit constraints; no explicit covector update.
    let (target_commitment, update_params) = round.code_switch().verify_for_implicit(
        vs,
        &mut state.sum,
        &opening.round_challenges,
        &state.irs_commitment,
    )?;
    let current_round = state.round_scale_factors.len();
    let domain_bits = msg_len.trailing_zeros();
    for (batching_weight, eval_point) in update_params
        .ood_rlc_coeffs
        .iter()
        .zip(&update_params.ood_eval_points)
    {
        state.constraints.push(ImplicitConstraint {
            eval_point: *eval_point,
            batching_weight: *batching_weight,
            domain_bits,
            added_at_round: current_round,
        });
    }
    for (batching_weight, eval_point) in update_params
        .in_domain_rlc_coeffs
        .iter()
        .zip(&update_params.in_domain_eval_points)
    {
        state.constraints.push(ImplicitConstraint {
            eval_point: *eval_point,
            batching_weight: *batching_weight,
            domain_bits,
            added_at_round: current_round,
        });
    }
    state
        .round_scale_factors
        .push(update_params.original_sl_coeff);
    state.challenges_at.push(state.all_round_challenges.len());
    state.irs_commitment = target_commitment;

    // Verify both mask trees and subtract cs_mask contribution from sum.
    masker.verify_and_discharge(
        &opening.round_challenges,
        msg_len,
        &update_params,
        vs,
        &mut state.sum,
    )?;

    Ok(state)
}
