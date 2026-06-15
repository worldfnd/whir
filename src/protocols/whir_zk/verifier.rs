use ark_ff::Field;
#[cfg(feature = "tracing")]
use tracing::instrument;

use super::{Commitment, Config};
use crate::{
    algebra::linear_form::{Covector, LinearForm},
    hash::Hash,
    protocols::whir_zk::utils::{
        build_blinding_forms, build_combined_and_subproof_claims,
        construct_batched_eq_weights_from_gammas, fold_weight_to_mask_size,
    },
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, VerificationResult,
        VerifierMessage, VerifierState,
    },
    verify,
};

/// Extract the first element of each `(mu+1)`-sized block: these are the
/// `M_eval = <w_folded, m_poly>` values needed for evaluation binding.
#[inline]
fn extract_m_evals<F: Field>(
    w_folded_blinding_evals: &[F],
    num_witness_variables_plus_1: usize,
) -> Vec<F> {
    w_folded_blinding_evals
        .chunks_exact(num_witness_variables_plus_1)
        .map(|block| block[0])
        .collect()
}

impl<F: Field> Config<F> {
    /// Verify a zkWHIR proof against the given evaluation claims.
    ///
    /// Replays the transcript, recomputes the expected blinding claims, and
    /// delegates to the inner witness-side and blinding-side WHIR verifiers.
    #[allow(clippy::too_many_lines)]
    #[cfg_attr(feature = "tracing", instrument(skip_all, name = "whir_zk::verify"))]
    pub fn verify<H>(
        &self,
        verifier_state: &mut VerifierState<'_, H>,
        weights: &[&dyn LinearForm<F>],
        evaluations: &[F],
        commitment: &Commitment<F>,
    ) -> VerificationResult<()>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(
            self.blinded_commitment.initial_committer.num_vectors(),
            1,
            "zkWHIR currently expects one vector per commitment"
        );
        let num_polynomials = commitment.f_hat.len();
        verify!(evaluations.len() == weights.len() * num_polynomials);

        let blinding_challenge: F = verifier_state.verifier_message();

        let num_witness_variables = self.num_witness_variables();
        let num_blinding_variables = self.num_blinding_variables();
        let num_witness_variables_plus_1 = num_witness_variables + 1;
        let num_w_folded_evals = weights.len() * num_polynomials * num_witness_variables_plus_1;
        let w_folded_blinding_evals: Vec<F> =
            verifier_state.prover_messages_vec(num_w_folded_evals)?;

        let m_evals = extract_m_evals(&w_folded_blinding_evals, num_witness_variables_plus_1);

        let masking_challenge: F = verifier_state.verifier_message();
        verify!(masking_challenge != F::ZERO);
        let commitments = commitment.f_hat.iter().collect::<Vec<_>>();
        let irs_commitments = commitments.iter().map(|c| &c.irs).collect::<Vec<_>>();
        let initial_in_domain = self
            .blinded_commitment
            .initial_committer
            .verify(verifier_state, &irs_commitments)?;

        // Expand base queries into coset points for the first folding round.
        let h_gammas = self.all_gammas(&initial_in_domain.points);
        let tau1: F = verifier_state.verifier_message();
        let tau2: F = verifier_state.verifier_message();
        let mut m_claims = vec![F::ZERO; num_polynomials];
        let mut g_hat_claims_per_poly = vec![vec![F::ZERO; num_witness_variables]; num_polynomials];
        let mut expected_batched_h_claims = vec![F::ZERO; num_polynomials];
        let mut tau2_power = F::ONE;
        for &gamma in &h_gammas {
            for poly_idx in 0..num_polynomials {
                // Parse (m_eval, g_hat_evals) per polynomial.
                let m_eval: F = verifier_state.prover_message()?;
                let mut h_value = m_eval;
                let mut blinding_power = blinding_challenge;
                let mut gamma_power = gamma;
                m_claims[poly_idx] += tau2_power * m_eval;
                for g_hat_claim in &mut g_hat_claims_per_poly[poly_idx] {
                    let g_hat_eval: F = verifier_state.prover_message()?;
                    *g_hat_claim += tau2_power * g_hat_eval;
                    h_value += blinding_power * gamma_power * g_hat_eval;
                    blinding_power *= blinding_challenge;
                    gamma_power = gamma_power.square();
                }
                expected_batched_h_claims[poly_idx] += tau2_power * h_value;
            }
            tau2_power *= tau2;
        }
        let combined_claims: Vec<F> = verifier_state.prover_messages_vec(num_polynomials)?;
        let batched_h_claims: Vec<F> = verifier_state.prover_messages_vec(num_polynomials)?;
        let modified_evaluations: Vec<F> = evaluations
            .iter()
            .zip(m_evals.iter())
            .map(|(&eval, &m_eval)| eval + m_eval)
            .collect();

        self.blinded_commitment
            .verify(verifier_state, &commitments, &modified_evaluations)?
            .verify(weights.iter().copied())?;

        verify!(batched_h_claims == expected_batched_h_claims);
        let g_hat_slices: Vec<&[F]> = g_hat_claims_per_poly.iter().map(Vec::as_slice).collect();
        let (expected_combined_claims, expected_batched_blinding_subproof_claims) =
            build_combined_and_subproof_claims(&m_claims, &g_hat_slices, tau1);
        verify!(combined_claims == expected_combined_claims);

        // eq_weights and weight folding are independent — overlap them.
        #[cfg(feature = "parallel")]
        let (beq_weights, w_folded_weights) = {
            use rayon::prelude::*;
            rayon::join(
                || {
                    construct_batched_eq_weights_from_gammas(
                        &h_gammas,
                        masking_challenge,
                        tau2,
                        num_blinding_variables,
                    )
                },
                || {
                    weights
                        .par_iter()
                        .map(|&w| {
                            fold_weight_to_mask_size(
                                w,
                                num_witness_variables,
                                num_blinding_variables,
                            )
                        })
                        .collect::<Vec<Covector<F>>>()
                },
            )
        };
        #[cfg(not(feature = "parallel"))]
        let (beq_weights, w_folded_weights) = {
            let beq = construct_batched_eq_weights_from_gammas(
                &h_gammas,
                masking_challenge,
                tau2,
                num_blinding_variables,
            );
            let folds: Vec<Covector<F>> = weights
                .iter()
                .map(|&w| {
                    fold_weight_to_mask_size(w, num_witness_variables, num_blinding_variables)
                })
                .collect();
            (beq, folds)
        };
        let blinding_forms = build_blinding_forms(&beq_weights, &w_folded_weights);
        let all_expected_blinding_claims = [
            &expected_batched_blinding_subproof_claims[..],
            &w_folded_blinding_evals[..],
        ]
        .concat();
        let blinding_commitments = vec![&commitment.blinding];
        self.blinding_commitment
            .verify(
                verifier_state,
                &blinding_commitments,
                &all_expected_blinding_claims,
            )?
            .verify(blinding_forms)?;

        Ok(())
    }
}
