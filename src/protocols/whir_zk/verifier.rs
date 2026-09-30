use ark_ff::FftField;
#[cfg(feature = "tracing")]
use tracing::instrument;

use super::{
    utils::{
        build_beq_tables_with_initial_ood, build_weight_covectors, compute_eq_weights,
        gamma_to_f_hat_indices, LambdaAccumulator, ProtocolDims,
    },
    Config,
};
use crate::{
    algebra::{
        dot,
        embedding::Identity,
        geometric_sequence,
        linear_form::{Covector, Evaluate, LinearForm, MultilinearExtension, UnivariateEvaluation},
        tensor_product, MultilinearPoint,
    },
    hash::Hash,
    protocols::{geometric_challenge::geometric_challenge, whir},
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, VerificationResult,
        VerifierMessage, VerifierState,
    },
    utils::zip_strict,
    verify,
};

/// Commitments for blinded and blinding polynomials.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitments<F: FftField> {
    pub(crate) blinded_commitment: whir::Commitment<F>,
    pub(crate) blinding_commitment: whir::Commitment<F>,
}

/// Intermediate result from verifying the blinded polynomial (Steps 2-6).
///
/// Carries the values needed by [`Config::verify_blinding_polynomial`] (Step 7)
/// and the blinded polynomial's deferred `FinalClaim`.
#[must_use]
#[derive(Debug)]
struct BlindedVerifyResult<F: FftField> {
    lambda: LambdaAccumulator<F>,
    eq_weights: Vec<F>,
    rho: F,
    alpha_coeffs: Vec<F>,
    dims: ProtocolDims,
    /// Deferred final claim for the blinded polynomial instance.
    blinded_final_claim: whir::FinalClaim<F>,
}

/// Result of Steps 2-4 (blinding claims, batching, combined claims, initial sumcheck).
#[must_use]
#[derive(Debug)]
struct VerifyPrepareResult<F> {
    beta_powers: Vec<F>,
    constraint_rlc_coeffs: Vec<F>,
    the_sum: F,
    initial_lambda: LambdaAccumulator<F>,
    rho: F,
    alpha_coeffs: Vec<F>,
    folding_randomness: MultilinearPoint<F>,
    eq_weights: Vec<F>,
    batching_weights: Vec<F>,
}

/// Result of Step 5 (OOD/STIR queries, remaining WHIR rounds, linear form RLC).
///
/// Includes all state forwarded from [`VerifyPrepareResult`] that Step 6 needs,
/// so that `ood_stir_and_rounds` consumes `prepare` by value — preventing
/// accidental reads of `Default`-valued fields after `mem::take`.
#[must_use]
#[derive(Debug)]
struct VerifyOodStirResult<F> {
    lambda: LambdaAccumulator<F>,
    gamma_points: Vec<F>,
    gamma_h_values: Vec<F>,
    /// Evaluation point for the blinded polynomial's deferred linear form check.
    evaluation_point: Vec<F>,
    /// RLC coefficients for the blinded polynomial's constraint combination.
    constraint_rlc_coeffs: Vec<F>,
    /// Claimed RLC value of the linear form MLEs at `evaluation_point`.
    linear_form_rlc: F,
    // Forwarded from `VerifyPrepareResult` for `gamma_check` (Step 6).
    batching_weights: Vec<F>,
    beta_powers: Vec<F>,
    rho: F,
    eq_weights: Vec<F>,
    alpha_coeffs: Vec<F>,
}

/// Context for verifying the blinded polynomial (Steps 2-6).
///
/// Bundles the constant context shared across protocol steps so that each step
/// method only needs its step-specific arguments.
struct BlindedVerifyCtx<'a, 'vs, F: FftField, H: DuplexSpongeInterface> {
    config: &'a Config<F>,
    verifier_state: &'a mut VerifierState<'vs, H>,
    commitments: &'a Commitments<F>,
    dims: ProtocolDims,
}

impl<F, H> BlindedVerifyCtx<'_, '_, F, H>
where
    F: FftField + Codec<[H::U]>,
    H: DuplexSpongeInterface,
    u8: Decoding<[H::U]>,
    [u8; 32]: Decoding<[H::U]>,
    U64: Codec<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    /// Steps 2-4: Blinding claims, multi-polynomial batching, combined claims, initial sumcheck.
    fn prepare_and_sumcheck(
        &mut self,
        weights: &[&dyn LinearForm<F>],
        evaluations: &[F],
    ) -> VerificationResult<VerifyPrepareResult<F>> {
        let num_vectors = self.dims.num_vectors;
        let num_g_polys = self.dims.num_g_polys();
        let num_forms = weights.len();
        assert_eq!(evaluations.len(), num_forms * num_vectors);

        // =====================================================================
        // Step 2: Blinding Polynomial Claim Generation
        //
        // V → P: β ←$ F_q
        // P → V: G_j = ⟨w_j, g⟩ where g(x̄) = Σ βⁱ·ĝᵢ(Φᵢ(x̄))
        // =====================================================================
        let beta: F = self.verifier_state.verifier_message();
        let beta_powers = geometric_sequence(beta, num_g_polys);
        let g_claims: Vec<F> = self.verifier_state.prover_messages_vec(num_forms)?;

        // =====================================================================
        // Step 2.5: Multi-polynomial batching
        //
        // V → P: α ←$ F_q (for n > 1; when n = 1, α = [1])
        // =====================================================================
        let alpha_coeffs: Vec<F> = geometric_challenge(self.verifier_state, 0, num_vectors);

        // =====================================================================
        // Step 3: Preparation for WHIR Sumcheck Rounds
        //
        // V → P: ρ ←$ F_q \ {0}
        // V computes combined_claim_j = ρ · Σᵢ αⁱ·eval[j,i] + G_j
        // =====================================================================
        let rho: F = self.verifier_state.verifier_message();
        verify!(rho != F::ZERO);

        let combined_claims: Vec<F> = (0..num_forms)
            .map(|j| {
                let row = &evaluations[j * num_vectors..(j + 1) * num_vectors];
                let combined_eval: F = alpha_coeffs.iter().zip(row).map(|(&a, &e)| a * e).sum();
                rho * combined_eval + g_claims[j]
            })
            .collect();

        let initial_ood = self.commitments.blinded_commitment.out_of_domain();
        let mut initial_lambda = LambdaAccumulator::new();
        let mut initial_ood_evals = Vec::with_capacity(initial_ood.points.len());
        for (&z, f_hat_values) in initial_ood
            .points
            .iter()
            .zip(initial_ood.matrix.chunks_exact(num_vectors))
        {
            let m_evals: Vec<F> = self.verifier_state.prover_messages_vec(num_vectors)?;
            let g_evals: Vec<F> = self.verifier_state.prover_messages_vec(self.dims.nu)?;
            let blinding_eval: F = m_evals.iter().copied().sum::<F>()
                + g_evals
                    .iter()
                    .zip(beta_powers.iter().skip(1))
                    .map(|(&claim, &beta)| beta * claim)
                    .sum::<F>();
            initial_ood_evals.push(rho * dot(&alpha_coeffs, f_hat_values) + blinding_eval);
            initial_lambda.push(z, m_evals, g_evals);
        }

        // =====================================================================
        // Step 4: WHIR Initial Round — sumcheck on f_zk
        //
        // P ↔ V: s-round sumcheck yielding folding randomness r̄
        // =====================================================================
        let constraint_rlc_coeffs: Vec<F> =
            geometric_challenge(self.verifier_state, 0, num_forms + initial_ood.points.len());
        let mut the_sum: F = constraint_rlc_coeffs[..num_forms]
            .iter()
            .zip(combined_claims.iter())
            .map(|(&c, &v)| c * v)
            .sum::<F>()
            + dot(&constraint_rlc_coeffs[num_forms..], &initial_ood_evals);

        let folding_randomness = self
            .config
            .blinded_polynomial
            .initial_sumcheck
            .verify(self.verifier_state, &mut the_sum)?;

        let eq_weights = compute_eq_weights(&folding_randomness.0);
        let batching_weights = tensor_product(&alpha_coeffs, &eq_weights);

        Ok(VerifyPrepareResult {
            beta_powers,
            constraint_rlc_coeffs,
            the_sum,
            initial_lambda,
            rho,
            alpha_coeffs,
            folding_randomness,
            eq_weights,
            batching_weights,
        })
    }

    /// Step 5: Virtual OOD and STIR queries, remaining WHIR rounds, linear form RLC check.
    ///
    /// Consumes `prepare` by value so that gamma_check receives all fields
    /// through the returned [`VerifyOodStirResult`] rather than reading a
    /// partially-hollowed struct.
    #[allow(clippy::too_many_lines)]
    fn ood_stir_and_rounds(
        &mut self,
        mut prepare: VerifyPrepareResult<F>,
        weights: &[&dyn LinearForm<F>],
    ) -> VerificationResult<VerifyOodStirResult<F>> {
        let nu = self.dims.nu;
        let num_vectors = self.dims.num_vectors;

        // =====================================================================
        // Step 5: Virtual OOD and STIR Queries
        //
        // V receives [[H]], then queries OOD/STIR points.
        // For each point z: reads ood_f̂, n m_evals, ν g_evals.
        // V reconstructs: f_zk(z) = ρ·fold(r̄, [[f̂]])(z) + Σ m_evals + Σ βⁱ·g_evals
        // All claims accumulate into Λ for Step 7.
        // =====================================================================
        let round_config = &self.config.blinded_polynomial.round_configs[0];
        let commitment_h = round_config
            .irs_committer
            .receive_commitment(self.verifier_state)?;
        round_config.pow.verify(self.verifier_state)?;
        let in_domain = self
            .config
            .blinded_polynomial
            .initial_committer
            .verify(self.verifier_state, &[&self.commitments.blinded_commitment])?;

        let mut lambda = prepare.initial_lambda;

        // --- 5d: OOD responses ---
        let one_weight = [F::ONE];
        // Collect is necessary: the iterator borrows commitment_h, which is reused below.
        #[allow(clippy::needless_collect)]
        let ood_h_evals: Vec<F> = commitment_h.out_of_domain().values(&one_weight).collect();

        for &z in &commitment_h.out_of_domain().points {
            // SECURITY
            // Consumed for Fiat-Shamir transcript binding only — NOT directly
            // verified. The prover sends MLE(f̂_combined, fold_args(r̄, z)),
            // which differs from the univariate fold evaluation fold(r̄, f̂)(z).
            // Soundness: the STIR constraint accumulation (Step 5f) binds [[H]]
            // evaluations via the sumcheck chain, and the Γ consistency check
            // (Step 6) verifies fold(r̄, [[f̂]])(γ) at FRI query indices.
            // Together these ensure the prover cannot forge the fold without
            // detection.
            let _ood_f_hat: F = self.verifier_state.prover_message()?;
            let m_evals: Vec<F> = self.verifier_state.prover_messages_vec(num_vectors)?;
            let g_evals: Vec<F> = self.verifier_state.prover_messages_vec(nu)?;
            lambda.push(z, m_evals, g_evals);
        }

        // --- 5e: In-domain blinding claims ---
        // Collected separately for STIR reconstruction, then moved into Λ.
        let mut in_domain_m_evals = Vec::with_capacity(in_domain.points.len());
        let mut in_domain_g_evals = Vec::with_capacity(in_domain.points.len());
        for &_z in &in_domain.points {
            let m_evals: Vec<F> = self.verifier_state.prover_messages_vec(num_vectors)?;
            let g_evals: Vec<F> = self.verifier_state.prover_messages_vec(nu)?;
            in_domain_m_evals.push(m_evals);
            in_domain_g_evals.push(g_evals);
        }

        // --- 5e': Reconstruct virtual STIR response ---
        // stir_{ρ·f+g}(z_j) = ρ·fold(r̄, [[f̂]])(z_j) + stir_{M_ρ}(z_j) + Σ βⁱ·stir_{ĝᵢ}(z_j)
        // fold(r̄, [[f̂_combined]])(z) uses tensor_product(α, eq_weights) for batching.
        let f_zk_in_domain_evals: Vec<F> = in_domain
            .values(&prepare.batching_weights)
            .zip(in_domain_m_evals.iter().zip(in_domain_g_evals.iter()))
            .map(|(f_hat_combined_fold, (m_evals, g_evals))| {
                // m_combined includes the g₀ contribution (absorbed into masking_coeffs_all[0]),
                // so g_evals here are g₁..g_ν, weighted by β¹..β^ν (not β⁰).
                let m_combined: F = m_evals.iter().copied().sum();
                let g_sum: F = g_evals
                    .iter()
                    .enumerate()
                    .map(|(i, &g)| prepare.beta_powers[i + 1] * g)
                    .sum();
                prepare.rho * f_hat_combined_fold + m_combined + g_sum
            })
            .collect();

        // Move in-domain claims into Λ (after reconstruction is done)
        for ((z, m), g) in in_domain
            .points
            .iter()
            .zip(in_domain_m_evals)
            .zip(in_domain_g_evals)
        {
            lambda.push(*z, m, g);
        }

        // --- 5f: STIR constraint accumulation ---
        let stir_challenges: Vec<UnivariateEvaluation<F>> = commitment_h
            .out_of_domain()
            .evaluators(round_config.initial_size())
            .chain(in_domain.evaluators(round_config.initial_size()))
            .collect();

        let stir_evaluations: Vec<F> = ood_h_evals
            .into_iter()
            .chain(f_zk_in_domain_evals)
            .collect();

        let stir_rlc_coeffs: Vec<F> =
            geometric_challenge(self.verifier_state, 1, stir_challenges.len());
        prepare.the_sum += dot(&stir_rlc_coeffs, &stir_evaluations);

        let mut round_constraints: Vec<(Vec<F>, Vec<UnivariateEvaluation<F>>)> =
            vec![(stir_rlc_coeffs, stir_challenges)];

        // --- 5g: Round 0 sumcheck ---
        let folding_randomness = round_config
            .sumcheck
            .verify(self.verifier_state, &mut prepare.the_sum)?;
        let mut round_folding_randomness = vec![prepare.folding_randomness];

        // =====================================================================
        // Step 5 (continued): Remaining standard WHIR rounds
        // =====================================================================
        let remaining = whir::rounds::verify_remaining_rounds(
            &self.config.blinded_polynomial.round_configs,
            &whir::rounds::FinalRoundConfig {
                sumcheck: &self.config.blinded_polynomial.final_sumcheck,
                pow: &self.config.blinded_polynomial.final_pow,
            },
            self.verifier_state,
            &mut prepare.the_sum,
            &commitment_h,
            &folding_randomness,
        )?;
        round_folding_randomness.push(folding_randomness);

        // Compute gamma_h_values from first in-domain opening, then move points out.
        let msg_len = round_config.irs_committer.message_length();
        let interleaving_depth = round_config.irs_committer.interleaving_depth;
        let gamma_h_values: Vec<F> = remaining
            .first_in_domain
            .points
            .iter()
            .zip(remaining.first_in_domain.rows())
            .map(|(&gamma, row)| {
                let gamma_step = gamma.pow([msg_len as u64]);
                let mut gamma_pow = F::ONE;
                let mut val = F::ZERO;
                for &r in row.iter().take(interleaving_depth) {
                    val += gamma_pow * r;
                    gamma_pow *= gamma_step;
                }
                val
            })
            .collect();
        let gamma_points = remaining.first_in_domain.points;

        round_constraints.extend(remaining.round_constraints);
        round_folding_randomness.extend(remaining.round_folding_randomness);
        round_folding_randomness.push(remaining.final_sumcheck_randomness.clone());

        // =====================================================================
        // Compute linear form RLC from the sumcheck chain.
        //
        // The blinded polynomial's FinalClaim is built here and returned to
        // the caller (via verify → verify_blinded_polynomial) for deferred
        // verification, matching the base WHIR pattern.
        // =====================================================================
        let evaluation_point: Vec<F> = round_folding_randomness
            .into_iter()
            .flat_map(|p| p.0.into_iter())
            .collect();

        let poly_eval = MultilinearExtension::new(remaining.final_sumcheck_randomness.0)
            .evaluate(&Identity::new(), &remaining.final_vector);
        verify!(poly_eval != F::ZERO);
        let mut linear_form_rlc = prepare.the_sum / poly_eval;

        for (idx, (rlc_coeffs, stir_weights)) in round_constraints.into_iter().enumerate() {
            let num_variables =
                self.config.blinded_polynomial.round_configs[idx].initial_num_variables();
            let start = evaluation_point.len().saturating_sub(num_variables);
            for (coeff, weight) in zip_strict(rlc_coeffs, stir_weights) {
                linear_form_rlc -= coeff * weight.mle_evaluate(&evaluation_point[start..]);
            }
        }

        // Inline linear form RLC check (blinded polynomial FinalClaim).
        // Also returned to the caller for deferred verification.
        let expected_forms: F = prepare.constraint_rlc_coeffs[..weights.len()]
            .iter()
            .zip(weights.iter())
            .map(|(&c, w)| c * w.mle_evaluate(&evaluation_point))
            .sum();
        let initial_ood_points = &self.commitments.blinded_commitment.out_of_domain().points;
        let expected_ood: F = prepare.constraint_rlc_coeffs[weights.len()..]
            .iter()
            .zip(initial_ood_points)
            .map(|(&c, &z)| {
                c * UnivariateEvaluation::new(z, self.dims.size).mle_evaluate(&evaluation_point)
            })
            .sum();
        let expected_rlc = expected_forms + expected_ood;
        verify!(expected_rlc == linear_form_rlc);

        Ok(VerifyOodStirResult {
            lambda,
            gamma_points,
            gamma_h_values,
            evaluation_point,
            constraint_rlc_coeffs: prepare.constraint_rlc_coeffs[..weights.len()].to_vec(),
            linear_form_rlc: linear_form_rlc - expected_ood,
            batching_weights: prepare.batching_weights,
            beta_powers: prepare.beta_powers,
            rho: prepare.rho,
            eq_weights: prepare.eq_weights,
            alpha_coeffs: prepare.alpha_coeffs,
        })
    }

    /// Step 6: Verifier Consistency Check — Γ point decomposition.
    fn gamma_check(
        &mut self,
        mut ood: VerifyOodStirResult<F>,
    ) -> VerificationResult<BlindedVerifyResult<F>> {
        let num_vectors = self.dims.num_vectors;
        let nu = self.dims.nu;

        // =====================================================================
        // Step 6: Verifier Consistency Check
        //
        // V computes Γ = {γ₁..γ_q} ⊆ Ω₁ (FRI query indices).
        // V locally folds: F̂_r̄(γ) = fold_k(r̄, [[f̂]])[γ]
        // V checks: [[L]](γ) = ρ·F̂_r̄(γ) + m̃ + Σ βⁱ·g̃ᵢ == [[H]](γ)
        // =====================================================================
        let gamma_f_hat_indices = gamma_to_f_hat_indices(&ood.gamma_points, self.config);

        let gamma_f_hat_evals = self
            .config
            .blinded_polynomial
            .initial_committer
            .verify_at_indices(
                self.verifier_state,
                &[&self.commitments.blinded_commitment],
                &gamma_f_hat_indices,
            )?;

        // fold(r̄, [[f̂_combined]])(γ) uses tensor_product(α, eq_weights)
        let f_hat_fold_at_gamma: Vec<F> = gamma_f_hat_evals.values(&ood.batching_weights).collect();

        // Read blinding claims and check decomposition
        for (idx, &gamma) in ood.gamma_points.iter().enumerate() {
            let m_evals: Vec<F> = self.verifier_state.prover_messages_vec(num_vectors)?;
            let g_evals: Vec<F> = self.verifier_state.prover_messages_vec(nu)?;

            // g₀ is absorbed into m_combined; g_evals are g₁..g_ν with β¹..β^ν.
            let m_combined: F = m_evals.iter().copied().sum();
            let g_sum: F = g_evals
                .iter()
                .enumerate()
                .map(|(i, &g)| ood.beta_powers[i + 1] * g)
                .sum();
            let l_gamma = ood.rho * f_hat_fold_at_gamma[idx] + m_combined + g_sum;
            verify!(l_gamma == ood.gamma_h_values[idx]);

            ood.lambda.push(gamma, m_evals, g_evals);
        }

        Ok(BlindedVerifyResult {
            lambda: ood.lambda,
            eq_weights: ood.eq_weights,
            rho: ood.rho,
            alpha_coeffs: ood.alpha_coeffs,
            dims: self.dims,
            blinded_final_claim: whir::FinalClaim {
                evaluation_point: ood.evaluation_point,
                rlc_coefficients: ood.constraint_rlc_coeffs,
                linear_form_rlc: ood.linear_form_rlc,
            },
        })
    }
}

impl<F: FftField> Config<F> {
    /// Receive the two commitments (blinded polynomial and blinding polynomial)
    /// from the transcript. This mirrors the commit phase in the prover.
    pub fn receive_commitments<H>(
        &self,
        verifier_state: &mut VerifierState<'_, H>,
    ) -> VerificationResult<Commitments<F>>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let blinded_commitment = self.blinded_polynomial.receive_commitment(verifier_state)?;
        let blinding_commitment = self
            .blinding_polynomial
            .receive_commitment(verifier_state)?;
        Ok(Commitments {
            blinded_commitment,
            blinding_commitment,
        })
    }

    /// Steps 2-6: Verify the blinded polynomial instance.
    ///
    /// Reads blinding claims (Step 2), reconstructs combined claims (Steps 2.5-3),
    /// runs the initial sumcheck (Step 4), verifies OOD/STIR/remaining rounds (Step 5),
    /// verifies the linear form RLC, and checks Γ consistency (Step 6).
    fn verify_blinded_polynomial<H>(
        &self,
        verifier_state: &mut VerifierState<'_, H>,
        weights: &[&dyn LinearForm<F>],
        evaluations: &[F],
        commitments: &Commitments<F>,
        protocol_dims: ProtocolDims,
    ) -> VerificationResult<BlindedVerifyResult<F>>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let mut ctx = BlindedVerifyCtx {
            config: self,
            verifier_state,
            commitments,
            dims: protocol_dims,
        };
        let prepare = ctx.prepare_and_sumcheck(weights, evaluations)?;
        let ood = ctx.ood_stir_and_rounds(prepare, weights)?;
        ctx.gamma_check(ood)
    }

    /// Step 7: Batched Proof on Blinding Polynomials.
    ///
    /// V → P: τ ←$ F_q (batching randomness)
    /// Both sides build beq tables and weight covectors wᵢ.
    /// V reads E[i][j] = ⟨wᵢ, vⱼ⟩ and checks diagonal against Λ claims:
    ///   E[i][i] = Σ_p τ^{p+1} · B[p][i]
    /// Then run second WHIR instance to verify batch opening.
    #[allow(clippy::needless_pass_by_value)]
    fn verify_blinding_polynomial<H>(
        &self,
        verifier_state: &mut VerifierState<'_, H>,
        commitments: &Commitments<F>,
        blinded: BlindedVerifyResult<F>,
    ) -> VerificationResult<whir::FinalClaim<F>>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let dims = blinded.dims;
        let blinded_final_claim = blinded.blinded_final_claim;
        let tau: F = verifier_state.verifier_message();

        let beq_tables = build_beq_tables_with_initial_ood(
            blinded.lambda.z_points(),
            self.blinded_polynomial.initial_committer.out_domain_samples,
            &blinded.eq_weights,
            tau,
            dims,
        );

        let weight_covectors =
            build_weight_covectors(&beq_tables, blinded.rho, &blinded.alpha_coeffs, dims);

        // Read eval_matrix from transcript
        let eval_matrix: Vec<F> =
            verifier_state.prover_messages_vec(dims.num_blinding_vecs * dims.num_blinding_vecs)?;

        // Verify diagonal: E[i][i] = Σ_p τ^{p+1} · claim_i_p
        let num_lambda = blinded.lambda.len();
        for i in 0..dims.num_blinding_vecs {
            let mut expected = F::ZERO;
            let mut tau_power = tau;
            for lambda_idx in 0..num_lambda {
                expected += tau_power * blinded.lambda.claim(lambda_idx, i, dims.num_vectors);
                tau_power *= tau;
            }
            verify!(eval_matrix[i * dims.num_blinding_vecs + i] == expected);
        }

        // Package weight covectors as LinearForm trait objects
        let blinding_forms: Vec<Box<dyn LinearForm<F>>> = weight_covectors
            .into_iter()
            .map(|cv| Box::new(Covector::new(cv)) as Box<dyn LinearForm<F>>)
            .collect();

        // Run blinding WHIR verifier.
        self.blinding_polynomial
            .verify(
                verifier_state,
                &[&commitments.blinding_commitment],
                &eval_matrix,
            )?
            .verify(
                blinding_forms
                    .iter()
                    .map(|l| l.as_ref() as &dyn LinearForm<F>),
            )?;

        Ok(blinded_final_claim)
    }

    /// zkWHIR 2.0 verifier — Alternative Randomness Sampling.
    ///
    /// Executes Steps 2-7 of the protocol.
    /// `evaluations` is row-major: `evaluations[j * n + i]` = ⟨wⱼ, fᵢ⟩.
    ///
    /// Returns a [`FinalClaim`](whir::FinalClaim) for the blinded polynomial instance.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn verify<H>(
        &self,
        verifier_state: &mut VerifierState<'_, H>,
        weights: &[&dyn LinearForm<F>],
        evaluations: &[F],
        commitments: &Commitments<F>,
    ) -> VerificationResult<whir::FinalClaim<F>>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let protocol_dims = ProtocolDims::new(self, commitments.blinded_commitment.num_vectors());
        let blinded = self.verify_blinded_polynomial(
            verifier_state,
            weights,
            evaluations,
            commitments,
            protocol_dims,
        )?;
        self.verify_blinding_polynomial(verifier_state, commitments, blinded)
    }
}
