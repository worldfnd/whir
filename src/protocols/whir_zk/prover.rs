use std::borrow::Cow;

use ark_ff::FftField;
use ark_std::rand::{CryptoRng, RngCore};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "tracing")]
use tracing::instrument;

use super::{
    utils::{
        build_beq_tables, build_fold_args, build_weight_covectors, compute_eq_weights,
        compute_rs_fold_blinding_coeffs, gamma_to_f_hat_indices, ProtocolDims, RsFoldCoeffs,
    },
    Config,
};
#[cfg(feature = "parallel")]
use crate::utils::workload_size;
use crate::{
    algebra::{
        dot,
        embedding::Identity,
        geometric_sequence,
        linear_form::{Covector, Evaluate, LinearForm, UnivariateEvaluation},
        multilinear_extend, univariate_evaluate, MultilinearPoint,
    },
    hash::Hash,
    protocols::{
        geometric_challenge::geometric_challenge, irs_commit, whir, whir_zk::committer::Witness,
    },
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState,
        VerifierMessage,
    },
};

/// Send m̃ and ĝ_i evaluations at a single point z.
///
/// Used at OOD, STIR, and Γ points in Steps 5-6.
fn send_blinding_evals<F, H, R>(
    prover_state: &mut ProverState<H, R>,
    z: F,
    masking_coeffs_all: &[Vec<F>],
    g_i_coeffs: &[Vec<F>],
) where
    F: FftField + Codec<[H::U]>,
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
{
    for m_coeffs in masking_coeffs_all {
        let m_eval = univariate_evaluate(m_coeffs, z);
        prover_state.prover_message(&m_eval);
    }
    for g_coeffs in g_i_coeffs {
        let g_eval = univariate_evaluate(g_coeffs, z);
        prover_state.prover_message(&g_eval);
    }
}

/// Intermediate result from proving the blinded polynomial (Steps 2-6).
///
/// Carries the values needed by Step 7 (blinding polynomial proof).
#[must_use]
#[derive(Debug)]
struct BlindedProveResult<F> {
    lambda_z_points: Vec<F>,
    eq_weights: Vec<F>,
    rho: F,
    alpha_coeffs: Vec<F>,
    dims: ProtocolDims,
}

/// Result of Steps 2-4 (blinding claims, batching, f_zk formation, initial sumcheck).
#[must_use]
#[derive(Debug)]
struct PrepareResult<F> {
    f_zk: Vec<F>,
    covector: Vec<F>,
    the_sum: F,
    rho: F,
    alpha_coeffs: Vec<F>,
    folding_randomness: MultilinearPoint<F>,
}

/// Result of Step 5 (OOD/STIR queries and remaining WHIR rounds).
#[must_use]
#[derive(Debug)]
struct OodStirResult<F> {
    lambda_z_points: Vec<F>,
    eq_weights: Vec<F>,
    masking_coeffs_all: Vec<Vec<F>>,
    g_i_coeffs: Vec<Vec<F>>,
    gamma_points: Vec<F>,
}

/// Shared context for proving the blinded polynomial (Steps 2-6).
///
/// Bundles config, transcript state, and protocol dimensions so that each
/// step method only needs its step-specific arguments.
struct BlindedProveCtx<'a, F: FftField, H: DuplexSpongeInterface, R: RngCore + CryptoRng> {
    config: &'a Config<F>,
    prover_state: &'a mut ProverState<H, R>,
    dims: ProtocolDims,
}

impl<F, H, R> BlindedProveCtx<'_, F, H, R>
where
    F: FftField + Codec<[H::U]>,
    H: DuplexSpongeInterface<U = u8>,
    R: RngCore + CryptoRng,
    [u8; 32]: Decoding<[H::U]>,
    U64: Codec<[H::U]>,
    u8: Decoding<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    /// Steps 2-4: Blinding claims, multi-polynomial batching, form f_zk, initial sumcheck.
    #[allow(clippy::too_many_lines)]
    fn prepare_and_sumcheck(
        &mut self,
        vectors: Vec<Cow<'_, [F]>>,
        g_polys: &[Vec<F>],
        linear_forms: &[Box<dyn LinearForm<F>>],
        evaluations: &[F],
    ) -> PrepareResult<F> {
        let num_vectors = self.dims.num_vectors;
        let num_forms = linear_forms.len();
        let size = self.dims.size;

        // =====================================================================
        // Step 2: Blinding Polynomial Claim Generation
        //
        // V → P: β ←$ F_q
        // P constructs g(x̄) = Σᵢ₌₀^ν βⁱ · ĝᵢ(Φᵢ(x̄))
        // P → V: G_j = ⟨w_j, g⟩ for each linear form w_j
        // =====================================================================
        let beta: F = self.prover_state.verifier_message();
        let beta_powers = geometric_sequence(beta, self.dims.num_g_polys());

        let compute_g = |hypercube_idx: usize| -> F {
            let mut sum = F::ZERO;
            for (i, &beta_pow) in beta_powers.iter().enumerate() {
                let idx = self.dims.phi_i_bits(hypercube_idx, i);
                sum += beta_pow * g_polys[i][idx];
            }
            sum
        };

        #[cfg(feature = "parallel")]
        let g_poly: Vec<F> = if size > workload_size::<F>() {
            (0..size).into_par_iter().map(compute_g).collect()
        } else {
            (0..size).map(compute_g).collect()
        };

        #[cfg(not(feature = "parallel"))]
        let g_poly: Vec<F> = (0..size).map(compute_g).collect();

        // G_j = ⟨w_j, g⟩ for each linear form (g is shared across all witnesses)
        let g_claims: Vec<F> = {
            let mut buf = vec![F::ZERO; size];
            let mut claims = Vec::with_capacity(linear_forms.len());
            for w in linear_forms {
                buf.fill(F::ZERO);
                w.accumulate(&mut buf, F::ONE);
                claims.push(dot(&buf, &g_poly));
            }
            claims
        };

        for g_claim in &g_claims {
            self.prover_state.prover_message(g_claim);
        }

        // =====================================================================
        // Step 2.5: Multi-polynomial batching
        //
        // V → P: α ←$ F_q (for n > 1; when n = 1, α = [1] with no transcript cost)
        // Used to form f_combined = Σ αⁱ fᵢ before applying ρ.
        // =====================================================================
        let alpha_coeffs: Vec<F> = geometric_challenge(self.prover_state, 0, num_vectors);

        // =====================================================================
        // Step 3: Preparation for WHIR Sumcheck Rounds
        //
        // V → P: ρ ←$ F_q \ {0}
        // P forms f_zk(x̄) = ρ · f(x̄) + g(x̄)
        // and proves: ρ·F + G = Σ_{b̄} w(f_zk(b̄), b̄)
        // =====================================================================
        let rho: F = self.prover_state.verifier_message();
        assert!(
            rho != F::ZERO,
            "rho must not be zero (negligible probability)"
        );

        // f_combined = Σ αⁱ fᵢ, then f_zk = ρ·f_combined + g
        let mut f_zk = {
            let mut iter = vectors.into_iter();
            let mut combined = iter.next().expect("vectors must be non-empty").into_owned();
            // alpha_coeffs[0] = ONE, so combined starts as vectors[0]
            for (vec_i, &alpha) in iter.zip(alpha_coeffs[1..].iter()) {
                for (f, v) in combined.iter_mut().zip(vec_i.iter()) {
                    *f += alpha * *v;
                }
            }
            combined
        };

        #[cfg(feature = "parallel")]
        if f_zk.len() > workload_size::<F>() {
            f_zk.par_iter_mut()
                .zip(g_poly.par_iter())
                .for_each(|(f, &g)| *f = rho * *f + g);
        } else {
            for (f, &g) in f_zk.iter_mut().zip(g_poly.iter()) {
                *f = rho * *f + g;
            }
        }

        #[cfg(not(feature = "parallel"))]
        for (f, &g) in f_zk.iter_mut().zip(g_poly.iter()) {
            *f = rho * *f + g;
        }
        drop(g_poly);

        // combined_eval_j = dot(α, evaluations[j*n..(j+1)*n])
        let combined_claims: Vec<F> = (0..num_forms)
            .map(|j| {
                let row = &evaluations[j * num_vectors..(j + 1) * num_vectors];
                let combined_eval: F = alpha_coeffs.iter().zip(row).map(|(&a, &e)| a * e).sum();
                rho * combined_eval + g_claims[j]
            })
            .collect();

        // =====================================================================
        // Step 4: WHIR Initial Round
        //
        // P ↔ V: s-round sumcheck on f_zk with weight w, yielding r̄ = {r₀..r_{s-1}}
        // P then sends [[H]] = fold_k(ρ·f + g, r̄)
        // =====================================================================
        let constraint_rlc_coeffs: Vec<F> =
            geometric_challenge(self.prover_state, 0, linear_forms.len());
        let mut covector = vec![F::ZERO; size];
        for (coeff, lf) in constraint_rlc_coeffs.iter().zip(linear_forms.iter()) {
            lf.accumulate(&mut covector, *coeff);
        }

        let mut the_sum: F = constraint_rlc_coeffs
            .iter()
            .zip(combined_claims.iter())
            .map(|(&c, &eval)| c * eval)
            .sum();

        let folding_randomness = self.config.blinded_polynomial.initial_sumcheck.prove(
            self.prover_state,
            &mut f_zk,
            &mut covector,
            &mut the_sum,
        );

        PrepareResult {
            f_zk,
            covector,
            the_sum,
            rho,
            alpha_coeffs,
            folding_randomness,
        }
    }

    /// Accumulate STIR constraints from OOD and in-domain evaluations into the
    /// sumcheck state and transcript.
    fn accumulate_stir_constraints(
        prover_state: &mut ProverState<H, R>,
        state: &mut whir::rounds::SumcheckState<'_, F>,
        commitment: &irs_commit::Witness<F, F>,
        in_domain: &irs_commit::Evaluations<F>,
        initial_size: usize,
    ) {
        let stir_challenges: Vec<UnivariateEvaluation<F>> = commitment
            .out_of_domain()
            .evaluators(initial_size)
            .chain(in_domain.evaluators(initial_size))
            .collect();

        let one_weight = [F::ONE];
        let ood_evals = commitment.out_of_domain().values(&one_weight);
        let num_ood = commitment.out_of_domain().points.len();
        let embedding = Identity::new();

        let stir_evaluations: Vec<F> = ood_evals
            .chain(
                stir_challenges[num_ood..]
                    .iter()
                    .map(|challenge| challenge.evaluate(&embedding, state.vector)),
            )
            .collect();

        let stir_rlc_coeffs: Vec<F> = geometric_challenge(prover_state, 1, stir_challenges.len());
        UnivariateEvaluation::accumulate_many(&stir_challenges, state.covector, &stir_rlc_coeffs);
        *state.the_sum += dot(&stir_rlc_coeffs, &stir_evaluations);

        debug_assert_eq!(
            dot(state.vector, state.covector),
            *state.the_sum,
            "invariant broken after STIR accumulation"
        );
    }

    /// Step 5: OOD/STIR queries, STIR constraint accumulation, and remaining WHIR rounds.
    ///
    /// Takes ownership of `f_hat_polys` so it can be freed after OOD evaluations,
    /// before the memory-intensive WHIR rounds begin.
    #[allow(clippy::too_many_arguments)]
    fn ood_stir_and_rounds(
        &mut self,
        state: &mut whir::rounds::SumcheckState<'_, F>,
        alpha_coeffs: &[F],
        rho: F,
        folding_randomness: MultilinearPoint<F>,
        f_hat_witness: &irs_commit::Witness<F, F>,
        f_hat_polys: Vec<Vec<F>>,
        masking_polys: &[Vec<F>],
        g_polys: &[Vec<F>],
    ) -> OodStirResult<F> {
        let mu = self.dims.mu;
        let size = self.dims.size;

        let round_config = &self.config.blinded_polynomial.round_configs[0];
        let folded_f_zk_commitment = round_config
            .irs_committer
            .commit(self.prover_state, &[state.vector.as_slice()]);
        round_config.pow.prove(self.prover_state);
        let in_domain = self
            .config
            .blinded_polynomial
            .initial_committer
            .open(self.prover_state, &[f_hat_witness]);

        let r_bar = folding_randomness.0;
        let eq_weights = compute_eq_weights(&r_bar);
        let RsFoldCoeffs {
            masking_coeffs_all,
            g_i_coeffs,
        } = compute_rs_fold_blinding_coeffs(
            &eq_weights,
            g_polys,
            masking_polys,
            alpha_coeffs,
            rho,
            self.dims,
        );

        let mut lambda_z_points: Vec<F> = Vec::new();

        // Precompute combined f̂ for OOD MLE evaluations.
        // When n=1, borrow directly to avoid a full 2^μ allocation.
        let f_hat_combined: Cow<'_, [F]> = if f_hat_polys.len() == 1 {
            Cow::Borrowed(&f_hat_polys[0])
        } else {
            Cow::Owned(
                (0..size)
                    .map(|k| {
                        alpha_coeffs
                            .iter()
                            .zip(f_hat_polys.iter())
                            .map(|(&a, p)| a * p[k])
                            .sum()
                    })
                    .collect(),
            )
        };

        // --- OOD responses ---
        for &z in &folded_f_zk_commitment.out_of_domain().points {
            let fold_point = build_fold_args(&r_bar, z, mu);
            let ood_f_hat = multilinear_extend(&f_hat_combined, &fold_point);
            self.prover_state.prover_message(&ood_f_hat);
            send_blinding_evals(self.prover_state, z, &masking_coeffs_all, &g_i_coeffs);
            lambda_z_points.push(z);
        }

        // Release f̂ data before WHIR rounds.
        drop(f_hat_combined);
        drop(f_hat_polys);

        // --- STIR responses ---
        for &z in &in_domain.points {
            send_blinding_evals(self.prover_state, z, &masking_coeffs_all, &g_i_coeffs);
            lambda_z_points.push(z);
        }

        Self::accumulate_stir_constraints(
            self.prover_state,
            state,
            &folded_f_zk_commitment,
            &in_domain,
            round_config.initial_size(),
        );

        // Round 0 sumcheck
        let folding_randomness = round_config.sumcheck.prove(
            self.prover_state,
            state.vector,
            state.covector,
            state.the_sum,
        );

        // Remaining standard WHIR rounds
        let remaining = whir::rounds::prove_remaining_rounds(
            &self.config.blinded_polynomial.round_configs,
            &whir::rounds::FinalRoundConfig {
                sumcheck: &self.config.blinded_polynomial.final_sumcheck,
                pow: &self.config.blinded_polynomial.final_pow,
            },
            self.prover_state,
            state,
            folded_f_zk_commitment,
            &folding_randomness,
        );

        OodStirResult {
            lambda_z_points,
            eq_weights,
            masking_coeffs_all,
            g_i_coeffs,
            gamma_points: remaining.first_in_domain_points,
        }
    }

    /// Step 6: Γ consistency check.
    ///
    /// Opens [[f̂]] at Γ indices and sends blinding evaluations for each γ ∈ Γ.
    fn gamma_check(
        &mut self,
        f_hat_witness: &irs_commit::Witness<F, F>,
        masking_coeffs_all: &[Vec<F>],
        g_i_coeffs: &[Vec<F>],
        gamma_points: &[F],
        lambda_z_points: &mut Vec<F>,
    ) {
        let gamma_f_hat_indices = gamma_to_f_hat_indices(gamma_points, self.config);

        // Writes [[f̂]] openings at Γ indices to the transcript.
        // The verifier uses these to reconstruct fold(r̄, [[f̂]])(γ).
        // Return value (Evaluations) is unused: the prover already knows the values.
        let _f_hat_openings = self
            .config
            .blinded_polynomial
            .initial_committer
            .open_at_indices(self.prover_state, &[f_hat_witness], &gamma_f_hat_indices);

        for &gamma in gamma_points {
            send_blinding_evals(self.prover_state, gamma, masking_coeffs_all, g_i_coeffs);
            lambda_z_points.push(gamma);
        }
    }
}

impl<F: FftField> Config<F> {
    /// Steps 2-6: Prove the blinded polynomial instance.
    ///
    /// `f_hat_polys` is taken by value and freed during OOD evaluations (Step 5),
    /// before the memory-intensive WHIR rounds begin.
    /// Other witness fields are borrowed; the caller frees them before Step 7.
    #[allow(clippy::too_many_arguments)]
    fn prove_blinded_polynomial<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        vectors: Vec<Cow<'_, [F]>>,
        f_hat_witness: &irs_commit::Witness<F, F>,
        f_hat_polys: Vec<Vec<F>>,
        masking_polys: &[Vec<F>],
        g_polys: &[Vec<F>],
        linear_forms: &[Box<dyn LinearForm<F>>],
        evaluations: &[F],
    ) -> BlindedProveResult<F>
    where
        H: DuplexSpongeInterface<U = u8>,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let num_vectors = vectors.len();
        let num_forms = linear_forms.len();
        assert_eq!(evaluations.len(), num_forms * num_vectors);

        assert!(
            vectors[0].len().is_power_of_two(),
            "vector length must be a power of 2"
        );
        let expected_size = vectors[0].len();
        for (i, v) in vectors.iter().enumerate() {
            assert_eq!(
                v.len(),
                expected_size,
                "vectors[{i}] has length {}, expected {expected_size}",
                v.len()
            );
        }

        let dims = ProtocolDims::new(self, num_vectors);
        let mut ctx = BlindedProveCtx {
            config: self,
            prover_state,
            dims,
        };

        let prep = ctx.prepare_and_sumcheck(vectors, g_polys, linear_forms, evaluations);
        let PrepareResult {
            mut f_zk,
            mut covector,
            mut the_sum,
            rho,
            alpha_coeffs,
            folding_randomness,
        } = prep;

        let OodStirResult {
            mut lambda_z_points,
            eq_weights,
            masking_coeffs_all,
            g_i_coeffs,
            gamma_points,
        } = ctx.ood_stir_and_rounds(
            &mut whir::rounds::SumcheckState {
                vector: &mut f_zk,
                covector: &mut covector,
                the_sum: &mut the_sum,
            },
            &alpha_coeffs,
            rho,
            folding_randomness,
            f_hat_witness,
            f_hat_polys,
            masking_polys,
            g_polys,
        );

        drop(f_zk);
        drop(covector);

        ctx.gamma_check(
            f_hat_witness,
            &masking_coeffs_all,
            &g_i_coeffs,
            &gamma_points,
            &mut lambda_z_points,
        );

        BlindedProveResult {
            lambda_z_points,
            eq_weights,
            rho,
            alpha_coeffs,
            dims: ctx.dims,
        }
    }

    /// Step 7: Batched Proof on Blinding Polynomials.
    ///
    /// V → P: τ ←$ F_q (batching randomness)
    /// Both sides build beq tables (batched eq polynomial) and weight
    /// covectors wᵢ for each of the n + ν committed blinding vectors.
    /// P sends evaluation matrix E[i][j] = ⟨wᵢ, vⱼ⟩.
    /// V checks diagonal: E[i][i] = Σ_p τ^{p+1} · claim_i_p (from Λ).
    /// Then run second WHIR instance to prove batch opening claims.
    fn prove_blinding_polynomial<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        blinding_vectors: &[Vec<F>],
        blinding_poly_witness: &irs_commit::Witness<F, F>,
        blinded: &BlindedProveResult<F>,
    ) where
        H: DuplexSpongeInterface<U = u8>,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let dims = blinded.dims;
        let tau: F = prover_state.verifier_message();

        // beq_tables has num_g_polys = ν+1 entries (one per Φ projection)
        let beq_tables = build_beq_tables(&blinded.lambda_z_points, &blinded.eq_weights, tau, dims);

        let weight_covectors =
            build_weight_covectors(&beq_tables, blinded.rho, &blinded.alpha_coeffs, dims);

        // Compute eval matrix E[i][j] = ⟨w_i, v_j⟩ (row-major, num_blinding_vecs²)
        let mut eval_matrix: Vec<F> =
            Vec::with_capacity(dims.num_blinding_vecs * dims.num_blinding_vecs);
        for w in &weight_covectors {
            for v in blinding_vectors {
                eval_matrix.push(dot(w, v));
            }
        }

        for eval in &eval_matrix {
            prover_state.prover_message(eval);
        }

        let blinding_forms: Vec<Box<dyn LinearForm<F>>> = weight_covectors
            .into_iter()
            .map(|cv| Box::new(Covector::new(cv)) as Box<dyn LinearForm<F>>)
            .collect();

        let blinding_vector_cows: Vec<Cow<'_, [F]>> = blinding_vectors
            .iter()
            .map(|v| Cow::Borrowed(v.as_slice()))
            .collect();
        // Final claim is internal to the blinding sub-protocol; not needed by caller.
        let _blinding_final_claim = self.blinding_polynomial.prove(
            prover_state,
            blinding_vector_cows,
            vec![Cow::Borrowed(blinding_poly_witness)],
            blinding_forms,
            Cow::Owned(eval_matrix),
        );
    }

    /// zkWHIR 2.0 prover — Alternative Randomness Sampling.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    #[allow(clippy::needless_pass_by_value)]
    pub fn prove<'a, H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        vectors: Vec<Cow<'a, [F]>>,
        witness: Witness<F>,
        linear_forms: Vec<Box<dyn LinearForm<F>>>,
        evaluations: Cow<'a, [F]>,
    ) where
        H: DuplexSpongeInterface<U = u8>,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let Witness {
            f_hat_witness,
            blinding_poly_witness,
            f_hat_polys,
            secrets,
        } = witness;

        // Steps 2-6: blinded polynomial proof.
        let blinded = self.prove_blinded_polynomial(
            prover_state,
            vectors,
            &f_hat_witness,
            f_hat_polys,
            &secrets.masking_polys,
            &secrets.g_polys,
            &linear_forms,
            &evaluations,
        );

        // Free fields only needed during Steps 2-6, before Step 7.
        drop(f_hat_witness);
        drop(linear_forms);

        // Step 7: batched blinding polynomial proof.
        self.prove_blinding_polynomial(
            prover_state,
            &secrets.blinding_vectors,
            &blinding_poly_witness,
            &blinded,
        );
    }
}
