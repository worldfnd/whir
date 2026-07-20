// IGNORE CHANGES TO THIS FILE - NOT FULLY PORTED TO PROPERLY USE BUFFER ABSTRACTION.

use std::borrow::Cow;

use ark_ff::Field;
use ark_std::rand::{distributions::Standard, prelude::Distribution};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "tracing")]
use tracing::instrument;

use super::{Config, Witness};
use crate::{
    algebra::{
        embedding::Identity,
        linear_form::{Covector, Evaluate, LinearForm},
        mixed_dot, scalar_mul_add,
    },
    buffer::Buffer,
    hash::Hash,
    protocols::{
        whir::FinalClaim,
        whir_zk::utils::{
            build_combined_and_subproof_claims, fill_eq_weights_at_gamma_half,
            fold_weight_to_mask_size, BlindingPolynomials,
        },
    },
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState,
        VerifierMessage,
    },
};

/// Evaluate blinding polynomials at all gamma query points and accumulate beq weights.
///
/// For each gamma in `h_gammas`, computes `(m_eval, g_hat_evals, h_val)` per polynomial
/// and accumulates `Sum_i tau2^i * beq_half(gamma_i, .)` into the batched beq covector.
///
/// Returns `(eval_results, beq_weight_accum)`:
/// - `eval_results`: flat buffer with per-gamma stride `num_polynomials * (num_witness_variables + 2)`,
///   laid out as `[m_eval, g_hat_0, ..., g_hat_{mu-1}, h_val]` per polynomial.
/// - `beq_weight_accum`: full-size `(ell+1)`-variable beq covector for the blinding subproof.
#[allow(clippy::too_many_lines)]
#[cfg_attr(
    feature = "tracing",
    instrument(skip_all, name = "evaluate_gamma_block")
)]
fn evaluate_gamma_block<F: Field>(
    blinding_polynomials: &[BlindingPolynomials<F>],
    h_gammas: &[F],
    masking_challenge: F,
    blinding_challenge: F,
    tau2: F,
    num_blinding_variables: usize,
    num_witness_variables: usize,
) -> (Vec<F>, Vec<F>) {
    let num_polynomials = blinding_polynomials.len();
    let half_size = 1usize << num_blinding_variables;
    let weight_size = 1usize << (num_blinding_variables + 1);
    let one_plus_rho = F::ONE + masking_challenge;
    let neg_rho = -masking_challenge;

    // Pre-fold m_poly over the -rho variable to halve all gamma-block dot products.
    // Uses the identity: beq_full[2j] = beq_half[j] * (1+rho),
    //                     beq_full[2j+1] = beq_half[j] * (-rho).
    let folded_m_polys: Vec<Vec<F>> = blinding_polynomials
        .iter()
        .map(|bp| {
            (0..half_size)
                .map(|j| one_plus_rho * bp.m_poly[2 * j] + neg_rho * bp.m_poly[2 * j + 1])
                .collect()
        })
        .collect();

    // Pre-compute tau2 powers for the parallel gamma loop.
    let num_gammas = h_gammas.len();
    let tau2_powers = {
        let mut powers = Vec::with_capacity(num_gammas);
        let mut p = F::ONE;
        for _ in 0..num_gammas {
            powers.push(p);
            p *= tau2;
        }
        powers
    };

    // Flat layout per gamma: [m_eval, g_hat_0, ..., g_hat_{mu-1}, h_val] × num_polynomials.
    let stride_per_poly = num_witness_variables + 2;
    let stride_per_gamma = num_polynomials * stride_per_poly;
    let mut eval_results = vec![F::ZERO; num_gammas * stride_per_gamma];

    #[cfg(feature = "parallel")]
    let beq_half_accum = {
        let batch = {
            let cores = rayon::current_num_threads();
            num_gammas.div_ceil(cores).max(1)
        };
        let batch_stride = batch * stride_per_gamma;
        eval_results
            .par_chunks_mut(batch_stride)
            .enumerate()
            .fold(
                || (vec![F::ZERO; half_size], vec![F::ZERO; half_size]),
                |(mut accum, mut eq_buf), (chunk_idx, chunk)| {
                    let base_gi = chunk_idx * batch;
                    let chunk_gammas = chunk.len() / stride_per_gamma;
                    let embedding = Identity::<F>::new();
                    for local in 0..chunk_gammas {
                        let gi = base_gi + local;
                        let gamma = h_gammas[gi];
                        let tau2_pow = tau2_powers[gi];
                        let slot =
                            &mut chunk[local * stride_per_gamma..(local + 1) * stride_per_gamma];
                        fill_eq_weights_at_gamma_half(&mut eq_buf, gamma, num_blinding_variables);
                        scalar_mul_add(&mut accum, tau2_pow, &eq_buf);
                        for (poly_idx, bp) in blinding_polynomials.iter().enumerate() {
                            let off = poly_idx * stride_per_poly;
                            slot[off] = eq_buf
                                .iter()
                                .zip(folded_m_polys[poly_idx].iter())
                                .map(|(&e, &f)| e * f)
                                .sum();
                            for (j, g_hat) in bp.g_hats.iter().enumerate() {
                                slot[off + 1 + j] =
                                    one_plus_rho * mixed_dot(&embedding, &eq_buf, g_hat);
                            }
                            let mut h = slot[off];
                            let mut bp_pow = blinding_challenge;
                            let mut gp = gamma;
                            for j in 0..bp.g_hats.len() {
                                h += bp_pow * gp * slot[off + 1 + j];
                                bp_pow *= blinding_challenge;
                                gp = gp.square();
                            }
                            slot[off + 1 + num_witness_variables] = h;
                        }
                    }
                    (accum, eq_buf)
                },
            )
            .map(|(accum, _)| accum)
            .reduce(
                || vec![F::ZERO; half_size],
                |mut a, b| {
                    for (x, &y) in a.iter_mut().zip(b.iter()) {
                        *x += y;
                    }
                    a
                },
            )
    };

    #[cfg(not(feature = "parallel"))]
    let beq_half_accum = {
        let embedding = Identity::<F>::new();
        let mut eq_buf = vec![F::ZERO; half_size];
        let mut accum = vec![F::ZERO; half_size];
        for (gi, &gamma) in h_gammas.iter().enumerate() {
            fill_eq_weights_at_gamma_half(&mut eq_buf, gamma, num_blinding_variables);
            let tau2_pow = tau2_powers[gi];
            scalar_mul_add(&mut accum, tau2_pow, &eq_buf);
            for (poly_idx, bp) in blinding_polynomials.iter().enumerate() {
                let off = gi * stride_per_gamma + poly_idx * stride_per_poly;
                eval_results[off] = eq_buf
                    .iter()
                    .zip(folded_m_polys[poly_idx].iter())
                    .map(|(&e, &f)| e * f)
                    .sum();
                for (j, g_hat) in bp.g_hats.iter().enumerate() {
                    eval_results[off + 1 + j] =
                        one_plus_rho * mixed_dot(&embedding, &eq_buf, g_hat);
                }
                let mut h = eval_results[off];
                let mut bp_pow = blinding_challenge;
                let mut gp = gamma;
                for j in 0..bp.g_hats.len() {
                    h += bp_pow * gp * eval_results[off + 1 + j];
                    bp_pow *= blinding_challenge;
                    gp = gp.square();
                }
                eval_results[off + 1 + num_witness_variables] = h;
            }
        }
        accum
    };

    // Reconstruct full-size beq_weight_accum from half-size accumulator.
    let beq_weight_accum = {
        let mut full = vec![F::ZERO; weight_size];
        for j in 0..half_size {
            full[2 * j] = one_plus_rho * beq_half_accum[j];
            full[2 * j + 1] = neg_rho * beq_half_accum[j];
        }
        full
    };

    (eval_results, beq_weight_accum)
}

impl<F: Field> Config<F> {
    /// Run the zkWHIR prover: prove evaluation claims on blinded polynomials.
    ///
    /// * `vectors` — original (unmasked) coefficient vectors.
    /// * `witness` — commitment witness produced by [`Config::commit`].
    /// * `linear_forms` — linear forms (one per evaluation query).
    /// * `evaluations` — row-major `linear_forms × vectors` evaluation matrix.
    ///
    /// Returns the final evaluation point and per-vector evaluations from the
    /// inner witness-side WHIR prover.
    #[allow(clippy::too_many_lines)]
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn prove<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        vectors: &[&Buffer<F>],
        witness: Witness<F>,
        linear_forms: Vec<Box<dyn LinearForm<F>>>,
        evaluations: Cow<'_, [F]>,
    ) -> FinalClaim<F>
    where
        Standard: Distribution<F>,
        H: DuplexSpongeInterface<U = u8>,
        R: ark_std::rand::RngCore + ark_std::rand::CryptoRng,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(
            self.blinded_commitment.initial_committer.num_vectors(),
            1,
            "zkWHIR currently expects one vector per commitment"
        );
        assert_eq!(
            vectors.len(),
            witness.f_hat_vectors.len(),
            "masked vector/polynomial length mismatch"
        );
        assert_eq!(
            vectors.len(),
            witness.f_hat_witnesses.len(),
            "witness/polynomial length mismatch"
        );
        assert_eq!(
            witness.blinding_vectors.len(),
            self.blinding_commitment.initial_committer.num_vectors(),
            "blinding vectors/witness mismatch"
        );
        assert_eq!(
            evaluations.len(),
            linear_forms.len() * vectors.len(),
            "evaluation matrix must be row-major linear_forms x vectors"
        );

        // Destructure early; blinding_polynomials is dropped after the gamma block.
        let Witness {
            f_hat_vectors,
            f_hat_witnesses,
            blinding_polynomials,
            blinding_vectors,
            blinding_witness,
        } = witness;

        let blinding_challenge: F = prover_state.verifier_message();

        let embedding = self.blinding_commitment.embedding();
        let num_polynomials = vectors.len();
        let num_witness_variables = self.num_witness_variables();
        let num_blinding_variables = self.num_blinding_variables();
        let num_witness_variables_plus_1 = num_witness_variables + 1;

        // Compute w_folded evaluations of all blinding vectors before rho for binding.
        let (w_folded_weights, m_evals, w_folded_blinding_evals) = {
            #[cfg(feature = "tracing")]
            let _span = tracing::info_span!("zk_w_folded_compute").entered();
            let mut w_folded_weights: Vec<Covector<F>> = Vec::with_capacity(linear_forms.len());
            let mut m_evals: Vec<F> = Vec::with_capacity(evaluations.len());
            let mut w_folded_blinding_evals: Vec<F> = Vec::with_capacity(
                linear_forms.len() * num_polynomials * num_witness_variables_plus_1,
            );

            for weight in &linear_forms {
                let w_folded = fold_weight_to_mask_size(
                    weight.as_ref(),
                    num_witness_variables,
                    num_blinding_variables,
                );
                for poly_idx in 0..num_polynomials {
                    let base = poly_idx * num_witness_variables_plus_1;
                    for v in 0..num_witness_variables_plus_1 {
                        let eval: F = w_folded.evaluate(embedding, &blinding_vectors[base + v]);
                        w_folded_blinding_evals.push(eval);
                        if v == 0 {
                            m_evals.push(eval);
                        }
                    }
                }
                w_folded_weights.push(w_folded);
            }
            (w_folded_weights, m_evals, w_folded_blinding_evals)
        };
        for eval in &w_folded_blinding_evals {
            prover_state.prover_message(eval);
        }

        // Sample non-zero rho after masking evaluations are committed.
        let masking_challenge: F = prover_state.verifier_message();
        assert_ne!(
            masking_challenge,
            F::ZERO,
            "zkWHIR requires non-zero masking challenge rho"
        );

        // modified_eval = F_eval + M_eval
        let modified_evaluations: Vec<F> = evaluations
            .iter()
            .zip(m_evals.iter())
            .map(|(&eval, &m)| eval + m)
            .collect();
        drop(evaluations);

        let initial_in_domain = {
            #[cfg(feature = "tracing")]
            let _span = tracing::info_span!("open_f_hat").entered();
            let witness_refs: Vec<_> = f_hat_witnesses.iter().map(|w| &w.irs).collect();
            self.blinded_commitment
                .initial_committer
                .open(prover_state, &witness_refs)
        };

        // Expand base queries into coset points for the first folding round.
        let h_gammas = self.all_gammas(&initial_in_domain.points);

        // tau1 batches the g_hat claims into the combined claim per polynomial.
        // tau2 batches across gamma query points.
        let tau1: F = prover_state.verifier_message();
        let tau2: F = prover_state.verifier_message();

        let (eval_results, beq_weight_accum) = evaluate_gamma_block(
            &blinding_polynomials,
            &h_gammas,
            masking_challenge,
            blinding_challenge,
            tau2,
            num_blinding_variables,
            num_witness_variables,
        );

        let num_gammas = h_gammas.len();
        let stride_per_poly = num_witness_variables + 2;
        let stride_per_gamma = num_polynomials * stride_per_poly;

        // Transcript writes and claim accumulation (must be sequential).
        let mut m_claims = vec![F::ZERO; num_polynomials];
        // Flat layout: g_hat_claims[poly_idx * num_witness_variables + j]
        let mut g_hat_claims = vec![F::ZERO; num_polynomials * num_witness_variables];
        let mut batched_h_claims = vec![F::ZERO; num_polynomials];
        {
            let mut tau2_pow = F::ONE;
            for gi in 0..num_gammas {
                let base = gi * stride_per_gamma;
                for poly_idx in 0..num_polynomials {
                    let off = base + poly_idx * stride_per_poly;
                    let m_eval = eval_results[off];
                    let g_hat_evals = &eval_results[off + 1..off + 1 + num_witness_variables];
                    let h_val = eval_results[off + 1 + num_witness_variables];

                    prover_state.prover_message(&m_eval);
                    for g_hat_eval in g_hat_evals {
                        prover_state.prover_message(g_hat_eval);
                    }

                    m_claims[poly_idx] += tau2_pow * m_eval;
                    for (j, &g) in g_hat_evals.iter().enumerate() {
                        g_hat_claims[poly_idx * num_witness_variables + j] += tau2_pow * g;
                    }
                    batched_h_claims[poly_idx] += tau2_pow * h_val;
                }
                tau2_pow *= tau2;
            }
        }
        drop(eval_results);
        drop(blinding_polynomials);

        let g_hat_slices: Vec<&[F]> = (0..num_polynomials)
            .map(|i| &g_hat_claims[i * num_witness_variables..(i + 1) * num_witness_variables])
            .collect();
        let (combined_claims, batched_blinding_subproof_claims) =
            build_combined_and_subproof_claims(&m_claims, &g_hat_slices, tau1);
        let beq_weights = Covector::new(beq_weight_accum);
        for claim in &combined_claims {
            prover_state.prover_message(claim);
        }
        for claim in &batched_h_claims {
            prover_state.prover_message(claim);
        }

        let result = {
            #[cfg(feature = "tracing")]
            let _span = tracing::info_span!("inner_blinded_prove").entered();
            let f_hat_buffers = f_hat_vectors
                .iter()
                .map(|v| Buffer::from(v.as_slice()))
                .collect::<Vec<_>>();
            let f_hat_refs = f_hat_buffers.iter().collect::<Vec<_>>();
            let f_hat_witness_refs = f_hat_witnesses.iter().collect::<Vec<_>>();
            self.blinded_commitment.prove(
                prover_state,
                &f_hat_refs,
                f_hat_witness_refs,
                linear_forms,
                Buffer::from(modified_evaluations),
            )
        };

        {
            #[cfg(feature = "tracing")]
            let _span = tracing::info_span!("inner_blinding_prove").entered();
            let blinding_forms: Vec<Box<dyn LinearForm<F>>> =
                std::iter::once(Box::new(beq_weights) as Box<dyn LinearForm<F>>)
                    .chain(
                        w_folded_weights
                            .into_iter()
                            .map(|wf| Box::new(wf) as Box<dyn LinearForm<F>>),
                    )
                    .collect();
            let all_blinding_claims = [
                &batched_blinding_subproof_claims[..],
                &w_folded_blinding_evals[..],
            ]
            .concat();
            // Blinding sub-proof result is discarded: the blinding WHIR's
            // evaluation point is not needed by the outer protocol.
            let blinding_buffers = blinding_vectors
                .iter()
                .map(|v| Buffer::from(v.as_slice()))
                .collect::<Vec<_>>();
            let blinding_refs = blinding_buffers.iter().collect::<Vec<_>>();
            let _ = self.blinding_commitment.prove(
                prover_state,
                &blinding_refs,
                vec![&blinding_witness],
                blinding_forms,
                Buffer::from(all_blinding_claims),
            );
        }

        result
    }
}
