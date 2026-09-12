/// zkWHIR 2.0 — Zero-Knowledge WHIR with poly-logarithmic overhead.
///
/// Uses the "Alternative Randomness Sampling" approach which samples only
/// ν + 1 = ⌊μ/ℓ⌋ + 1 blinding polynomials (instead of μ + 1), reducing proof
/// size to (ν + 1) · q(δ) field elements.
///
/// Two WHIR instances run as sub-protocols:
///   1. `blinded_polynomial`: over the μ-variate masked witness f̂ = f + msk(Φ₀)
///   2. `blinding_polynomial`: over (ℓ+1)-variate committed vectors M and ĝ₁..ĝ_ν
///
/// Protocol phases:
///   Step 1: Commitment — sample msk, ĝ₀..ĝ_ν; commit [[f̂]], [[M]], [[ĝᵢ]]
///   Step 2: Blinding claims — V samples β; P builds g(x̄) = Σ βⁱ·ĝᵢ(Φᵢ(x̄)), sends G
///   Step 3: Combination — V samples ρ ≠ 0; P forms f_zk = ρ·f + g
///   Step 4: Initial sumcheck on f_zk; P sends [[H]] = fold_k(f_zk, r̄)
///   Step 5: Virtual OOD/STIR queries + remaining WHIR rounds
///   Step 6: Γ consistency check — verify [[f̂]] openings match [[H]]
///   Step 7: Batched blinding proof via second WHIR instance
use std::fmt;

use ark_ff::FftField;
use serde::{Deserialize, Serialize};

use crate::algebra::embedding::Embedding;

mod committer;
mod prover;
mod utils;
mod verifier;

pub use self::{committer::Witness, verifier::Commitments};
use crate::{
    algebra::embedding::Identity,
    bits::Bits,
    parameters::ProtocolParameters,
    protocols::{irs_commit, whir},
};

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize, Debug)]
#[serde(bound = "")]
pub struct Config<F: FftField> {
    /// First WHIR instance: proves claims about f_zk = ρ·f + g over 2^μ evaluations.
    pub blinded_polynomial: whir::Config<Identity<F>>,
    /// Second WHIR instance: batched proof of blinding polynomial evaluations
    /// over 2^(ℓ+1) evaluations with n + ν committed vectors.
    pub blinding_polynomial: whir::Config<Identity<F>>,
}

impl<F: FftField> fmt::Display for Config<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "zkWHIR 2.0 — Alternative Randomness Sampling")?;
        writeln!(f, "Blinded polynomial instance:")?;
        write!(f, "{}", self.blinded_polynomial)?;
        writeln!(f, "Blinding polynomial instance:")?;
        write!(f, "{}", self.blinding_polynomial)
    }
}

impl<F: FftField> Config<F> {
    /// Check whether the configured PoW bits exceed the specified maximum.
    pub fn check_max_pow_bits(&self, bits: Bits) -> bool {
        self.blinded_polynomial.check_max_pow_bits(bits)
    }

    /// Disable proof-of-work on both WHIR sub-instances (for testing).
    #[cfg(test)]
    pub(crate) fn disable_pow(&mut self) {
        self.blinded_polynomial.disable_pow();
        self.blinding_polynomial.disable_pow();
    }

    /// Return a reference to the embedding used by the blinded polynomial instance.
    pub fn embedding(&self) -> &Identity<F> {
        self.blinded_polynomial.embedding()
    }

    /// Number of variables in the witness polynomial (μ = log₂ of the evaluation domain size).
    pub fn num_witness_variables(&self) -> usize {
        self.blinded_polynomial.initial_num_variables()
    }

    /// Security levels (in bits) for the two WHIR sub-instances.
    ///
    /// Returns `(blinded_security, blinding_security)` where:
    /// - `blinded_security` is for the witness instance (f̂ with `num_vectors` committed polynomials
    ///   and `num_linear_forms` constraints)
    /// - `blinding_security` is for the blinding instance (M and ĝ vectors)
    pub fn security_levels(&self, num_vectors: usize, num_linear_forms: usize) -> (f64, f64) {
        let num_blinding_vecs = self.blinding_polynomial.initial_committer.num_vectors;
        let blinded_sec = self
            .blinded_polynomial
            .security_level(num_vectors, num_linear_forms);
        let blinding_sec = self
            .blinding_polynomial
            .security_level(num_blinding_vecs, num_blinding_vecs);
        (blinded_sec, blinding_sec)
    }

    pub fn new(num_variables_main: usize, params: &ProtocolParameters) -> Self {
        assert!(
            !params.unique_decoding,
            "zkWHIR 2.0 requires list decoding (unique_decoding must be false). \
             The protocol relies on OOD queries in Step 5 for blinding claim \
             generation; unique decoding sets out_domain_samples = 0, making \
             Commitment::num_vectors() undefined."
        );
        let blinded_config: whir::Config<Identity<F>> =
            whir::Config::new(1 << num_variables_main, params);
        assert!(
            !blinded_config.round_configs.is_empty(),
            "zkWHIR 2.0 requires at least one WHIR round \
             (num_variables_main too small for folding_factor)"
        );
        let witness_sec = params.security_level.saturating_sub(params.pow_bits) as f64;
        let blinding_sec = params.security_level as f64;

        // T(δ) for the witness instance: the polynomial size sent in the final WHIR round.
        let witness_t_delta = blinded_config.final_sumcheck.initial_size;

        let mut witness_leak = InstanceLeak::new(
            params,
            witness_sec,
            &blinded_config.initial_committer,
            num_variables_main,
        );
        witness_leak.t_delta = witness_t_delta;

        // Blinding leak: use μ for num_variables (conservative, per paper §"Query Complexity
        // Computation": q_ub ≤ leak(δ₁,k₁,μ,d) + leak(δ₂,k₂,μ,3)). Using the witness IRS
        // config for ood_delta is also conservative since the blinding instance is smaller.
        let blinding_leak = InstanceLeak::new(
            params,
            blinding_sec,
            &blinded_config.initial_committer,
            num_variables_main,
        );
        let ell = ell_from_q_ub(query_upper_bound(&witness_leak, &blinding_leak));
        assert!(
            ell + 1 < num_variables_main,
            "blinding variables ell+1={} must be < mu={num_variables_main}",
            ell + 1
        );
        // Configuration validation: if ell < initial_folding_factor, build_beq_tables
        // will underflow when computing m_cap. Catch misconfiguration here with a
        // clear message instead of a confusing panic deep in the proving path.
        assert!(
            ell >= params.initial_folding_factor,
            "ell={ell} must be >= initial_folding_factor={} \
             (parameters too aggressive for ZK sizing)",
            params.initial_folding_factor
        );

        // nu = ⌊mu/ell⌋ — highest blinding polynomial index (nu+1 total g-polynomials).
        // batch_size = n + ν: the blinding instance commits n M-polynomials (one per
        // witness, each embedding g₀ + mskᵢ) plus ν embedded ĝ-polynomials (ĝ₁..ĝ_ν).
        let nu = num_variables_main / ell;
        let blinding_params = ProtocolParameters {
            batch_size: params.batch_size + nu,
            ..*params
        };

        Self {
            blinded_polynomial: blinded_config,
            blinding_polynomial: whir::Config::new(1 << (ell + 1), &blinding_params),
        }
    }
}

/// Maximum degree of the sumcheck round polynomial.
///
/// In each sumcheck round the prover evaluates `eq(r, x) · f(x)` where `eq` is degree 2
/// and `f` contributes degree 1 from folding, giving a combined degree of 3.
/// This bounds the (d+1)·μ term in the leakage formula.
const MAX_SUMCHECK_DEGREE: usize = 3;

/// Per-instance leakage parameters for a single WHIR execution.
///
/// `leak(δ, k, μ, d) := k · [q(δ) + stir(δ)] + T(δ) + ood(δ) + (d+1) · μ`
struct InstanceLeak {
    /// Folding factor `k = 2^s` for the first round.
    k: usize,
    /// Number of sumcheck rounds (= number of witness variables).
    mu: usize,
    /// Max degree of sumcheck round polynomial.
    d: usize,
    /// `q(δ)`: query complexity.
    q_delta: usize,
    /// `stir(δ)`: STIR queries during the first folding round.
    stir_delta: usize,
    /// `ood(δ)`: out-of-domain queries.
    ood_delta: usize,
    /// `T(δ)`: number of raw coefficients sent during the last sumcheck round.
    /// Invariant: `T(δ) ≥ q(δ)`.
    t_delta: usize,
}

impl InstanceLeak {
    /// Construct leak parameters for one WHIR instance.
    ///
    /// `security_target` is `λ - pow_bits` for the witness side
    /// or full `λ` for the blinding side.
    fn new<M>(
        params: &ProtocolParameters,
        security_target: f64,
        irs_config: &irs_commit::Config<M>,
        num_variables: usize,
    ) -> Self
    where
        M: Embedding,
        M::Source: FftField,
        M::Target: FftField,
    {
        #[allow(clippy::cast_possible_wrap)]
        let rate = 0.5_f64.powi(params.starting_log_inv_rate as i32);
        let (q, stir) = Self::query_counts(
            params.unique_decoding,
            security_target,
            rate,
            params.initial_folding_factor,
        );

        Self {
            k: 1 << params.initial_folding_factor,
            mu: num_variables,
            d: MAX_SUMCHECK_DEGREE,
            q_delta: q,
            stir_delta: stir,
            ood_delta: irs_config.out_domain_samples,
            t_delta: q, // conservative default; overridden for witness instance in Config::new
        }
    }

    /// `leak(δ, k, μ, d) := k · [q(δ) + stir(δ)] + T(δ) + ood(δ) + (d+1) · μ`
    const fn leak(&self) -> usize {
        self.k
            .saturating_mul(self.q_delta.saturating_add(self.stir_delta))
            .saturating_add(self.t_delta)
            .saturating_add(self.ood_delta)
            .saturating_add(self.d.saturating_add(1).saturating_mul(self.mu))
    }

    /// Compute `q(δ)` and `stir(δ)` for a given security target and rate.
    ///
    /// - `q(δ) = ⌈λ / log₂(1/(1-δ))⌉`
    /// - `stir(δ) ≈ ⌈λ / (s + log₂(1/(1-δ)))⌉`
    #[allow(clippy::cast_sign_loss)]
    fn query_counts(
        unique_decoding: bool,
        security_target: f64,
        rate: f64,
        folding_factor: usize,
    ) -> (usize, usize) {
        let q = irs_commit::num_in_domain_queries(unique_decoding, security_target, rate);
        let s = folding_factor as f64;
        let slack = irs_commit::johnson_slack(unique_decoding, rate);
        let per_sample = if unique_decoding {
            f64::midpoint(1., rate)
        } else {
            rate.sqrt() + slack
        };
        let stir = (security_target / (s + (-per_sample.log2()))).ceil() as usize;
        (q, stir)
    }
}

/// Compute `q_ub` — the total leakage upper bound across both WHIR instances.
///
/// `q_ub ≤ leak(δ₁, k₁, μ, d) + leak(δ₂, k₂, μ, 3)`
const fn query_upper_bound(witness: &InstanceLeak, blinding: &InstanceLeak) -> usize {
    witness.leak().saturating_add(blinding.leak())
}

/// Smallest `ell` such that `2^ell > q_ub`.
const fn ell_from_q_ub(q_ub: usize) -> usize {
    assert!(q_ub > 0, "query upper bound must be positive");
    (usize::BITS - q_ub.leading_zeros()) as usize
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use ark_ff::{AdditiveGroup, Field};

    use super::{
        committer::Witness,
        utils::{
            build_beq_tables, build_fold_args, build_weight_covectors, compute_eq_weights,
            compute_rs_fold_blinding_coeffs, gamma_to_f_hat_indices, ProtocolDims, RsFoldCoeffs,
        },
        Config,
    };
    use crate::{
        algebra::{
            dot,
            embedding::Identity,
            fields::Field64,
            geometric_sequence,
            linear_form::{
                Covector, Evaluate, LinearForm, MultilinearExtension, UnivariateEvaluation,
            },
            multilinear_extend, univariate_evaluate, MultilinearPoint,
        },
        hash,
        parameters::ProtocolParameters,
        protocols::{geometric_challenge::geometric_challenge, whir},
        transcript::{
            codecs::Empty, DomainSeparator, Proof, ProverState, VerifierMessage, VerifierState,
        },
    };

    type F = Field64;

    const TEST_NUM_VARIABLES: usize = 12;
    const TEST_NUM_COEFFS: usize = 1 << TEST_NUM_VARIABLES;

    fn make_test_config() -> Config<F> {
        make_test_config_batch(1)
    }

    /// Materialize linear forms into Covectors for the prover.
    fn to_prove_forms(
        forms: &[Box<dyn LinearForm<F>>],
        size: usize,
    ) -> Vec<Box<dyn LinearForm<F>>> {
        forms
            .iter()
            .map(|f| {
                let mut cv = vec![F::ZERO; size];
                f.accumulate(&mut cv, F::ONE);
                Box::new(Covector::new(cv)) as Box<dyn LinearForm<F>>
            })
            .collect()
    }

    /// Run a full prove → verify cycle for zkWHIR 2.0.
    /// Convenience wrapper around `honest_proof_and_verify` that discards
    /// the returned `(ds, proof)` — used by functional tests that don't
    /// need to attempt forgery afterwards.
    #[allow(clippy::needless_pass_by_value)]
    fn prove_and_verify(
        config: &Config<F>,
        vectors: Vec<Vec<F>>,
        forms: Vec<Box<dyn LinearForm<F>>>,
        evaluations: &[F],
    ) {
        let vec_refs: Vec<&[F]> = vectors.iter().map(|v| v.as_slice()).collect();
        let _ = honest_proof_and_verify(config, &vec_refs, &forms, evaluations);
    }

    #[test]
    fn test_zk_prove_verify_single_point() {
        let mut rng = ark_std::test_rng();
        let config = make_test_config();

        let vector = vec![F::ONE; TEST_NUM_COEFFS];
        let point = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let form = MultilinearExtension { point: point.0 };
        let evaluation = form.evaluate(config.embedding(), &vector);

        prove_and_verify(&config, vec![vector], vec![Box::new(form)], &[evaluation]);
    }

    #[test]
    fn test_zk_prove_verify_multiple_points() {
        let mut rng = ark_std::test_rng();
        let config = make_test_config();

        let vector: Vec<F> = (0..TEST_NUM_COEFFS)
            .map(|i| F::from(i as u64 + 1))
            .collect();

        let p0 = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let p1 = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let f0 = MultilinearExtension { point: p0.0 };
        let f1 = MultilinearExtension { point: p1.0 };

        let embedding = config.embedding();
        let eval0 = f0.evaluate(embedding, &vector);
        let eval1 = f1.evaluate(embedding, &vector);

        prove_and_verify(
            &config,
            vec![vector],
            vec![Box::new(f0), Box::new(f1)],
            &[eval0, eval1],
        );
    }

    #[test]
    fn test_zk_prove_verify_with_covector() {
        let mut rng = ark_std::test_rng();
        let config = make_test_config();

        let vector = vec![F::ONE; TEST_NUM_COEFFS];

        let point = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let mle_form = MultilinearExtension { point: point.0 };
        let embedding = config.embedding();
        let mle_eval = mle_form.evaluate(embedding, &vector);

        let cov = Covector::new((0..TEST_NUM_COEFFS).map(|i| F::from(i as u64)).collect());
        let cov_eval = cov.evaluate(embedding, &vector);

        prove_and_verify(
            &config,
            vec![vector],
            vec![Box::new(mle_form), Box::new(cov)],
            &[mle_eval, cov_eval],
        );
    }

    fn make_test_config_batch(batch_size: usize) -> Config<F> {
        let whir_params = ProtocolParameters {
            unique_decoding: false,
            security_level: 16,
            pow_bits: 0,
            initial_folding_factor: 2,
            folding_factor: 2,
            starting_log_inv_rate: 1,
            batch_size,
            hash_id: hash::SHA2,
        };
        let mut config = Config::new(TEST_NUM_VARIABLES, &whir_params);
        config.disable_pow();
        config
    }

    /// Round-trip test with `num_variables` chosen so that `mu % ell != 0`,
    /// exercising the uneven-tiling code path where Φ₀ and Φ₁ extract
    /// different bit windows.
    #[test]
    fn test_zk_prove_verify_nonzero_rem() {
        const NUM_VARS: usize = 14;
        const NUM_COEFFS: usize = 1 << NUM_VARS;

        let mut rng = ark_std::test_rng();
        let whir_params = ProtocolParameters {
            unique_decoding: false,
            security_level: 16,
            pow_bits: 0,
            initial_folding_factor: 2,
            folding_factor: 2,
            starting_log_inv_rate: 1,
            batch_size: 1,
            hash_id: hash::SHA2,
        };
        let mut config = Config::new(NUM_VARS, &whir_params);
        config.disable_pow();

        // Verify rem != 0 for this parameter set.
        let ell = config.blinding_polynomial.initial_num_variables() - 1;
        assert_ne!(NUM_VARS % ell, 0, "test requires non-zero rem");

        let vector = vec![F::ONE; NUM_COEFFS];
        let point = MultilinearPoint::rand(&mut rng, NUM_VARS);
        let form = MultilinearExtension { point: point.0 };
        let evaluation = form.evaluate(config.embedding(), &vector);

        prove_and_verify(&config, vec![vector], vec![Box::new(form)], &[evaluation]);
    }

    #[test]
    fn test_zk_prove_verify_multi_vector() {
        let mut rng = ark_std::test_rng();
        let config = make_test_config_batch(2);

        let v0: Vec<F> = (0..TEST_NUM_COEFFS)
            .map(|i| F::from(i as u64 + 1))
            .collect();
        let v1: Vec<F> = (0..TEST_NUM_COEFFS)
            .map(|i| F::from(i as u64 * 3 + 7))
            .collect();

        let p0 = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let f0 = MultilinearExtension { point: p0.0 };

        let embedding = config.embedding();
        // evaluations[j * n + i] = ⟨wⱼ, fᵢ⟩
        // 1 form, 2 vectors: evaluations = [⟨w₀, f₀⟩, ⟨w₀, f₁⟩]
        let eval_0_0 = f0.evaluate(embedding, &v0);
        let eval_0_1 = f0.evaluate(embedding, &v1);

        prove_and_verify(
            &config,
            vec![v0, v1],
            vec![Box::new(f0)],
            &[eval_0_0, eval_0_1],
        );
    }

    #[test]
    fn test_zk_prove_verify_multi_vector_multi_form() {
        let mut rng = ark_std::test_rng();
        let config = make_test_config_batch(2);

        let v0: Vec<F> = (0..TEST_NUM_COEFFS)
            .map(|i| F::from(i as u64 + 1))
            .collect();
        let v1: Vec<F> = (0..TEST_NUM_COEFFS)
            .map(|i| F::from(i as u64 * 3 + 7))
            .collect();

        let p0 = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let p1 = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let f0 = MultilinearExtension { point: p0.0 };
        let f1 = MultilinearExtension { point: p1.0 };

        let embedding = config.embedding();
        // Row-major: evaluations[j * n + i] = ⟨wⱼ, fᵢ⟩
        // 2 forms × 2 vectors = 4 evaluations
        let eval_0_0 = f0.evaluate(embedding, &v0);
        let eval_0_1 = f0.evaluate(embedding, &v1);
        let eval_1_0 = f1.evaluate(embedding, &v0);
        let eval_1_1 = f1.evaluate(embedding, &v1);

        prove_and_verify(
            &config,
            vec![v0, v1],
            vec![Box::new(f0), Box::new(f1)],
            &[eval_0_0, eval_0_1, eval_1_0, eval_1_1],
        );
    }

    // =====================================================================
    // Soundness / negative tests
    // =====================================================================

    /// Verification must reject when the public evaluations are tampered with.
    /// Both `Err` and a panic (from debug transcript checks) count as rejection.
    #[test]
    fn test_zk_rejects_wrong_evaluations() {
        let mut rng = ark_std::test_rng();
        let config = make_test_config();

        let vector: Vec<F> = (0..TEST_NUM_COEFFS)
            .map(|i| F::from(i as u64 + 1))
            .collect();
        let p0 = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let p1 = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let f0 = MultilinearExtension { point: p0.0 };
        let f1 = MultilinearExtension { point: p1.0 };

        let embedding = config.embedding();
        let evaluations = vec![
            f0.evaluate(embedding, &vector),
            f1.evaluate(embedding, &vector),
        ];

        let forms: Vec<Box<dyn LinearForm<F>>> = vec![Box::new(f0), Box::new(f1)];
        let prove_forms = to_prove_forms(&forms, vector.len());

        let ds = DomainSeparator::protocol(&config)
            .session(&format!("zk2-wrong-eval {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);
        let witness = config.commit(&mut prover_state, &[&vector]);
        config.prove(
            &mut prover_state,
            vec![Cow::Owned(vector)],
            witness,
            prove_forms,
            Cow::Borrowed(&evaluations),
        );
        let proof = prover_state.proof();

        let mut wrong_evaluations = evaluations;
        wrong_evaluations[0] += F::ONE;

        let weight_refs: Vec<&dyn LinearForm<F>> = forms
            .iter()
            .map(|f| f.as_ref() as &dyn LinearForm<F>)
            .collect();

        let verify_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut verifier_state = VerifierState::new_std(&ds, &proof);
            let commitments = config
                .receive_commitments(&mut verifier_state)
                .expect("receive_commitments");
            config
                .verify(
                    &mut verifier_state,
                    &weight_refs,
                    &wrong_evaluations,
                    &commitments,
                )?
                .verify(weight_refs.iter().copied())
        }));
        if let Ok(result) = verify_outcome {
            assert!(
                result.is_err(),
                "verification should reject wrong public evaluations"
            );
        }
    }

    /// Verification must reject when the proof transcript is corrupted.
    #[test]
    fn test_zk_rejects_tampered_proof() {
        let mut rng = ark_std::test_rng();
        let config = make_test_config();

        let vector: Vec<F> = (0..TEST_NUM_COEFFS)
            .map(|i| F::from(i as u64 + 1))
            .collect();
        let p0 = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let p1 = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let f0 = MultilinearExtension { point: p0.0 };
        let f1 = MultilinearExtension { point: p1.0 };

        let embedding = config.embedding();
        let evaluations = vec![
            f0.evaluate(embedding, &vector),
            f1.evaluate(embedding, &vector),
        ];

        let forms: Vec<Box<dyn LinearForm<F>>> = vec![Box::new(f0), Box::new(f1)];
        let prove_forms = to_prove_forms(&forms, vector.len());

        let ds = DomainSeparator::protocol(&config)
            .session(&format!("zk2-tamper {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);
        let witness = config.commit(&mut prover_state, &[&vector]);
        config.prove(
            &mut prover_state,
            vec![Cow::Owned(vector)],
            witness,
            prove_forms,
            Cow::Borrowed(&evaluations),
        );

        let mut tampered_proof = prover_state.proof();
        if let Some(last) = tampered_proof.narg_string.last_mut() {
            *last ^= 1;
        } else {
            panic!("expected non-empty proof transcript");
        }

        let weight_refs: Vec<&dyn LinearForm<F>> = forms
            .iter()
            .map(|f| f.as_ref() as &dyn LinearForm<F>)
            .collect();

        let verify_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut verifier_state = VerifierState::new_std(&ds, &tampered_proof);
            let commitments = config
                .receive_commitments(&mut verifier_state)
                .expect("receive_commitments");
            config
                .verify(
                    &mut verifier_state,
                    &weight_refs,
                    &evaluations,
                    &commitments,
                )?
                .verify(weight_refs.iter().copied())
        }));
        if let Ok(result) = verify_outcome {
            assert!(
                result.is_err(),
                "verification should reject tampered proof bytes"
            );
        }
        // A panic is also a valid rejection (debug transcript checks).
    }

    /// Verify that `unique_decoding: true` is rejected at config construction.
    ///
    /// zkWHIR 2.0's "Alternative Randomness Sampling" requires OOD queries
    /// (Step 5d: the prover sends ood_f̂, m_evals, g_evals at each OOD point).
    /// With unique decoding, `out_domain_samples = 0`, which breaks the
    /// protocol in two ways:
    ///
    /// 1. `Commitment::num_vectors()` derives the vector count from the OOD
    ///    evaluation matrix as `matrix.len() / points.len()`. With 0 OOD
    ///    points this division is undefined, silently returning 0.
    ///
    /// 2. The verifier uses `num_vectors()` to build `ProtocolDims`, so all
    ///    subsequent transcript reads are misaligned — the verifier expects
    ///    0 m_evals per point while the prover sent 1.
    ///
    /// This is by design: the paper's ZK construction relies on OOD openings
    /// to bind the blinding polynomial claims into the Fiat-Shamir transcript.
    /// Without them the simulator cannot produce indistinguishable transcripts.
    #[test]
    #[should_panic(expected = "zkWHIR 2.0 requires list decoding")]
    fn test_zk_unique_decoding_unsupported() {
        let whir_params = ProtocolParameters {
            unique_decoding: true,
            security_level: 32,
            pow_bits: 0,
            initial_folding_factor: 2,
            folding_factor: 2,
            starting_log_inv_rate: 1,
            batch_size: 1,
            hash_id: hash::SHA2,
        };
        Config::<F>::new(TEST_NUM_VARIABLES, &whir_params);
    }

    /// Round-trip test where `mu % ell == 0` (rem = 0).
    ///
    /// When rem = 0, Φ₀ and Φ₁ extract the same bit window. This exercises
    /// the overlapping-window path through `build_beq_tables`,
    /// `build_weight_covectors`, and the Step 7 diagonal check.
    ///
    /// Parameters: num_vars=20, security_level=16 → ell=10, rem = 20 % 10 = 0.
    #[test]
    fn test_zk_prove_verify_zero_rem() {
        const NUM_VARS: usize = 20;
        const NUM_COEFFS: usize = 1 << NUM_VARS;

        let mut rng = ark_std::test_rng();
        let whir_params = ProtocolParameters {
            unique_decoding: false,
            security_level: 16,
            pow_bits: 0,
            initial_folding_factor: 2,
            folding_factor: 2,
            starting_log_inv_rate: 1,
            batch_size: 1,
            hash_id: hash::SHA2,
        };
        let mut config = Config::new(NUM_VARS, &whir_params);
        config.disable_pow();

        let ell = config.blinding_polynomial.initial_num_variables() - 1;
        assert_eq!(NUM_VARS % ell, 0, "test requires rem == 0");

        let vector = vec![F::ONE; NUM_COEFFS];
        let point = MultilinearPoint::rand(&mut rng, NUM_VARS);
        let form = MultilinearExtension { point: point.0 };
        let evaluation = form.evaluate(config.embedding(), &vector);

        prove_and_verify(&config, vec![vector], vec![Box::new(form)], &[evaluation]);
    }

    /// Generate an honest zkWHIR proof and sanity-check that it verifies.
    /// Returns the domain separator and proof for use in forgery tests.
    fn honest_proof_and_verify(
        config: &Config<F>,
        vectors: &[&[F]],
        forms: &[Box<dyn LinearForm<F>>],
        evals: &[F],
    ) -> (DomainSeparator<'static, Empty>, Proof) {
        let ds = DomainSeparator::protocol(config)
            .session(&format!("audit {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut ps = ProverState::new_std(&ds);
        let witness = config.commit(&mut ps, vectors);
        config.prove(
            &mut ps,
            vectors.iter().map(|&v| Cow::Borrowed(v)).collect(),
            witness,
            to_prove_forms(forms, vectors[0].len()),
            Cow::Borrowed(evals),
        );
        let proof = ps.proof();

        let weights: Vec<&dyn LinearForm<F>> = forms
            .iter()
            .map(|f| f.as_ref() as &dyn LinearForm<F>)
            .collect();
        let mut vs = VerifierState::new_std(&ds, &proof);
        let commitments = config.receive_commitments(&mut vs).unwrap();
        config
            .verify(&mut vs, &weights, evals, &commitments)
            .unwrap()
            .verify(weights.iter().copied())
            .unwrap();

        (ds, proof)
    }

    /// Run the zkWHIR verifier with the given evaluations.
    /// Returns `true` if accepted (soundness bug), `false` if rejected.
    ///
    /// Uses `catch_unwind` because the `verify!` macro can either return
    /// `Err` or panic depending on build configuration — both count as
    /// correct rejection.
    fn verifier_accepts(
        config: &Config<F>,
        ds: &DomainSeparator<'_, Empty>,
        proof: &Proof,
        forms: &[Box<dyn LinearForm<F>>],
        claimed_evals: &[F],
    ) -> bool {
        let weights: Vec<&dyn LinearForm<F>> = forms
            .iter()
            .map(|f| f.as_ref() as &dyn LinearForm<F>)
            .collect();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut vs = VerifierState::new_std(ds, proof);
            let commitments = config.receive_commitments(&mut vs).unwrap();
            config
                .verify(&mut vs, &weights, claimed_evals, &commitments)
                .and_then(|fc| fc.verify(weights.iter().copied()))
        }));
        matches!(result, Ok(Ok(())))
    }

    /// α-cancelling forgery across batched vectors (n=2, f=1) is rejected.
    /// Extracts α from transcript, constructs `[+Δ, −Δ/α]` preserving the
    /// batched sum — verifier rejects because evals are individually bound.
    #[test]
    fn test_rejects_alpha_cancelling_forgery() {
        let config = make_test_config_batch(2);
        let mut rng = ark_std::test_rng();

        let v0: Vec<F> = (0..TEST_NUM_COEFFS)
            .map(|i| F::from(i as u64 + 1))
            .collect();
        let v1: Vec<F> = (0..TEST_NUM_COEFFS)
            .map(|i| F::from(i as u64 * 3 + 7))
            .collect();
        let point = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let form = MultilinearExtension { point: point.0 };
        let embedding = config.embedding();
        let evals = vec![form.evaluate(embedding, &v0), form.evaluate(embedding, &v1)];
        let forms: Vec<Box<dyn LinearForm<F>>> = vec![Box::new(form)];

        let (ds, proof) = honest_proof_and_verify(&config, &[&v0, &v1], &forms, &evals);

        // Replay the verifier transcript to extract the batching coefficient α.
        //
        // zkWHIR transcript after receive_commitments (n=2, f=1):
        //   1. V → P: β                          (verifier challenge)
        //   2. P → V: G = ⟨w, g⟩                 (1 g_claim for 1 form)
        //   3. P → V: eval₀, eval₁               (2 evals — the fix)
        //   4. V → P: α via geometric_challenge(2) → [1, α]
        let alpha = {
            let mut vs = VerifierState::new_std(&ds, &proof);
            let _ = config.receive_commitments(&mut vs).unwrap();
            let _beta: F = vs.verifier_message(); // step 1
            let _g_claim: F = vs.prover_message().unwrap(); // step 2
            let _eval_0: F = vs.prover_message().unwrap(); // step 3
            let _eval_1: F = vs.prover_message().unwrap(); // step 3
            geometric_challenge::<_, F>(&mut vs, 2)[1] // step 4
        };

        let delta = F::from(42u64);
        let mut forged = evals.clone();
        forged[0] += delta;
        forged[1] -= delta / alpha;
        assert_eq!(evals[0] + alpha * evals[1], forged[0] + alpha * forged[1]);

        assert!(
            !verifier_accepts(&config, &ds, &proof, &forms, &forged),
            "REGRESSION issue #1: α-cancelling forgery must be rejected"
        );
    }

    /// G-claim forgery compensated via ρ (n=1, f=1) is rejected.
    ///
    /// Full manual transcript replay with a malicious prover that:
    /// 1. Commits honestly.
    /// 2. Sends forged G' = G + Δ.
    /// 3. Absorbs honest eval (must commit before ρ is sampled).
    /// 4. After ρ is sampled, constructs e' = e − Δ/ρ to preserve ρ·e + G.
    /// 5. Completes the rest of the proof honestly.
    ///
    /// Verifier reads the honest eval from the transcript and rejects e'.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_rejects_g_claim_forgery_via_rho() {
        let mut rng = ark_std::test_rng();
        let config = make_test_config();

        let vector = vec![F::ONE; TEST_NUM_COEFFS];
        let point = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let form = MultilinearExtension { point: point.0 };
        let honest_eval = form.evaluate(config.embedding(), &vector);

        let forms: Vec<Box<dyn LinearForm<F>>> = vec![Box::new(form)];
        let weight_refs: Vec<&dyn LinearForm<F>> = forms
            .iter()
            .map(|f| f.as_ref() as &dyn LinearForm<F>)
            .collect();

        let ds = DomainSeparator::protocol(&config)
            .session(&format!("audit {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);
        let witness = config.commit(&mut prover_state, &[&vector]);

        let Witness {
            f_hat_witness,
            blinding_poly_witness,
            f_hat_polys,
            secrets,
        } = witness;

        let dims = ProtocolDims::new(&config, 1);
        let size = dims.size;

        // ── MALICIOUS Step 2: send g_claim + Δ ──

        let beta: F = prover_state.verifier_message();
        let beta_powers = geometric_sequence(beta, dims.num_g_polys());
        let g_poly: Vec<F> = (0..size)
            .map(|idx| {
                beta_powers
                    .iter()
                    .enumerate()
                    .map(|(i, &bp)| bp * secrets.g_polys[i][dims.phi_i_bits(idx, i)])
                    .sum()
            })
            .collect();
        let honest_g_claim: F = {
            let mut buf = vec![F::ZERO; size];
            forms[0].accumulate(&mut buf, F::ONE);
            dot(&buf, &g_poly)
        };

        let delta = F::from(77u64);
        prover_state.prover_message(&(honest_g_claim + delta)); // forged G'
        prover_state.prover_message(&honest_eval); // must absorb before ρ

        let alpha_coeffs: Vec<F> = geometric_challenge(&mut prover_state, 1);
        let rho: F = prover_state.verifier_message();
        assert_ne!(rho, F::ZERO);

        // f_zk = ρ·f + g (honest)
        let mut f_zk: Vec<F> = vector.iter().map(|&v| rho * v).collect();
        for (f, &g) in f_zk.iter_mut().zip(g_poly.iter()) {
            *f += g;
        }
        drop(g_poly);

        let combined_claim = rho * honest_eval + honest_g_claim;

        // Step 4: sumcheck
        let constraint_rlc: Vec<F> = geometric_challenge(&mut prover_state, 1);
        let mut covector = vec![F::ZERO; size];
        for (coeff, lf) in constraint_rlc.iter().zip(forms.iter()) {
            lf.accumulate(&mut covector, *coeff);
        }
        let mut the_sum: F = constraint_rlc[0] * combined_claim;
        let folding_randomness = config.blinded_polynomial.initial_sumcheck.prove(
            &mut prover_state,
            &mut f_zk,
            &mut covector,
            &mut the_sum,
        );

        // Steps 5-6: honest
        let r_bar = &folding_randomness.0;
        let eq_weights_vec = compute_eq_weights(r_bar);
        let RsFoldCoeffs {
            masking_coeffs_all,
            g_i_coeffs,
        } = compute_rs_fold_blinding_coeffs(
            &eq_weights_vec,
            &secrets.g_polys,
            &secrets.masking_polys,
            &alpha_coeffs,
            rho,
            dims,
        );

        let round_config = &config.blinded_polynomial.round_configs[0];
        let folded_commit = round_config
            .irs_committer
            .commit(&mut prover_state, &[&f_zk]);
        round_config.pow.prove(&mut prover_state);
        let f_hat_refs: Vec<&[F]> = f_hat_polys.iter().map(Vec::as_slice).collect();
        let in_domain = config
            .blinded_polynomial
            .initial_committer
            .open_from_coeffs(&mut prover_state, &[&f_hat_refs], &[&f_hat_witness]);

        let mut lambda_z_points: Vec<F> = Vec::new();
        let send_blinding = |ps: &mut ProverState<_, _>, z: F| {
            for m in &masking_coeffs_all {
                ps.prover_message(&univariate_evaluate(m, z));
            }
            for g in &g_i_coeffs {
                ps.prover_message(&univariate_evaluate(g, z));
            }
        };

        let f_hat_combined = &f_hat_polys[0];
        let mu = dims.mu;
        for &z in &folded_commit.out_of_domain().points {
            prover_state.prover_message(&multilinear_extend(
                f_hat_combined,
                &build_fold_args(r_bar, z, mu),
            ));
            send_blinding(&mut prover_state, z);
            lambda_z_points.push(z);
        }
        for &z in &in_domain.points {
            send_blinding(&mut prover_state, z);
            lambda_z_points.push(z);
        }
        {
            let stir_challenges: Vec<UnivariateEvaluation<F>> = folded_commit
                .out_of_domain()
                .evaluators(round_config.initial_size())
                .chain(in_domain.evaluators(round_config.initial_size()))
                .collect();
            let ood_evals = folded_commit.out_of_domain().values(&[F::ONE]);
            let num_ood = folded_commit.out_of_domain().points.len();
            let embedding = Identity::new();
            let stir_evals: Vec<F> = ood_evals
                .chain(
                    stir_challenges[num_ood..]
                        .iter()
                        .map(|ch| ch.evaluate(&embedding, &f_zk)),
                )
                .collect();
            let stir_rlc: Vec<F> = geometric_challenge(&mut prover_state, stir_challenges.len());
            UnivariateEvaluation::accumulate_many(&stir_challenges, &mut covector, &stir_rlc);
            the_sum += dot(&stir_rlc, &stir_evals);
        }
        let round0_folding =
            round_config
                .sumcheck
                .prove(&mut prover_state, &mut f_zk, &mut covector, &mut the_sum);
        let remaining = whir::rounds::prove_remaining_rounds(
            &config.blinded_polynomial.round_configs,
            &whir::rounds::FinalRoundConfig {
                sumcheck: &config.blinded_polynomial.final_sumcheck,
                pow: &config.blinded_polynomial.final_pow,
            },
            &mut prover_state,
            &mut whir::rounds::SumcheckState {
                vector: &mut f_zk,
                covector: &mut covector,
                the_sum: &mut the_sum,
            },
            folded_commit,
            &round0_folding,
        );
        let gamma_points = remaining.first_in_domain_points;
        let _ = config
            .blinded_polynomial
            .initial_committer
            .open_at_indices_from_coeffs(
                &mut prover_state,
                &[&f_hat_refs],
                &[&f_hat_witness],
                &gamma_to_f_hat_indices(&gamma_points, &config),
            );
        drop(f_hat_polys);
        for &gamma in &gamma_points {
            send_blinding(&mut prover_state, gamma);
            lambda_z_points.push(gamma);
        }
        drop(f_zk);
        drop(covector);

        // Step 7: blinding proof (honest)
        let tau: F = prover_state.verifier_message();
        let beq_tables = build_beq_tables(&lambda_z_points, &eq_weights_vec, tau, dims);
        let weight_covectors = build_weight_covectors(&beq_tables, rho, &alpha_coeffs, dims);
        let mut eval_matrix = Vec::with_capacity(dims.num_blinding_vecs * dims.num_blinding_vecs);
        for w in &weight_covectors {
            for v in &secrets.blinding_vectors {
                eval_matrix.push(dot(w, v));
            }
        }
        for e in &eval_matrix {
            prover_state.prover_message(e);
        }
        let blinding_forms: Vec<Box<dyn LinearForm<F>>> = weight_covectors
            .into_iter()
            .map(|cv| Box::new(Covector::new(cv)) as _)
            .collect();
        let blinding_cows: Vec<Cow<'_, [F]>> = secrets
            .blinding_vectors
            .iter()
            .map(|v| Cow::Borrowed(v.as_slice()))
            .collect();
        // Re-encode blinding_poly_witness.matrix (cleared at commit time);
        // mirrors prover.rs::prove_blinded_polynomial before
        // `prove_blinding_polynomial`.
        let blinding_refs: Vec<&[F]> = secrets
            .blinding_vectors
            .iter()
            .map(|v| v.as_slice())
            .collect();
        let mut blinding_poly_witness = blinding_poly_witness;
        blinding_poly_witness.matrix = crate::algebra::ntt::interleaved_rs_encode(
            &blinding_refs,
            config.blinding_polynomial.initial_committer.codeword_length,
            config.blinding_polynomial.initial_committer.interleaving_depth,
        );
        drop(blinding_refs);
        let _ = config.blinding_polynomial.prove(
            &mut prover_state,
            blinding_cows,
            vec![Cow::Borrowed(&blinding_poly_witness)],
            blinding_forms,
            Cow::Owned(eval_matrix),
        );

        // Verify with forged e' = e − Δ/ρ.
        let proof = prover_state.proof();
        let forged_eval = honest_eval - delta / rho;
        assert_ne!(forged_eval, honest_eval);

        let attack = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut vs = VerifierState::new_std(&ds, &proof);
            let commitments = config.receive_commitments(&mut vs).unwrap();
            config
                .verify(&mut vs, &weight_refs, &[forged_eval], &commitments)
                .and_then(|fc| fc.verify(weight_refs.iter().copied()))
        }));
        assert!(
            !matches!(attack, Ok(Ok(()))),
            "REGRESSION issue #2: g_claim forgery (G'=G+Δ, e'=e−Δ/ρ) must be rejected"
        );
    }

    /// Constraint-RLC-cancelling forgery across forms (n=1, f=2) is rejected.
    /// Extracts c₁ from transcript, constructs `[+Δ, −Δ/c₁]` preserving the
    /// weighted sum — verifier rejects because evals are individually bound.
    #[test]
    fn test_rejects_constraint_rlc_cancelling_forgery() {
        let config = make_test_config();
        let mut rng = ark_std::test_rng();

        let vector: Vec<F> = (0..TEST_NUM_COEFFS)
            .map(|i| F::from(i as u64 + 1))
            .collect();
        let p0 = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let p1 = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let f0 = MultilinearExtension { point: p0.0 };
        let f1 = MultilinearExtension { point: p1.0 };
        let embedding = config.embedding();
        let evals = vec![
            f0.evaluate(embedding, &vector),
            f1.evaluate(embedding, &vector),
        ];
        let forms: Vec<Box<dyn LinearForm<F>>> = vec![Box::new(f0), Box::new(f1)];

        let (ds, proof) = honest_proof_and_verify(&config, &[&vector], &forms, &evals);

        // Replay the verifier transcript to extract constraint RLC coefficient c₁.
        //
        // zkWHIR transcript after receive_commitments (n=1, f=2):
        //   1. V → P: β                                    (verifier challenge)
        //   2. P → V: G₀, G₁                               (2 g_claims for 2 forms)
        //   3. P → V: eval₀, eval₁                         (2 evals — the fix)
        //   4. V → P: α via geometric_challenge(1) → [1]   (no transcript squeeze for n=1)
        //   5. V → P: ρ                                    (verifier challenge)
        //   6. V → P: c via geometric_challenge(2) → [1, c₁]
        let c1 = {
            let mut vs = VerifierState::new_std(&ds, &proof);
            let _ = config.receive_commitments(&mut vs).unwrap();
            let _beta: F = vs.verifier_message(); // step 1
            for _ in 0..2 {
                let _: F = vs.prover_message().unwrap();
            } // step 2: g_claims
            for _ in 0..2 {
                let _: F = vs.prover_message().unwrap();
            } // step 3: evals
              // step 4: geometric_challenge(1) returns [ONE] without squeezing,
              // but must be called to keep the transcript replay in sync.
            let _alpha: Vec<F> = geometric_challenge(&mut vs, 1);
            let _rho: F = vs.verifier_message(); // step 5
            geometric_challenge::<_, F>(&mut vs, 2)[1] // step 6: c₁
        };

        let delta = F::from(99u64);
        let mut forged = evals.clone();
        forged[0] += delta;
        forged[1] -= delta / c1;
        assert_eq!(evals[0] + c1 * evals[1], forged[0] + c1 * forged[1]);

        assert!(
            !verifier_accepts(&config, &ds, &proof, &forms, &forged),
            "REGRESSION issue #3: constraint-RLC-cancelling forgery must be rejected"
        );
    }

    /// All forgery surfaces combined (n=2, f=2, 4 evaluations).
    /// Tests single-entry, cross-vector, and cross-form forgeries
    /// in one proof to exercise α, ρ, and constraint_rlc binding together.
    #[test]
    fn test_rejects_all_forgery_patterns_n2_f2() {
        let config = make_test_config_batch(2);
        let mut rng = ark_std::test_rng();

        let v0: Vec<F> = (0..TEST_NUM_COEFFS)
            .map(|i| F::from(i as u64 + 1))
            .collect();
        let v1: Vec<F> = (0..TEST_NUM_COEFFS)
            .map(|i| F::from(i as u64 * 3 + 7))
            .collect();
        let p0 = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let p1 = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let f0 = MultilinearExtension { point: p0.0 };
        let f1 = MultilinearExtension { point: p1.0 };
        let embedding = config.embedding();
        let evals = vec![
            f0.evaluate(embedding, &v0),
            f0.evaluate(embedding, &v1),
            f1.evaluate(embedding, &v0),
            f1.evaluate(embedding, &v1),
        ];
        let forms: Vec<Box<dyn LinearForm<F>>> = vec![Box::new(f0), Box::new(f1)];

        let (ds, proof) = honest_proof_and_verify(&config, &[&v0, &v1], &forms, &evals);

        let mut fa = evals.clone();
        fa[0] += F::from(1u64);
        assert!(
            !verifier_accepts(&config, &ds, &proof, &forms, &fa),
            "single-entry forgery must be rejected"
        );

        let mut fb = evals.clone();
        fb[0] += F::from(99u64);
        fb[1] -= F::from(99u64);
        assert!(
            !verifier_accepts(&config, &ds, &proof, &forms, &fb),
            "cross-vector forgery must be rejected"
        );

        let mut fc = evals.clone();
        fc[0] += F::from(55u64);
        fc[2] -= F::from(55u64);
        assert!(
            !verifier_accepts(&config, &ds, &proof, &forms, &fc),
            "cross-form forgery must be rejected"
        );
    }

    /// Linear form replay: w' with same MLE at the final point
    /// but different inner product. Verifier must reject when the caller
    /// binds weights into the transcript before calling prove/verify.
    ///
    /// This test demonstrates the recommended pattern: the caller absorbs
    /// weights into the transcript at the call site (caller responsibility).
    /// With this in place, a forged w' is caught at the binding read step.
    #[test]
    fn test_rejects_forged_linear_form_with_same_mle_at_eval_point() {
        use crate::algebra::eval_eq;

        let config = make_test_config();
        let mut rng = ark_std::test_rng();

        let vector: Vec<F> = (0..TEST_NUM_COEFFS)
            .map(|i| F::from(i as u64 + 1))
            .collect();
        let point = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let form = MultilinearExtension { point: point.0 };
        let embedding = config.embedding();
        let honest_eval = form.evaluate(embedding, &vector);
        let forms: Vec<Box<dyn LinearForm<F>>> = vec![Box::new(form)];

        // Honest prover with caller-side weight binding before `prove`.
        let ds = DomainSeparator::protocol(&config)
            .session(&format!("audit {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut ps = ProverState::new_std(&ds);
        let witness = config.commit(&mut ps, &[&vector]);

        // Caller binds weight covectors into the transcript.
        for f in &forms {
            let mut cv = vec![F::ZERO; vector.len()];
            f.accumulate(&mut cv, F::ONE);
            for v in &cv {
                ps.prover_message(v);
            }
        }

        config.prove(
            &mut ps,
            vec![Cow::Borrowed(vector.as_slice())],
            witness,
            to_prove_forms(&forms, vector.len()),
            Cow::Borrowed(&[honest_eval]),
        );
        let proof = ps.proof();

        // Honest verification with the matching caller-side binding — succeeds
        // and yields the evaluation_point used by the attack.
        let eval_point = {
            let weights: Vec<&dyn LinearForm<F>> = forms
                .iter()
                .map(|f| f.as_ref() as &dyn LinearForm<F>)
                .collect();
            let mut vs = VerifierState::new_std(&ds, &proof);
            let commitments = config.receive_commitments(&mut vs).unwrap();

            for f in &forms {
                let mut cv = vec![F::ZERO; vector.len()];
                f.accumulate(&mut cv, F::ONE);
                for &expected in &cv {
                    let read: F = vs.prover_message().unwrap();
                    assert_eq!(read, expected);
                }
            }

            config
                .verify(&mut vs, &weights, &[honest_eval], &commitments)
                .unwrap()
                .evaluation_point
        };

        // Build forged form w' via two-index delta on the honest covector.
        // w'[a] += eq(r, b), w'[b] -= eq(r, a) preserves the MLE at r.
        let mut forged_weights = vec![F::ZERO; TEST_NUM_COEFFS];
        forms[0].accumulate(&mut forged_weights, F::ONE);

        let mut eq_r = vec![F::ZERO; TEST_NUM_COEFFS];
        eval_eq(&mut eq_r, &eval_point, F::ONE);

        let (a, b) = (0, 1);
        forged_weights[a] += eq_r[b];
        forged_weights[b] -= eq_r[a];
        let forged_form = Covector::new(forged_weights);

        // Sanity: MLE agrees at r, but inner product differs.
        assert_eq!(
            forged_form.mle_evaluate(&eval_point),
            forms[0].as_ref().mle_evaluate(&eval_point),
        );
        assert_ne!(forged_form.evaluate(embedding, &vector), honest_eval);

        // Attack: try to verify with forged form w' against the proof.
        // The caller-side binding catches the forgery — when the verifier
        // absorbs w'.accumulate() it doesn't match the bytes the prover
        // wrote (which encoded w), so the read check fails.
        let forged_forms: Vec<Box<dyn LinearForm<F>>> = vec![Box::new(forged_form)];
        let attack_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let weights: Vec<&dyn LinearForm<F>> = forged_forms
                .iter()
                .map(|f| f.as_ref() as &dyn LinearForm<F>)
                .collect();
            let mut vs = VerifierState::new_std(&ds, &proof);
            let commitments = config.receive_commitments(&mut vs)?;

            // Caller-side binding with the FORGED form catches the attack:
            // the bytes in the proof encode w (what the honest prover bound),
            // but the verifier here computes w'.accumulate() — they mismatch.
            for f in &forged_forms {
                let mut cv = vec![F::ZERO; vector.len()];
                f.accumulate(&mut cv, F::ONE);
                for &expected in &cv {
                    let read: F = vs.prover_message()?;
                    crate::verify!(read == expected);
                }
            }

            config
                .verify(&mut vs, &weights, &[honest_eval], &commitments)
                .and_then(|fc| fc.verify(weights.iter().copied()))
        }));

        assert!(
            !matches!(attack_result, Ok(Ok(()))),
            "REGRESSION : caller-bound w must reject forged w' with matching MLE at r"
        );
    }
}
