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
        let blinded_sec = self.blinded_polynomial.security_level(
            num_vectors,
            num_linear_forms + self.blinded_polynomial.initial_committer.out_domain_samples,
        );
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

    use super::Config;
    use crate::{
        algebra::{
            fields::Field64,
            linear_form::{Covector, Evaluate, LinearForm, MultilinearExtension},
            MultilinearPoint,
        },
        hash,
        parameters::ProtocolParameters,
        transcript::{codecs::Empty, DomainSeparator, ProverState, VerifierState},
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

    /// Helper: run a full prove → verify cycle for zkWHIR 2.0.
    /// `vectors` is a list of witness polynomial evaluation tables.
    /// `evaluations` is row-major: `evaluations[j * n + i]` = ⟨wⱼ, fᵢ⟩.
    #[allow(clippy::needless_pass_by_value)]
    fn prove_and_verify(
        config: &Config<F>,
        vectors: Vec<Vec<F>>,
        forms: Vec<Box<dyn LinearForm<F>>>,
        evaluations: &[F],
    ) {
        let prove_forms = to_prove_forms(forms.as_slice(), vectors[0].len());

        let ds = DomainSeparator::protocol(config)
            .session(&format!("zk2-pv {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);

        let poly_refs: Vec<&[F]> = vectors.iter().map(|v| v.as_slice()).collect();
        let witness = config.commit(&mut prover_state, &poly_refs);
        config.prove(
            &mut prover_state,
            vectors.into_iter().map(Cow::Owned).collect(),
            witness,
            prove_forms,
            Cow::Borrowed(evaluations),
        );

        let proof = prover_state.proof();
        let mut verifier_state = VerifierState::new_std(&ds, &proof);

        let commitments = config
            .receive_commitments(&mut verifier_state)
            .expect("receive_commitments failed");

        let weight_refs: Vec<&dyn LinearForm<F>> = forms
            .iter()
            .map(|f| f.as_ref() as &dyn LinearForm<F>)
            .collect();

        // Blinded polynomial FinalClaim: verify the linear form RLC.
        // (Blinding polynomial FinalClaim is verified internally by verify().)
        config
            .verify(&mut verifier_state, &weight_refs, evaluations, &commitments)
            .expect("verification failed")
            .verify(weight_refs)
            .expect("blinded polynomial final claim check failed");
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
    fn test_zk_prove_verify_with_only_initial_ood_claims() {
        let config = make_test_config();
        let initial_ood_samples = config
            .blinded_polynomial
            .initial_committer
            .out_domain_samples;
        assert_ne!(initial_ood_samples, 0);
        let vector = vec![F::ONE; TEST_NUM_COEFFS];

        prove_and_verify(&config, vec![vector], Vec::new(), &[]);
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

    /// Soundness: a malicious prover who generates a proof for a wrong evaluation
    /// must be rejected. If verify() accepts, it means the prover can forge
    /// arbitrary evaluation claims.
    #[test]
    fn test_zk_malicious_prover_wrong_evaluation() {
        let mut rng = ark_std::test_rng();
        let config = make_test_config();

        let vector = vec![F::ONE; TEST_NUM_COEFFS];
        let point = MultilinearPoint::rand(&mut rng, TEST_NUM_VARIABLES);
        let form = MultilinearExtension { point: point.0 };
        let correct_evaluation = form.evaluate(config.embedding(), &vector);
        let wrong_evaluation = correct_evaluation + F::from(42u64);

        let forms: Vec<Box<dyn LinearForm<F>>> = vec![Box::new(form)];
        let prove_forms = to_prove_forms(&forms, vector.len());
        let weight_refs: Vec<&dyn LinearForm<F>> = forms
            .iter()
            .map(|f| f.as_ref() as &dyn LinearForm<F>)
            .collect();

        let ds = DomainSeparator::protocol(&config)
            .session(&format!("zk2-malicious {}:{}", file!(), line!()))
            .instance(&Empty);

        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut prover_state = ProverState::new_std(&ds);
            let witness = config.commit(&mut prover_state, &[&vector]);
            config.prove(
                &mut prover_state,
                vec![Cow::Borrowed(vector.as_slice())],
                witness,
                prove_forms,
                Cow::Owned(vec![wrong_evaluation]),
            );

            let proof = prover_state.proof();
            let mut verifier_state = VerifierState::new_std(&ds, &proof);
            let commitments = config
                .receive_commitments(&mut verifier_state)
                .expect("receive_commitments");
            config
                .verify(
                    &mut verifier_state,
                    &weight_refs,
                    &[wrong_evaluation],
                    &commitments,
                )?
                .verify(weight_refs.iter().copied())
        }));

        if let Ok(result) = outcome {
            assert!(
                result.is_err(),
                "SOUNDNESS BUG: verifier accepted wrong evaluation from malicious prover \
                 (correct={correct_evaluation:?}, claimed={wrong_evaluation:?})"
            );
        }
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
}
