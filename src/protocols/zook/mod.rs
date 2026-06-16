//! Zook ZK protocol — Construction 9.7.

pub mod commit;
pub mod prover;
pub mod verifier;

pub use commit::{Commitment, CommittedWitness};

pub use crate::protocols::params::protocol_config::ProtocolConfig;
use crate::{algebra::linear_form::LinearForm, transcript::VerificationResult, verify};

/// Output of [`ProtocolConfig::verify`].
///
/// The verifier has completed all round checks. The caller must finish
/// verification by checking that the input forms evaluate to the claimed value:
///
/// ```text
/// initial_claim_scale × Σ_j rlc_coefficients[j] × form_j.mle_evaluate(evaluation_point) == linear_forms_contribution
/// ```
#[must_use]
#[derive(Clone, Debug)]
pub struct FinalClaim<F: ark_ff::Field> {
    /// All sumcheck challenges from all rounds concatenated with basecase
    /// evaluation points. Length = log2(vector_size).
    pub evaluation_point: Vec<F>,
    /// Cumulative product of all code-switch `original_sl_coeff` values.
    pub initial_claim_scale: F,
    /// `opening.linear_form_evaluation − Σ_c constraint_contributions`.
    /// The portion of the protocol sum attributable to the input linear forms.
    pub linear_forms_contribution: F,
    /// Fiat-Shamir RLC coefficients for each input form.
    /// `rlc_coefficients[0] == F::ONE` always.
    pub rlc_coefficients: Vec<F>,
}

impl<F: ark_ff::Field> FinalClaim<F> {
    /// Complete the verification started by [`ProtocolConfig::verify`].
    ///
    /// Checks `initial_claim_scale × Σ_j rlc_coefficients[j] × form_j.mle_evaluate(evaluation_point) == linear_forms_contribution`.
    ///
    /// For `MultilinearExtension` forms this runs in O(num_forms × log N).
    /// For `Covector` forms the default `mle_evaluate` is O(N).
    pub fn verify(&self, linear_forms: &[&dyn LinearForm<F>]) -> VerificationResult<()>
    where
        F: Default,
    {
        assert_eq!(
            linear_forms.len(),
            self.rlc_coefficients.len(),
            "linear_forms.len() must match rlc_coefficients.len()"
        );
        let form_mle_sum: F = linear_forms
            .iter()
            .zip(&self.rlc_coefficients)
            .map(|(form, &g)| g * form.mle_evaluate(&self.evaluation_point))
            .sum();
        verify!(self.initial_claim_scale * form_mle_sum == self.linear_forms_contribution);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    use super::ProtocolConfig;
    use crate::{
        algebra::{
            embedding::Identity,
            fields::{Field256, Field64},
            linear_form::{Evaluate, LinearForm, MultilinearExtension},
            random_vector,
        },
        hash,
        protocols::params::spec::{
            DecodingRegime, FoldingFactor, Mode, PowBudget, RateSchedule, SecuritySpec, TuningSpec,
        },
        transcript::{codecs::Empty, DomainSeparator, ProverState, VerifierState},
    };

    // ── Small-test field and config (Field64, λ=40, fast) ───────────────────

    type SmallF = Field64;
    type SmallEmbed = Identity<SmallF>;

    fn small_spec(mode: Mode) -> SecuritySpec {
        SecuritySpec {
            mode,
            decoding_regime: DecodingRegime::Johnson,
            target_security_bits: 40,
            pow_budget: PowBudget::per_slot(10),
            hash_id: hash::BLAKE3,
        }
    }

    /// 2^8 witness, folding_factor 2 → multiple rounds.
    fn multi_round_tuning() -> TuningSpec {
        TuningSpec {
            vector_size: 1 << 8,
            starting_log_inv_rate: 1,
            folding_factor: FoldingFactor::Constant(2),
            rate_schedule: RateSchedule::Stepping,
        }
    }

    /// 2^3 witness → basecase-only (too small for any round).
    fn basecase_tuning() -> TuningSpec {
        TuningSpec {
            vector_size: 1 << 3,
            starting_log_inv_rate: 1,
            folding_factor: FoldingFactor::Constant(2),
            rate_schedule: RateSchedule::Stepping,
        }
    }

    // ── Test helpers ─────────────────────────────────────────────────────────

    /// A shared `&'static Empty` so that the returned `DomainSeparator` carries
    /// a `'static` lifetime (required by `VerifierState::new_std`).
    static EMPTY: Empty = Empty;

    /// Construct a deterministic `DomainSeparator` for the given label.
    fn make_ds(label: &str) -> DomainSeparator<'static, Empty> {
        DomainSeparator::protocol(&"zook-test")
            .session(&label.to_string())
            .instance(&EMPTY)
    }

    /// Build a witness, compute true evaluations, prove, and return the proof
    /// along with the domain separator, forms, and values for further checks.
    fn build_and_prove(
        config: &ProtocolConfig<SmallEmbed>,
        num_claims: usize,
        seed: u64,
        label: &str,
    ) -> (
        crate::transcript::Proof,
        DomainSeparator<'static, Empty>,
        Vec<MultilinearExtension<SmallF>>,
        Vec<SmallF>,
    ) {
        let embedding = <SmallEmbed as Default>::default();
        let mut rng = StdRng::seed_from_u64(seed);
        let witness: Vec<SmallF> = random_vector(&mut rng, config.tuning().vector_size);
        let mu = config.tuning().vector_size.trailing_zeros() as usize;

        let forms: Vec<MultilinearExtension<SmallF>> = (0..num_claims)
            .map(|_| MultilinearExtension {
                point: random_vector::<SmallF>(&mut rng, mu),
            })
            .collect();
        let values: Vec<SmallF> = forms
            .iter()
            .map(|f| f.evaluate(&embedding, &witness))
            .collect();
        let form_refs: Vec<&dyn LinearForm<SmallF>> =
            forms.iter().map(|f| f as &dyn LinearForm<SmallF>).collect();

        let ds = make_ds(label);
        let mut ps = ProverState::new_std(&ds);
        let committed = config.commit(&mut ps, &witness);
        config.prove(&mut ps, committed, &form_refs, &values);
        let proof = ps.proof();

        (proof, ds, forms, values)
    }

    /// Run a full roundtrip: commit → prove → verify → FinalClaim::verify.
    /// Panics if anything fails.
    fn full_roundtrip(
        config: &ProtocolConfig<SmallEmbed>,
        num_claims: usize,
        seed: u64,
        label: &str,
    ) {
        let (proof, ds, forms, values) = build_and_prove(config, num_claims, seed, label);
        let form_refs: Vec<&dyn LinearForm<SmallF>> =
            forms.iter().map(|f| f as &dyn LinearForm<SmallF>).collect();

        let mut vs = VerifierState::new_std(&ds, &proof);
        let commitment = config.receive_commitment(&mut vs).unwrap();
        let claim = config
            .verify(&mut vs, commitment, &form_refs, &values)
            .unwrap();
        claim.verify(&form_refs).expect("FinalClaim::verify failed");
        vs.check_eof().unwrap();
    }

    /// Expect verification to fail (handles both `verifier_panics` and normal builds).
    fn assert_verify_rejected(verify: impl FnOnce() -> crate::transcript::VerificationResult<()>) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(verify));
        match result {
            Err(_) | Ok(Err(_)) => {} // panicked or returned Err — failure as expected
            Ok(Ok(())) => panic!("expected verification to fail but it succeeded"),
        }
    }

    // ── Positive tests: happy paths ──────────────────────────────────────────

    #[test]
    fn roundtrip_zk_with_rounds() {
        let config = ProtocolConfig::<SmallEmbed>::derive(
            small_spec(Mode::ZeroKnowledge),
            multi_round_tuning(),
        )
        .unwrap();
        assert!(!config.rounds().is_empty(), "expected at least one round");
        full_roundtrip(&config, 1, 0, "roundtrip_zk_with_rounds");
    }

    #[test]
    fn roundtrip_standard_with_rounds() {
        let config =
            ProtocolConfig::<SmallEmbed>::derive(small_spec(Mode::Standard), multi_round_tuning())
                .unwrap();
        assert!(!config.rounds().is_empty(), "expected at least one round");
        full_roundtrip(&config, 1, 1, "roundtrip_standard_with_rounds");
    }

    #[test]
    fn roundtrip_zk_basecase_only() {
        let config = ProtocolConfig::<SmallEmbed>::derive(
            small_spec(Mode::ZeroKnowledge),
            basecase_tuning(),
        )
        .unwrap();
        assert!(config.rounds().is_empty(), "expected basecase-only plan");
        full_roundtrip(&config, 1, 2, "roundtrip_zk_basecase_only");
    }

    #[test]
    fn roundtrip_standard_basecase_only() {
        let config =
            ProtocolConfig::<SmallEmbed>::derive(small_spec(Mode::Standard), basecase_tuning())
                .unwrap();
        assert!(config.rounds().is_empty(), "expected basecase-only plan");
        full_roundtrip(&config, 1, 3, "roundtrip_standard_basecase_only");
    }

    #[test]
    fn roundtrip_multiple_claims_zk() {
        let config = ProtocolConfig::<SmallEmbed>::derive(
            small_spec(Mode::ZeroKnowledge),
            multi_round_tuning(),
        )
        .unwrap();
        full_roundtrip(&config, 4, 4, "roundtrip_multiple_claims_zk");
    }

    #[test]
    fn roundtrip_multiple_claims_standard() {
        let config =
            ProtocolConfig::<SmallEmbed>::derive(small_spec(Mode::Standard), multi_round_tuning())
                .unwrap();
        full_roundtrip(&config, 4, 5, "roundtrip_multiple_claims_standard");
    }

    // ── Negative tests: input validation ─────────────────────────────────────

    #[test]
    #[should_panic(expected = "linear_forms.len() != evaluations.len()")]
    fn prove_rejects_form_evaluation_count_mismatch() {
        let config =
            ProtocolConfig::<SmallEmbed>::derive(small_spec(Mode::Standard), multi_round_tuning())
                .unwrap();
        let mut rng = StdRng::seed_from_u64(0);
        let witness: Vec<SmallF> = random_vector(&mut rng, config.tuning().vector_size);
        let mu = config.tuning().vector_size.trailing_zeros() as usize;
        let form: MultilinearExtension<SmallF> = MultilinearExtension {
            point: random_vector(&mut rng, mu),
        };
        let ds = DomainSeparator::protocol(&"zook-test")
            .session(&"count-mismatch".to_string())
            .instance(&Empty);
        let mut ps = ProverState::new_std(&ds);
        let committed = config.commit(&mut ps, &witness);
        // 1 form but 2 values — should panic
        config.prove(
            &mut ps,
            committed,
            &[&form as &dyn LinearForm<SmallF>],
            &[SmallF::from(1u64), SmallF::from(2u64)],
        );
    }

    #[test]
    #[should_panic(expected = "zook requires ≥ 1")]
    fn prove_rejects_empty_forms() {
        let config =
            ProtocolConfig::<SmallEmbed>::derive(small_spec(Mode::Standard), multi_round_tuning())
                .unwrap();
        let mut rng = StdRng::seed_from_u64(0);
        let witness: Vec<SmallF> = random_vector(&mut rng, config.tuning().vector_size);
        let ds = DomainSeparator::protocol(&"zook-test")
            .session(&"empty-forms".to_string())
            .instance(&Empty);
        let mut ps = ProverState::new_std(&ds);
        let committed = config.commit(&mut ps, &witness);
        // No forms at all — should panic
        config.prove(&mut ps, committed, &[], &[]);
    }

    // ── Negative tests: wrong claimed values ──────────────────────────────────

    #[test]
    fn verify_rejects_wrong_evaluation_zk() {
        let config = ProtocolConfig::<SmallEmbed>::derive(
            small_spec(Mode::ZeroKnowledge),
            multi_round_tuning(),
        )
        .unwrap();
        let (proof, ds, forms, true_values) =
            build_and_prove(&config, 1, 10, "verify_rejects_wrong_evaluation_zk");

        let form_refs: Vec<&dyn LinearForm<SmallF>> =
            forms.iter().map(|f| f as &dyn LinearForm<SmallF>).collect();
        // Shift the claimed value by 1 — the proof is now inconsistent with this claim
        let wrong_values = vec![true_values[0] + SmallF::from(1u64)];

        assert_verify_rejected(|| {
            let mut vs = VerifierState::new_std(&ds, &proof);
            let commitment = config.receive_commitment(&mut vs)?;
            let claim = config.verify(&mut vs, commitment, &form_refs, &wrong_values)?;
            claim.verify(&form_refs)?;
            vs.check_eof()
        });
    }

    #[test]
    fn verify_rejects_wrong_evaluation_standard() {
        let config =
            ProtocolConfig::<SmallEmbed>::derive(small_spec(Mode::Standard), multi_round_tuning())
                .unwrap();
        let (proof, ds, forms, true_values) =
            build_and_prove(&config, 1, 11, "verify_rejects_wrong_evaluation_standard");

        let form_refs: Vec<&dyn LinearForm<SmallF>> =
            forms.iter().map(|f| f as &dyn LinearForm<SmallF>).collect();
        let wrong_values = vec![true_values[0] + SmallF::from(1u64)];

        assert_verify_rejected(|| {
            let mut vs = VerifierState::new_std(&ds, &proof);
            let commitment = config.receive_commitment(&mut vs)?;
            let claim = config.verify(&mut vs, commitment, &form_refs, &wrong_values)?;
            claim.verify(&form_refs)?;
            vs.check_eof()
        });
    }

    #[test]
    fn verify_rejects_wrong_evaluation_multiple_claims() {
        let config = ProtocolConfig::<SmallEmbed>::derive(
            small_spec(Mode::ZeroKnowledge),
            multi_round_tuning(),
        )
        .unwrap();
        let (proof, ds, forms, mut values) = build_and_prove(
            &config,
            3,
            12,
            "verify_rejects_wrong_evaluation_multiple_claims",
        );

        let form_refs: Vec<&dyn LinearForm<SmallF>> =
            forms.iter().map(|f| f as &dyn LinearForm<SmallF>).collect();
        // Corrupt the second claim
        values[1] += SmallF::from(42u64);

        assert_verify_rejected(|| {
            let mut vs = VerifierState::new_std(&ds, &proof);
            let commitment = config.receive_commitment(&mut vs)?;
            let claim = config.verify(&mut vs, commitment, &form_refs, &values)?;
            claim.verify(&form_refs)?;
            vs.check_eof()
        });
    }

    #[test]
    fn final_claim_rejects_wrong_form() {
        // Correct proof + correct verify, but FinalClaim::verify called with a
        // different form → must be detected.
        let config = ProtocolConfig::<SmallEmbed>::derive(
            small_spec(Mode::ZeroKnowledge),
            multi_round_tuning(),
        )
        .unwrap();
        let (proof, ds, forms, values) =
            build_and_prove(&config, 1, 20, "final_claim_rejects_wrong_form");

        let form_refs: Vec<&dyn LinearForm<SmallF>> =
            forms.iter().map(|f| f as &dyn LinearForm<SmallF>).collect();

        let mut vs = VerifierState::new_std(&ds, &proof);
        let commitment = config.receive_commitment(&mut vs).unwrap();
        let claim = config
            .verify(&mut vs, commitment, &form_refs, &values)
            .unwrap();

        // Construct a different form at a different evaluation point
        let mut rng = StdRng::seed_from_u64(999);
        let mu = config.tuning().vector_size.trailing_zeros() as usize;
        let wrong_form: MultilinearExtension<SmallF> = MultilinearExtension {
            point: random_vector(&mut rng, mu),
        };

        assert_verify_rejected(|| claim.verify(&[&wrong_form as &dyn LinearForm<SmallF>]));
    }

    // ── Negative tests: tampered proof ────────────────────────────────────────

    #[test]
    fn verify_rejects_tampered_proof() {
        let config = ProtocolConfig::<SmallEmbed>::derive(
            small_spec(Mode::ZeroKnowledge),
            multi_round_tuning(),
        )
        .unwrap();
        let (mut proof, ds, forms, values) =
            build_and_prove(&config, 1, 30, "verify_rejects_tampered_proof");

        let form_refs: Vec<&dyn LinearForm<SmallF>> =
            forms.iter().map(|f| f as &dyn LinearForm<SmallF>).collect();

        // Flip a byte in the middle of the transcript to corrupt the proof
        let mid = proof.narg_string.len() / 2;
        proof.narg_string[mid] ^= 0xff;

        assert_verify_rejected(|| {
            let mut vs = VerifierState::new_std(&ds, &proof);
            let commitment = config.receive_commitment(&mut vs)?;
            let claim = config.verify(&mut vs, commitment, &form_refs, &values)?;
            claim.verify(&form_refs)?;
            vs.check_eof()
        });
    }

    // ── Large integration tests (Field256, λ=128, realistic) ─────────────────
    // These cover the full security target with a 2^19 witness and multiple
    // folding rounds. Kept for end-to-end regression coverage.

    type LargeF = Field256;
    type LargeEmbed = Identity<LargeF>;

    fn large_spec(mode: Mode) -> SecuritySpec {
        SecuritySpec {
            mode,
            decoding_regime: DecodingRegime::Johnson,
            target_security_bits: 128,
            pow_budget: PowBudget::per_slot(10),
            hash_id: hash::SHA2,
        }
    }

    fn large_tuning() -> TuningSpec {
        TuningSpec {
            vector_size: 1 << 19,
            starting_log_inv_rate: 2,
            folding_factor: FoldingFactor::ConstantFromSecondRound {
                initial: 3,
                rest: 3,
            },
            rate_schedule: RateSchedule::Stepping,
        }
    }

    fn large_roundtrip(mode: Mode, seed: u64) {
        crate::tests::init();
        let config =
            ProtocolConfig::<LargeEmbed>::derive(large_spec(mode), large_tuning()).unwrap();

        let mut rng = StdRng::seed_from_u64(seed);
        let witness: Vec<LargeF> = random_vector(&mut rng, config.tuning().vector_size);
        let mu = config.tuning().vector_size.trailing_zeros() as usize;
        let embedding = <LargeEmbed as Default>::default();

        let forms: Vec<MultilinearExtension<LargeF>> = (0..3)
            .map(|_| MultilinearExtension {
                point: random_vector::<LargeF>(&mut rng, mu),
            })
            .collect();
        let values: Vec<LargeF> = forms
            .iter()
            .map(|f| f.evaluate(&embedding, &witness))
            .collect();
        let form_refs: Vec<&dyn LinearForm<LargeF>> =
            forms.iter().map(|f| f as &dyn LinearForm<LargeF>).collect();

        let ds = DomainSeparator::protocol(&"zook-large-test")
            .session(&format!("three-claims mode={mode:?} seed={seed}"))
            .instance(&Empty);

        let mut ps = ProverState::new_std(&ds);
        let committed = config.commit(&mut ps, &witness);
        config.prove(&mut ps, committed, &form_refs, &values);
        let proof = ps.proof();

        let mut vs = VerifierState::new_std(&ds, &proof);
        let commitment = config.receive_commitment(&mut vs).unwrap();
        let claim = config
            .verify(&mut vs, commitment, &form_refs, &values)
            .unwrap();
        claim.verify(&form_refs).expect("FinalClaim::verify failed");
        vs.check_eof().unwrap();
    }

    #[test]
    fn roundtrip_2_pow_20_three_claims_zk() {
        large_roundtrip(Mode::ZeroKnowledge, 0);
    }

    #[test]
    fn roundtrip_2_pow_20_three_claims_standard() {
        large_roundtrip(Mode::Standard, 1);
    }
}
