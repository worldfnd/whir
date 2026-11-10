use super::*;
use crate::protocols::{
    code_switch::CodeSwitchMode,
    irs_commit::IrsParams,
    params::{
        config::BasecasePlan,
        derive::{solve_irs, solve_node, solve_t_ood},
        masks::solve_commit,
        spec::{Mode, RoundContext, SecuritySpec},
        DeriveError, SecurityEvent,
    },
    proof_of_work::Config as PowConfig,
    sumcheck::SumcheckMode,
};

mod bounds {
    use super::*;
    use crate::protocols::params::test_utils::assert_close;

    /// OOD per-sample Schwartz–Zippel: `log₂((k−1) / |F|) = log₂(k−1) − field_bits`.
    #[test]
    fn ood_per_sample_log2_formula() {
        // `k = 129` so `k − 1 = 128 = 2^7` for exact `log2`.
        const K: usize = 129;
        const FIELD_BITS: f64 = 64.0;

        let got = ood_per_sample_log2(K, FIELD_BITS);
        let expected = ((K - 1) as f64).log2() - FIELD_BITS;
        assert_close(got, expected);
        // (k−1)/|F| < 1 for sane parameters ⇒ log is negative.
        assert!(got < 0.0);
    }

    #[test]
    fn log_arithmetic_handles_tiny_errors_and_small_challenge_domains() {
        let zero_error = log2_sum(f64::NEG_INFINITY, f64::NEG_INFINITY);
        assert!(zero_error.is_infinite() && zero_error.is_sign_negative());
        assert_close(log2_sum(f64::NEG_INFINITY, -80.0), -80.0);
        assert!((log2_sum(-2000.0, -2000.0) + 1999.0).abs() < 1e-12);
        assert!((out_of_domain_bits(8.0, 1).unwrap() - 255.0_f64.log2()).abs() < 1e-12);
        assert!((out_of_domain_bits(8.0, 128).unwrap() - 7.0).abs() < 1e-12);
        assert_eq!(
            out_of_domain_bits(8.0, 256),
            Err(ZookSecurityError::EmptyOutOfDomain)
        );
    }
}

mod regime {
    use super::*;
    use crate::protocols::params::test_utils::assert_close;

    const TIGHT_EPS: f64 = 1e-12;

    fn johnson(slack: f64) -> DecodingRegimeParams {
        DecodingRegimeParams::Johnson {
            slack: OrderedFloat(slack),
        }
    }

    fn capacity(slack: f64) -> DecodingRegimeParams {
        DecodingRegimeParams::Capacity {
            slack: OrderedFloat(slack),
        }
    }

    /// Johnson list size: `|Λ| = 1 / (2η√ρ)`, log₂ form.
    #[test]
    fn list_size_log2_johnson_formula() {
        let got = johnson(0.1).list_size_log2(/* log_degree */ 4.0, 2.0);
        let expected = -1.0 - 0.1_f64.log2() + 0.5 * 2.0;
        assert_close(got, expected);
    }

    /// Capacity list size: `|Λ| = d / (ρ · η)`, log₂ form.
    #[test]
    fn list_size_log2_capacity_formula() {
        let got = capacity(0.125).list_size_log2(4.0, 2.0);
        let expected = 4.0 + 2.0 - 0.125_f64.log2();
        assert_close(got, expected);
    }

    /// Unique-decoding regime gives `|Λ| = 1`, i.e. log = 0.
    #[test]
    fn list_size_log2_unique_decoding_is_zero() {
        assert_close(DecodingRegimeParams::Unique.list_size_log2(4.0, 2.0), 0.0);
    }

    /// `η = √ρ/20` ⇒ `|Λ| = 10/ρ` ⇒ `list_size(_, b) = 10 · 2^b`.
    #[test]
    fn johnson_list_size_closed_form() {
        for b in [1.0, 2.0, 3.0, 5.0] {
            let got = DecodingRegimeParams::from_policy(DecodingRegime::Johnson, rate(b))
                .list_size(4.0, b);
            let expected = 10.0 * 2_f64.powf(b);
            assert!(
                (got - expected).abs() / expected < TIGHT_EPS,
                "log_inv_rate={b}: got {got} vs {expected}",
            );
        }
    }

    /// `η = ρ/20` ⇒ `|Λ| = 20 · d / ρ²`.
    #[test]
    fn capacity_list_size_closed_form() {
        for (log_d, b) in [(4.0, 1.0), (6.0, 2.0), (8.0, 3.0)] {
            let got = DecodingRegimeParams::from_policy(DecodingRegime::Capacity, rate(b))
                .list_size(log_d, b);
            let expected = 20.0 * 2_f64.powf(log_d) * 2_f64.powf(2.0 * b);
            assert!(
                (got - expected).abs() / expected < TIGHT_EPS,
                "log_d={log_d}, log_inv_rate={b}: got {got} vs {expected}",
            );
        }
    }

    #[test]
    fn johnson_list_size_matches_config_list_size() {
        use crate::{
            algebra::{embedding::Identity, fields::Field64},
            hash,
            protocols::irs_commit::{Config, IrsMode, IrsParams},
        };
        const PLACEHOLDER_SECURITY_TARGET_BITS: f64 = 80.0;
        const PLACEHOLDER_NUM_VECTORS: usize = 2;
        const PLACEHOLDER_VECTOR_SIZE: usize = 8;
        const PLACEHOLDER_INTERLEAVING_DEPTH: usize = 1;
        const LOG_INV_RATE: u32 = 2;

        let config: Config<Identity<Field64>> = Config::new(IrsParams {
            security_target: PLACEHOLDER_SECURITY_TARGET_BITS,
            decoding_regime: DecodingRegime::Johnson,
            hash_id: hash::BLAKE3,
            num_vectors: PLACEHOLDER_NUM_VECTORS,
            vector_size: PLACEHOLDER_VECTOR_SIZE,
            interleaving_depth: PLACEHOLDER_INTERLEAVING_DEPTH,
            rate: 2_f64.powf(-f64::from(LOG_INV_RATE)),
            mode: IrsMode::Standard,
        });
        let log_degree = (config.masked_message_length() as f64).log2();
        let got = DecodingRegimeParams::from_policy(DecodingRegime::Johnson, config.rate())
            .list_size(log_degree, f64::from(LOG_INV_RATE));
        let expected = config.list_size();
        assert!(
            (got - expected).abs() / expected < TIGHT_EPS,
            "regime helper ({got}) vs Config::list_size ({expected})",
        );
    }

    /// `1 − δ` in unique-decoding mode: midpoint of 1 and ρ.
    #[test]
    fn one_minus_distance_log2_unique() {
        let log_inv_rate = 2.0;
        let got = DecodingRegimeParams::Unique.one_minus_distance_log2(log_inv_rate);
        let rho = 2_f64.powf(-log_inv_rate);
        let expected = f64::midpoint(1.0, rho).log2();
        assert_close(got, expected);
    }

    /// `1 − δ` in Johnson regime: `√ρ + η`.
    #[test]
    fn one_minus_distance_log2_johnson() {
        let log_inv_rate = 2.0;
        let eta = 0.1;
        let got = johnson(eta).one_minus_distance_log2(log_inv_rate);
        let rho = 2_f64.powf(-log_inv_rate);
        let expected = (rho.sqrt() + eta).log2();
        assert_close(got, expected);
    }

    /// `1 − δ` in Capacity regime: `ρ + η`.
    #[test]
    fn one_minus_distance_log2_capacity() {
        let log_inv_rate = 2.0;
        let eta = 0.05;
        let got = capacity(eta).one_minus_distance_log2(log_inv_rate);
        let rho = 2_f64.powf(-log_inv_rate);
        let expected = (rho + eta).log2();
        assert_close(got, expected);
    }

    /// `ood_security_bits = t · (|F| − log d) − 2·log|Λ| + 1`. Returns 0
    /// under Unique.
    #[test]
    fn ood_security_bits_formula() {
        const LOG_DEGREE: f64 = 6.0;
        const LOG_INV_RATE: f64 = 2.0;
        const FIELD_BITS: f64 = 64.0;
        const OOD: usize = 3;

        // Unique → 0 (no soundness from OOD).
        let unique = DecodingRegimeParams::Unique.ood_security_bits(
            LOG_DEGREE,
            LOG_INV_RATE,
            FIELD_BITS,
            OOD,
        );
        assert_close(unique, 0.0);

        let slack = 2_f64.powf(-LOG_INV_RATE).sqrt() / 20.0;
        let got = johnson(slack).ood_security_bits(LOG_DEGREE, LOG_INV_RATE, FIELD_BITS, OOD);
        let log_list = johnson(slack).list_size_log2(LOG_DEGREE, LOG_INV_RATE);
        let expected = (OOD as f64) * (FIELD_BITS - LOG_DEGREE) - 2.0 * log_list + 1.0;
        assert_close(got, expected);
    }

    const MCA_MESSAGE_LENGTH: usize = 16;
    const MCA_LOG_INV_RATE: f64 = 2.0;
    const MCA_FIELD_BITS: f64 = 64.0;

    /// Compare the MCA formulas with independent theorem anchors.
    #[test]
    fn eps_mca_log2_formulas_and_theorem_anchors() {
        let log_k = (MCA_MESSAGE_LENGTH as f64).log2();
        let rho = 2_f64.powf(-MCA_LOG_INV_RATE);
        // Unique: ε ≈ k/|F| · ρ⁻¹ gives log₂ε = 4 + 2 − 64 = −58.
        let unique = log_k + MCA_LOG_INV_RATE - MCA_FIELD_BITS;
        // Johnson, BCSS25 Theorem 1.5, canonical η = √ρ/20 and m = 10:
        // ε = (2·10.5⁵/3) · n · ρ^{−3/2} / |F|, with n = k/ρ = 64.
        // log₂ε = 16.376624613… + 6 + 3 − 64 = −38.623375386…
        let johnson_bound =
            (2.0 * 10.5_f64.powi(5) / 3.0).log2() + log_k + 2.5 * MCA_LOG_INV_RATE - MCA_FIELD_BITS;
        // Capacity, STIR Conjecture 5.6, canonical η = ρ/20:
        // ε = d / (η · ρ²) / |F|, where d = k = 16, η = 1/80, and ρ² = 1/16.
        // log₂ε = log₂20480 − 64 = 14.321928094… − 64 = −49.678071905…
        let capacity_bound = log_k + 3.0 * MCA_LOG_INV_RATE + LOG2_10 + 1.0 - MCA_FIELD_BITS;
        for (regime, formula, anchor) in [
            (DecodingRegimeParams::Unique, unique, -58.0),
            (
                johnson(rho.sqrt() / 20.0),
                johnson_bound,
                -38.623_375_386_827_36,
            ),
            (capacity(rho / 20.0), capacity_bound, -49.678_071_905_112_64),
        ] {
            let actual = regime.eps_mca_log2(MCA_LOG_INV_RATE, MCA_MESSAGE_LENGTH, MCA_FIELD_BITS);
            assert_close(actual, formula);
            assert_close(actual, anchor);
        }
    }

    /// `from_policy` uses the canonical slack for each regime.
    #[test]
    fn from_policy_matches_canonical() {
        assert_eq!(
            DecodingRegimeParams::from_policy(DecodingRegime::Unique, 0.25),
            DecodingRegimeParams::Unique,
        );
        assert_eq!(
            DecodingRegimeParams::from_policy(DecodingRegime::Johnson, 0.25),
            johnson(0.25_f64.sqrt() / 20.0),
        );
        assert_eq!(
            DecodingRegimeParams::from_policy(DecodingRegime::Capacity, 0.25),
            capacity(0.25 / 20.0),
        );
    }
}

mod irs_commit {
    use std::num::NonZeroUsize;

    use super::*;
    use crate::{
        algebra::{
            embedding::Identity,
            fields::{Field64, Field64_2},
        },
        hash::BLAKE3,
        protocols::params::{
            spec::DecodingRegime,
            test_utils::{johnson_code, johnson_code_with_vectors},
        },
    };
    type F = Field64_2;

    #[test]
    fn joint_masks_require_two_full_length_columns() {
        for vectors in [1, 2, 3] {
            for depth in [1, 2] {
                let code = johnson_code_with_vectors::<Identity<F>>(128, depth, 0.25, vectors);
                let expected = if vectors == 2 && depth == 1 {
                    Ok(())
                } else {
                    Err(ZookSecurityError::ExpectedMaskAndBlinder)
                };
                assert_eq!(validate_joint_mask_code(&code), expected);
                assert_eq!(
                    validate_single_polynomial(&code).is_ok(),
                    vectors == 1 && depth == 1
                );
            }
        }
    }

    #[test]
    fn rejects_appended_masks_and_unsupported_regimes() {
        for (mode, regime, expected) in [
            (
                IrsMode::ZeroKnowledge {
                    mask_length: NonZeroUsize::new(16).unwrap(),
                },
                DecodingRegime::Johnson,
                ZookSecurityError::AppendedMasks,
            ),
            (
                IrsMode::Standard,
                DecodingRegime::Unique,
                ZookSecurityError::UnsupportedRegime,
            ),
            (
                IrsMode::Standard,
                DecodingRegime::Capacity,
                ZookSecurityError::UnsupportedRegime,
            ),
        ] {
            let config = IrsConfig::<Identity<F>>::new(IrsParams {
                security_target: 80.0,
                decoding_regime: regime,
                hash_id: BLAKE3,
                num_vectors: 1,
                vector_size: 128,
                interleaving_depth: 1,
                rate: 0.25,
                mode,
            });
            assert_eq!(validate_zook_code(&config), Err(expected));
        }
    }

    #[test]
    fn rejects_proximity_below_half_distance() {
        let small = johnson_code::<Identity<Field64>>(2, 1, 0.5);
        assert_eq!(
            validate_zook_code(&small),
            Err(ZookSecurityError::InvalidProximity)
        );
    }
}

mod sumcheck {
    use super::*;
    use crate::protocols::params::{
        spec::{DecodingRegime, PowBudget},
        test_utils::{assert_close, TestEmbedding, TestField},
    };

    #[test]
    fn standard_sumcheck_preserves_dimensions_and_grinding() {
        for regime in [
            DecodingRegime::Unique,
            DecodingRegime::Johnson,
            DecodingRegime::Capacity,
        ] {
            let spec = SecuritySpec::new(80)
                .with_decoding_regime(regime)
                .with_pow_budget(PowBudget::per_slot(60));
            for folds in 1..=3 {
                let ctx = RoundContext {
                    vector_size: 256,
                    log_inv_rate: 2,
                    folding_factor: folds,
                };
                let source = solve_irs::<TestEmbedding>(&spec, &ctx).unwrap();
                let target = solve_irs::<Identity<TestField>>(
                    &spec,
                    &RoundContext {
                        vector_size: source.message_length(),
                        log_inv_rate: ctx.log_inv_rate + folds - 1,
                        folding_factor: 0,
                    },
                )
                .unwrap();
                let expected = source
                    .rbr_soundness_fold_prox_gaps()
                    .min(TestField::field_size_bits() - source.list_size().log2() - 1.0)
                    .max(0.0);
                assert_close(
                    f64::from(sumcheck_error_bits(&source, false).unwrap()),
                    expected,
                );
                let node = solve_node(&spec, source, Some(&target), 0).unwrap();
                let round = node.round(&target, 0, false);
                let solved = round.sumcheck();
                assert_eq!(solved.initial_size(), ctx.vector_size);
                assert_eq!(solved.num_rounds(), folds as usize);
                assert!(matches!(solved.mode(), SumcheckMode::Standard));
                assert!(expected + f64::from(solved.round_pow().difficulty()) >= 80.0 - 1e-9);
            }
        }
    }
}

mod code_switch {
    use super::*;
    use crate::protocols::params::{
        spec::{DecodingRegime, PowBudget, RoundContext},
        test_utils::{assert_close, TestEmbedding, TestField},
    };

    #[test]
    fn standard_code_switch_preserves_query_bounds_and_grinding() {
        for regime in [
            DecodingRegime::Unique,
            DecodingRegime::Johnson,
            DecodingRegime::Capacity,
        ] {
            let spec = SecuritySpec::new(80)
                .with_decoding_regime(regime)
                .with_pow_budget(PowBudget::per_slot(60));
            let source = solve_irs::<TestEmbedding>(
                &spec,
                &RoundContext {
                    vector_size: 256,
                    log_inv_rate: 2,
                    folding_factor: 2,
                },
            )
            .unwrap();
            let target = solve_irs::<TestEmbedding>(
                &spec,
                &RoundContext {
                    vector_size: 64,
                    log_inv_rate: 3,
                    folding_factor: 0,
                },
            )
            .unwrap();
            let t_ood = solve_t_ood(&spec, &source, &target, 0).unwrap();
            let list = target.list_size();
            let field_bits = TestField::field_size_bits();
            let ood = (t_ood as f64) * (field_bits - ((source.message_length() - 1) as f64).log2())
                - (list * (list - 1.0) / 2.0).log2();
            let count = t_ood + source.in_domain_samples() * source.interleaving_depth();
            let combination = field_bits - (count as f64).log2() - list.log2();
            let expected = ood.min(source.rbr_queries()).min(combination).max(0.0);
            let node = solve_node(&spec, source, Some(&target), 0).unwrap();
            let round = node.round(&target, 0, false);
            let solved = round.code_switch();
            assert_eq!(solved.out_domain_samples(), t_ood);
            assert_close(
                f64::from(
                    code_switch_error_bits(solved.source(), solved.target(), t_ood, false).unwrap(),
                ),
                expected,
            );
            assert!(matches!(solved.mode(), CodeSwitchMode::Standard));
            assert!(expected + f64::from(solved.pow().difficulty()) >= 80.0 - 1e-9);
        }
    }
}

mod basecase {
    use super::*;
    use crate::{
        algebra::fields::{Field64, Field64_2, FieldWithSize},
        protocols::params::{
            spec::PowBudget,
            test_utils::{johnson_code, johnson_code_with_vectors},
        },
    };

    #[test]
    fn final_sumcheck_is_standard_and_receives_no_grinding_credit() {
        let code = johnson_code::<Identity<Field64_2>>(128, 1, 0.25);
        let spec = SecuritySpec::new(80).with_mode(Mode::ZeroKnowledge);
        let node = solve_node(&spec, code, None, 0).unwrap();
        let BasecasePlan::ZeroKnowledge { sumcheck, .. } = node.basecase(true, 0) else {
            panic!("expected Zook final sumcheck");
        };
        assert!(matches!(sumcheck.mode(), SumcheckMode::Standard));
        assert_eq!(sumcheck.initial_size(), 128);
        assert_eq!(sumcheck.final_size(), 1);
        assert_eq!(sumcheck.round_pow(), PowConfig::none());
        assert_eq!(
            revealed_sumcheck_error_bits::<Field64_2>(),
            Bits::new(Field64_2::field_size_bits() - 1.0)
        );

        let code = johnson_code::<Identity<Field64>>(128, 1, 0.25);
        let spec = SecuritySpec::new(64)
            .with_mode(Mode::ZeroKnowledge)
            .with_pow_budget(PowBudget::per_slot(20));
        assert!(matches!(
            solve_node(&spec, code, None, 0),
            Err(DeriveError::SecurityTargetNotMet {
                pow: SecurityEvent::BasecaseSumcheck,
                ..
            })
        ));
    }

    #[test]
    fn final_queries_lose_one_bit_when_errors_are_equal() {
        let code = johnson_code::<Identity<Field64_2>>(128, 1, 0.25);
        let masks = johnson_code_with_vectors::<Identity<Field64_2>>(128, 1, 0.25, 2);
        let result = ZookFinalOpeningSoundness::for_codes(&code, &masks).unwrap();
        assert!((f64::from(result.queries) - (code.rbr_queries() - 1.0)).abs() < 1e-10);
    }

    #[test]
    fn final_opening_uses_weaker_agreement_bound() {
        let witness = johnson_code::<Identity<Field64_2>>(64, 1, 0.25);
        let masks = johnson_code_with_vectors::<Identity<Field64_2>>(1024, 1, 0.25, 2);
        let result = ZookFinalOpeningSoundness::for_codes(&witness, &masks).unwrap();
        assert_eq!(
            result.combination,
            constrained_agreement_bits(&masks).unwrap()
        );
    }
}

mod mask_proximity {
    use super::*;
    use crate::{
        algebra::fields::{Field64_2, Field64_3},
        protocols::params::{
            spec::{Mode, PowBudget},
            test_utils::{assert_probability_bits, johnson_code_with_vectors},
        },
    };

    #[test]
    fn mask_message_keeps_its_length_when_codeword_is_rounded() {
        let spec = SecuritySpec::new(128)
            .with_mode(Mode::ZeroKnowledge)
            .with_pow_budget(PowBudget::Forbidden);
        let commit = solve_commit::<Field64_3>(&spec, 4495, 3).unwrap();
        assert_eq!(commit.vector_size(), 4566);
        assert!(commit.codeword_length() >= 16 * commit.vector_size());
        assert!(4495 + commit.in_domain_samples() + 1 + 3 <= commit.vector_size());
        validate_joint_mask_code(&commit).unwrap();
        assert!(zook_mask_binding_bits(&commit).unwrap() >= Bits::from(128.0));
    }

    #[test]
    fn mask_binding_includes_frobenius_error() {
        let masks = johnson_code_with_vectors::<Identity<Field64_2>>(256, 1, 0.25, 2);
        let challenges = 2.0_f64.powf(Field64_2::field_size_bits()) - 1.0;
        let list = masks.list_size();
        let degree = masks.message_length() as f64;
        assert_probability_bits(
            zook_mask_binding_bits(&masks).unwrap(),
            list * (list - 1.0) / 2.0 * degree / challenges + list * degree / challenges,
        );
    }
}

mod rounds {
    use super::*;
    use crate::{
        algebra::{
            embedding::Basefield,
            fields::{Field64_2, FieldWithSize},
        },
        protocols::params::test_utils::{assert_probability_bits, johnson_code},
    };

    #[test]
    fn round_uses_extension_field_and_full_folded_degree() {
        let source = johnson_code::<Basefield<Field64_2>>(256, 4, 0.25);
        let target = johnson_code::<Identity<Field64_2>>(64, 2, 0.125);
        let field = 2.0_f64.powf(Field64_2::field_size_bits());
        let agreement = sumcheck_error_bits(&source, true).unwrap();
        let (queries, aggregation) = zook_query_error_bits(&source, &target).unwrap();
        let l = source.list_size();
        assert_probability_bits(
            agreement,
            l * ((2.0 / 3.0) * l.powi(4) * source.rate() + 1.0) * source.codeword_length() as f64
                / field,
        );
        let deep_error = target.list_size().powi(2) / 2.0 * target.vector_size() as f64
            / (field - target.codeword_length() as f64);
        let rho = source.rate();
        let query_error = (rho.sqrt() * 1.05).powf(source.in_domain_samples() as f64);
        assert_probability_bits(queries, deep_error + query_error);
        assert_probability_bits(
            aggregation,
            target.list_size() * source.in_domain_samples() as f64 / field,
        );
    }

    #[test]
    fn rejects_invalid_shapes() {
        let interleaved = johnson_code::<Identity<Field64_2>>(128, 2, 0.25);
        assert_eq!(
            zook_mask_binding_bits(&interleaved),
            Err(ZookSecurityError::ExpectedMaskAndBlinder)
        );
        let wrong_target = johnson_code::<Identity<Field64_2>>(32, 1, 0.25);
        assert_eq!(
            zook_query_error_bits(&interleaved, &wrong_target),
            Err(ZookSecurityError::InvalidRoundShape)
        );
    }
}

mod grinding {
    use super::*;
    use crate::protocols::params::spec::PowBudget;

    #[test]
    fn standard_grinding_closes_fractional_gaps() {
        let spec = SecuritySpec::new(80).with_pow_budget(PowBudget::per_slot(60));
        for analytic in [20.3422, 35.623_375_386_491_45] {
            let analytic = Bits::new(analytic);
            let pow = grind_to_at(&spec, analytic, SecurityEvent::BasecaseSumcheck).unwrap();
            assert!(analytic + pow.difficulty() >= Bits::new(80.0 - 1e-9));
            assert!(pow.difficulty() <= Bits::new(60.0));
        }
    }

    #[test]
    fn grinding_still_enforces_budget_and_zook_target() {
        let spec = SecuritySpec::new(80).with_pow_budget(PowBudget::per_slot(59));
        assert!(matches!(
            grind_to_at(&spec, Bits::new(20.3422), SecurityEvent::BasecaseSumcheck),
            Err(DeriveError::PowBudgetExceeded { .. }),
        ));
        let spec = spec.with_mode(Mode::ZeroKnowledge);
        assert!(matches!(
            grind_to_at(&spec, Bits::new(79.9999), SecurityEvent::JointMaskBinding),
            Err(DeriveError::SecurityTargetNotMet { .. }),
        ));
    }
}
