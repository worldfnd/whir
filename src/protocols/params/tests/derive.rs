use super::*;

#[cfg(test)]
mod derive_tests {
    use super::*;
    use crate::{
        algebra::{embedding::Basefield, fields::Field64_2},
        protocols::params::{
            config::BasecasePlan,
            spec::{FoldingFactor, KneeWeight, PowBudget, RateSchedule},
        },
    };

    type M = Basefield<Field64_2>;

    fn spec() -> SecuritySpec {
        SecuritySpec::new(80)
            .with_mode(Mode::ZeroKnowledge)
            .with_pow_budget(PowBudget::Forbidden)
    }

    fn tuning(vector_size: usize) -> TuningSpec {
        TuningSpec {
            vector_size,
            starting_log_inv_rate: 2,
            folding_factor: FoldingFactor::Constant(2),
            rate_schedule: RateSchedule::Stepping,
        }
    }

    #[test]
    fn derives_one_joint_mask_commitment_with_all_components() {
        let config = ProtocolConfig::<M>::derive(spec(), tuning(4096)).unwrap();
        assert!(config.has_rounds());
        let joint = config.joint_masks().unwrap();
        assert_eq!(joint.commit().num_vectors(), 2);
        assert_eq!(joint.commit().interleaving_depth(), 1);
        assert_eq!(joint.commit().mask_length(), 0);
        assert_eq!(joint.sumcheck_masks().len(), config.num_rounds());
        assert_eq!(joint.witness_masks().len(), config.num_rounds() + 1);
        assert_eq!(joint.basefield_blinder().len(), 2);
        assert_eq!(
            joint.blinder().len(),
            joint.commit().in_domain_samples() + 1
        );
        let masks = config
            .first_round()
            .into_iter()
            .filter_map(RoundConfig::masks)
            .chain(config.tail_rounds().iter().filter_map(RoundConfig::masks));
        for (mask, range) in masks.zip(joint.sumcheck_masks()) {
            assert_eq!(range.len(), 5);
            assert_eq!(range.len(), mask.sumcheck_coefficients);
        }
        config.validate().unwrap();
    }

    #[test]
    fn overlays_fit_every_interleaved_column_and_final_deep_query() {
        let config = ProtocolConfig::<M>::derive(spec(), tuning(4096)).unwrap();
        let first = config.first_round().unwrap();
        let source = first.code_switch().source();
        let initial = first.masks().unwrap().witness;
        assert_eq!(
            initial.coefficients_per_polynomial,
            source.in_domain_samples()
        );
        assert_eq!(initial.num_polynomials, source.interleaving_depth());
        assert!(initial.coefficients_per_polynomial <= source.message_length());
        for round in config.tail_rounds() {
            let source = round.code_switch().source();
            let mask = round.masks().unwrap().witness;
            assert_eq!(
                mask.coefficients_per_polynomial,
                source.in_domain_samples() + 1
            );
            assert!(mask.coefficients_per_polynomial <= source.message_length());
            assert_eq!(source.mask_length(), 0);
        }
        let final_code = config.basecase().commit();
        let final_mask = config.basecase().witness_mask().unwrap();
        assert_eq!(
            final_mask.coefficients_per_polynomial,
            final_code.in_domain_samples() + 1
        );
        assert!(final_mask.coefficients_per_polynomial <= final_code.message_length());
    }

    #[test]
    fn pads_small_witness_before_sampling_masks() {
        let config = ProtocolConfig::<M>::derive(spec(), tuning(8)).unwrap();
        assert_eq!(config.tuning().vector_size, 8);
        assert!(!config.has_rounds());
        assert!(config.padded_vector_size() > 8);
        assert_eq!(
            config.padded_vector_size(),
            config.basecase().commit().vector_size()
        );
        assert_eq!(config.joint_masks().unwrap().basefield_blinder().len(), 1);
    }

    #[test]
    fn validates_final_sumcheck_after_deserialization() {
        let config = ProtocolConfig::<M>::derive(spec(), tuning(4096)).unwrap();
        for field in ["num_rounds", "initial_size"] {
            let mut encoded = serde_json::to_value(&config).unwrap();
            let value = &mut encoded["basecase"]["ZeroKnowledge"]["sumcheck"][field];
            assert!(value.as_u64().unwrap() > 0);
            *value = serde_json::json!(0);
            assert!(serde_json::from_value::<ProtocolConfig<M>>(encoded).is_err());
        }
    }

    #[test]
    fn grinding_budget_does_not_reduce_zook_queries() {
        let config = ProtocolConfig::<M>::derive(spec(), tuning(4096)).unwrap();
        let with_pow = ProtocolConfig::<M>::derive(
            spec().with_pow_budget(PowBudget::per_slot(60)),
            tuning(4096),
        )
        .unwrap();
        assert_eq!(config.joint_masks(), with_pow.joint_masks());
        assert_eq!(config.first_round(), with_pow.first_round());
        assert_eq!(config.tail_rounds(), with_pow.tail_rounds());
        assert!(config
            .security_report()
            .unwrap()
            .iter()
            .all(|slot| slot.pow.difficulty().is_zero()));
    }

    #[test]
    fn rejects_zero_rate_cap_without_panicking() {
        for mode in [Mode::Standard, Mode::ZeroKnowledge] {
            let mut tuning = tuning(256);
            tuning.rate_schedule = RateSchedule::Capped {
                max_log_inv_rate: 0,
            };
            assert!(matches!(
                ProtocolConfig::<M>::derive(spec().with_mode(mode), tuning),
                Err(DeriveError::Layout(LayoutError::RateCapBelowOne))
            ));
        }
    }

    #[test]
    fn rejects_unsupported_zook_regimes() {
        for regime in [DecodingRegime::Unique, DecodingRegime::Capacity] {
            assert!(matches!(
                ProtocolConfig::<M>::derive(spec().with_decoding_regime(regime), tuning(4096)),
                Err(DeriveError::ZookSecurity(
                    ZookSecurityError::UnsupportedRegime
                )),
            ));
        }
    }

    #[test]
    fn validates_mask_ranges_after_deserialization() {
        let config = ProtocolConfig::<M>::derive(spec(), tuning(4096)).unwrap();
        let json = serde_json::to_string(&config).unwrap();
        let restored: ProtocolConfig<M> = serde_json::from_str(&json).unwrap();
        assert_eq!(config, restored);
        let mut value = serde_json::to_value(&restored).unwrap();
        let end = value["joint_masks"]["witness_masks"][0]["end"]
            .as_u64()
            .unwrap();
        value["joint_masks"]["witness_masks"][0]["end"] = (end + 1).into();
        assert!(serde_json::from_value::<ProtocolConfig<M>>(value).is_err());
    }

    #[test]
    fn rejects_inconsistent_rounds_after_deserialization() {
        let config = ProtocolConfig::<M>::derive(spec(), tuning(4096)).unwrap();
        for path in [
            "/first_round/round_index",
            "/first_round/sumcheck/initial_size",
            "/first_round/code_switch/out_domain_samples",
        ] {
            let mut value = serde_json::to_value(&config).unwrap();
            let field = value.pointer_mut(path).unwrap();
            *field = (field.as_u64().unwrap() + 1).into();
            assert!(
                serde_json::from_value::<ProtocolConfig<M>>(value).is_err(),
                "accepted mutation at {path}"
            );
        }
    }

    #[test]
    fn adaptive_rates_use_the_joint_mask_and_final_opening_bounds() {
        let mut dims = tuning(4096);
        dims.rate_schedule = RateSchedule::Adaptive {
            knee_weight: KneeWeight::DEFAULT,
        };
        let config = ProtocolConfig::<M>::derive(spec(), dims).unwrap();
        config.validate().unwrap();
    }

    #[test]
    fn standard_mode_has_no_joint_masks_or_padding() {
        for size in [8, 256, 4096] {
            for schedule in [
                RateSchedule::Stepping,
                RateSchedule::Capped {
                    max_log_inv_rate: 4,
                },
                RateSchedule::Adaptive {
                    knee_weight: KneeWeight::DEFAULT,
                },
            ] {
                let mut dims = tuning(size);
                dims.rate_schedule = schedule;
                let config = ProtocolConfig::<M>::derive(SecuritySpec::new(40), dims).unwrap();
                assert!(config.joint_masks().is_none());
                assert_eq!(config.padded_vector_size(), size);
                assert!(matches!(config.basecase(), BasecasePlan::Standard { .. }));
                config.validate().unwrap();
                let restored: ProtocolConfig<M> =
                    serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
                assert_eq!(config, restored);
            }
        }
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;
    use crate::protocols::params::{
        spec::{FoldingFactor, Mode, PowBudget, RateSchedule, SecuritySpec},
        test_utils::TestEmbedding,
    };

    const FIXTURE_FOLDING_FACTOR: usize = 2;
    const FIXTURE_LOG_INV_RATE: u32 = 1;

    /// Spec fixture suitable for Standard-mode layout tests — RateSchedule
    /// dispatch is exercised here, not security solving, so any feasible
    /// spec works.
    fn layout_spec() -> SecuritySpec {
        SecuritySpec::new(40)
            .with_mode(Mode::Standard)
            .with_pow_budget(PowBudget::per_slot(60))
    }

    /// Exercise layout construction in Standard mode.
    /// Derivation tests cover adaptive schedules and complete configurations.
    fn layout(tuning: &TuningSpec) -> Result<RoundLayout, DeriveError> {
        round_layout::<TestEmbedding>(&layout_spec(), tuning)
    }

    const LOG_VECTOR_SIZE_NO_ROUNDS: u32 = 3;
    const LOG_VECTOR_SIZE_MULTI_ROUND: u32 = 8;

    const VARIED_INITIAL_FOLDING: usize = 3;
    const VARIED_STEADY_FOLDING: usize = 2;

    const RATE_STEPPING_STARTING_LOG_INV_RATE: u32 = 2;
    const MIN_ROUNDS_FOR_CHAINING_TEST: usize = 2;

    fn tuning_with(vector_size: usize) -> TuningSpec {
        TuningSpec {
            vector_size,
            starting_log_inv_rate: FIXTURE_LOG_INV_RATE,
            folding_factor: FoldingFactor::Constant(FIXTURE_FOLDING_FACTOR),
            rate_schedule: RateSchedule::Stepping,
        }
    }

    #[test]
    fn round_layout_rate_steps_up_by_folding_minus_one() {
        let tuning = TuningSpec {
            vector_size: 1 << LOG_VECTOR_SIZE_MULTI_ROUND,
            starting_log_inv_rate: RATE_STEPPING_STARTING_LOG_INV_RATE,
            folding_factor: FoldingFactor::ConstantFromSecondRound {
                initial: VARIED_INITIAL_FOLDING,
                rest: VARIED_STEADY_FOLDING,
            },
            rate_schedule: RateSchedule::Stepping,
        };
        let layout = layout(&tuning).unwrap();

        let mut expected_log_inv_rate = RATE_STEPPING_STARTING_LOG_INV_RATE;
        for shape in &layout.shapes {
            assert_eq!(shape.source_log_inv_rate, expected_log_inv_rate);
            expected_log_inv_rate += shape.source_folding_factor.saturating_sub(1);
        }
        assert_eq!(layout.basecase_log_inv_rate, expected_log_inv_rate);
    }

    #[test]
    fn round_layout_chains_target_to_next_source_folding() {
        let tuning = TuningSpec {
            vector_size: 1 << LOG_VECTOR_SIZE_MULTI_ROUND,
            starting_log_inv_rate: FIXTURE_LOG_INV_RATE,
            folding_factor: FoldingFactor::ConstantFromSecondRound {
                initial: VARIED_INITIAL_FOLDING,
                rest: VARIED_STEADY_FOLDING,
            },
            rate_schedule: RateSchedule::Stepping,
        };
        let layout = layout(&tuning).unwrap();
        assert!(
            layout.shapes.len() >= MIN_ROUNDS_FOR_CHAINING_TEST,
            "need ≥ {MIN_ROUNDS_FOR_CHAINING_TEST} rounds to test chaining",
        );
        for window in layout.shapes.windows(2) {
            assert_eq!(
                window[0].target_folding_factor,
                window[1].source_folding_factor
            );
        }
    }

    #[test]
    fn round_layout_basecase_size_consumes_remaining_num_vars() {
        let tuning = tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND);
        let layout = layout(&tuning).unwrap();
        let consumed: u32 = layout.shapes.iter().map(|s| s.source_folding_factor).sum();
        let initial_num_vars = tuning.vector_size.trailing_zeros();
        let remaining = initial_num_vars - consumed;
        assert_eq!(layout.basecase_vector_size, 1usize << remaining);
    }

    #[test]
    fn round_layout_stops_when_no_room_for_source_plus_target() {
        let vector_size = 1usize << LOG_VECTOR_SIZE_NO_ROUNDS;
        let tuning = tuning_with(vector_size);
        let layout = layout(&tuning).unwrap();
        assert_eq!(layout.shapes.len(), 0);
        assert_eq!(layout.basecase_vector_size, vector_size);
        assert_eq!(layout.basecase_log_inv_rate, FIXTURE_LOG_INV_RATE);
    }

    #[test]
    fn round_layout_rejects_non_pow2_vector_size() {
        let tuning = TuningSpec {
            vector_size: 12,
            starting_log_inv_rate: FIXTURE_LOG_INV_RATE,
            folding_factor: FoldingFactor::Constant(FIXTURE_FOLDING_FACTOR),
            rate_schedule: RateSchedule::Stepping,
        };
        let err = layout(&tuning).expect_err("non-pow2 vector_size must fail");
        assert!(
            matches!(
                err,
                DeriveError::Layout(LayoutError::VectorSizeNotPowerOfTwo { vector_size: 12 })
            ),
            "got {err:?}",
        );
    }

    #[test]
    fn round_layout_rejects_zero_folding_factor() {
        let tuning = TuningSpec {
            vector_size: 1 << LOG_VECTOR_SIZE_MULTI_ROUND,
            starting_log_inv_rate: FIXTURE_LOG_INV_RATE,
            folding_factor: FoldingFactor::Constant(0),
            rate_schedule: RateSchedule::Stepping,
        };
        let err = layout(&tuning).expect_err("folding_factor = 0 must fail");
        assert!(
            matches!(
                err,
                DeriveError::Layout(LayoutError::FoldingFactorBelowOne { min: 0 })
            ),
            "got {err:?}",
        );
    }

    #[test]
    fn round_layout_rejects_zero_starting_rate() {
        let tuning = TuningSpec {
            vector_size: 1 << LOG_VECTOR_SIZE_MULTI_ROUND,
            starting_log_inv_rate: 0,
            folding_factor: FoldingFactor::Constant(FIXTURE_FOLDING_FACTOR),
            rate_schedule: RateSchedule::Stepping,
        };
        let err = layout(&tuning).expect_err("starting_log_inv_rate = 0 must fail");
        assert!(
            matches!(err, DeriveError::Layout(LayoutError::StartingRateBelowOne)),
            "got {err:?}",
        );
    }
}
