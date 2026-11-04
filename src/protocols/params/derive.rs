//! Derive commitment dimensions and all mask lengths before proving.

use crate::{
    algebra::embedding::{Embedding, Identity},
    protocols::params::{
        basecase::{solve as solve_basecase, solve_with_commit},
        branch::RoundBuildMode,
        build_round::build_round_config,
        error::{DeriveError, ZookSecurityError},
        layout::{round_layout, RoundLayout},
        mask_proximity::solve as solve_joint_masks,
        protocol_config::{embedding_degree, ProtocolConfig, RoundConfig},
        spec::{DecodingRegime, Mode, SecuritySpec, TuningSpec},
    },
};

impl<M: Embedding + Default> ProtocolConfig<M> {
    /// Build and validate the commitment and mask configuration.
    ///
    /// Zook derives IOP parameters without grinding credit. Its serialized
    /// configuration includes mask ranges and padded witness dimensions.
    pub fn derive(spec: SecuritySpec, tuning: TuningSpec) -> Result<Self, DeriveError> {
        let mode = match spec.mode {
            Mode::Standard => RoundBuildMode::Standard,
            Mode::ZeroKnowledge => {
                if !matches!(spec.decoding_regime, DecodingRegime::Johnson) {
                    return Err(ZookSecurityError::UnsupportedRegime.into());
                }
                RoundBuildMode::ZeroKnowledge(())
            }
        };
        let RoundLayout {
            padded_vector_size,
            shapes,
            basecase_vector_size,
            basecase_log_inv_rate,
        } = round_layout::<M>(&spec, &tuning, mode)?;
        let mut shapes = shapes.iter();
        let first_round: Option<RoundConfig<M>> = shapes
            .next()
            .map(|shape| build_round_config(&spec, shape, mode))
            .transpose()?;
        let tail_rounds: Vec<RoundConfig<Identity<M::Target>>> = shapes
            .map(|shape| build_round_config(&spec, shape, mode))
            .collect::<Result<_, _>>()?;
        let last_target = tail_rounds
            .last()
            .map(|round| round.code_switch().target())
            .or_else(|| {
                first_round
                    .as_ref()
                    .map(|round| round.code_switch().target())
            });
        let basecase = match last_target {
            Some(target) => solve_with_commit(&spec, target.clone(), 1)?,
            None => solve_basecase(&spec, basecase_vector_size, basecase_log_inv_rate, 0)?,
        };
        let joint_masks = if mode.is_zk() {
            let masks = first_round
                .iter()
                .filter_map(RoundConfig::masks)
                .chain(tail_rounds.iter().filter_map(RoundConfig::masks));
            let mut sumcheck_lengths = Vec::new();
            let mut witness_lengths = Vec::new();
            for mask in masks {
                sumcheck_lengths.push(mask.sumcheck_coefficients);
                witness_lengths.push(mask.witness.num_coefficients()?);
            }
            witness_lengths.push(
                basecase
                    .witness_mask()
                    .ok_or(DeriveError::InvalidMaskLayout)?
                    .num_coefficients()?,
            );
            // A basecase-only witness is lifted before committing, so its
            // mask and blinder live entirely in the target field.
            let degree = if first_round.is_some() {
                embedding_degree::<M>()?
            } else {
                1
            };
            Some(solve_joint_masks(
                &spec,
                &sumcheck_lengths,
                &witness_lengths,
                basecase.commit(),
                degree,
            )?)
        } else {
            None
        };
        let config = Self::new(
            spec,
            tuning,
            padded_vector_size,
            first_round,
            tail_rounds,
            basecase,
            joint_masks,
        );
        config.validate()?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        algebra::{embedding::Basefield, fields::Field64_2},
        protocols::params::{
            protocol_config::BasecasePlan,
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
        assert!(config.check_all_invariants());
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
            let value = &mut encoded["basecase"]["ZeroKnowledge"]["sumcheck"]["config"][field];
            assert!(value.as_u64().unwrap() > 0);
            *value = serde_json::json!(0);
            let restored: ProtocolConfig<M> = serde_json::from_value(encoded).unwrap();
            assert!(matches!(
                restored.validate(),
                Err(DeriveError::ZookSecurity(
                    ZookSecurityError::InvalidRoundShape
                ))
            ));
        }
        let mut encoded = serde_json::to_value(&config).unwrap();
        encoded["basecase"]["ZeroKnowledge"]["sumcheck"]["analytic"] = serde_json::json!(200.0);
        let restored: ProtocolConfig<M> = serde_json::from_value(encoded).unwrap();
        assert!(matches!(
            restored.validate(),
            Err(DeriveError::AnalyticDrift { .. })
        ));
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
            .pow_slots()
            .unwrap()
            .iter()
            .all(|slot| slot.pow.difficulty().is_zero()));
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
        let restored: ProtocolConfig<M> = serde_json::from_value(value).unwrap();
        assert!(matches!(
            restored.validate(),
            Err(DeriveError::InvalidMaskLayout)
        ));
    }

    #[test]
    fn rejects_inconsistent_rounds_after_deserialization() {
        let config = ProtocolConfig::<M>::derive(spec(), tuning(4096)).unwrap();
        for path in [
            "/first_round/round_index",
            "/first_round/sumcheck/config/initial_size",
            "/first_round/code_switch/config/out_domain_samples",
        ] {
            let mut value = serde_json::to_value(&config).unwrap();
            let field = value.pointer_mut(path).unwrap();
            *field = (field.as_u64().unwrap() + 1).into();
            let restored: ProtocolConfig<M> = serde_json::from_value(value).unwrap();
            assert!(restored.validate().is_err(), "accepted mutation at {path}");
        }
    }

    #[test]
    fn adaptive_rates_use_the_joint_mask_and_final_opening_bounds() {
        let mut dims = tuning(4096);
        dims.rate_schedule = RateSchedule::Adaptive {
            knee_weight: KneeWeight::DEFAULT,
        };
        let config = ProtocolConfig::<M>::derive(spec(), dims).unwrap();
        assert!(config.check_all_invariants());
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
                assert!(config.check_all_invariants());
                let restored: ProtocolConfig<M> =
                    serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
                assert_eq!(config, restored);
            }
        }
    }
}
