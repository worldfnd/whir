//! Add one selector fold to the first round of an existing zook configuration.

use super::{
    branch::SolveMode,
    build_round::build_mask_oracle,
    code_switch,
    error::Pow,
    protocol_config::{ProtocolConfig, RoundConfig, RoundMode},
    spec::{FoldingFactor, LogInvRate, OodSampleBudget, RateSchedule, RoundContext},
    sumcheck,
};
use crate::{
    algebra::embedding::Embedding,
    transcript::{VerificationError, VerificationResult},
};

impl<M: Embedding + Default> ProtocolConfig<M> {
    pub(crate) fn pair_config(&self) -> VerificationResult<Self> {
        let tuning = self.tuning();
        // Check untrusted dimensions before the parameter solver performs arithmetic.
        if tuning.vector_size.checked_mul(2).is_none()
            || tuning.starting_log_inv_rate >= usize::BITS
            || matches!(
                tuning.rate_schedule,
                RateSchedule::Capped {
                    max_log_inv_rate: 0
                }
            )
        {
            return Err(VerificationError);
        }
        let canonical =
            Self::derive(self.security().clone(), tuning.clone()).map_err(|_| VerificationError)?;
        if self != &canonical {
            return Err(VerificationError);
        }
        let first = self.first_round().ok_or(VerificationError)?;
        let source = first
            .code_switch()
            .source()
            .joint_pair_source()
            .ok_or(VerificationError)?;
        let context = RoundContext {
            vector_size: source.vector_size(),
            folding_factor: source.interleaving_depth().ilog2(),
            log_inv_rate: tuning.starting_log_inv_rate,
        };
        let t_ood = first.code_switch().out_domain_samples();
        let (round_mode, solve_mode) = if let Some(zk_spec) = self.security().as_zk() {
            // Keep the original mask code rate and add one selector mask.
            let code = first
                .mask_oracle()
                .ok_or(VerificationError)?
                .sumcheck_masks()
                .c_zk_commit();
            let log_inv_rate = (code.codeword_length() / code.vector_size()).ilog2();
            let oracle = build_mask_oracle(
                zk_spec,
                &context,
                &source,
                t_ood,
                LogInvRate::new(log_inv_rate),
                first.round_index(),
            )
            .map_err(|_| VerificationError)?;
            let solve_mode = SolveMode::ZeroKnowledge(oracle.info());
            (
                RoundMode::ZeroKnowledge {
                    t_ood: OodSampleBudget::new(t_ood),
                    mask_oracle: Box::new(oracle),
                },
                solve_mode,
            )
        } else {
            (RoundMode::Standard, SolveMode::Standard)
        };
        let sumcheck = sumcheck::solve(
            self.security(),
            &context,
            &source,
            solve_mode,
            Pow::RoundSumcheck {
                index: first.round_index(),
            },
        )
        .map_err(|_| VerificationError)?;
        let code_switch = code_switch::solve(
            self.security(),
            source,
            first.code_switch().target().clone(),
            t_ood,
            solve_mode,
            first.round_index(),
        )
        .map_err(|_| VerificationError)?;
        let mut tuning = tuning.clone();
        tuning.vector_size = context.vector_size;
        tuning.folding_factor = FoldingFactor::ConstantFromSecondRound {
            initial: context.folding_factor as usize,
            rest: tuning.folding_factor.at_round(1),
        };
        let pair = Self::new(
            self.security().clone(),
            tuning,
            Some(RoundConfig::new(
                first.round_index(),
                sumcheck,
                code_switch,
                round_mode,
            )),
            self.tail_rounds().to_vec(),
            self.basecase_plan().clone(),
        );
        pair.validate().map_err(|_| VerificationError)?;
        Ok(pair)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        algebra::{
            embedding::{Basefield, Identity},
            fields::{Field256, Field64_3},
        },
        bits::Bits,
        protocols::params::{Mode, SecuritySpec, TuningSpec},
    };

    fn source<M: Embedding + Default>(mode: Mode, vector_size: usize) -> ProtocolConfig<M> {
        ProtocolConfig::derive(
            SecuritySpec::new(128).with_mode(mode),
            TuningSpec {
                vector_size,
                starting_log_inv_rate: 2,
                folding_factor: FoldingFactor::Constant(2),
                rate_schedule: RateSchedule::Stepping,
            },
        )
        .unwrap()
    }

    fn check_pair<M: Embedding + Default>(mode: Mode) {
        let source = source::<M>(mode, 256);
        let config = source.pair_config().unwrap();
        let original = source.first_round().unwrap();
        let first = config.first_round().unwrap();
        let original_irs = original.code_switch().source();
        let joint_irs = first.code_switch().source();

        // Two 256-entry witnesses become one 512-entry virtual witness.
        assert_eq!(joint_irs.vector_size(), 512);
        assert_eq!(original_irs.interleaving_depth(), 4);
        assert_eq!(joint_irs.interleaving_depth(), 8);
        assert_eq!(first.sumcheck().num_rounds(), 3);
        assert_eq!(
            first.sumcheck().num_rounds(),
            original.sumcheck().num_rounds() + 1
        );

        // The encoded polynomials and query positions remain the same.
        assert_eq!(joint_irs.message_length(), original_irs.message_length());
        assert_eq!(joint_irs.codeword_length(), original_irs.codeword_length());
        assert_eq!(
            joint_irs.in_domain_samples(),
            original_irs.in_domain_samples()
        );
        assert_eq!(joint_irs.mask_length(), original_irs.mask_length());
        assert_eq!(joint_irs.rate().to_bits(), original_irs.rate().to_bits());
        assert_eq!(joint_irs.num_cols(), 2 * original_irs.num_cols());

        // The selector disappears before the unchanged target and remaining rounds.
        assert_eq!(
            first.code_switch().target(),
            original.code_switch().target()
        );
        assert_eq!(config.tail_rounds(), source.tail_rounds());
        assert_eq!(config.basecase_plan(), source.basecase_plan());
        assert_eq!(config.security(), source.security());
        assert_eq!(config.security().target_security_bits, 128);
        config.validate_security_target_met().unwrap();
        config.validate_pow_budget().unwrap();

        if let Some(oracle) = first.mask_oracle() {
            // Three sumcheck rounds need three masks and three fresh masks.
            assert_eq!(oracle.sumcheck_masks().num_masks(), 3);
            assert_eq!(oracle.sumcheck_masks().c_zk_commit().num_vectors(), 6);
            assert_eq!(oracle.cs_mask(), original.mask_oracle().unwrap().cs_mask());
        }
    }

    #[test]
    fn bn254_standard_keeps_commitments_and_adds_one_selector() {
        check_pair::<Identity<Field256>>(Mode::Standard);
    }

    #[test]
    fn bn254_zero_knowledge_adds_one_selector_mask() {
        check_pair::<Identity<Field256>>(Mode::ZeroKnowledge);
    }

    #[test]
    fn extension_standard_keeps_base_field_commitment_encoding() {
        check_pair::<Basefield<Field64_3>>(Mode::Standard);
    }

    #[test]
    fn extension_zero_knowledge_adds_one_selector_mask() {
        check_pair::<Basefield<Field64_3>>(Mode::ZeroKnowledge);
    }

    #[test]
    fn basecase_only_sources_are_rejected() {
        assert!(source::<Identity<Field256>>(Mode::Standard, 8)
            .validate_pair()
            .is_err());
    }

    #[test]
    fn changed_source_security_data_is_rejected() {
        let mut source = source::<Basefield<Field64_3>>(Mode::Standard, 256);
        source.corrupt_round_sumcheck_analytic_for_test(0, Bits::new(1.0));
        assert!(source.validate_pair().is_err());
    }

    #[test]
    fn serialized_source_keeps_its_checks() {
        type M = Basefield<Field64_3>;
        let source = source::<M>(Mode::Standard, 256);
        let encoded = serde_json::to_value(&source).unwrap();
        let decoded: ProtocolConfig<M> = serde_json::from_value(encoded.clone()).unwrap();
        decoded.validate_pair().unwrap();

        for (pointer, value) in [
            (
                "/first_round/code_switch/config/source/in_domain_samples",
                1,
            ),
            (
                "/first_round/code_switch/config/source/interleaving_depth",
                0,
            ),
            ("/tuning/starting_log_inv_rate", u32::MAX),
        ] {
            let mut malformed = encoded.clone();
            *malformed.pointer_mut(pointer).unwrap() = serde_json::json!(value);
            let decoded: ProtocolConfig<M> = serde_json::from_value(malformed).unwrap();
            assert!(decoded.validate_pair().is_err(), "{pointer}");
        }
    }
}
