//! First-round parameters for two separately committed, equal-sized witnesses.

use crate::{
    algebra::embedding::Embedding,
    bits::Bits,
    protocols::params::{
        config::{ProtocolConfig, RoundConfig},
        derive::{solve_code_switch, solve_sumcheck},
        masks::{mask_coefficients, RoundMasks},
        spec::{RateSchedule, RoundContext},
        SecurityEvent, WitnessMaskConfig,
    },
    transcript::{VerificationError, VerificationResult},
};

impl<M: Embedding + Default> ProtocolConfig<M> {
    pub(crate) fn pair_round(&self) -> VerificationResult<RoundConfig<M>> {
        let tuning = self.tuning();
        if tuning.vector_size != self.padded_vector_size()
            || tuning.vector_size.checked_mul(2).is_none()
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
        let first = self.first_round().ok_or(VerificationError)?;
        if first.sumcheck().num_rounds() == 0 {
            return Err(VerificationError);
        }
        let source = first
            .code_switch()
            .source()
            .joint_pair_source()
            .ok_or(VerificationError)?;
        // Shared source queries give a tuple Reed–Solomon code with the same
        // polynomial degree, rate, distance, and Johnson list bound. Only the
        // number of columns and quadratic sumcheck steps increases.
        let context = RoundContext {
            vector_size: source.vector_size(),
            folding_factor: source.interleaving_depth().ilog2(),
            log_inv_rate: tuning.starting_log_inv_rate,
        };
        let masks = match first.masks() {
            Some(masks) => {
                let joint = self.joint_masks().ok_or(VerificationError)?;
                // Two initial binding events share this error budget. Both mask
                // roots are opened at common indices in the final tuple check.
                if joint.analytic_bits().map_err(|_| VerificationError)? - Bits::new(1.0)
                    < Bits::new(f64::from(self.security().target_security_bits))
                {
                    return Err(VerificationError);
                }
                Some(RoundMasks {
                    witness: WitnessMaskConfig {
                        num_polynomials: masks
                            .witness
                            .num_polynomials
                            .checked_mul(2)
                            .ok_or(VerificationError)?,
                        coefficients_per_polynomial: masks.witness.coefficients_per_polynomial,
                    },
                    sumcheck_coefficients: mask_coefficients(context.folding_factor as usize)
                        .map_err(|_| VerificationError)?,
                })
            }
            None => None,
        };
        let sumcheck = solve_sumcheck(
            self.security(),
            &context,
            &source,
            SecurityEvent::RoundSumcheck { index: 0 },
        )
        .map_err(|_| VerificationError)?;
        let code_switch = solve_code_switch(
            self.security(),
            source,
            first.code_switch().target().clone(),
            first.code_switch().out_domain_samples(),
            0,
        )
        .map_err(|_| VerificationError)?;
        Ok(RoundConfig::new(0, sumcheck, code_switch, masks))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{from_value, to_value};

    use super::*;
    use crate::{
        algebra::{embedding::Identity, fields::Field256},
        protocols::params::{FoldingFactor, Mode, PowBudget, SecuritySpec, TuningSpec},
    };

    fn source(mode: Mode) -> ProtocolConfig<Identity<Field256>> {
        ProtocolConfig::derive(
            SecuritySpec::new(128)
                .with_mode(mode)
                .with_pow_budget(PowBudget::Forbidden),
            TuningSpec {
                vector_size: 4096,
                starting_log_inv_rate: 2,
                folding_factor: FoldingFactor::Constant(2),
                rate_schedule: RateSchedule::Stepping,
            },
        )
        .unwrap()
    }

    #[test]
    fn selector_changes_only_the_first_source_columns_and_fold_count() {
        for mode in [Mode::Standard, Mode::ZeroKnowledge] {
            let config = source(mode);
            let original = config.first_round().unwrap();
            let pair = config.pair_round().unwrap();
            assert_eq!(
                pair.sumcheck().num_rounds(),
                original.sumcheck().num_rounds() + 1
            );
            assert_eq!(
                pair.sumcheck().initial_size(),
                2 * original.sumcheck().initial_size()
            );
            assert_eq!(
                pair.sumcheck().final_size(),
                original.sumcheck().final_size()
            );
            let a = original.code_switch().source();
            let b = pair.code_switch().source();
            assert_eq!(b.interleaving_depth(), 2 * a.interleaving_depth());
            assert_eq!(b.message_length(), a.message_length());
            assert_eq!(b.codeword_length(), a.codeword_length());
            assert_eq!(b.in_domain_samples(), a.in_domain_samples());
            assert_eq!(b.regime(), a.regime());
            assert_eq!(pair.code_switch().target(), original.code_switch().target());
            if let (Some(a), Some(b)) = (original.masks(), pair.masks()) {
                assert_eq!(b.sumcheck_coefficients, a.sumcheck_coefficients + 2);
                assert_eq!(b.witness.num_polynomials, 2 * a.witness.num_polynomials);
                assert_eq!(
                    b.witness.coefficients_per_polynomial,
                    a.witness.coefficients_per_polynomial
                );
            }
        }
    }

    #[test]
    fn pair_rejects_serialized_configuration_drift_and_overflow() {
        let config = source(Mode::ZeroKnowledge);
        for field in ["vector_size", "starting_log_inv_rate"] {
            let mut encoded = to_value(&config).unwrap();
            encoded["tuning"][field] = to_value(if field == "vector_size" {
                usize::MAX as u64
            } else {
                u64::from(u32::MAX)
            })
            .unwrap();
            if let Ok(changed) = from_value::<ProtocolConfig<Identity<Field256>>>(encoded) {
                assert!(changed.validate_pair().is_err());
            }
        }
        let mut encoded = to_value(&config).unwrap();
        encoded["joint_masks"]["commit"]["in_domain_samples"] = to_value(1).unwrap();
        assert!(from_value::<ProtocolConfig<Identity<Field256>>>(encoded).is_err());
    }
}
