//! Pair openings add one selector fold to the shared canonical first transition.

use crate::{
    algebra::embedding::Embedding,
    bits::Bits,
    protocols::params::{
        config::{Node, ProtocolConfig, RoundConfig},
        derive::solve_node,
    },
    transcript::{VerificationError, VerificationResult},
};

impl<M: Embedding + Default> ProtocolConfig<M> {
    pub(crate) fn pair_source(&self) -> VerificationResult<Node<M>> {
        if self.tuning().vector_size != self.padded_vector_size()
            || self.tuning().vector_size.checked_mul(2).is_none()
        {
            return Err(VerificationError);
        }
        let first = self.first_round().ok_or(VerificationError)?;
        if let Some(joint) = self.joint_masks() {
            // Two initial mask-binding events share this error budget.
            if joint.analytic_bits().map_err(|_| VerificationError)? - Bits::new(1.0)
                < Bits::new(f64::from(self.security().target_security_bits))
            {
                return Err(VerificationError);
            }
        }
        // The tuple code retains degree, rate, distance, and list size.
        let source = first
            .code_switch()
            .source()
            .joint_pair_source()
            .ok_or(VerificationError)?;
        solve_node(
            self.security(),
            source,
            Some(first.code_switch().target()),
            0,
        )
        .map_err(|_| VerificationError)
    }

    pub(crate) fn pair_round<'a>(
        &'a self,
        node: &'a Node<M>,
    ) -> VerificationResult<RoundConfig<'a, M>> {
        let first = self.first_round().ok_or(VerificationError)?;
        Ok(node.round(
            first.code_switch().target(),
            0,
            self.security().mode.is_zk(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{from_value, to_value};

    use super::*;
    use crate::{
        algebra::{embedding::Identity, fields::Field256},
        protocols::params::{
            FoldingFactor, Mode, PowBudget, RateSchedule, SecuritySpec, TuningSpec,
        },
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
            let source = config.pair_source().unwrap();
            let pair = config.pair_round(&source).unwrap();
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
