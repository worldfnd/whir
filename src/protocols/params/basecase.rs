//! Basecase (Construction 7.2, p.43) parameter selection + γ-combination bound.

use ark_ff::Field;

use crate::{
    algebra::{embedding::Identity, fields::FieldWithSize},
    bits::Bits,
    protocols::{
        basecase::{self, Config as BasecaseConfig},
        irs_commit::Config as IrsConfig,
        params::{
            error::{grind_to_at, DeriveError, Pow},
            irs_commit as irs_params,
            protocol_config::BasecasePlan,
            spec::{Mode as SpecMode, OodSampleBudget, RoundContext, SecuritySpec},
            sumcheck as sumcheck_params,
        },
        proof_of_work::Config as PowConfig,
        sumcheck::{self, Config as SumcheckConfig},
    },
};

pub fn solve<F: Field>(
    spec: &SecuritySpec,
    vector_size: usize,
    log_inv_rate: u32,
) -> Result<BasecasePlan<F>, DeriveError> {
    assert!(vector_size > 0, "basecase requires vector_size ≥ 1");

    let ctx = RoundContext {
        vector_size,
        log_inv_rate,
        folding_factor: 0,
    };
    let commit = irs_params::solve(spec, &ctx, OodSampleBudget::ZERO)?;
    solve_with_commit(spec, commit)
}

/// Same as [`solve`] but with a pre-built IRS config — used by `derive` when
/// the last round's `code_switch.target` is being reused as the basecase
/// commit (Phase 2: no recommit of the folded message). The shared IRS is
/// built once at the previous-round layer; this function only owns the
/// basecase-specific sumcheck + γ-combination PoW grind.
pub fn solve_with_commit<F: Field>(
    spec: &SecuritySpec,
    commit: IrsConfig<Identity<F>>,
) -> Result<BasecasePlan<F>, DeriveError> {
    let vector_size = commit.vector_size();
    assert!(vector_size > 0, "basecase requires vector_size ≥ 1");

    let sumcheck_analytic = sumcheck_params::analytic_error_bits(&commit, None);
    let sumcheck_pow = grind_to_at(spec, sumcheck_analytic, Pow::BasecaseSumcheck)?;
    let sumcheck = SumcheckConfig::new(
        vector_size,
        sumcheck_pow,
        vector_size.next_power_of_two().trailing_zeros() as usize,
        sumcheck::SumcheckMode::Standard,
    );

    let gamma_analytic = analytic_error_bits(&commit);
    let (mode, pow) = match spec.mode {
        SpecMode::Standard => (basecase::BasecaseMode::Standard, PowConfig::none()),
        SpecMode::ZeroKnowledge => (
            basecase::BasecaseMode::ZeroKnowledge,
            grind_to_at(spec, gamma_analytic, Pow::BasecaseGammaCombination)?,
        ),
    };

    Ok(BasecasePlan::new(
        BasecaseConfig::new(commit, sumcheck, mode, pow),
        sumcheck_analytic,
        gamma_analytic,
    ))
}

/// γ-combination soundness (Lemma 7.4 combination-randomness slot, paper p.45).
pub fn analytic_error_bits<F: Field>(commit: &IrsConfig<Identity<F>>) -> Bits {
    let field_bits = F::field_size_bits();
    let log_list = commit.list_size().log2();
    let prox_gaps = commit.rbr_soundness_fold_prox_gaps();
    let poly_id = field_bits - log_list;
    Bits::new(prox_gaps.min(poly_id).max(0.0))
}

impl<F: Field> BasecaseConfig<F> {
    /// Analytic soundness bits (excluding PoW): `min(sumcheck round error, γ-slot error)`.
    pub fn analytic_bits(&self) -> Bits {
        let sumcheck_term = f64::from(sumcheck_params::analytic_error_bits(self.commit(), None));
        let min_bits = match self.mode() {
            basecase::BasecaseMode::Standard => sumcheck_term,
            basecase::BasecaseMode::ZeroKnowledge => {
                sumcheck_term.min(f64::from(analytic_error_bits(self.commit())))
            }
        };
        Bits::new(min_bits.max(0.0))
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::protocols::params::test_utils::{
        arb_standard_spec, arb_zk_spec, assert_close, assert_pow_closes_gap, deterministic_spec,
        TestField, TEST_TARGET_RANGE,
    };

    const FIXTURE_VECTOR_SIZE: usize = 16;
    const FIXTURE_LOG_INV_RATE: u32 = 2;

    fn arb_dims() -> impl Strategy<Value = (u32, u32)> {
        (1u32..=4, 1u32..=3)
    }

    #[test]
    fn analytic_error_formula() {
        use crate::protocols::params::{
            irs_commit as irs_params,
            spec::{Mode, OodSampleBudget, RoundContext},
        };

        let spec = deterministic_spec(Mode::ZeroKnowledge);
        let ctx = RoundContext {
            vector_size: FIXTURE_VECTOR_SIZE,
            log_inv_rate: FIXTURE_LOG_INV_RATE,
            folding_factor: 0,
        };
        let commit: IrsConfig<Identity<TestField>> =
            irs_params::solve(&spec, &ctx, OodSampleBudget::ZERO).expect("IRS fixture must solve");

        let got = f64::from(analytic_error_bits(&commit));
        let field_bits = TestField::field_size_bits();
        let log_list = commit.list_size().log2();
        let prox_gaps = commit.rbr_soundness_fold_prox_gaps();
        let poly_id = field_bits - log_list;
        let expected = prox_gaps.min(poly_id).max(0.0);

        assert_close(got, expected);
    }

    #[test]
    fn analytic_error_uses_eps_mca_when_limiting() {
        use crate::protocols::params::{
            irs_commit as irs_params,
            spec::{Mode, OodSampleBudget, RoundContext},
        };

        let spec = deterministic_spec(Mode::ZeroKnowledge);
        let ctx = RoundContext {
            vector_size: FIXTURE_VECTOR_SIZE,
            log_inv_rate: 1,
            folding_factor: 0,
        };
        let commit: IrsConfig<Identity<TestField>> =
            irs_params::solve(&spec, &ctx, OodSampleBudget::ZERO).expect("IRS fixture must solve");

        let field_bits = TestField::field_size_bits();
        let log_list = commit.list_size().log2();
        let prox_gaps = commit.rbr_soundness_fold_prox_gaps();
        let poly_id = field_bits - log_list;
        assert!(
            prox_gaps < poly_id,
            "fixture wants prox_gaps to bind: prox_gaps {prox_gaps} ≥ poly_id {poly_id}",
        );

        let got = f64::from(analytic_error_bits(&commit));
        assert_close(got, prox_gaps.max(0.0));
    }

    proptest! {
        #[test]
        fn solve_standard_assembles(
            spec in arb_standard_spec(TEST_TARGET_RANGE),
            (log_size, log_inv_rate) in arb_dims(),
        ) {
            let config = solve::<TestField>(&spec, 1usize << log_size, log_inv_rate).unwrap();
            prop_assert!(matches!(config.mode(), basecase::BasecaseMode::Standard));
            prop_assert_eq!(config.commit().interleaving_depth(), 1);
            prop_assert_eq!(config.commit().num_vectors(), 1);
            prop_assert_eq!(config.commit().vector_size(), config.sumcheck().initial_size());
        }

        #[test]
        fn solve_zk_assembles(
            spec in arb_zk_spec(TEST_TARGET_RANGE),
            (log_size, log_inv_rate) in arb_dims(),
        ) {
            let config = solve::<TestField>(&spec, 1usize << log_size, log_inv_rate).unwrap();
            prop_assert!(matches!(config.mode(), basecase::BasecaseMode::ZeroKnowledge));
            prop_assert!(config.commit().mask_length() > 0);
        }

        #[test]
        fn pow_closes_gap_to_target_zk(
            spec in arb_zk_spec(TEST_TARGET_RANGE),
            (log_size, log_inv_rate) in arb_dims(),
        ) {
            let config = solve::<TestField>(&spec, 1usize << log_size, log_inv_rate).unwrap();
            assert_pow_closes_gap(&spec, analytic_error_bits(config.commit()), &config.pow());
        }

        #[test]
        fn standard_mode_has_no_pow(
            spec in arb_standard_spec(TEST_TARGET_RANGE),
            (log_size, log_inv_rate) in arb_dims(),
        ) {
            let config = solve::<TestField>(&spec, 1usize << log_size, log_inv_rate).unwrap();
            prop_assert_eq!(config.pow(), PowConfig::none());
        }
    }
}
