//! Sumcheck configuration and per-step security.

use ark_ff::Field;

use crate::{
    algebra::{embedding::Embedding, fields::FieldWithSize},
    bits::Bits,
    protocols::{
        irs_commit::Config as IrsConfig,
        params::{
            branch::SolveMode,
            error::{grind_to_at, DeriveError, Pow, ZookSecurityError},
            regime::constrained_agreement_bits,
            solved::Solved,
            spec::{RoundContext, SecuritySpec},
        },
        sumcheck::{Config as SumcheckConfig, SumcheckMaskLen, SumcheckMode},
    },
};

pub fn solve<M: Embedding>(
    spec: &SecuritySpec,
    ctx: &RoundContext,
    source: &IrsConfig<M>,
    mode: SolveMode,
    pow: Pow,
) -> Result<Solved<SumcheckConfig<M::Target>>, DeriveError> {
    let analytic = analytic_error_bits(source, mode.is_zk())?;
    let round_pow = grind_to_at(spec, analytic, pow)?;
    let output_mode = match mode {
        SolveMode::Standard => SumcheckMode::Standard,
        SolveMode::ZeroKnowledge(()) => SumcheckMode::ZeroKnowledge {
            mask_length: SumcheckMaskLen::new(3),
        },
    };
    Ok(Solved::new(
        SumcheckConfig::new(
            ctx.vector_size,
            round_pow,
            ctx.folding_factor as usize,
            output_mode,
        ),
        analytic,
    ))
}

/// Per-step security of the configured sumcheck protocol.
pub fn analytic_error_bits<M: Embedding>(
    source: &IrsConfig<M>,
    is_zk: bool,
) -> Result<Bits, ZookSecurityError> {
    if is_zk {
        return zook_sumcheck_error_bits(source);
    }
    let polynomial_identity = M::Target::field_size_bits() - source.list_size().log2() - 1.0;
    Ok(Bits::new(
        source
            .rbr_soundness_fold_prox_gaps()
            .min(polynomial_identity)
            .max(0.0),
    ))
}

/// A quadratic sumcheck step on a revealed vector fails with probability at most `2 / |F|`.
///
/// The vector is fixed before these challenges, so there is no codeword-list factor.
/// This is the additional reduction after §5.4, “The final opening protocol”.
pub(super) fn revealed_sumcheck_error_bits<F: Field>() -> Bits {
    Bits::new((F::field_size_bits() - 1.0).max(0.0))
}

/// Sparse sumcheck mask length: one constant and two coefficients per variable.
///
/// ProveKit Prover Whitepaper §5.4, “ZO0K”, “The required masks”: `2 * folds + 1`.
pub(super) fn mask_coefficients(folds: usize) -> Result<usize, DeriveError> {
    folds
        .checked_mul(2)
        .and_then(|n| n.checked_add(1))
        .ok_or(DeriveError::SizeOverflow)
}

/// Security of each quadratic sumcheck step after joint-mask binding:
/// `epsilon <= L * ((2/3) * L^4 * rho + 1) * n / Q`.
/// Here `L`, `rho`, and `n` are the source code's list bound, rate, and
/// codeword length; `Q` is the target-field size.
///
/// ProveKit Prover Whitepaper §6.4, “Soundness”, theorem “ZO0K soundness under
/// mask binding”, sumcheck phase; the agreement bound is in Appendix A,
/// “Correlated agreement”. Conditioning on mask binding removes the mask-list
/// multiplicity; this is a bound for one step, not a union over all steps.
pub(super) fn zook_sumcheck_error_bits<M: Embedding>(
    source: &IrsConfig<M>,
) -> Result<Bits, ZookSecurityError> {
    constrained_agreement_bits(source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::params::{
        irs_commit::solve as solve_irs,
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
                let solved = solve(
                    &spec,
                    &ctx,
                    &source,
                    SolveMode::Standard,
                    Pow::RoundSumcheck { index: 0 },
                )
                .unwrap();
                let expected = source
                    .rbr_soundness_fold_prox_gaps()
                    .min(TestField::field_size_bits() - source.list_size().log2() - 1.0)
                    .max(0.0);
                assert_close(f64::from(solved.analytic()), expected);
                assert_eq!(solved.initial_size(), ctx.vector_size);
                assert_eq!(solved.num_rounds(), folds as usize);
                assert!(matches!(solved.mode(), SumcheckMode::Standard));
                assert!(expected + f64::from(solved.round_pow().difficulty()) >= 80.0 - 1e-9);
            }
        }
    }
}
