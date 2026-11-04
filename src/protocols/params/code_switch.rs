//! Code-switch configuration and query/aggregation security.

use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        fields::FieldWithSize,
    },
    bits::Bits,
    protocols::{
        code_switch::{CodeSwitchMode, Config as CodeSwitchConfig},
        irs_commit::Config as IrsConfig,
        params::{
            bounds::{error_bits, log2_sum, out_of_domain_bits, usize_to_f64},
            branch::SolveMode,
            error::{grind_to_at, DeriveError, Pow, ZookSecurityError},
            irs_commit::validate_zook_code,
            solved::Solved,
            spec::SecuritySpec,
        },
    },
};

pub fn solve<M: Embedding>(
    spec: &SecuritySpec,
    source: IrsConfig<M>,
    target: IrsConfig<Identity<M::Target>>,
    t_ood: usize,
    mode: SolveMode,
    round_index: usize,
) -> Result<Solved<CodeSwitchConfig<M>>, DeriveError> {
    let analytic = analytic_error_bits(&source, &target, t_ood, mode.is_zk())?;
    let pow = grind_to_at(spec, analytic, Pow::RoundCodeSwitch { index: round_index })?;
    // Witness overlays and mask claims belong to Zook. The code-switch
    // primitive receives ordinary IRS commitments without appended masks.
    Ok(Solved::new(
        CodeSwitchConfig::new(source, target, t_ood, CodeSwitchMode::Standard, pow),
        analytic,
    ))
}

/// Per-step query and aggregation security.
pub fn analytic_error_bits<M: Embedding>(
    source: &IrsConfig<M>,
    target: &IrsConfig<Identity<M::Target>>,
    t_ood: usize,
    is_zk: bool,
) -> Result<Bits, ZookSecurityError> {
    if is_zk {
        let (queries, aggregation) = zook_query_error_bits(source, target)?;
        return Ok(queries.min(aggregation));
    }
    let field_bits = M::Target::field_size_bits();
    let list = target.list_size();
    let degree = source.message_length();
    let ood = usize_to_f64(t_ood) * (field_bits - usize_to_f64(degree.saturating_sub(1)).log2())
        - (list * (list - 1.0) / 2.0).log2();
    let count = usize_to_f64(t_ood + source.in_domain_samples() * source.interleaving_depth());
    let combination = field_bits - count.log2() - list.log2();
    Ok(Bits::new(
        ood.min(source.rbr_queries()).min(combination).max(0.0),
    ))
}

/// Query and aggregation security, conditional on joint-mask binding:
/// `epsilon_query <= L^2 * m / (2*(Q - n)) + (1 - theta)^s`;
/// `epsilon_aggregation <= L * s / Q`.
/// Here `L = target.list_size()`, `m = target.vector_size()`,
/// `n = target.codeword_length()`, and `Q` is the target-field size.
/// The source has proximity radius `theta` and `s = source.in_domain_samples()`.
/// The DEEP challenge is uniform outside the target domain of size `n`.
///
/// ProveKit Prover Whitepaper §6.4, “Soundness”, theorem “ZO0K soundness under
/// mask binding”, query phase and aggregation step.
/// The query error is the sum of DEEP and in-domain errors.
pub(super) fn zook_query_error_bits<M: Embedding>(
    source: &IrsConfig<M>,
    target: &IrsConfig<Identity<M::Target>>,
) -> Result<(Bits, Bits), ZookSecurityError> {
    validate_zook_code(source)?;
    validate_zook_code(target)?;
    if source.num_vectors() != 1
        || target.num_vectors() != 1
        || source.message_length() != target.vector_size()
    {
        return Err(ZookSecurityError::InvalidRoundShape);
    }
    let field_bits = M::Target::field_size_bits();
    let list_bits = target.list_size().log2();
    let outside_bits = out_of_domain_bits(field_bits, target.codeword_length())?;
    // The DEEP evaluation is of the whole folded polynomial, before its
    // coefficients are split into the target's interleaved columns.
    let deep_log_error =
        2.0 * list_bits - 1.0 + usize_to_f64(target.vector_size()).log2() - outside_bits;
    let query_log_error = -source.rbr_queries();
    Ok((
        error_bits(log2_sum(deep_log_error, query_log_error)),
        error_bits(list_bits + usize_to_f64(source.in_domain_samples()).log2() - field_bits),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::params::{
        build_round::solve_t_ood,
        irs_commit::solve as solve_irs,
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
            let solved = solve(&spec, source, target, t_ood, SolveMode::Standard, 0).unwrap();
            assert_close(f64::from(solved.analytic()), expected);
            assert!(matches!(solved.mode(), CodeSwitchMode::Standard));
            assert!(expected + f64::from(solved.pow().difficulty()) >= 80.0 - 1e-9);
        }
    }
}
