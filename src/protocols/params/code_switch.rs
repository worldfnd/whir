//! Code-switching IOR (Construction 9.7, p.55) builder + Lemma 9.9 OOD bound.

use std::num::NonZeroUsize;

use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        fields::FieldWithSize,
    },
    bits::Bits,
    protocols::{
        code_switch::{self, Config as CodeSwitchConfig},
        irs_commit::Config as IrsConfig,
        params::{
            bounds::usize_to_f64,
            branch::SolveMode,
            error::{grind_to_at, DeriveError, Pow},
            protocol_config::MaskOracleInfo,
            solved::Solved,
            spec::SecuritySpec,
        },
    },
};

/// Per-round code-switch builder.
pub fn solve<M: Embedding>(
    spec: &SecuritySpec,
    source: IrsConfig<M>,
    target: IrsConfig<Identity<M::Target>>,
    t_ood: usize,
    mode: SolveMode,
    round_index: usize,
) -> Result<Solved<CodeSwitchConfig<M>>, DeriveError> {
    let (mask_oracle, output_mode) = match mode {
        SolveMode::Standard => (None, code_switch::CodeSwitchMode::Standard),
        SolveMode::ZeroKnowledge(mask_oracle) => {
            let l_zk = mask_oracle.l_zk.get();
            assert!(
                l_zk >= source.mask_length().saturating_add(t_ood),
                "ℓ_zk ({l_zk}) < r + t_ood ({} + {}) — violates Theorem 9.6 witness sizing",
                source.mask_length(),
                t_ood,
            );
            (
                Some(mask_oracle),
                code_switch::CodeSwitchMode::ZeroKnowledge {
                    message_mask_length: NonZeroUsize::new(l_zk).expect("ℓ_zk > 0"),
                },
            )
        }
    };

    let analytic = analytic_error_bits(&source, &target, t_ood, mask_oracle);
    let pow = grind_to_at(spec, analytic, Pow::RoundCodeSwitch { index: round_index })?;

    Ok(Solved::new(
        CodeSwitchConfig::new(source, target, t_ood, output_mode, pow),
        analytic,
    ))
}

/// Per-round code-switch soundness in bits: `min` over Lemma 9.9's three RBR
/// error slots (OOD, in-domain, combination).
pub fn analytic_error_bits<M: Embedding>(
    source: &IrsConfig<M>,
    target: &IrsConfig<Identity<M::Target>>,
    t_ood: usize,
    mask_oracle: Option<MaskOracleInfo>,
) -> Bits {
    assert!(t_ood > 0, "code-switch requires t_ood ≥ 1");

    let field_bits = M::Target::field_size_bits();
    let combined_list =
        target.list_size() * mask_oracle.map_or(1.0, |info| info.c_zk_list_size.get());
    // OOD polynomial is over witness `[f; r_C; s]` of length `ℓ + ℓ_zk` (ZK) or
    // `ℓ` (Standard).
    let degree = mask_oracle.map_or_else(
        || source.message_length(),
        |info| source.message_length().saturating_add(info.l_zk.get()),
    );
    let t_ood_f = usize_to_f64(t_ood);

    // OOD term — Lemma 9.9, term 1.
    let log_degree_minus_1 = usize_to_f64(degree.saturating_sub(1)).log2();
    let log_l_choose_2 = (combined_list * (combined_list - 1.0) / 2.0).log2();
    let ood_term = t_ood_f * (field_bits - log_degree_minus_1) - log_l_choose_2;

    // In-domain term — Lemma 9.9, term 2.
    let in_domain_term = source.rbr_queries();

    // Combination term — Lemma 9.9, term 3 (γ-RLC, bounds doc §5.1).
    let log_count = usize_to_f64(
        t_ood.saturating_add(source.in_domain_samples() * source.interleaving_depth()),
    )
    .log2();
    let combination_term = field_bits - log_count - combined_list.log2();

    Bits::new(ood_term.min(in_domain_term).min(combination_term).max(0.0))
}

/// Number of `(r ‖ s)` mask polynomials code-switch contributes to C_zk per
/// round.
pub const fn masks_required() -> usize {
    1
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::protocols::params::{
        branch::OodMode,
        build_round::{compute_l_zk, solve_t_ood},
        irs_commit as irs_params,
        spec::{
            ListSize, LogInvRate, MaskCodeMessageLen, Mode, OodSampleBudget, PowBudget,
            RoundContext, SecuritySpec, ZkSpec,
        },
        test_utils::{
            arb_standard_spec as utils_standard_spec, arb_zk_spec as utils_zk_spec, assert_close,
            assert_pow_closes_gap, build_round_io, deterministic_spec, TestEmbedding,
            TestExtensionField, TestField, TestNonIdentityEmbedding, TEST_TARGET_RANGE,
        },
    };

    type M = TestEmbedding;

    fn arb_zk_spec() -> impl Strategy<Value = SecuritySpec> {
        utils_zk_spec(TEST_TARGET_RANGE)
    }

    fn arb_standard_spec() -> impl Strategy<Value = SecuritySpec> {
        utils_standard_spec(TEST_TARGET_RANGE)
    }

    const NUM_VARS_HEADROOM: u32 = 4;

    fn arb_dims() -> impl Strategy<Value = (u32, u32, u32)> {
        (1u32..=3, 1u32..=2).prop_flat_map(|(log_inv_rate, folding_factor)| {
            let min_num_vars = 2 * folding_factor;
            (
                Just(log_inv_rate),
                Just(folding_factor),
                min_num_vars..=(min_num_vars + NUM_VARS_HEADROOM),
            )
        })
    }

    const FORMULA_LOG_INV_RATE: u32 = 1;
    const FORMULA_FOLDING_FACTOR: u32 = 2;
    const FORMULA_NUM_VARS: u32 = 6;

    #[test]
    fn analytic_error_standard_formula() {
        let spec: SecuritySpec = deterministic_spec(Mode::Standard);
        let (source, target, t_ood) = build_round_io::<M>(
            &spec,
            FORMULA_LOG_INV_RATE,
            FORMULA_FOLDING_FACTOR,
            FORMULA_NUM_VARS,
            None,
        );
        let got = f64::from(analytic_error_bits(&source, &target, t_ood, None));

        let field_bits = <TestField as FieldWithSize>::field_size_bits();
        let target_list = target.list_size();
        let degree = source.message_length();
        let log_deg_m1 = ((degree - 1) as f64).log2();
        let l_choose_2 = target_list * (target_list - 1.0) / 2.0;
        let ood = (t_ood as f64) * (field_bits - log_deg_m1) - l_choose_2.log2();
        let in_domain = source.rbr_queries();
        let count = t_ood + source.in_domain_samples() * source.interleaving_depth();
        let comb = field_bits - (count as f64).log2() - target_list.log2();
        let expected = ood.min(in_domain).min(comb).max(0.0);

        assert_close(got, expected);
    }

    #[test]
    fn analytic_error_zk_formula() {
        const C_ZK_LIST_SIZE: f64 = 4.0;
        const L_ZK_USIZE: usize = 8;

        let spec: SecuritySpec = deterministic_spec(Mode::ZeroKnowledge);
        let mask_oracle = MaskOracleInfo {
            c_zk_list_size: ListSize::new(C_ZK_LIST_SIZE),
            l_zk: MaskCodeMessageLen::new(L_ZK_USIZE),
        };
        let (source, target, t_ood) = build_round_io::<M>(
            &spec,
            FORMULA_LOG_INV_RATE,
            FORMULA_FOLDING_FACTOR,
            FORMULA_NUM_VARS,
            Some(FORMULA_LOG_INV_RATE),
        );
        let got = f64::from(analytic_error_bits(
            &source,
            &target,
            t_ood,
            Some(mask_oracle),
        ));

        let field_bits = <TestField as FieldWithSize>::field_size_bits();
        let target_list = target.list_size();
        let combined_list = target_list * C_ZK_LIST_SIZE;
        let degree = source.message_length() + L_ZK_USIZE;
        let log_deg_m1 = ((degree - 1) as f64).log2();
        let l_choose_2 = combined_list * (combined_list - 1.0) / 2.0;
        let ood = (t_ood as f64) * (field_bits - log_deg_m1) - l_choose_2.log2();
        let in_domain = source.rbr_queries();
        let count = t_ood + source.in_domain_samples() * source.interleaving_depth();
        let comb = field_bits - (count as f64).log2() - target_list.log2() - C_ZK_LIST_SIZE.log2();
        let expected = ood.min(in_domain).min(comb).max(0.0);

        assert_close(got, expected);
    }

    #[test]
    fn analytic_error_uses_in_domain_when_limiting() {
        const LIMITING_TARGET_BITS: u32 = 16;
        const LIMITING_LOG_INV_RATE: u32 = 1;
        const LIMITING_FOLDING_FACTOR: u32 = 1;
        const LIMITING_NUM_VARS: u32 = 4;

        let spec = SecuritySpec::new(LIMITING_TARGET_BITS).with_pow_budget(PowBudget::Forbidden);
        let (source, target, t_ood) = build_round_io::<M>(
            &spec,
            LIMITING_LOG_INV_RATE,
            LIMITING_FOLDING_FACTOR,
            LIMITING_NUM_VARS,
            None,
        );

        let field_bits = <TestField as FieldWithSize>::field_size_bits();
        let target_list = target.list_size();
        let degree = source.message_length();
        let log_deg_m1 = ((degree - 1) as f64).log2();
        let l_choose_2 = target_list * (target_list - 1.0) / 2.0;
        let ood = (t_ood as f64) * (field_bits - log_deg_m1) - l_choose_2.log2();
        let in_domain = source.rbr_queries();
        let count = t_ood + source.in_domain_samples() * source.interleaving_depth();
        let comb = field_bits - (count as f64).log2() - target_list.log2();
        assert!(
            in_domain < ood && in_domain < comb,
            "fixture wants in_domain to bind: in_domain {in_domain}, ood {ood}, comb {comb}",
        );

        let got = f64::from(analytic_error_bits(&source, &target, t_ood, None));
        assert_close(got, in_domain.max(0.0));
    }

    proptest! {
        #[test]
        fn solve_standard_assembles(
            spec in arb_standard_spec(),
            (log_inv_rate, folding_factor, num_vars) in arb_dims(),
        ) {
            let (source, target, t_ood) =
                build_round_io::<M>(&spec, log_inv_rate, folding_factor, num_vars, None);
            let config = solve(&spec, source, target, t_ood, SolveMode::Standard, 0).unwrap();
            prop_assert!(matches!(config.mode(), code_switch::CodeSwitchMode::Standard));
            prop_assert!(config.out_domain_samples() >= 1);
        }

        #[test]
        fn solve_zk_mask_equals_padded_r_plus_t_ood(
            spec in arb_zk_spec(),
            (log_inv_rate, folding_factor, num_vars) in arb_dims(),
        ) {
            let (source, target, t_ood) = build_round_io::<M>(
                &spec, log_inv_rate, folding_factor, num_vars, Some(log_inv_rate),
            );
            let r = source.mask_length();
            let l_zk = compute_l_zk(&source, t_ood);
            let zk_spec = ZkSpec::try_new(&spec).expect("arb_zk_spec");
            let c_zk = irs_params::solve_mask_code::<M>(
                zk_spec,
                l_zk,
                r,
                LogInvRate::new(log_inv_rate),
                2,
            )
            .expect("C_zk fixture must solve");
            let mask_oracle = MaskOracleInfo {
                c_zk_list_size: ListSize::new(c_zk.list_size()),
                l_zk,
            };
            let config = solve(
                &spec,
                source,
                target,
                t_ood,
                SolveMode::ZeroKnowledge(mask_oracle),
                0,
            )
            .unwrap();
            prop_assert_eq!(config.message_mask_length(), (r + t_ood).next_power_of_two());
        }

        #[test]
        fn pow_closes_gap_to_target_standard(
            spec in arb_standard_spec(),
            (log_inv_rate, folding_factor, num_vars) in arb_dims(),
        ) {
            let (source, target, t_ood) =
                build_round_io::<M>(&spec, log_inv_rate, folding_factor, num_vars, None);
            let error = analytic_error_bits(&source, &target, t_ood, None);
            let config = solve(&spec, source, target, t_ood, SolveMode::Standard, 0).unwrap();
            assert_pow_closes_gap(&spec, error, &config.pow());
        }
    }

    fn non_identity_smoke_ctxs() -> (RoundContext, RoundContext) {
        const SOURCE_VECTOR_SIZE: usize = 64;
        const SOURCE_LOG_INV_RATE: u32 = 1;
        const FOLDING_FACTOR: u32 = 2;

        let source_ctx = RoundContext {
            vector_size: SOURCE_VECTOR_SIZE,
            log_inv_rate: SOURCE_LOG_INV_RATE,
            folding_factor: FOLDING_FACTOR,
        };
        let target_ctx = RoundContext {
            vector_size: source_ctx.vector_size / (1 << source_ctx.folding_factor),
            log_inv_rate: source_ctx.log_inv_rate + source_ctx.folding_factor - 1,
            folding_factor: source_ctx.folding_factor,
        };
        (source_ctx, target_ctx)
    }

    #[test]
    #[should_panic(expected = "violates Theorem 9.6")]
    fn solve_zk_rejects_l_zk_below_r_plus_t_ood() {
        const TOO_SMALL_L_ZK: usize = 1;

        let spec: SecuritySpec = deterministic_spec(Mode::ZeroKnowledge);
        let (source, target, t_ood) = build_round_io::<M>(
            &spec,
            FORMULA_LOG_INV_RATE,
            FORMULA_FOLDING_FACTOR,
            FORMULA_NUM_VARS,
            Some(FORMULA_LOG_INV_RATE),
        );
        assert!(source.mask_length() + t_ood > TOO_SMALL_L_ZK);

        let mask_oracle = MaskOracleInfo {
            c_zk_list_size: ListSize::new(SMOKE_C_ZK_LIST_SIZE),
            l_zk: MaskCodeMessageLen::new(TOO_SMALL_L_ZK),
        };
        let _ = solve(
            &spec,
            source,
            target,
            t_ood,
            SolveMode::ZeroKnowledge(mask_oracle),
            0,
        );
    }

    #[test]
    fn solve_works_with_basefield_embedding_standard() {
        let spec: SecuritySpec = deterministic_spec(Mode::Standard);
        let (source_ctx, target_ctx) = non_identity_smoke_ctxs();
        let target_log_degree =
            f64::from((source_ctx.vector_size / (1 << source_ctx.folding_factor)).trailing_zeros());
        let target_list_size = spec
            .decoding_regime
            .list_size_estimate(target_log_degree, f64::from(target_ctx.log_inv_rate));
        let (source, t_ood) = solve_t_ood::<TestNonIdentityEmbedding>(
            &spec,
            &source_ctx,
            target_list_size,
            OodMode::Standard,
            0,
        )
        .unwrap();
        let target = irs_params::solve::<Identity<TestExtensionField>>(
            &spec,
            &target_ctx,
            OodSampleBudget::ZERO,
        )
        .expect("target IRS fixture must solve");

        let config = solve(&spec, source, target, t_ood, SolveMode::Standard, 0).unwrap();
        assert!(matches!(
            config.mode(),
            code_switch::CodeSwitchMode::Standard
        ));
    }

    const SMOKE_C_ZK_LIST_SIZE: f64 = 4.0;

    #[test]
    fn solve_works_with_basefield_embedding_zk() {
        let spec: SecuritySpec = deterministic_spec(Mode::ZeroKnowledge);
        let (source_ctx, target_ctx) = non_identity_smoke_ctxs();
        let target_log_degree =
            f64::from((source_ctx.vector_size / (1 << source_ctx.folding_factor)).trailing_zeros());
        let target_list_size = spec
            .decoding_regime
            .list_size_estimate(target_log_degree, f64::from(target_ctx.log_inv_rate));
        let (source, t_ood) = solve_t_ood::<TestNonIdentityEmbedding>(
            &spec,
            &source_ctx,
            target_list_size,
            OodMode::ZeroKnowledge(LogInvRate::new(source_ctx.log_inv_rate)),
            0,
        )
        .unwrap();
        let target = irs_params::solve::<Identity<TestExtensionField>>(
            &spec,
            &target_ctx,
            OodSampleBudget::new(t_ood),
        )
        .expect("target IRS fixture must solve");

        let mask_oracle = MaskOracleInfo {
            c_zk_list_size: ListSize::new(SMOKE_C_ZK_LIST_SIZE),
            l_zk: MaskCodeMessageLen::new((source.mask_length() + t_ood).next_power_of_two()),
        };
        let config = solve(
            &spec,
            source,
            target,
            t_ood,
            SolveMode::ZeroKnowledge(mask_oracle),
            0,
        )
        .unwrap();
        assert!(matches!(
            config.mode(),
            code_switch::CodeSwitchMode::ZeroKnowledge { .. }
        ));
    }
}
