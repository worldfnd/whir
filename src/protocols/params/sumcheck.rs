//! Sumcheck parameter selection. ZK mode adds a degree-2 mask per round
//! (Lemma 6.4, p.38).

use crate::{
    algebra::{embedding::Embedding, fields::FieldWithSize},
    bits::Bits,
    protocols::{
        irs_commit::Config as IrsConfig,
        params::{
            bounds::usize_to_f64,
            branch::SolveMode,
            error::{grind_to_at, DeriveError, Pow},
            protocol_config::MaskOracleInfo,
            solved::Solved,
            spec::{RoundContext, SecuritySpec},
        },
        sumcheck::{self, Config as SumcheckConfig, SumcheckMaskLen},
    },
};

/// Per-round sumcheck builder.
pub fn solve<M: Embedding>(
    spec: &SecuritySpec,
    ctx: &RoundContext,
    source_irs: &IrsConfig<M>,
    mode: SolveMode,
    pow: Pow,
) -> Result<Solved<SumcheckConfig<M::Target>>, DeriveError> {
    let (mask_oracle, output_mode) = match mode {
        SolveMode::Standard => (None, sumcheck::SumcheckMode::Standard),
        SolveMode::ZeroKnowledge(mask_oracle) => (
            Some(mask_oracle),
            sumcheck::SumcheckMode::ZeroKnowledge {
                mask_length: zk_mask_length(),
            },
        ),
    };
    let analytic = analytic_error_bits(source_irs, mask_oracle);
    let round_pow = grind_to_at(spec, analytic, pow)?;
    Ok(Solved::new(
        SumcheckConfig::new(
            ctx.vector_size,
            round_pow,
            num_sumcheck_rounds(ctx),
            output_mode,
        ),
        analytic,
    ))
}

/// Per-sumcheck-round soundness in bits: `min(ε_mca, poly_identity_term)`.
///
/// - Standard (degree-2): `log|F| − log|Λ(C)| − 1`.
/// - ZK (Lemma 6.5, p.40): `log|F| − log|Λ(C)| − log|Λ(C_zk)| − log ℓ_zk`.
pub fn analytic_error_bits<M: Embedding>(
    source_irs: &IrsConfig<M>,
    mask_oracle: Option<MaskOracleInfo>,
) -> Bits {
    let field_bits = M::Target::field_size_bits();
    let log_list_size = source_irs.list_size().log2();
    let prox_gaps = source_irs.rbr_soundness_fold_prox_gaps();

    let poly_id = mask_oracle.map_or(field_bits - log_list_size - 1.0, |info| {
        let log_list_size_c_zk = info.c_zk_list_size.log2();
        let log_l_zk = usize_to_f64(info.l_zk.get()).log2();
        field_bits - log_list_size - log_list_size_c_zk - log_l_zk
    });

    Bits::new(prox_gaps.min(poly_id).max(0.0))
}

/// Number of degree-2 round-polynomial masks sumcheck contributes to C_zk
/// per round (Lemma 6.4).
pub const fn masks_required(ctx: &RoundContext) -> usize {
    num_sumcheck_rounds(ctx)
}

const fn num_sumcheck_rounds(ctx: &RoundContext) -> usize {
    ctx.folding_factor as usize
}

/// Construction 6.3 step 4(a) sends `h_j ∈ F^{<max{2, ℓ_zk}}[X]`. WHIR's round
/// polynomial is degree-2, so 3 coefficients suffice; `ℓ_zk = 3` is the
/// smallest value that masks it (Lemma 6.4 requires only `ℓ_zk ≥ 2`).
pub const fn zk_mask_length() -> SumcheckMaskLen {
    SumcheckMaskLen::new(3)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::protocols::params::{
        irs_commit as irs_params,
        spec::{ListSize, MaskCodeMessageLen, Mode, OodSampleBudget},
        test_utils::{
            arb_round_ctx, arb_standard_spec, arb_zk_spec, assert_close, assert_pow_closes_gap,
            build_minimal_mask_oracle, deterministic_spec, TestEmbedding, TestField,
            TestNonIdentityEmbedding, EPS, TEST_TARGET_RANGE,
        },
    };

    const FIXTURE_C_ZK_LIST_SIZE: f64 = 4.0;
    const FIXTURE_L_ZK: usize = 8;

    fn build_source_irs(spec: &SecuritySpec, ctx: &RoundContext) -> IrsConfig<TestEmbedding> {
        irs_params::solve(spec, ctx, OodSampleBudget::ZERO).expect("source IRS fixture must solve")
    }

    const FIXTURE_LOG_VECTOR_SIZE: u32 = 4;
    const FIXTURE_LOG_INV_RATE: u32 = 1;
    const FIXTURE_FOLDING_FACTOR: u32 = 2;

    fn fixture_ctx() -> RoundContext {
        RoundContext {
            vector_size: 1 << FIXTURE_LOG_VECTOR_SIZE,
            log_inv_rate: FIXTURE_LOG_INV_RATE,
            folding_factor: FIXTURE_FOLDING_FACTOR,
        }
    }

    #[test]
    fn zk_mode_has_three_mask_coefficients() {
        let spec = deterministic_spec(Mode::ZeroKnowledge);
        let ctx = fixture_ctx();
        let source_irs = build_source_irs(&spec, &ctx);
        let mask_oracle =
            build_minimal_mask_oracle(&spec).expect("ZK spec must produce a mask oracle");
        let config = solve(
            &spec,
            &ctx,
            &source_irs,
            SolveMode::ZeroKnowledge(mask_oracle),
            Pow::RoundSumcheck { index: 0 },
        )
        .unwrap();
        match config.mode() {
            sumcheck::SumcheckMode::ZeroKnowledge { mask_length } => {
                assert_eq!(mask_length.get(), 3);
            }
            sumcheck::SumcheckMode::Standard => panic!("expected ZK"),
        }
    }

    #[test]
    fn analytic_error_standard_formula() {
        let spec = deterministic_spec(Mode::Standard);
        let ctx = fixture_ctx();
        let irs = build_source_irs(&spec, &ctx);

        let got = f64::from(analytic_error_bits::<TestEmbedding>(&irs, None));

        let field_bits = TestField::field_size_bits();
        let log_list = irs.list_size().log2();
        let prox = irs.rbr_soundness_fold_prox_gaps();
        let expected = prox.min(field_bits - log_list - 1.0).max(0.0);

        assert_close(got, expected);
    }

    #[test]
    fn analytic_error_zk_formula() {
        let log_c_zk_list = FIXTURE_C_ZK_LIST_SIZE.log2();
        let log_l_zk = (FIXTURE_L_ZK as f64).log2();

        let spec = deterministic_spec(Mode::ZeroKnowledge);
        let ctx = fixture_ctx();
        let irs = build_source_irs(&spec, &ctx);
        let info = MaskOracleInfo {
            c_zk_list_size: ListSize::new(FIXTURE_C_ZK_LIST_SIZE),
            l_zk: MaskCodeMessageLen::new(FIXTURE_L_ZK),
        };

        let got = f64::from(analytic_error_bits::<TestEmbedding>(&irs, Some(info)));

        let field_bits = TestField::field_size_bits();
        let log_list = irs.list_size().log2();
        let prox = irs.rbr_soundness_fold_prox_gaps();
        let expected = prox
            .min(field_bits - log_list - log_c_zk_list - log_l_zk)
            .max(0.0);

        assert_close(got, expected);
    }

    #[test]
    fn analytic_error_clamps_to_zero() {
        const OVERSIZED_LOG_C_ZK_LIST: i32 = 60;
        const OVERSIZED_LOG_L_ZK: u32 = 30;

        let spec = deterministic_spec(Mode::ZeroKnowledge);
        let ctx = fixture_ctx();
        let irs = build_source_irs(&spec, &ctx);
        let huge = MaskOracleInfo {
            c_zk_list_size: ListSize::new(2_f64.powi(OVERSIZED_LOG_C_ZK_LIST)),
            l_zk: MaskCodeMessageLen::new(1 << OVERSIZED_LOG_L_ZK),
        };
        let bits = f64::from(analytic_error_bits::<TestEmbedding>(&irs, Some(huge)));
        assert_close(bits, 0.0);
    }

    proptest! {
        #[test]
        fn standard_mode_propagates(
            spec in arb_standard_spec(TEST_TARGET_RANGE),
            ctx in arb_round_ctx(),
        ) {
            let source_irs = build_source_irs(&spec, &ctx);
            let pow = Pow::RoundSumcheck { index: 0 };
            let config = solve(&spec, &ctx, &source_irs, SolveMode::Standard, pow).unwrap();
            prop_assert!(matches!(config.mode(), sumcheck::SumcheckMode::Standard));
        }

        #[test]
        fn num_rounds_matches_folding_factor(
            spec in prop_oneof![
                arb_standard_spec(TEST_TARGET_RANGE),
                arb_zk_spec(TEST_TARGET_RANGE),
            ],
            ctx in arb_round_ctx(),
        ) {
            let source_irs = build_source_irs(&spec, &ctx);
            let pow = Pow::RoundSumcheck { index: 0 };
            let mode = build_minimal_mask_oracle(&spec)
                .map_or(SolveMode::Standard, SolveMode::ZeroKnowledge);
            let config = solve(&spec, &ctx, &source_irs, mode, pow).unwrap();
            prop_assert_eq!(config.num_rounds(), ctx.folding_factor as usize);
        }

        #[test]
        fn zk_error_le_standard_error(
            spec in arb_zk_spec(TEST_TARGET_RANGE),
            ctx in arb_round_ctx(),
        ) {
            let irs = build_source_irs(&spec, &ctx);
            let mo = build_minimal_mask_oracle(&spec);
            let zk = f64::from(analytic_error_bits::<TestEmbedding>(&irs, mo));
            let standard = f64::from(analytic_error_bits::<TestEmbedding>(&irs, None));
            prop_assert!(zk <= standard + EPS, "zk {} > standard {}", zk, standard);
        }

        #[test]
        fn round_pow_closes_gap_to_target(
            spec in prop_oneof![
                arb_standard_spec(TEST_TARGET_RANGE),
                arb_zk_spec(TEST_TARGET_RANGE),
            ],
            ctx in arb_round_ctx(),
        ) {
            let source_irs = build_source_irs(&spec, &ctx);
            let mask_oracle = build_minimal_mask_oracle(&spec);
            let error = analytic_error_bits(&source_irs, mask_oracle);
            let pow = Pow::RoundSumcheck { index: 0 };
            let mode = mask_oracle.map_or(SolveMode::Standard, SolveMode::ZeroKnowledge);
            let config = solve(&spec, &ctx, &source_irs, mode, pow).unwrap();
            assert_pow_closes_gap(&spec, error, &config.round_pow());
        }
    }

    #[test]
    fn solve_works_with_basefield_embedding_zk() {
        let spec = deterministic_spec(Mode::ZeroKnowledge);
        let ctx = fixture_ctx();
        let source_irs: IrsConfig<TestNonIdentityEmbedding> =
            irs_params::solve(&spec, &ctx, OodSampleBudget::ZERO)
                .expect("source IRS fixture must solve");
        let info = MaskOracleInfo {
            c_zk_list_size: ListSize::new(FIXTURE_C_ZK_LIST_SIZE),
            l_zk: MaskCodeMessageLen::new(FIXTURE_L_ZK),
        };
        let config = solve(
            &spec,
            &ctx,
            &source_irs,
            SolveMode::ZeroKnowledge(info),
            Pow::RoundSumcheck { index: 0 },
        )
        .unwrap();
        assert!(matches!(
            config.mode(),
            sumcheck::SumcheckMode::ZeroKnowledge { .. }
        ));
    }
}
