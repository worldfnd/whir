//! Per-round build: turns a [`RoundShape`] into a [`RoundConfig`].
//!
//! Solves the `t_ood` fix-point, builds source/target IRS configs, and
//! (in ZK) assembles the per-round mask oracle. Consumed by
//! [`super::derive`], which drives the per-round loop.

use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        fields::FieldWithSize,
    },
    protocols::{
        irs_commit::Config as IrsConfig,
        mask_proximity::Config as MaskProximityConfig,
        params::{
            bounds::usize_to_f64,
            branch::{Branch, OodMode, RoundBuildMode, RoundBuildPayload, SolveMode},
            code_switch as code_switch_params,
            error::{DeriveError, Pow},
            irs_commit as irs_params,
            layout::{round_context, target_context, RoundShape},
            mask_proximity as mask_proximity_params,
            protocol_config::{MaskOracleConfig, RoundConfig, RoundMode},
            spec::{
                DecodingRegime, LogInvRate, MaskCodeMessageLen, OodSampleBudget, RoundContext,
                SecuritySpec, ZkSpec,
            },
            sumcheck as sumcheck_params,
        },
    },
};

const T_OOD_MAX_ITER: usize = 32;

pub(super) fn build_round_config<M: Embedding + Default>(
    spec: &SecuritySpec,
    shape: &RoundShape,
    mode: RoundBuildMode<'_>,
) -> Result<RoundConfig<M>, DeriveError> {
    let ctx = round_context(shape);
    let ood_mode = mode.map(|p| p.c_zk_log_inv_rate);
    let (source, t_ood) = solve_round_source::<M>(spec, shape, ood_mode)?;

    let (target_budget, solve_mode, round_mode) = match mode {
        Branch::Standard => (
            OodSampleBudget::ZERO,
            SolveMode::Standard,
            RoundMode::Standard,
        ),
        Branch::ZeroKnowledge(RoundBuildPayload {
            zk_spec,
            c_zk_log_inv_rate,
        }) => {
            let num_masks =
                sumcheck_params::masks_required(&ctx) + code_switch_params::masks_required();
            let mask_oracle = build_mask_oracle::<M>(
                zk_spec,
                &source,
                t_ood,
                num_masks,
                c_zk_log_inv_rate,
                shape.round_index,
            )?;
            let solve_mode = SolveMode::ZeroKnowledge(mask_oracle.info());
            (
                OodSampleBudget::new(t_ood),
                solve_mode,
                RoundMode::ZeroKnowledge {
                    t_ood: OodSampleBudget::new(t_ood),
                    mask_oracle: Box::new(mask_oracle),
                },
            )
        }
    };

    let target: IrsConfig<Identity<M::Target>> =
        irs_params::solve(spec, &target_context(shape, &source), target_budget)?;
    let sumcheck = sumcheck_params::solve(
        spec,
        &ctx,
        &source,
        solve_mode,
        Pow::RoundSumcheck {
            index: shape.round_index,
        },
    )?;
    let code_switch =
        code_switch_params::solve(spec, source, target, t_ood, solve_mode, shape.round_index)?;

    Ok(RoundConfig::new(
        shape.round_index,
        sumcheck,
        code_switch,
        round_mode,
    ))
}

fn solve_round_source<M: Embedding + Default>(
    spec: &SecuritySpec,
    shape: &RoundShape,
    ood_mode: OodMode,
) -> Result<(IrsConfig<M>, usize), DeriveError> {
    let src_ctx = round_context(shape);
    let target_log_inv_rate = f64::from(
        shape
            .source_log_inv_rate
            .saturating_add(shape.source_folding_factor.saturating_sub(1)),
    );
    let target_log_degree = f64::from(
        shape
            .source_vector_size
            .trailing_zeros()
            .saturating_sub(shape.source_folding_factor),
    );
    let target_list_size = spec
        .decoding_regime
        .list_size_estimate(target_log_degree, target_log_inv_rate);
    solve_t_ood::<M>(
        spec,
        &src_ctx,
        target_list_size,
        ood_mode,
        shape.round_index,
    )
}

/// ZK-only: assemble the per-round mask oracle (C_zk codeword + mask-proximity
/// check).
fn build_mask_oracle<M: Embedding>(
    zk_spec: ZkSpec<'_>,
    source: &IrsConfig<M>,
    t_ood: usize,
    num_masks: usize,
    c_zk_log_inv_rate: LogInvRate,
    round_index: usize,
) -> Result<MaskOracleConfig<M::Target>, DeriveError> {
    let spec = zk_spec.as_inner();
    let l_zk = compute_l_zk(source, t_ood);
    let c_zk: IrsConfig<Identity<M::Target>> = irs_params::solve_mask_code(
        zk_spec,
        l_zk,
        source.mask_length(),
        c_zk_log_inv_rate,
        MaskProximityConfig::<M::Target>::num_vectors_for(num_masks),
    )?;
    let c_zk_list_size_estimate = spec.decoding_regime.list_size_estimate(
        (l_zk.get() as f64).log2(),
        f64::from(c_zk_log_inv_rate.get()),
    );
    debug_assert!(
        (c_zk.list_size() - c_zk_list_size_estimate).abs()
            < 1e-9 * c_zk_list_size_estimate.max(1.0),
        "c_zk.list_size() {} drifted from planner estimate {}",
        c_zk.list_size(),
        c_zk_list_size_estimate,
    );
    let mask_proximity = mask_proximity_params::solve(spec, c_zk.clone(), num_masks, round_index)?;
    Ok(MaskOracleConfig::new(c_zk, l_zk, mask_proximity))
}

/// `ℓ_zk = next_pow2(r + t_ood)` (Theorem 9.6 + Lemma 9.3).
pub(super) const fn compute_l_zk<M: Embedding>(
    source: &IrsConfig<M>,
    t_ood: usize,
) -> MaskCodeMessageLen {
    MaskCodeMessageLen::new(
        source
            .mask_length()
            .saturating_add(t_ood)
            .next_power_of_two(),
    )
}

/// Per-round `(source, t_ood)`.
///
/// Under `Unique`, `t_ood = 1` is pinned (the `log(L·(L−1)/2)` term degenerates
/// when `L = 1`, and Construction 9.7 requires `out_domain_samples ≥ 1`).
/// Otherwise linear search over `t_ood = 1..=T_OOD_MAX_ITER` for the smallest
/// value where [`ood_security_bits_at`] meets `protocol_security_target_bits`.
pub(super) fn solve_t_ood<M: Embedding + Default>(
    spec: &SecuritySpec,
    src_ctx: &RoundContext,
    target_list_size: f64,
    ood_mode: OodMode,
    round_index: usize,
) -> Result<(IrsConfig<M>, usize), DeriveError> {
    if matches!(spec.decoding_regime, DecodingRegime::Unique) {
        let source = irs_params::solve(spec, src_ctx, OodSampleBudget::new(1))?;
        return Ok((source, 1));
    }

    let security_target = f64::from(spec.protocol_security_target_bits());
    let field_bits = M::Target::field_size_bits();

    for t_ood in 1..=T_OOD_MAX_ITER {
        let source: IrsConfig<M> = irs_params::solve(spec, src_ctx, OodSampleBudget::new(t_ood))?;
        let bits =
            ood_security_bits_at(spec, &source, t_ood, target_list_size, ood_mode, field_bits);
        if bits >= security_target {
            return Ok((source, t_ood));
        }
    }
    Err(DeriveError::FixedPointDidNotConverge { round_index })
}

/// OOD security bits at candidate `t_ood`, per STIR Lemma 4.5:
/// `bits = t · (|F| − log d) − log(L · (L − 1) / 2) ≈ t·(|F| − log d) − 2·log L + 1`.
fn ood_security_bits_at<M: Embedding>(
    spec: &SecuritySpec,
    source: &IrsConfig<M>,
    t_ood: usize,
    target_list_size: f64,
    ood_mode: OodMode,
    field_bits: f64,
) -> f64 {
    let (log_degree, log_combined_list) = match ood_mode {
        Branch::Standard => (
            usize_to_f64(source.message_length()).log2(),
            target_list_size.log2(),
        ),
        Branch::ZeroKnowledge(c_zk_log_inv_rate) => {
            let l_zk = source
                .mask_length()
                .saturating_add(t_ood)
                .next_power_of_two();
            let c_zk_list = spec.decoding_regime.list_size_estimate(
                usize_to_f64(l_zk).log2(),
                f64::from(c_zk_log_inv_rate.get()),
            );
            (
                usize_to_f64(source.message_length().saturating_add(l_zk)).log2(),
                (target_list_size * c_zk_list).log2(),
            )
        }
    };
    let ood = usize_to_f64(t_ood);
    ood * (field_bits - log_degree) - 2.0 * log_combined_list + 1.0
}
