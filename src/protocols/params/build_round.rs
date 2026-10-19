//! Build one folding round from its chosen dimensions.

use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        fields::FieldWithSize,
    },
    protocols::{
        irs_commit::Config as IrsConfig,
        params::{
            bounds::usize_to_f64,
            branch::{RoundBuildMode, SolveMode},
            code_switch::solve as solve_code_switch,
            error::{DeriveError, Pow},
            irs_commit::solve as solve_irs,
            layout::{round_context, target_context, RoundShape},
            protocol_config::{RoundConfig, RoundMasks, RoundMode, WitnessMaskConfig},
            spec::{DecodingRegime, SecuritySpec},
            sumcheck::{mask_coefficients, solve as solve_sumcheck},
        },
    },
};

const T_OOD_MAX_ITER: usize = 32;

pub(super) fn build_round_config<M: Embedding + Default>(
    spec: &SecuritySpec,
    shape: &RoundShape,
    mode: RoundBuildMode,
) -> Result<RoundConfig<M>, DeriveError> {
    let ctx = round_context(shape);
    let source = solve_irs::<M>(spec, &ctx)?;
    let target = solve_irs::<Identity<M::Target>>(spec, &target_context(shape, &source))?;
    let (t_ood, round_mode) = match mode {
        RoundBuildMode::Standard => (
            solve_t_ood(spec, &source, &target, shape.round_index)?,
            RoundMode::Standard,
        ),
        RoundBuildMode::ZeroKnowledge(()) => {
            let witness =
                WitnessMaskConfig::for_code(&source, usize::from(shape.round_index != 0))?;
            WitnessMaskConfig::for_code(&target, 1)?;
            (
                1,
                RoundMode::ZeroKnowledge(RoundMasks {
                    witness,
                    sumcheck_coefficients: mask_coefficients(ctx.folding_factor as usize)?,
                }),
            )
        }
    };
    let solve_mode: SolveMode = mode;
    let sumcheck = solve_sumcheck(
        spec,
        &ctx,
        &source,
        solve_mode,
        Pow::RoundSumcheck {
            index: shape.round_index,
        },
    )?;
    let code_switch =
        solve_code_switch(spec, source, target, t_ood, solve_mode, shape.round_index)?;
    Ok(RoundConfig::new(
        shape.round_index,
        sumcheck,
        code_switch,
        round_mode,
    ))
}

/// Standard WHIR can repeat DEEP samples to reach its analytic target.
pub(super) fn solve_t_ood<M: Embedding>(
    spec: &SecuritySpec,
    source: &IrsConfig<M>,
    target: &IrsConfig<Identity<M::Target>>,
    round_index: usize,
) -> Result<usize, DeriveError> {
    if matches!(spec.decoding_regime, DecodingRegime::Unique) {
        return Ok(1);
    }
    let security_target = f64::from(spec.protocol_security_target_bits());
    let per_sample = M::Target::field_size_bits() - usize_to_f64(source.message_length()).log2();
    let list_bits = target.list_size().log2();
    (1..=T_OOD_MAX_ITER)
        .find(|&t| usize_to_f64(t) * per_sample - 2.0 * list_bits + 1.0 >= security_target)
        .ok_or(DeriveError::FixedPointDidNotConverge { round_index })
}
