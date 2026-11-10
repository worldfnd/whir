//! Derive commitment dimensions, rounds, and masks before proving.

use ark_ff::Field;
use thiserror::Error;

use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        fields::FieldWithSize,
    },
    protocols::{
        basecase::{BasecaseMode, Config as BasecaseConfig},
        code_switch::{CodeSwitchMode, Config as CodeSwitchConfig},
        irs_commit::{Config as IrsConfig, IrsMode, IrsParams},
        params::{
            config::{embedding_degree, BasecasePlan, ProtocolConfig, RoundConfig},
            error::{DeriveError, SecurityEvent, ZookSecurityError},
            masks::{mask_coefficients, solve as solve_joint_masks, RoundMasks, WitnessMaskConfig},
            search::plan_adaptive_rates,
            security::{
                code_switch_error_bits, grind_to_at, rate, revealed_sumcheck_error_bits,
                sumcheck_error_bits, usize_to_f64, validate_single_polynomial,
            },
            spec::{DecodingRegime, Mode, RateSchedule, RoundContext, SecuritySpec, TuningSpec},
        },
        proof_of_work::Config as PowConfig,
        sumcheck::{Config as SumcheckConfig, SumcheckMaskLen, SumcheckMode},
    },
};

impl<M: Embedding + Default> ProtocolConfig<M> {
    /// Build and validate the commitment and mask configuration.
    ///
    /// Zook derives IOP parameters without grinding credit. Its serialized
    /// configuration includes mask ranges and padded witness dimensions.
    pub fn derive(spec: SecuritySpec, tuning: TuningSpec) -> Result<Self, DeriveError> {
        if spec.mode.is_zk() && !matches!(spec.decoding_regime, DecodingRegime::Johnson) {
            return Err(ZookSecurityError::UnsupportedRegime.into());
        }
        let RoundLayout {
            padded_vector_size,
            shapes,
            basecase_vector_size,
            basecase_log_inv_rate,
        } = round_layout::<M>(&spec, &tuning)?;
        let mut shapes = shapes.iter();
        let first_round: Option<RoundConfig<M>> = shapes
            .next()
            .map(|shape| build_round_config(&spec, shape))
            .transpose()?;
        let tail_rounds: Vec<RoundConfig<Identity<M::Target>>> = shapes
            .map(|shape| build_round_config(&spec, shape))
            .collect::<Result<_, _>>()?;
        let last_target = tail_rounds
            .last()
            .map(|round| round.code_switch().target())
            .or_else(|| {
                first_round
                    .as_ref()
                    .map(|round| round.code_switch().target())
            });
        let basecase = match last_target {
            Some(target) => solve_with_commit(&spec, target.clone(), 1)?,
            None => solve_basecase(&spec, basecase_vector_size, basecase_log_inv_rate, 0)?,
        };
        let joint_masks = if spec.mode.is_zk() {
            let masks = first_round
                .iter()
                .filter_map(RoundConfig::masks)
                .chain(tail_rounds.iter().filter_map(RoundConfig::masks));
            let mut sumcheck_lengths = Vec::new();
            let mut witness_lengths = Vec::new();
            for mask in masks {
                sumcheck_lengths.push(mask.sumcheck_coefficients);
                witness_lengths.push(mask.witness.num_coefficients()?);
            }
            witness_lengths.push(
                basecase
                    .witness_mask()
                    .ok_or(DeriveError::InvalidMaskLayout)?
                    .num_coefficients()?,
            );
            // A basecase-only witness is lifted before committing, so its
            // mask and blinder live entirely in the target field.
            let degree = if first_round.is_some() {
                embedding_degree::<M>()?
            } else {
                1
            };
            Some(solve_joint_masks(
                &spec,
                &sumcheck_lengths,
                &witness_lengths,
                basecase.commit(),
                degree,
            )?)
        } else {
            None
        };
        let config = Self::new(
            spec,
            tuning,
            padded_vector_size,
            first_round,
            tail_rounds,
            basecase,
            joint_masks,
        );
        config.validate()?;
        Ok(config)
    }
}

/// Reasons a [`TuningSpec`] cannot produce a valid round layout.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum LayoutError {
    /// `tuning.vector_size` must be a power of 2.
    #[error("tuning.vector_size ({vector_size}) must be a power of 2; pad the vector")]
    VectorSizeNotPowerOfTwo { vector_size: usize },

    /// `tuning.folding_factor` must yield at least 1 at every round.
    #[error("tuning.folding_factor min ({min}) must be ≥ 1")]
    FoldingFactorBelowOne { min: usize },

    /// `tuning.starting_log_inv_rate` must be ≥ 1 (i.e. rate < 1). At rate == 1
    /// the code has zero decoding distance, so query soundness is undefined and
    /// the in-domain query count would diverge.
    #[error("tuning.starting_log_inv_rate must be ≥ 1 (rate < 1); got 0")]
    StartingRateBelowOne,

    #[error("rate cap must be at least one")]
    RateCapBelowOne,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct RoundShape {
    pub(super) round_index: usize,
    pub(super) source_vector_size: usize,
    pub(super) source_log_inv_rate: u32,
    pub(super) source_folding_factor: u32,
    pub(super) target_folding_factor: u32,
    /// Per-round inverse rate for this round's `code_switch.target` IRS.
    /// Equal to `RateSchedule::step(source_log_inv_rate, source_folding_factor)`
    /// for `Stepping` / `Capped`; overwritten by [`plan_adaptive_rates`] for
    /// `Adaptive`.
    pub(super) target_log_inv_rate: u32,
}

#[derive(Debug)]
struct RoundLayout {
    padded_vector_size: usize,
    shapes: Vec<RoundShape>,
    basecase_vector_size: usize,
    basecase_log_inv_rate: u32,
}

fn round_layout<M: Embedding + Default>(
    spec: &SecuritySpec,
    tuning: &TuningSpec,
) -> Result<RoundLayout, DeriveError> {
    let padded_vector_size = initial_vector_size::<M>(spec, tuning)?;
    let mut num_vars = padded_vector_size.trailing_zeros() as usize;
    let mut log_inv_rate = tuning.starting_log_inv_rate;
    let mut shapes = Vec::new();

    loop {
        let round = shapes.len();
        let source_folding = tuning.folding_factor.at_round(round);
        let next_folding = tuning.folding_factor.at_round(round + 1);
        let Some(remaining) = num_vars.checked_sub(source_folding) else {
            break;
        };
        if !spec.mode.is_zk() && remaining < next_folding {
            break;
        }
        let source_folding_factor =
            u32::try_from(source_folding).map_err(|_| DeriveError::SizeOverflow)?;
        let target_log_inv_rate = tuning
            .rate_schedule
            .step(log_inv_rate, source_folding_factor);
        let mut target_folding_factor =
            u32::try_from(next_folding).map_err(|_| DeriveError::SizeOverflow)?;

        if spec.mode.is_zk() {
            let source_fits = if round == 0 {
                mask_fits::<M>(
                    spec,
                    1usize << num_vars,
                    log_inv_rate,
                    source_folding_factor,
                    0,
                )?
            } else {
                mask_fits::<Identity<M::Target>>(
                    spec,
                    1usize << num_vars,
                    log_inv_rate,
                    source_folding_factor,
                    1,
                )?
            };
            if !source_fits
                || !mask_fits::<Identity<M::Target>>(
                    spec,
                    1usize << remaining,
                    target_log_inv_rate,
                    0,
                    1,
                )?
            {
                break;
            }
            if remaining < next_folding
                || !mask_fits::<Identity<M::Target>>(
                    spec,
                    1usize << remaining,
                    target_log_inv_rate,
                    target_folding_factor,
                    1,
                )?
            {
                target_folding_factor = 0;
            }
        }
        shapes.push(RoundShape {
            round_index: round,
            source_vector_size: 1usize << num_vars,
            source_log_inv_rate: log_inv_rate,
            source_folding_factor,
            target_folding_factor,
            target_log_inv_rate,
        });
        num_vars = remaining;
        log_inv_rate = target_log_inv_rate;
        if spec.mode.is_zk() && target_folding_factor == 0 {
            break;
        }
    }
    if let Some(last) = shapes.last_mut() {
        last.target_folding_factor = 0;
    }
    let basecase_vector_size = 1usize << num_vars;
    let basecase_log_inv_rate = select_rates::<M>(
        spec,
        tuning,
        &mut shapes,
        basecase_vector_size,
        log_inv_rate,
    )?;
    Ok(RoundLayout {
        padded_vector_size,
        shapes,
        basecase_vector_size,
        basecase_log_inv_rate,
    })
}

fn select_rates<M: Embedding + Default>(
    spec: &SecuritySpec,
    tuning: &TuningSpec,
    shapes: &mut [RoundShape],
    basecase_vector_size: usize,
    log_inv_rate: u32,
) -> Result<u32, DeriveError> {
    let rate = match tuning.rate_schedule {
        RateSchedule::Adaptive { knee_weight } if !shapes.is_empty() => {
            let rates =
                plan_adaptive_rates::<M>(spec, tuning, knee_weight, shapes, basecase_vector_size)?;
            for (shape, &rate) in shapes.iter_mut().zip(&rates) {
                shape.target_log_inv_rate = rate;
            }
            for i in 1..shapes.len() {
                shapes[i].source_log_inv_rate = shapes[i - 1].target_log_inv_rate;
            }
            rates
                .last()
                .copied()
                .ok_or(DeriveError::AdaptiveNoFeasibleSchedule)?
        }
        _ => log_inv_rate,
    };
    Ok(rate)
}

fn initial_vector_size<M: Embedding + Default>(
    spec: &SecuritySpec,
    tuning: &TuningSpec,
) -> Result<usize, DeriveError> {
    if !tuning.vector_size.is_power_of_two() {
        return Err(LayoutError::VectorSizeNotPowerOfTwo {
            vector_size: tuning.vector_size,
        }
        .into());
    }
    let min = tuning.folding_factor.min();
    if min < 1 {
        return Err(LayoutError::FoldingFactorBelowOne { min }.into());
    }
    if tuning.starting_log_inv_rate < 1 {
        return Err(LayoutError::StartingRateBelowOne.into());
    }

    if matches!(
        tuning.rate_schedule,
        RateSchedule::Capped {
            max_log_inv_rate: 0
        }
    ) {
        return Err(LayoutError::RateCapBelowOne.into());
    }

    let mut padded_vector_size = tuning.vector_size;
    if spec.mode.is_zk() {
        while !mask_fits::<Identity<M::Target>>(
            spec,
            padded_vector_size,
            tuning.starting_log_inv_rate,
            0,
            0,
        )? {
            padded_vector_size = padded_vector_size
                .checked_mul(2)
                .ok_or(DeriveError::SizeOverflow)?;
        }
    }
    Ok(padded_vector_size)
}

fn mask_fits<M: Embedding + Default>(
    spec: &SecuritySpec,
    vector_size: usize,
    log_inv_rate: u32,
    folding_factor: u32,
    incoming_deep: usize,
) -> Result<bool, DeriveError> {
    let code = solve_irs::<M>(
        spec,
        &RoundContext {
            vector_size,
            log_inv_rate,
            folding_factor,
        },
    )?;
    match WitnessMaskConfig::for_code(&code, incoming_deep) {
        Ok(_) => Ok(true),
        Err(DeriveError::MaskDoesNotFit { .. }) => Ok(false),
        Err(error) => Err(error),
    }
}

const T_OOD_MAX_ITER: usize = 32;

pub(super) fn build_round_config<M: Embedding + Default>(
    spec: &SecuritySpec,
    shape: &RoundShape,
) -> Result<RoundConfig<M>, DeriveError> {
    let ctx = RoundContext {
        vector_size: shape.source_vector_size,
        log_inv_rate: shape.source_log_inv_rate,
        folding_factor: shape.source_folding_factor,
    };
    let source = solve_irs::<M>(spec, &ctx)?;
    let target = solve_irs::<Identity<M::Target>>(
        spec,
        &RoundContext {
            vector_size: source.message_length(),
            log_inv_rate: shape.target_log_inv_rate,
            folding_factor: shape.target_folding_factor,
        },
    )?;
    let (t_ood, masks) = match spec.mode {
        Mode::Standard => (
            solve_t_ood(spec, &source, &target, shape.round_index)?,
            None,
        ),
        Mode::ZeroKnowledge => {
            let witness =
                WitnessMaskConfig::for_code(&source, usize::from(shape.round_index != 0))?;
            WitnessMaskConfig::for_code(&target, 1)?;
            (
                1,
                Some(RoundMasks {
                    witness,
                    sumcheck_coefficients: mask_coefficients(ctx.folding_factor as usize)?,
                }),
            )
        }
    };
    let sumcheck = solve_sumcheck(
        spec,
        &ctx,
        &source,
        SecurityEvent::RoundSumcheck {
            index: shape.round_index,
        },
    )?;
    let code_switch = solve_code_switch(spec, source, target, t_ood, shape.round_index)?;
    Ok(RoundConfig::new(
        shape.round_index,
        sumcheck,
        code_switch,
        masks,
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

/// Encode the supplied vector without appending randomness.
pub(super) fn solve_irs<M: Embedding + Default>(
    spec: &SecuritySpec,
    ctx: &RoundContext,
) -> Result<IrsConfig<M>, DeriveError> {
    solve_with_num_vectors(spec, ctx, 1)
}

/// Encode equal-length vectors in one commitment with shared query paths.
pub(super) fn solve_with_num_vectors<M: Embedding + Default>(
    spec: &SecuritySpec,
    ctx: &RoundContext,
    num_vectors: usize,
) -> Result<IrsConfig<M>, DeriveError> {
    let depth = 1_usize
        .checked_shl(ctx.folding_factor)
        .ok_or(DeriveError::SizeOverflow)?;
    if ctx.vector_size == 0 || !ctx.vector_size.is_multiple_of(depth) {
        return Err(DeriveError::InvalidMaskLayout);
    }
    let rate = rate(f64::from(ctx.log_inv_rate));
    if rate == 0.0 {
        return Err(DeriveError::SizeOverflow);
    }
    // Zook query steps sum two error terms; reserve half the error budget
    // for the in-domain term. No grinding credit is used for these IOP bounds.
    let security_target = match spec.mode {
        Mode::Standard => f64::from(spec.protocol_security_target_bits()),
        Mode::ZeroKnowledge => f64::from(spec.target_security_bits) + 1.0,
    };
    Ok(IrsConfig::try_new(IrsParams {
        security_target,
        decoding_regime: spec.decoding_regime,
        hash_id: spec.hash_id,
        num_vectors,
        vector_size: ctx.vector_size,
        interleaving_depth: depth,
        rate,
        mode: IrsMode::Standard,
    })?)
}

pub(super) fn solve_sumcheck<M: Embedding>(
    spec: &SecuritySpec,
    ctx: &RoundContext,
    source: &IrsConfig<M>,
    pow: SecurityEvent,
) -> Result<SumcheckConfig<M::Target>, DeriveError> {
    let analytic = sumcheck_error_bits(source, spec.mode.is_zk())?;
    let round_pow = grind_to_at(spec, analytic, pow)?;
    let output_mode = match spec.mode {
        Mode::Standard => SumcheckMode::Standard,
        Mode::ZeroKnowledge => SumcheckMode::ZeroKnowledge {
            mask_length: SumcheckMaskLen::new(3),
        },
    };
    Ok(SumcheckConfig::new(
        ctx.vector_size,
        round_pow,
        ctx.folding_factor as usize,
        output_mode,
    ))
}

pub(super) fn solve_code_switch<M: Embedding>(
    spec: &SecuritySpec,
    source: IrsConfig<M>,
    target: IrsConfig<Identity<M::Target>>,
    t_ood: usize,
    round_index: usize,
) -> Result<CodeSwitchConfig<M>, DeriveError> {
    let analytic = code_switch_error_bits(&source, &target, t_ood, spec.mode.is_zk())?;
    let pow = grind_to_at(
        spec,
        analytic,
        SecurityEvent::RoundCodeSwitch { index: round_index },
    )?;
    // Witness overlays and mask claims belong to Zook. The code-switch
    // primitive receives ordinary IRS commitments without appended masks.
    Ok(CodeSwitchConfig::new(
        source,
        target,
        t_ood,
        CodeSwitchMode::Standard,
        pow,
    ))
}

pub(super) fn solve_basecase<F: Field>(
    spec: &SecuritySpec,
    vector_size: usize,
    log_inv_rate: u32,
    incoming_deep: usize,
) -> Result<BasecasePlan<F>, DeriveError> {
    let commit = solve_irs(
        spec,
        &RoundContext {
            vector_size,
            log_inv_rate,
            folding_factor: 0,
        },
    )?;
    solve_with_commit(spec, commit, incoming_deep)
}

pub(super) fn solve_with_commit<F: Field>(
    spec: &SecuritySpec,
    commit: IrsConfig<Identity<F>>,
    incoming_deep: usize,
) -> Result<BasecasePlan<F>, DeriveError> {
    if matches!(spec.mode, Mode::ZeroKnowledge) {
        validate_single_polynomial(&commit)?;
        let witness_mask = WitnessMaskConfig::for_code(&commit, incoming_deep)?;
        let analytic = revealed_sumcheck_error_bits::<F>();
        let pow = grind_to_at(spec, analytic, SecurityEvent::BasecaseSumcheck)?;
        let sumcheck = SumcheckConfig::new(
            commit.vector_size(),
            pow,
            commit.vector_size().trailing_zeros() as usize,
            SumcheckMode::Standard,
        );
        return Ok(BasecasePlan::ZeroKnowledge {
            commit,
            witness_mask,
            sumcheck,
        });
    }
    let analytic = sumcheck_error_bits(&commit, false)?;
    let pow = grind_to_at(spec, analytic, SecurityEvent::BasecaseSumcheck)?;
    let sumcheck = SumcheckConfig::new(
        commit.vector_size(),
        pow,
        commit.vector_size().trailing_zeros() as usize,
        SumcheckMode::Standard,
    );
    Ok(BasecasePlan::Standard {
        config: BasecaseConfig::new(commit, sumcheck, BasecaseMode::Standard, PowConfig::none()),
    })
}

#[cfg(test)]
#[path = "tests/derive.rs"]
mod tests;
