//! Derive commitment dimensions, rounds, and masks before proving.

use std::collections::VecDeque;

use ark_ff::Field;
use thiserror::Error;

use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        fields::FieldWithSize,
    },
    protocols::{
        irs_commit::{Config as IrsConfig, IrsMode, IrsParams},
        params::{
            config::{
                embedding_degree, BasecasePlan, Node, ProtocolConfig, ProtocolSpec, RoundConfig,
            },
            error::{DeriveError, SecurityEvent, ZookSecurityError},
            masks::{solve as solve_joint_masks, WitnessMaskConfig},
            search::plan_adaptive_rates,
            security::{
                code_switch_error_bits, grind_to_at, rate, revealed_sumcheck_error_bits,
                sumcheck_error_bits, usize_to_f64, validate_single_polynomial,
            },
            spec::{DecodingRegime, Mode, RateSchedule, RoundContext, SecuritySpec, TuningSpec},
        },
    },
};

impl<M: Embedding + Default> ProtocolConfig<M> {
    /// Derive a canonical commitment sequence before sampling prover randomness.
    pub fn derive(spec: SecuritySpec, tuning: TuningSpec) -> Result<Self, DeriveError> {
        if spec.mode.is_zk() && !matches!(spec.decoding_regime, DecodingRegime::Johnson) {
            return Err(ZookSecurityError::UnsupportedRegime.into());
        }
        let layout = round_layout::<M>(&spec, &tuning)?;
        let has_rounds = layout.len() > 1;
        let mut codes = layout
            .iter()
            .skip(usize::from(has_rounds))
            .map(|context| solve_irs::<Identity<M::Target>>(&spec, context))
            .collect::<Result<VecDeque<_>, _>>()?;
        let first = if has_rounds {
            Some(solve_node(
                &spec,
                solve_irs::<M>(&spec, &layout[0])?,
                codes.front(),
                0,
            )?)
        } else {
            None
        };
        let mut targets = Vec::with_capacity(codes.len());
        while let Some(code) = codes.pop_front() {
            let index = targets.len() + usize::from(first.is_some());
            targets.push(solve_node(&spec, code, codes.front(), index)?);
        }
        let mut config = Self {
            request: ProtocolSpec {
                embedding: crate::type_info::Typed::new(M::default()),
                security: spec,
                tuning,
            },
            first,
            targets,
            joint_masks: None,
        };
        if config.security().mode.is_zk() {
            let sumchecks = config
                .first_round()
                .into_iter()
                .map(|r| r.masks().unwrap().sumcheck_coefficients)
                .chain(
                    config
                        .tail_rounds()
                        .map(|r| r.masks().unwrap().sumcheck_coefficients),
                )
                .collect::<Vec<_>>();
            let witnesses = (0..=config.num_rounds())
                .map(|i| config.witness_mask(i).unwrap().num_coefficients())
                .collect::<Result<Vec<_>, _>>()?;
            let degree = if config.has_rounds() {
                embedding_degree::<M>()?
            } else {
                1
            };
            config.joint_masks = Some(solve_joint_masks(
                config.security(),
                &sumchecks,
                &witnesses,
                config.final_commit(),
                degree,
            )?);
        }
        config.validate()?;
        Ok(config)
    }
}

pub(super) fn solve_node<M: Embedding>(
    spec: &SecuritySpec,
    code: IrsConfig<M>,
    target: Option<&IrsConfig<Identity<M::Target>>>,
    index: usize,
) -> Result<Node<M>, DeriveError> {
    if spec.mode.is_zk() {
        WitnessMaskConfig::for_code(&code, usize::from(index != 0))?;
    }
    let (analytic, event, switch) = if let Some(target) = target {
        let ood = if spec.mode.is_zk() {
            WitnessMaskConfig::for_code(target, 1)?;
            1
        } else {
            solve_t_ood(spec, &code, target, index)?
        };
        let switch_pow = grind_to_at(
            spec,
            code_switch_error_bits(&code, target, ood, spec.mode.is_zk())?,
            SecurityEvent::RoundCodeSwitch { index },
        )?;
        (
            sumcheck_error_bits(&code, spec.mode.is_zk())?,
            SecurityEvent::RoundSumcheck { index },
            Some((ood, switch_pow)),
        )
    } else {
        let analytic = if spec.mode.is_zk() {
            validate_single_polynomial(&code)?;
            revealed_sumcheck_error_bits::<M::Target>()
        } else {
            sumcheck_error_bits(&code, false)?
        };
        (analytic, SecurityEvent::BasecaseSumcheck, None)
    };
    Ok(Node {
        code,
        sumcheck_pow: grind_to_at(spec, analytic, event)?,
        switch,
    })
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

fn round_layout<M: Embedding + Default>(
    spec: &SecuritySpec,
    tuning: &TuningSpec,
) -> Result<Vec<RoundContext>, DeriveError> {
    let padded_vector_size = initial_vector_size::<M>(spec, tuning)?;
    let mut num_vars = padded_vector_size.trailing_zeros() as usize;
    let mut log_inv_rate = tuning.starting_log_inv_rate;
    let mut layout = Vec::new();

    loop {
        let round = layout.len().saturating_sub(1);
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
        if layout.is_empty() {
            layout.push(RoundContext {
                vector_size: 1usize << num_vars,
                log_inv_rate,
                folding_factor: source_folding_factor,
            });
        }
        layout.push(RoundContext {
            vector_size: 1usize << remaining,
            log_inv_rate: target_log_inv_rate,
            folding_factor: target_folding_factor,
        });
        num_vars = remaining;
        log_inv_rate = target_log_inv_rate;
        if spec.mode.is_zk() && target_folding_factor == 0 {
            break;
        }
    }
    if layout.is_empty() {
        layout.push(RoundContext {
            vector_size: 1usize << num_vars,
            log_inv_rate,
            folding_factor: 0,
        });
    }
    layout.last_mut().unwrap().folding_factor = 0;
    if let RateSchedule::Adaptive { knee_weight } = tuning.rate_schedule {
        if layout.len() > 1 {
            let rates = plan_adaptive_rates::<M>(spec, tuning, knee_weight, &layout)?;
            for (context, rate) in layout.iter_mut().skip(1).zip(rates) {
                context.log_inv_rate = rate;
            }
        }
    }
    Ok(layout)
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

pub(super) fn with_round_config<M: Embedding + Default, R>(
    spec: &SecuritySpec,
    source: &RoundContext,
    target: &RoundContext,
    index: usize,
    f: impl FnOnce(RoundConfig<'_, M>) -> R,
) -> Result<R, DeriveError> {
    let source = solve_irs::<M>(spec, source)?;
    let target = solve_irs(spec, target)?;
    let node = solve_node(spec, source, Some(&target), index)?;
    Ok(f(node.round(&target, index, spec.mode.is_zk())))
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

pub(super) fn with_basecase<F: Field, R>(
    spec: &SecuritySpec,
    vector_size: usize,
    log_inv_rate: u32,
    incoming_deep: usize,
    f: impl FnOnce(BasecasePlan<'_, F>) -> R,
) -> Result<R, DeriveError> {
    let code = solve_irs(
        spec,
        &RoundContext {
            vector_size,
            log_inv_rate,
            folding_factor: 0,
        },
    )?;
    let node = solve_node(spec, code, None, incoming_deep)?;
    Ok(f(node.basecase(spec.mode.is_zk(), incoming_deep)))
}

#[cfg(test)]
#[path = "tests/derive.rs"]
mod tests;
