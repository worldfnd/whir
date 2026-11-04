//! Round dimensions chosen before masks are sampled.

use thiserror::Error;

use crate::{
    algebra::embedding::{Embedding, Identity},
    protocols::{
        irs_commit::Config as IrsConfig,
        params::{
            adaptive::plan_adaptive_rates,
            branch::RoundBuildMode,
            error::DeriveError,
            irs_commit::solve as solve_irs,
            protocol_config::WitnessMaskConfig,
            spec::{RateSchedule, RoundContext, SecuritySpec, TuningSpec},
        },
    },
};

/// Reasons a [`TuningSpec`] cannot produce a valid round layout.
///
/// Nested into [`super::error::DeriveError`] via `#[from]`, so
/// [`round_layout`] failures propagate through
/// [`super::derive::ProtocolConfig::derive`] with `?`.
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
pub(super) struct RoundLayout {
    pub(super) padded_vector_size: usize,
    pub(super) shapes: Vec<RoundShape>,
    pub(super) basecase_vector_size: usize,
    pub(super) basecase_log_inv_rate: u32,
}

pub(super) fn round_layout<M: Embedding + Default>(
    spec: &SecuritySpec,
    tuning: &TuningSpec,
    mode: RoundBuildMode,
) -> Result<RoundLayout, DeriveError> {
    let padded_vector_size = initial_vector_size::<M>(spec, tuning, mode)?;
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
        if !mode.is_zk() && remaining < next_folding {
            break;
        }
        let source_folding_factor =
            u32::try_from(source_folding).map_err(|_| DeriveError::SizeOverflow)?;
        let target_log_inv_rate = tuning
            .rate_schedule
            .step(log_inv_rate, source_folding_factor);
        let mut target_folding_factor =
            u32::try_from(next_folding).map_err(|_| DeriveError::SizeOverflow)?;

        if mode.is_zk() {
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
        if mode.is_zk() && target_folding_factor == 0 {
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
        mode,
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
    mode: RoundBuildMode,
    log_inv_rate: u32,
) -> Result<u32, DeriveError> {
    let rate = match tuning.rate_schedule {
        RateSchedule::Adaptive { knee_weight } if !shapes.is_empty() => {
            let rates = plan_adaptive_rates::<M>(
                spec,
                tuning,
                knee_weight,
                shapes,
                basecase_vector_size,
                mode,
            )?;
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
    mode: RoundBuildMode,
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

    let mut padded_vector_size = tuning.vector_size;
    if mode.is_zk() {
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

pub(super) const fn round_context(shape: &RoundShape) -> RoundContext {
    RoundContext {
        vector_size: shape.source_vector_size,
        log_inv_rate: shape.source_log_inv_rate,
        folding_factor: shape.source_folding_factor,
    }
}

pub(super) fn target_context<M: Embedding>(
    shape: &RoundShape,
    source: &IrsConfig<M>,
) -> RoundContext {
    RoundContext {
        vector_size: source.message_length(),
        log_inv_rate: shape.target_log_inv_rate,
        folding_factor: shape.target_folding_factor,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::params::{
        branch::Branch,
        spec::{FoldingFactor, Mode, PowBudget, RateSchedule, SecuritySpec},
        test_utils::TestEmbedding,
    };

    const FIXTURE_FOLDING_FACTOR: usize = 2;
    const FIXTURE_LOG_INV_RATE: u32 = 1;

    /// Spec fixture suitable for Standard-mode layout tests — RateSchedule
    /// dispatch is exercised here, not security solving, so any feasible
    /// spec works.
    fn layout_spec() -> SecuritySpec {
        SecuritySpec::new(40)
            .with_mode(Mode::Standard)
            .with_pow_budget(PowBudget::per_slot(60))
    }

    /// Convenience wrapper so tests don't repeat the generic + spec + mode
    /// boilerplate. Standard mode is enough to exercise the layout walk;
    /// adaptive coverage lives in `derive` tests where the full pipeline
    /// runs end-to-end.
    fn layout(tuning: &TuningSpec) -> Result<RoundLayout, DeriveError> {
        round_layout::<TestEmbedding>(&layout_spec(), tuning, Branch::Standard)
    }

    const LOG_VECTOR_SIZE_NO_ROUNDS: u32 = 3;
    const LOG_VECTOR_SIZE_MULTI_ROUND: u32 = 8;

    const VARIED_INITIAL_FOLDING: usize = 3;
    const VARIED_STEADY_FOLDING: usize = 2;

    const RATE_STEPPING_STARTING_LOG_INV_RATE: u32 = 2;
    const MIN_ROUNDS_FOR_CHAINING_TEST: usize = 2;

    fn tuning_with(vector_size: usize) -> TuningSpec {
        TuningSpec {
            vector_size,
            starting_log_inv_rate: FIXTURE_LOG_INV_RATE,
            folding_factor: FoldingFactor::Constant(FIXTURE_FOLDING_FACTOR),
            rate_schedule: RateSchedule::Stepping,
        }
    }

    #[test]
    fn round_layout_rate_steps_up_by_folding_minus_one() {
        let tuning = TuningSpec {
            vector_size: 1 << LOG_VECTOR_SIZE_MULTI_ROUND,
            starting_log_inv_rate: RATE_STEPPING_STARTING_LOG_INV_RATE,
            folding_factor: FoldingFactor::ConstantFromSecondRound {
                initial: VARIED_INITIAL_FOLDING,
                rest: VARIED_STEADY_FOLDING,
            },
            rate_schedule: RateSchedule::Stepping,
        };
        let layout = layout(&tuning).unwrap();

        let mut expected_log_inv_rate = RATE_STEPPING_STARTING_LOG_INV_RATE;
        for shape in &layout.shapes {
            assert_eq!(shape.source_log_inv_rate, expected_log_inv_rate);
            expected_log_inv_rate += shape.source_folding_factor.saturating_sub(1);
        }
        assert_eq!(layout.basecase_log_inv_rate, expected_log_inv_rate);
    }

    #[test]
    fn round_layout_chains_target_to_next_source_folding() {
        let tuning = TuningSpec {
            vector_size: 1 << LOG_VECTOR_SIZE_MULTI_ROUND,
            starting_log_inv_rate: FIXTURE_LOG_INV_RATE,
            folding_factor: FoldingFactor::ConstantFromSecondRound {
                initial: VARIED_INITIAL_FOLDING,
                rest: VARIED_STEADY_FOLDING,
            },
            rate_schedule: RateSchedule::Stepping,
        };
        let layout = layout(&tuning).unwrap();
        assert!(
            layout.shapes.len() >= MIN_ROUNDS_FOR_CHAINING_TEST,
            "need ≥ {MIN_ROUNDS_FOR_CHAINING_TEST} rounds to test chaining",
        );
        for window in layout.shapes.windows(2) {
            assert_eq!(
                window[0].target_folding_factor,
                window[1].source_folding_factor
            );
        }
    }

    #[test]
    fn round_layout_basecase_size_consumes_remaining_num_vars() {
        let tuning = tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND);
        let layout = layout(&tuning).unwrap();
        let consumed: u32 = layout.shapes.iter().map(|s| s.source_folding_factor).sum();
        let initial_num_vars = tuning.vector_size.trailing_zeros();
        let remaining = initial_num_vars - consumed;
        assert_eq!(layout.basecase_vector_size, 1usize << remaining);
    }

    #[test]
    fn round_layout_stops_when_no_room_for_source_plus_target() {
        let vector_size = 1usize << LOG_VECTOR_SIZE_NO_ROUNDS;
        let tuning = tuning_with(vector_size);
        let layout = layout(&tuning).unwrap();
        assert_eq!(layout.shapes.len(), 0);
        assert_eq!(layout.basecase_vector_size, vector_size);
        assert_eq!(layout.basecase_log_inv_rate, FIXTURE_LOG_INV_RATE);
    }

    #[test]
    fn round_layout_rejects_non_pow2_vector_size() {
        let tuning = TuningSpec {
            vector_size: 12,
            starting_log_inv_rate: FIXTURE_LOG_INV_RATE,
            folding_factor: FoldingFactor::Constant(FIXTURE_FOLDING_FACTOR),
            rate_schedule: RateSchedule::Stepping,
        };
        let err = layout(&tuning).expect_err("non-pow2 vector_size must fail");
        assert!(
            matches!(
                err,
                DeriveError::Layout(LayoutError::VectorSizeNotPowerOfTwo { vector_size: 12 })
            ),
            "got {err:?}",
        );
    }

    #[test]
    fn round_layout_rejects_zero_folding_factor() {
        let tuning = TuningSpec {
            vector_size: 1 << LOG_VECTOR_SIZE_MULTI_ROUND,
            starting_log_inv_rate: FIXTURE_LOG_INV_RATE,
            folding_factor: FoldingFactor::Constant(0),
            rate_schedule: RateSchedule::Stepping,
        };
        let err = layout(&tuning).expect_err("folding_factor = 0 must fail");
        assert!(
            matches!(
                err,
                DeriveError::Layout(LayoutError::FoldingFactorBelowOne { min: 0 })
            ),
            "got {err:?}",
        );
    }

    #[test]
    fn round_layout_rejects_zero_starting_rate() {
        let tuning = TuningSpec {
            vector_size: 1 << LOG_VECTOR_SIZE_MULTI_ROUND,
            starting_log_inv_rate: 0,
            folding_factor: FoldingFactor::Constant(FIXTURE_FOLDING_FACTOR),
            rate_schedule: RateSchedule::Stepping,
        };
        let err = layout(&tuning).expect_err("starting_log_inv_rate = 0 must fail");
        assert!(
            matches!(err, DeriveError::Layout(LayoutError::StartingRateBelowOne)),
            "got {err:?}",
        );
    }
}
