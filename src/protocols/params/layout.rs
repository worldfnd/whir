//! Round-skeleton layout: pure-data walk over the witness shape.
//!
//! Produces per-round shapes (vector size, log_inv_rate, folding factors)
//! independent of [`SecuritySpec`] and IRS solving. Consumed by
//! [`super::build_round`] to instantiate per-round configs and by
//! [`super::derive`] to drive the round/basecase split.

use thiserror::Error;

use crate::{
    algebra::embedding::Embedding,
    protocols::{
        irs_commit::Config as IrsConfig,
        params::spec::{RoundContext, TuningSpec},
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
}

#[derive(Debug)]
pub(super) struct RoundLayout {
    pub(super) shapes: Vec<RoundShape>,
    pub(super) basecase_vector_size: usize,
    pub(super) basecase_log_inv_rate: u32,
}

pub(super) fn round_layout(tuning: &TuningSpec) -> Result<RoundLayout, LayoutError> {
    if !tuning.vector_size.is_power_of_two() {
        return Err(LayoutError::VectorSizeNotPowerOfTwo {
            vector_size: tuning.vector_size,
        });
    }
    let min_folding = tuning.folding_factor.min();
    if min_folding < 1 {
        return Err(LayoutError::FoldingFactorBelowOne { min: min_folding });
    }
    if tuning.starting_log_inv_rate < 1 {
        return Err(LayoutError::StartingRateBelowOne);
    }

    let mut num_vars = tuning.vector_size.trailing_zeros() as usize;
    let mut log_inv_rate = tuning.starting_log_inv_rate;
    let mut shapes = Vec::new();

    loop {
        let round = shapes.len();
        let source_folding = tuning.folding_factor.at_round(round);
        let target_folding = tuning.folding_factor.at_round(round.saturating_add(1));
        if num_vars < source_folding.saturating_add(target_folding) {
            break;
        }
        shapes.push(RoundShape {
            round_index: round,
            source_vector_size: 1usize << num_vars,
            source_log_inv_rate: log_inv_rate,
            source_folding_factor: source_folding as u32,
            target_folding_factor: target_folding as u32,
        });
        num_vars = num_vars.saturating_sub(source_folding);
        log_inv_rate = log_inv_rate.saturating_add((source_folding as u32).saturating_sub(1));
    }

    Ok(RoundLayout {
        shapes,
        basecase_vector_size: 1usize << num_vars,
        basecase_log_inv_rate: log_inv_rate,
    })
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
        log_inv_rate: shape
            .source_log_inv_rate
            .saturating_add(shape.source_folding_factor.saturating_sub(1)),
        folding_factor: shape.target_folding_factor,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::params::spec::FoldingFactor;

    const FIXTURE_FOLDING_FACTOR: usize = 2;
    const FIXTURE_LOG_INV_RATE: u32 = 1;

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
        };
        let layout = round_layout(&tuning).unwrap();

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
        };
        let layout = round_layout(&tuning).unwrap();
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
        let layout = round_layout(&tuning).unwrap();
        let consumed: u32 = layout.shapes.iter().map(|s| s.source_folding_factor).sum();
        let initial_num_vars = tuning.vector_size.trailing_zeros();
        let remaining = initial_num_vars - consumed;
        assert_eq!(layout.basecase_vector_size, 1usize << remaining);
    }

    #[test]
    fn round_layout_stops_when_no_room_for_source_plus_target() {
        let vector_size = 1usize << LOG_VECTOR_SIZE_NO_ROUNDS;
        let tuning = tuning_with(vector_size);
        let layout = round_layout(&tuning).unwrap();
        assert!(layout.shapes.is_empty());
        assert_eq!(layout.basecase_vector_size, vector_size);
        assert_eq!(layout.basecase_log_inv_rate, FIXTURE_LOG_INV_RATE);
    }

    #[test]
    fn round_layout_rejects_non_pow2_vector_size() {
        let tuning = TuningSpec {
            vector_size: 12,
            starting_log_inv_rate: FIXTURE_LOG_INV_RATE,
            folding_factor: FoldingFactor::Constant(FIXTURE_FOLDING_FACTOR),
        };
        let err = round_layout(&tuning).expect_err("non-pow2 vector_size must fail");
        assert!(
            matches!(
                err,
                LayoutError::VectorSizeNotPowerOfTwo { vector_size: 12 }
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
        };
        let err = round_layout(&tuning).expect_err("folding_factor = 0 must fail");
        assert!(
            matches!(err, LayoutError::FoldingFactorBelowOne { min: 0 }),
            "got {err:?}",
        );
    }

    #[test]
    fn round_layout_rejects_zero_starting_rate() {
        let tuning = TuningSpec {
            vector_size: 1 << LOG_VECTOR_SIZE_MULTI_ROUND,
            starting_log_inv_rate: 0,
            folding_factor: FoldingFactor::Constant(FIXTURE_FOLDING_FACTOR),
        };
        let err = round_layout(&tuning).expect_err("starting_log_inv_rate = 0 must fail");
        assert!(
            matches!(err, LayoutError::StartingRateBelowOne),
            "got {err:?}",
        );
    }
}
