//! Errors raised by [`super::ProtocolConfig::derive`] and the
//! sub-protocol solvers.

use std::fmt::{self, Display, Formatter};

use thiserror::Error;

use crate::{
    bits::Bits,
    protocols::{
        irs_commit::CodewordLengthError, params::derive::LayoutError, proof_of_work::PowError,
    },
};

/// A code configuration outside the assumptions of the Zook bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum ZookSecurityError {
    /// Witness masks must be supplied by Zook without increasing the IRS degree.
    #[error("Zook bounds require standard IRS encoding")]
    AppendedMasks,
    /// Only the Johnson decoding bound is implemented here.
    #[error("Zook bounds require Johnson decoding")]
    UnsupportedRegime,
    /// The correlated-agreement bound requires half-distance ≤ theta < Johnson radius.
    #[error("proximity parameter is outside the correlated-agreement interval")]
    InvalidProximity,
    /// The final witness commitment encodes one polynomial.
    #[error("expected a single non-interleaved polynomial")]
    ExpectedSinglePolynomial,
    /// Joint masks and their final-opening blinder occupy two full-length columns.
    #[error("expected joint masks and their final-opening blinder as two non-interleaved vectors")]
    ExpectedMaskAndBlinder,
    /// One round commits the polynomial obtained by folding its source columns.
    #[error("target vector size must equal source polynomial length")]
    InvalidRoundShape,
    /// A DEEP challenge must have at least one point outside the code domain.
    #[error("code domain exhausts the challenge field")]
    EmptyOutOfDomain,
}

/// Identifies a security bound in the derived protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityEvent {
    /// Binding the single joint-mask commitment before folding.
    JointMaskBinding,
    /// Combining the sparse sumcheck mask with the current claim.
    RoundMaskCombination { index: usize },
    /// Common combination of final witness and joint masks.
    FinalOpeningCombination,
    /// Independent queries to the final witness and joint masks.
    FinalOpeningQueries,
    /// Basecase sumcheck grind.
    BasecaseSumcheck,
    /// Per-round sumcheck grind at `index`.
    RoundSumcheck { index: usize },
    /// Per-round code-switch grind at `index`.
    RoundCodeSwitch { index: usize },
}

impl Display for SecurityEvent {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::JointMaskBinding => f.write_str("joint-mask binding"),
            Self::RoundMaskCombination { index } => write!(f, "round {index} mask combination"),
            Self::FinalOpeningCombination => f.write_str("final-opening combination"),
            Self::FinalOpeningQueries => f.write_str("final-opening queries"),
            Self::BasecaseSumcheck => f.write_str("basecase sumcheck"),
            Self::RoundSumcheck { index } => write!(f, "round {index} sumcheck"),
            Self::RoundCodeSwitch { index } => write!(f, "round {index} code-switch"),
        }
    }
}

/// Failure modes for [`super::ProtocolConfig::derive`] and the
/// sub-protocol solvers it calls.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DeriveError {
    /// A code is outside the assumptions of the Zook security bounds.
    #[error(transparent)]
    ZookSecurity(#[from] ZookSecurityError),

    /// A dimension or mask offset cannot be represented by usize.
    #[error("protocol dimensions overflow usize")]
    SizeOverflow,

    /// A mask cannot fit without increasing its polynomial's degree.
    #[error("mask requires {required} coefficients, polynomial has {available}")]
    MaskDoesNotFit { required: usize, available: usize },

    /// Mask ranges, commitment dimensions, or protocol modes disagree.
    #[error("invalid joint-mask layout")]
    InvalidMaskLayout,
    /// The `t_ood` fixed-point in `solve_t_ood` ran out of
    /// iterations — usually the field is too small for the security target.
    #[error(
        "t_ood fixed-point did not converge for round {round_index}; \
         lower target_security_bits or use a larger field"
    )]
    FixedPointDidNotConverge { round_index: usize },

    /// A PoW grind cannot close the analytic-to-target gap — the spec is too
    /// tight for any single grind to reach `target_security_bits`.
    #[error(
        "{pow} cannot be ground: {source}; lower target_security_bits or \
         switch to a less conservative decoding regime (Johnson/Capacity)"
    )]
    PowUngrindable {
        pow: SecurityEvent,
        #[source]
        source: PowError,
    },

    /// A PoW grind fits the grind cap but exceeds the per-slot budget set by
    /// [`super::spec::SecuritySpec::pow_budget`].
    #[error(
        "{pow} requires {required} bits, exceeds spec.pow_budget = {max}; \
         raise pow_budget or lower target_security_bits"
    )]
    PowBudgetExceeded {
        pow: SecurityEvent,
        required: Bits,
        max: Bits,
    },

    /// Computed codeword length exceeds the NTT engine's supported order.
    #[error(
        "codeword length {length} exceeds the NTT engine's supported order; \
         reduce vector_size or starting_log_inv_rate"
    )]
    CodewordExceedsNtt { length: usize },

    /// The tuning spec cannot produce a valid round layout.
    #[error(transparent)]
    Layout(#[from] LayoutError),

    /// A step's analytic bound and permitted grinding do not meet the target.
    #[error("{pow} soundness gap: analytic {analytic} + pow {pow_bits} < target {target}")]
    SecurityTargetNotMet {
        pow: SecurityEvent,
        analytic: Bits,
        pow_bits: Bits,
        target: Bits,
    },

    /// No searched rate schedule satisfies the configured size and security bounds.
    #[error("adaptive rate planner found no schedule satisfying the size and security bounds")]
    AdaptiveNoFeasibleSchedule,
}

impl From<CodewordLengthError> for DeriveError {
    fn from(e: CodewordLengthError) -> Self {
        Self::CodewordExceedsNtt { length: e.length }
    }
}
