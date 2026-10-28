//! Errors raised by [`super::derive::ProtocolConfig::derive`] and the
//! sub-protocol solvers.

use std::fmt::{self, Display, Formatter};

use thiserror::Error;

use crate::{
    bits::Bits,
    protocols::{
        irs_commit::CodewordLengthError,
        params::{
            layout::LayoutError,
            spec::{Mode, SecuritySpec},
        },
        proof_of_work::{Config as PowConfig, PowError},
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

/// Identifies a single PoW grind in the derived protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pow {
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

impl Display for Pow {
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

/// Origin side of a [`DeriveError::RoundChainBroken`]: either a numbered round
/// or the pre-round `tuning` shape (for plans with no rounds at all).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainSource {
    Tuning,
    Round(usize),
}

impl Display for ChainSource {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tuning => f.write_str("tuning"),
            Self::Round(i) => write!(f, "round {i}"),
        }
    }
}

/// Destination side of a [`DeriveError::RoundChainBroken`]: the next round in
/// sequence, or the basecase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainTarget {
    NextRound(usize),
    Basecase,
}

impl Display for ChainTarget {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::NextRound(i) => write!(f, "round {i}"),
            Self::Basecase => f.write_str("basecase"),
        }
    }
}

/// Failure modes for [`super::derive::ProtocolConfig::derive`] and the
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
    /// The `t_ood` fixed-point in [`super::build_round::solve_t_ood`] ran out of
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
        pow: Pow,
        #[source]
        source: PowError,
    },

    /// A PoW grind fits the grind cap but exceeds the per-slot budget set by
    /// [`super::spec::SecuritySpec::pow_budget`].
    #[error(
        "{pow} requires {required} bits, exceeds spec.pow_budget = {max}; \
         raise pow_budget or lower target_security_bits"
    )]
    PowBudgetExceeded { pow: Pow, required: Bits, max: Bits },

    /// Computed codeword length exceeds the NTT engine's supported order.
    #[error(
        "codeword length {length} exceeds the NTT engine's supported order; \
         reduce vector_size or starting_log_inv_rate"
    )]
    CodewordExceedsNtt { length: usize },

    /// The tuning spec cannot produce a valid round layout.
    #[error(transparent)]
    Layout(#[from] LayoutError),

    /// Cross-round (or round → basecase) shape chain broken: the next
    /// component's source `vector_size` does not match the previous
    /// component's target `vector_size`. Surfaced by
    /// [`super::protocol_config::ProtocolConfig::validate_round_chaining`].
    #[error("chain broken: {from} → {to} expected vector_size {expected}, found {found}")]
    RoundChainBroken {
        from: ChainSource,
        to: ChainTarget,
        expected: usize,
        found: usize,
    },

    /// A PoW slot's `analytic + pow.difficulty()` is below `target_security_bits`.
    /// `grind_to_at` guarantees this at construction; this error fires only if
    /// the analytic-error formulas applied at validate-time disagree with what
    /// the per-protocol `solve` functions consumed (e.g., a planner regression
    /// drifts away from the actual configured IRS rate). Catches the case where
    /// the rate-schedule plumbing under-reports a per-slot rate.
    #[error("{pow} soundness gap: analytic {analytic} + pow {pow_bits} < target {target}")]
    SecurityTargetNotMet {
        pow: Pow,
        analytic: Bits,
        pow_bits: Bits,
        target: Bits,
    },

    /// The analytic floor recorded at solve time disagrees with a fresh
    /// recompute from the same config's state. Indicates that the inputs to
    /// the `analytic_error_bits` formula drifted between solve and validate
    /// (e.g., an IRS field was overwritten after construction).
    #[error("{pow} analytic drift: recorded {recorded} vs recompute {recompute}")]
    AnalyticDrift {
        pow: Pow,
        recorded: Bits,
        recompute: Bits,
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

/// Lift `Result<T, PowError>` into `Result<T, DeriveError>` by attaching a
/// [`Pow`] label.
pub trait PowResultExt<T> {
    fn at(self, pow: Pow) -> Result<T, DeriveError>;
}

impl<T> PowResultExt<T> for Result<T, PowError> {
    fn at(self, pow: Pow) -> Result<T, DeriveError> {
        self.map_err(|source| DeriveError::PowUngrindable { pow, source })
    }
}

/// Enforce the target, allowing grinding credit only in Standard mode.
///
/// Zook requires the full analytic target until a grinding reduction is supplied.
pub fn grind_to_at(
    spec: &SecuritySpec,
    analytic: Bits,
    pow_kind: Pow,
) -> Result<PowConfig, DeriveError> {
    let target = Bits::new(f64::from(spec.target_security_bits));
    if matches!(spec.mode, Mode::ZeroKnowledge) {
        if analytic < target {
            return Err(DeriveError::SecurityTargetNotMet {
                pow: pow_kind,
                analytic,
                pow_bits: Bits::new(0.0),
                target,
            });
        }
        return Ok(PowConfig::none());
    }
    let mut pow = PowConfig::grind_to(target, analytic, spec.hash_id).at(pow_kind)?;
    // grind_to rounds the threshold upward. Tighten a fractional gap by one
    // threshold unit instead of accepting less than the configured target.
    if analytic + pow.difficulty() < target {
        pow.threshold = pow.threshold.saturating_sub(1);
    }
    let required = pow.difficulty();
    let max = Bits::new(f64::from(spec.pow_budget.bits()));
    if required > max {
        return Err(DeriveError::PowBudgetExceeded {
            pow: pow_kind,
            required,
            max,
        });
    }
    Ok(pow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::params::spec::PowBudget;

    #[test]
    fn standard_grinding_closes_fractional_gaps() {
        let spec = SecuritySpec::new(80).with_pow_budget(PowBudget::per_slot(60));
        for analytic in [20.3422, 35.623_375_386_491_45] {
            let analytic = Bits::new(analytic);
            let pow = grind_to_at(&spec, analytic, Pow::BasecaseSumcheck).unwrap();
            assert!(analytic + pow.difficulty() >= Bits::new(80.0 - 1e-9));
            assert!(pow.difficulty() <= Bits::new(60.0));
        }
    }

    #[test]
    fn grinding_still_enforces_budget_and_zook_target() {
        let spec = SecuritySpec::new(80).with_pow_budget(PowBudget::per_slot(59));
        assert!(matches!(
            grind_to_at(&spec, Bits::new(20.3422), Pow::BasecaseSumcheck),
            Err(DeriveError::PowBudgetExceeded { .. }),
        ));
        let spec = spec.with_mode(Mode::ZeroKnowledge);
        assert!(matches!(
            grind_to_at(&spec, Bits::new(79.9999), Pow::JointMaskBinding),
            Err(DeriveError::SecurityTargetNotMet { .. }),
        ));
    }
}
