//! Errors raised by [`super::derive::ProtocolConfig::derive`] and the
//! sub-protocol solvers.

use std::fmt::{self, Display, Formatter};

use thiserror::Error;

use crate::{
    bits::Bits,
    protocols::{
        irs_commit::CodewordLengthError,
        params::{layout::LayoutError, spec::SecuritySpec},
        proof_of_work::{Config as PowConfig, PowError},
    },
};

/// Identifies a single PoW grind in the derived protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pow {
    /// Basecase γ-RLC grind (Lemma 7.4) — ZK mode only.
    BasecaseGammaCombination,
    /// Basecase sumcheck grind.
    BasecaseSumcheck,
    /// Per-round sumcheck grind at `index`.
    RoundSumcheck { index: usize },
    /// Per-round code-switch grind at `index`.
    RoundCodeSwitch { index: usize },
    /// Per-round mask-proximity grind at `index` — ZK mode only.
    RoundMaskProximity { index: usize },
}

impl Display for Pow {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::BasecaseGammaCombination => f.write_str("basecase γ-combination"),
            Self::BasecaseSumcheck => f.write_str("basecase sumcheck"),
            Self::RoundSumcheck { index } => write!(f, "round {index} sumcheck"),
            Self::RoundCodeSwitch { index } => write!(f, "round {index} code-switch"),
            Self::RoundMaskProximity { index } => write!(f, "round {index} mask-proximity"),
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

/// Grind `analytic → spec.target_security_bits`, then check the result against
/// `spec.pow_budget`.
pub fn grind_to_at(
    spec: &SecuritySpec,
    analytic: Bits,
    pow_kind: Pow,
) -> Result<PowConfig, DeriveError> {
    let target = Bits::new(f64::from(spec.target_security_bits));
    let pow = PowConfig::grind_to(target, analytic, spec.hash_id).at(pow_kind)?;
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
