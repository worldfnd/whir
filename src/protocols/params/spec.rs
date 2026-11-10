use std::{
    fmt::{self, Display, Formatter},
    num::NonZeroU32,
    str::FromStr,
};

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{bits::Bits, engines::EngineId, hash};

/// Default per-slot PoW budget when the user does not specify one.
///
/// 16 bits balances prover grinding cost against the security credit it buys:
/// higher values slow the prover on every slot; lower values shrink the PoW
/// contribution and push the analytic floor (and thus proof size) up.
pub const DEFAULT_POW_BUDGET_BITS: u32 = 16;

/// Per-slot proof-of-work policy.
///
/// `bits` plays two roles:
/// - **Planning credit**: subtracted from `target_security_bits` so solvers
///   know the analytic floor they must reach.
/// - **Validation cap**: rejects any per-slot PoW that exceeds `bits`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PowBudget {
    Forbidden,
    PerSlot { bits: NonZeroU32 },
}

impl PowBudget {
    /// `Forbidden` when `bits == 0`, else `PerSlot { bits }`.
    pub const fn per_slot(bits: u32) -> Self {
        match NonZeroU32::new(bits) {
            Some(bits) => Self::PerSlot { bits },
            None => Self::Forbidden,
        }
    }

    /// Bits of grinding allowed per slot. `0` for [`PowBudget::Forbidden`].
    pub const fn bits(self) -> u32 {
        match self {
            Self::Forbidden => 0,
            Self::PerSlot { bits } => bits.get(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecuritySpec {
    pub mode: Mode,
    pub decoding_regime: DecodingRegime,
    pub target_security_bits: u32,
    pub pow_budget: PowBudget,
    pub hash_id: EngineId,
}

impl SecuritySpec {
    /// Spec with canonical defaults: [`Mode::Standard`],
    /// [`DecodingRegime::Johnson`], BLAKE3, and a
    /// [`DEFAULT_POW_BUDGET_BITS`]-bit per-slot PoW budget.
    ///
    /// Override individual choices with the `with_*` methods or struct-update
    /// syntax.
    pub const fn new(target_security_bits: u32) -> Self {
        Self {
            mode: Mode::Standard,
            decoding_regime: DecodingRegime::Johnson,
            target_security_bits,
            pow_budget: PowBudget::per_slot(DEFAULT_POW_BUDGET_BITS),
            hash_id: hash::BLAKE3,
        }
    }

    #[must_use]
    pub const fn with_mode(mut self, mode: Mode) -> Self {
        self.mode = mode;
        self
    }

    #[must_use]
    pub const fn with_decoding_regime(mut self, decoding_regime: DecodingRegime) -> Self {
        self.decoding_regime = decoding_regime;
        self
    }

    #[must_use]
    pub const fn with_pow_budget(mut self, pow_budget: PowBudget) -> Self {
        self.pow_budget = pow_budget;
        self
    }

    #[must_use]
    pub const fn with_hash(mut self, hash_id: EngineId) -> Self {
        self.hash_id = hash_id;
        self
    }

    pub fn protocol_security_target_bits(&self) -> Bits {
        let pow = self.pow_budget.bits();
        Bits::new(f64::from(self.target_security_bits.saturating_sub(pow)))
    }
}

/// Per-round folding strategy. `at_round(i)` returns the factor for round `i`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FoldingFactor {
    /// Same folding factor across all rounds.
    Constant(usize),
    /// `at_round(0) = initial`; `at_round(i) = rest` for `i ≥ 1`.
    ConstantFromSecondRound { initial: usize, rest: usize },
}

impl FoldingFactor {
    pub const fn at_round(&self, round: usize) -> usize {
        match self {
            Self::Constant(f) => *f,
            Self::ConstantFromSecondRound { initial, rest } => {
                if round == 0 {
                    *initial
                } else {
                    *rest
                }
            }
        }
    }

    /// Smallest factor across rounds.
    pub const fn min(&self) -> usize {
        match self {
            Self::Constant(f) => *f,
            Self::ConstantFromSecondRound { initial, rest } => {
                if *initial < *rest {
                    *initial
                } else {
                    *rest
                }
            }
        }
    }
}

/// Proof-size / prover-time / soundness-margin tradeoffs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuningSpec {
    pub vector_size: usize,
    pub starting_log_inv_rate: u32,
    pub folding_factor: FoldingFactor,
    /// Per-round inverse-rate schedule (see [`RateSchedule`]).
    pub rate_schedule: RateSchedule,
}

/// Pareto-knee bias for [`RateSchedule::Adaptive`]'s planner. Controls the
/// tradeoff between prover-time and proof-size axes when picking per-round
/// inverse rates. Clamped to `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KneeWeight(OrderedFloat<f64>);

impl KneeWeight {
    /// Pure geometric knee — equal weighting on both axes.
    pub const DEFAULT: Self = Self(OrderedFloat(0.5));

    /// Build a `KneeWeight`, clamping `value` to `[0, 1]`. Out-of-range
    /// inputs (including NaN) are silently corrected — the planner has no
    /// meaningful behavior outside `[0, 1]`, so reject-by-clamp at the API
    /// boundary rather than propagating a `Result`.
    pub const fn new(value: f64) -> Self {
        let clamped = if value.is_nan() {
            0.5
        } else {
            value.clamp(0.0, 1.0)
        };
        Self(OrderedFloat(clamped))
    }

    pub const fn get(self) -> f64 {
        self.0 .0
    }
}

impl Default for KneeWeight {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Per-round inverse-rate schedule.
///
/// The Section 10 code-switching IOPP (Theorem 10.2) commits a fresh codeword
/// each round, so the per-round rate is a free parameter — the only structural
/// constraint is on message lengths (`2^{k_{i+1}} · ℓ_{i+1} ≥ ℓ_i`).
///
/// Capping the inverse rate trades a small increase in per-round query count
/// (cheap on tiny late-round Merkle trees) for dramatically smaller late-round
/// and basecase NTTs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RateSchedule {
    /// Step `log_inv_rate += folding − 1` per round, unbounded — the
    /// canonical legacy in-place WHIR behavior. Late-round and basecase NTTs
    /// grow with depth.
    Stepping,
    /// Step like [`Self::Stepping`], then clamp at `max_log_inv_rate`. Set
    /// `max_log_inv_rate == tuning.starting_log_inv_rate` for a constant-rate
    /// schedule.
    Capped { max_log_inv_rate: u32 },
    /// Per-round rates chosen at `derive` time by a planner that minimises a
    /// `(prover_time_proxy, proof_size_proxy)` pareto knee under the security
    /// target. No user-supplied budget; the knee is scale-free. The
    /// [`KneeWeight`] biases the picker between the two axes.
    Adaptive { knee_weight: KneeWeight },
}

impl RateSchedule {
    /// Compute the next round's inverse rate from the current one and the
    /// current round's folding factor.
    ///
    /// `Adaptive` returns the unbounded step here; the planner runs after the
    /// skeleton is built and overwrites every per-round rate.
    pub const fn step(self, current_log_inv_rate: u32, folding_factor: u32) -> u32 {
        let stepped = current_log_inv_rate.saturating_add(folding_factor.saturating_sub(1));
        match self {
            Self::Stepping | Self::Adaptive { .. } => stepped,
            Self::Capped { max_log_inv_rate } => {
                if stepped < max_log_inv_rate {
                    stepped
                } else {
                    max_log_inv_rate
                }
            }
        }
    }

    pub const fn is_adaptive(self) -> bool {
        matches!(self, Self::Adaptive { .. })
    }
}

/// Dimensions and rate of one commitment.
#[derive(Debug, Clone, Copy)]
pub struct RoundContext {
    pub vector_size: usize,
    pub log_inv_rate: u32,
    pub folding_factor: u32,
}

/// Standard vs. zero-knowledge selection. Orthogonal to [`DecodingRegime`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mode {
    Standard,
    ZeroKnowledge,
}

impl Mode {
    pub const fn is_zk(self) -> bool {
        matches!(self, Self::ZeroKnowledge)
    }
}

/// Reed–Solomon decoding regime selection.
///
/// - `Unique`: `δ < (1 − ρ)/2`, list size 1, no conjectures.
/// - `Johnson`: `δ < 1 − √ρ − η`, canonical `η = √ρ/20`. Proximity-gap error
///   per the BCHKS25 improvement to BCIKS '20.
/// - `Capacity`: `δ < 1 − ρ − η`, canonical `η = ρ/20`. Conjectured list size
///   `d/(ρ·η)` and proximity-gap error per STIR Conjecture 5.6.
///
/// High security targets or large codewords can make `Unique` exceed the
/// per-round PoW cap. [`super::ProtocolConfig::derive`] then returns
/// `PowUngrindable`. `Johnson` or `Capacity` can provide different bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecodingRegime {
    Unique,
    Johnson,
    Capacity,
}

impl Display for DecodingRegime {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unique => f.write_str("Unique"),
            Self::Johnson => f.write_str("Johnson"),
            Self::Capacity => f.write_str("Capacity"),
        }
    }
}

/// Error returned by [`DecodingRegime`]'s [`FromStr`] impl.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("invalid decoding regime: {0}, options are: Unique, Johnson, Capacity")]
pub struct ParseDecodingRegimeError(String);

impl FromStr for DecodingRegime {
    type Err = ParseDecodingRegimeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Unique" => Ok(Self::Unique),
            "Johnson" => Ok(Self::Johnson),
            "Capacity" => Ok(Self::Capacity),
            _ => Err(ParseDecodingRegimeError(s.to_owned())),
        }
    }
}

#[cfg(test)]
#[path = "tests/spec.rs"]
mod tests;
