use std::{
    fmt::{self, Debug, Display, Formatter},
    hash::{Hash, Hasher},
    marker::PhantomData,
    num::NonZeroU32,
    ops::Deref,
    str::FromStr,
};

use ordered_float::OrderedFloat;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
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

/// Phantom-typed newtype.
///
/// Trait impls are written by hand so that bounds apply to `T` only — tag
/// types stay bare, uninhabited enums.
pub struct Tagged<T, Tag>(T, PhantomData<Tag>);

impl<T: Copy, Tag> Tagged<T, Tag> {
    pub const fn new(v: T) -> Self {
        Self(v, PhantomData)
    }

    pub const fn get(self) -> T {
        self.0
    }
}

impl<T: Debug, Tag> Debug for Tagged<T, Tag> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Tagged").field(&self.0).finish()
    }
}

impl<T: Clone, Tag> Clone for Tagged<T, Tag> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

impl<T: Copy, Tag> Copy for Tagged<T, Tag> {}

impl<T: PartialEq, Tag> PartialEq for Tagged<T, Tag> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: Eq, Tag> Eq for Tagged<T, Tag> {}

impl<T: Hash, Tag> Hash for Tagged<T, Tag> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<T: Serialize, Tag> Serialize for Tagged<T, Tag> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de, T: Deserialize<'de>, Tag> Deserialize<'de> for Tagged<T, Tag> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        T::deserialize(deserializer).map(|value| Self(value, PhantomData))
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

    /// Borrow this spec as a [`ZkSpec`] proof, or `None` in standard mode.
    /// Prefer branching on this over matching `mode` and re-proving with
    /// [`ZkSpec::try_new`].
    pub fn as_zk(&self) -> Option<ZkSpec<'_>> {
        ZkSpec::try_new(self)
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

/// Per-round context handed to a sub-protocol builder.
#[derive(Debug, Clone)]
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

/// A `SecuritySpec` borrow proven to be in [`Mode::ZeroKnowledge`].
#[derive(Debug, Clone, Copy)]
pub struct ZkSpec<'a>(&'a SecuritySpec);

impl<'a> ZkSpec<'a> {
    pub fn try_new(spec: &'a SecuritySpec) -> Option<Self> {
        matches!(spec.mode, Mode::ZeroKnowledge).then_some(Self(spec))
    }

    pub const fn as_inner(self) -> &'a SecuritySpec {
        self.0
    }
}

impl Deref for ZkSpec<'_> {
    type Target = SecuritySpec;
    fn deref(&self) -> &SecuritySpec {
        self.0
    }
}

/// Reed–Solomon decoding regime selection.
///
/// - `Unique`: `δ < (1 − ρ)/2`, list size 1, no conjectures.
/// - `Johnson`: `δ < 1 − √ρ − η`, canonical `η = √ρ/20`. Proximity-gap error
///   per the BCSS25 improvement to BCIKS '20.
/// - `Capacity`: `δ < 1 − ρ − η`, canonical `η = ρ/20`. Conjectured list size
///   `d/(ρ·η)` and proximity-gap error per STIR Conjecture 5.6.
///
/// WHIR's rate stepping (each round bumps `log_inv_rate` by
/// `folding_factor − 1`) pushes ρ → 1, shrinking the unique-decoding
/// radius. At high security targets or deep folding, `Unique` may exceed
/// the grind cap on per-round PoW and [`super::derive::ProtocolConfig::derive`]
/// will return `PowUngrindable` — pick `Johnson` or `Capacity` for those.
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
mod decoding_regime_tests {
    use super::*;

    #[test]
    fn from_str_round_trips_display() {
        for r in [
            DecodingRegime::Unique,
            DecodingRegime::Johnson,
            DecodingRegime::Capacity,
        ] {
            assert_eq!(r.to_string().parse::<DecodingRegime>().unwrap(), r);
        }
    }

    #[test]
    fn from_str_rejects_unknown() {
        assert!("johnson".parse::<DecodingRegime>().is_err()); // case-sensitive
        assert!("".parse::<DecodingRegime>().is_err());
        assert!("capacity".parse::<DecodingRegime>().is_err());
    }
}

pub enum OodSampleBudgetTag {}
pub enum MaskCodeMessageLenTag {}
pub enum LogInvRateTag {}

/// OOD-sample budget (Lemma 9.9 / bounds doc §5.2).
pub type OodSampleBudget = Tagged<usize, OodSampleBudgetTag>;

impl Tagged<usize, OodSampleBudgetTag> {
    /// Sentinel for "no OOD samples".
    pub const ZERO: Self = Self::new(0);
}

/// C_zk message length (Theorem 9.6: `ℓ_zk ≥ source mask length`).
pub type MaskCodeMessageLen = Tagged<usize, MaskCodeMessageLenTag>;

/// `rate = 2^-log_inv_rate`.
pub type LogInvRate = Tagged<u32, LogInvRateTag>;

/// Reed–Solomon list-decoding ball size `|Λ(C, δ)|`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ListSize(OrderedFloat<f64>);

impl ListSize {
    pub const fn new(v: f64) -> Self {
        Self(OrderedFloat(v))
    }

    pub const fn get(self) -> f64 {
        self.0 .0
    }

    /// `log₂ |Λ|` — the form every analytic-error formula consumes.
    pub fn log2(self) -> f64 {
        self.get().log2()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash;

    const TARGET_BITS: u32 = 100;

    fn spec(pow_budget: PowBudget) -> SecuritySpec {
        SecuritySpec::new(TARGET_BITS)
            .with_mode(Mode::ZeroKnowledge)
            .with_pow_budget(pow_budget)
    }

    #[test]
    fn new_uses_documented_defaults() {
        let spec = SecuritySpec::new(128);
        assert_eq!(spec.mode, Mode::Standard);
        assert_eq!(spec.decoding_regime, DecodingRegime::Johnson);
        assert_eq!(spec.target_security_bits, 128);
        assert_eq!(
            spec.pow_budget,
            PowBudget::per_slot(DEFAULT_POW_BUDGET_BITS)
        );
        assert_eq!(spec.hash_id, hash::BLAKE3);
    }

    #[test]
    fn forbidden_means_no_pow_credit() {
        assert_eq!(
            spec(PowBudget::Forbidden).protocol_security_target_bits(),
            Bits::new(f64::from(TARGET_BITS)),
        );
    }

    #[test]
    fn per_slot_zero_collapses_to_forbidden() {
        assert_eq!(PowBudget::per_slot(0), PowBudget::Forbidden);
    }

    #[test]
    fn per_slot_bits_round_trip() {
        assert_eq!(PowBudget::per_slot(20).bits(), 20);
        assert_eq!(PowBudget::Forbidden.bits(), 0);
    }

    #[test]
    fn pow_credit_shifts_analytic_floor() {
        assert_eq!(
            spec(PowBudget::per_slot(20)).protocol_security_target_bits(),
            Bits::new(80.0),
        );
        assert_eq!(
            spec(PowBudget::per_slot(60)).protocol_security_target_bits(),
            Bits::new(40.0),
        );
    }

    #[test]
    fn pow_exceeding_target_saturates_to_zero() {
        let pow_over_target = TARGET_BITS + 100;
        assert_eq!(
            spec(PowBudget::per_slot(pow_over_target)).protocol_security_target_bits(),
            Bits::new(0.0),
        );
    }

    #[test]
    fn rate_schedule_stepping_unbounded() {
        // Stepping adds `folding − 1` per round regardless of magnitude
        // (legacy unbounded-stepping behavior).
        assert_eq!(RateSchedule::Stepping.step(2, 3), 4);
        assert_eq!(RateSchedule::Stepping.step(100, 5), 104);
    }

    #[test]
    fn rate_schedule_capped_clamps_above_cap() {
        let cap = RateSchedule::Capped {
            max_log_inv_rate: 5,
        };
        assert_eq!(cap.step(2, 3), 4); // below cap → step normally
        assert_eq!(cap.step(4, 3), 5); // would step to 6 → clamp to 5
        assert_eq!(cap.step(5, 3), 5); // already at cap → stays
        assert_eq!(cap.step(10, 3), 5); // above cap → snaps back to cap
    }

    #[test]
    fn rate_schedule_folding_factor_one_never_steps() {
        // folding == 1 means rate step is `1 − 1 = 0`. The IRS layer disallows
        // folding < 1, but the math here should still be consistent.
        assert_eq!(RateSchedule::Stepping.step(2, 1), 2);
        assert_eq!(
            RateSchedule::Capped {
                max_log_inv_rate: 5
            }
            .step(2, 1),
            2,
        );
    }
}
