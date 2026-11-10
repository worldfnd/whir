//! Parameter selection for WHIR and Zook.
//!
//! Requests contain user choices. Construction owns each commitment once and checks its bounds.
//! Borrowed execution views use that canonical sequence; deserialization repeats construction.
//! Mathematical references use the pinned [ProveKit whitepaper].
//!
//! [ProveKit whitepaper]: https://github.com/worldfnd/provekit/tree/6605cf5242567e17899a26c27db829c7ce9eb478/whitepaper

use std::{
    fmt::{self, Display, Formatter},
    str::FromStr,
};

use ark_ff::Field;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    algebra::embedding::{Embedding, Identity},
    engines::EngineId,
    protocols::{
        basecase::{BasecaseMode, Config as BasecaseConfig},
        code_switch::Config as CodeSwitchConfig,
        irs_commit::Config as IrsConfig,
        proof_of_work::Config as PowConfig,
        sumcheck::Config as SumcheckConfig,
    },
    type_info::Typed,
};

mod derive;
mod masks;
#[cfg(test)]
mod parameter_regressions;
mod search;
pub(crate) mod security;
#[cfg(test)]
mod tests;

pub use derive::DeriveError;
type Result<T> = std::result::Result<T, DeriveError>;
pub use masks::{JointMaskConfig, WitnessMaskConfig};

/// Security and layout choices. Rates and folds use base-two logarithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Params {
    pub zk: bool,
    pub regime: DecodingRegime,
    /// Required security bits for each checked protocol step.
    pub bits: u32,
    /// Maximum grinding bits per step; Zook receives no grinding credit.
    pub pow: u32,
    pub hash: EngineId,
    pub size: usize,
    pub rate: u32,
    /// Log2 folding factors for the first round and all later rounds.
    pub folds: [usize; 2],
    pub schedule: RateSchedule,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RateSchedule {
    Stepping,
    Capped(u32),
    /// Weighted Pareto knee: zero favors proof size; one favors encoding work.
    Adaptive(KneeWeight),
}
/// Clamp weights to `[0,1]`; NaN selects equal weighting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KneeWeight(OrderedFloat<f64>);
impl From<f64> for KneeWeight {
    fn from(value: f64) -> Self {
        Self(OrderedFloat(if value.is_nan() {
            0.5
        } else {
            value.clamp(0.0, 1.0)
        }))
    }
}
impl RateSchedule {
    fn step(self, rate: u32, folds: u32) -> u32 {
        let rate = rate.saturating_add(folds.saturating_sub(1));
        match self {
            Self::Capped(cap) => rate.min(cap),
            _ => rate,
        }
    }
}
#[derive(Debug, Clone, Copy)]
struct RoundContext {
    size: usize,
    rate: u32,
    folds: u32,
}
/// Unique, Johnson, and conjectured capacity decoding bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecodingRegime {
    Unique,
    Johnson,
    Capacity,
}
impl Display for DecodingRegime {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl FromStr for DecodingRegime {
    type Err = DeriveError;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "Unique" => Ok(Self::Unique),
            "Johnson" => Ok(Self::Johnson),
            "Capacity" => Ok(Self::Capacity),
            _ => Err(DeriveError("invalid decoding regime")),
        }
    }
}

/// The serialized request is the only external configuration input.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, bound = "")]
pub struct ProtocolSpec<M: Embedding> {
    pub embedding: Typed<M>,
    pub params: Params,
}

/// Each IRS configuration occurs once. The last target is the final commitment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolConfig<M: Embedding> {
    request: ProtocolSpec<M>,
    first: Option<IrsConfig<M>>,
    targets: Vec<IrsConfig<Identity<M::Target>>>,
    joint_masks: Option<JointMaskConfig<M::Target>>,
}

impl<M: Embedding> Serialize for ProtocolConfig<M> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        self.request.serialize(serializer)
    }
}
impl<'de, M: Embedding + Default> Deserialize<'de> for ProtocolConfig<M> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let request = ProtocolSpec::<M>::deserialize(deserializer)?;
        if request.embedding.0 != M::default() {
            return Err(serde::de::Error::custom(
                "configuration requires the default embedding",
            ));
        }
        Self::derive(request.params).map_err(serde::de::Error::custom)
    }
}

impl<M: Embedding> ProtocolConfig<M> {
    pub const fn params(&self) -> &Params {
        &self.request.params
    }
    pub fn padded_vector_size(&self) -> usize {
        self.first.as_ref().map_or_else(
            || self.final_commit().vector_size(),
            |code| code.vector_size(),
        )
    }
    pub fn first_round(&self) -> Option<RoundConfig<'_, M>> {
        self.first
            .as_ref()
            .map(|source| derive::round(self.params(), source, &self.targets[0], 0).unwrap())
    }
    pub fn tail_rounds(
        &self,
    ) -> impl ExactSizeIterator<Item = RoundConfig<'_, Identity<M::Target>>> {
        self.targets
            .windows(2)
            .enumerate()
            .map(|(i, pair)| derive::round(self.params(), &pair[0], &pair[1], i + 1).unwrap())
    }
    pub const fn has_rounds(&self) -> bool {
        self.first.is_some()
    }
    pub fn num_rounds(&self) -> usize {
        self.targets.len() - 1 + usize::from(self.has_rounds())
    }
    pub fn final_commit(&self) -> &IrsConfig<Identity<M::Target>> {
        self.targets.last().expect("final commitment")
    }
    /// Standard terminal primitive; Zook uses only its sumcheck after blinding the final claim.
    pub(crate) fn basecase(&self) -> BasecaseConfig<'_, M::Target> {
        BasecaseConfig::new(
            self.final_commit(),
            derive::terminal(
                self.params(),
                self.final_commit(),
                usize::from(self.has_rounds()),
            )
            .unwrap(),
            BasecaseMode::Standard,
            PowConfig::none(),
        )
    }
    pub const fn joint_masks(&self) -> Option<&JointMaskConfig<M::Target>> {
        self.joint_masks.as_ref()
    }
    /// Mask dimensions come from the commitment that owns the corresponding witness.
    pub fn witness_mask(&self, index: usize) -> Option<WitnessMaskConfig> {
        assert!(index <= self.num_rounds());
        if !self.params().zk {
            return None;
        }
        let mask = if index == 0 && self.has_rounds() {
            WitnessMaskConfig::for_code(self.first.as_ref().unwrap(), 0)
        } else {
            let node = &self.targets[index - usize::from(self.has_rounds())];
            WitnessMaskConfig::for_code(node, usize::from(index != 0))
        };
        Some(mask.expect("construction checked witness mask dimensions"))
    }
}

/// A temporary execution view; each borrowed IRS has one owner in ProtocolConfig.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoundConfig<'a, M: Embedding> {
    pub index: usize,
    pub sumcheck: SumcheckConfig<M::Target>,
    pub code_switch: CodeSwitchConfig<'a, M>,
    pub masks: Option<WitnessMaskConfig>,
}

fn embedding_degree<M: Embedding>() -> Result<usize> {
    let source = M::Source::extension_degree();
    let target = M::Target::extension_degree();
    if source == 0 || !target.is_multiple_of(source) {
        return Err(DeriveError("invalid embedding degree"));
    }
    usize::try_from(target / source).map_err(|_| DeriveError("dimension overflow"))
}

fn require(condition: bool, message: &'static str) -> Result<()> {
    condition.then_some(()).ok_or(DeriveError(message))
}
