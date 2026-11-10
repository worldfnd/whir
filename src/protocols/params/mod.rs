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
use thiserror::Error;

use crate::{
    algebra::embedding::{Embedding, Identity},
    bits::Bits,
    engines::EngineId,
    hash,
    protocols::{
        basecase::{BasecaseMode, Config as BasecaseConfig},
        code_switch::{CodeSwitchMode, Config as CodeSwitchConfig},
        irs_commit::Config as IrsConfig,
        proof_of_work::Config as PowConfig,
        sumcheck::{Config as SumcheckConfig, SumcheckMaskLen, SumcheckMode},
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
use masks::mask_coefficients;
pub use masks::{JointMaskConfig, RoundMasks, WitnessMaskConfig};

pub const DEFAULT_POW_BUDGET_BITS: u32 = 16;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecuritySpec {
    pub mode: Mode,
    pub decoding_regime: DecodingRegime,
    pub target_security_bits: u32,
    /// Per-step grinding budget in bits; zero forbids grinding.
    pub pow_budget: u32,
    pub hash_id: EngineId,
}

impl SecuritySpec {
    pub const fn new(target_security_bits: u32) -> Self {
        Self {
            mode: Mode::Standard,
            decoding_regime: DecodingRegime::Johnson,
            target_security_bits,
            pow_budget: DEFAULT_POW_BUDGET_BITS,
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
    pub const fn with_pow_budget(mut self, pow_budget: u32) -> Self {
        self.pow_budget = pow_budget;
        self
    }
    #[must_use]
    pub const fn with_hash(mut self, hash_id: EngineId) -> Self {
        self.hash_id = hash_id;
        self
    }
    /// Standard planning credit; Zook always uses the full target.
    pub fn protocol_security_target_bits(&self) -> Bits {
        Bits::new(f64::from(
            self.target_security_bits.saturating_sub(self.pow_budget),
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuningSpec {
    pub vector_size: usize,
    pub starting_log_inv_rate: u32,
    /// Folding factors for the first round and all later rounds.
    pub folding_factor: [usize; 2],
    pub rate_schedule: RateSchedule,
}

/// Adaptive search bias toward encoding cost; zero favors proof size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KneeWeight(OrderedFloat<f64>);
impl KneeWeight {
    pub const DEFAULT: Self = Self(OrderedFloat(0.5));
    /// Clamp to `[0,1]`; NaN selects equal weighting.
    pub const fn new(value: f64) -> Self {
        Self(OrderedFloat(if value.is_nan() {
            0.5
        } else {
            value.clamp(0.0, 1.0)
        }))
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RateSchedule {
    /// Increase log2 inverse rate by folding_factor - 1.
    Stepping,
    /// Apply the same increase, then cap the result.
    Capped { max_log_inv_rate: u32 },
    /// Select the same cost/proof Pareto knee as the original planner.
    Adaptive { knee_weight: KneeWeight },
}
impl RateSchedule {
    pub fn step(self, current: u32, folds: u32) -> u32 {
        let stepped = current.saturating_add(folds.saturating_sub(1));
        match self {
            Self::Capped { max_log_inv_rate } => stepped.min(max_log_inv_rate),
            _ => stepped,
        }
    }
}

/// One commitment in the selected schedule.
#[derive(Debug, Clone, Copy)]
struct RoundContext {
    pub vector_size: usize,
    pub log_inv_rate: u32,
    pub folding_factor: u32,
}

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

/// Unique uses distance (1-rho)/2; Johnson uses 1-sqrt(rho)-eta.
/// Capacity uses the conjectured 1-rho-eta bound. See security.rs for eta and citations.
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

/// The serialized request is the only external configuration input.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, bound = "")]
pub struct ProtocolSpec<M: Embedding> {
    pub embedding: Typed<M>,
    pub security: SecuritySpec,
    pub tuning: TuningSpec,
}

/// Each IRS configuration occurs once. The last target is the final commitment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolConfig<M: Embedding> {
    request: ProtocolSpec<M>,
    first: Option<Node<M>>,
    targets: Vec<Node<Identity<M::Target>>>,
    joint_masks: Option<JointMaskConfig<M::Target>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Node<M: Embedding> {
    code: IrsConfig<M>,
    sumcheck_pow: PowConfig,
    switch: Option<(usize, PowConfig)>,
}

impl<M: Embedding> Serialize for ProtocolConfig<M> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.request.serialize(serializer)
    }
}
impl<'de, M: Embedding + Default> Deserialize<'de> for ProtocolConfig<M> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let request = ProtocolSpec::<M>::deserialize(deserializer)?;
        if request.embedding.0 != M::default() {
            return Err(serde::de::Error::custom(
                "configuration requires the default embedding",
            ));
        }
        Self::derive(request.security, request.tuning).map_err(serde::de::Error::custom)
    }
}

impl<M: Embedding> ProtocolConfig<M> {
    pub const fn security(&self) -> &SecuritySpec {
        &self.request.security
    }
    pub const fn tuning(&self) -> &TuningSpec {
        &self.request.tuning
    }
    pub fn padded_vector_size(&self) -> usize {
        self.first.as_ref().map_or_else(
            || self.final_commit().vector_size(),
            |node| node.code.vector_size(),
        )
    }
    pub fn first_round(&self) -> Option<RoundConfig<'_, M>> {
        self.first
            .as_ref()
            .map(|node| node.round(&self.targets[0].code, 0, self.security().mode.is_zk()))
    }
    pub fn tail_rounds(
        &self,
    ) -> impl ExactSizeIterator<Item = RoundConfig<'_, Identity<M::Target>>> {
        self.targets
            .windows(2)
            .enumerate()
            .map(|(i, pair)| pair[0].round(&pair[1].code, i + 1, self.security().mode.is_zk()))
    }
    pub const fn has_rounds(&self) -> bool {
        self.first.is_some()
    }
    pub fn num_rounds(&self) -> usize {
        self.targets.len() - 1 + usize::from(self.has_rounds())
    }
    pub fn final_commit(&self) -> &IrsConfig<Identity<M::Target>> {
        &self.targets.last().expect("final commitment").code
    }
    /// Standard terminal primitive; Zook uses only its sumcheck after blinding the final claim.
    pub(crate) fn basecase(&self) -> BasecaseConfig<'_, M::Target> {
        let node = self.targets.last().expect("final commitment");
        BasecaseConfig::new(
            &node.code,
            node.sumcheck(node.code.vector_size().ilog2() as usize, false),
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
        if !self.security().mode.is_zk() {
            return None;
        }
        let mask = if index == 0 && self.has_rounds() {
            WitnessMaskConfig::for_code(&self.first.as_ref().unwrap().code, 0)
        } else {
            let node = &self.targets[index - usize::from(self.has_rounds())];
            WitnessMaskConfig::for_code(&node.code, usize::from(index != 0))
        };
        Some(mask.expect("construction checked witness mask dimensions"))
    }
}

impl<M: Embedding> Node<M> {
    fn round<'a>(
        &'a self,
        target: &'a IrsConfig<Identity<M::Target>>,
        index: usize,
        is_zk: bool,
    ) -> RoundConfig<'a, M> {
        let folds = self.code.interleaving_depth().ilog2() as usize;
        let (ood, pow) = self.switch.expect("nonterminal commitment");
        let masks = is_zk.then(|| RoundMasks {
            witness: WitnessMaskConfig::for_code(&self.code, usize::from(index != 0))
                .expect("construction checked witness mask dimensions"),
            sumcheck_coefficients: mask_coefficients(folds).expect("bounded folding factor"),
        });
        RoundConfig {
            index,
            sumcheck: self.sumcheck(folds, is_zk),
            code_switch: CodeSwitchConfig::new(
                &self.code,
                target,
                ood,
                CodeSwitchMode::Standard,
                pow,
            ),
            masks,
        }
    }
    fn sumcheck(&self, folds: usize, is_zk: bool) -> SumcheckConfig<M::Target> {
        let mode = if is_zk {
            SumcheckMode::ZeroKnowledge {
                mask_length: SumcheckMaskLen::new(3),
            }
        } else {
            SumcheckMode::Standard
        };
        SumcheckConfig::new(self.code.vector_size(), self.sumcheck_pow, folds, mode)
    }
}

/// A temporary execution view; each borrowed IRS has one owner in ProtocolConfig.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoundConfig<'a, M: Embedding> {
    pub index: usize,
    pub sumcheck: SumcheckConfig<M::Target>,
    pub code_switch: CodeSwitchConfig<'a, M>,
    pub masks: Option<RoundMasks>,
}

fn embedding_degree<M: Embedding>() -> Result<usize, DeriveError> {
    let source = M::Source::extension_degree();
    let target = M::Target::extension_degree();
    if source == 0 || !target.is_multiple_of(source) {
        return Err(DeriveError::InvalidMaskLayout);
    }
    usize::try_from(target / source).map_err(|_| DeriveError::SizeOverflow)
}
