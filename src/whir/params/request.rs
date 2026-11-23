//! User-facing parameter request: mode, decoding regime, security, sizes, and rate schedule.
//! `ProtocolConfig::derive` turns a request into concrete round parameters.

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use crate::{
    algebra::embedding::Embedding, decoding::DecodingRegime, engines::EngineId, type_info::Typed,
};

/// Select standard WHIR or zero-knowledge WHIR with overlaid masks.
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

/// Security and layout choices. Rates and folds use base-two logarithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Params {
    pub mode: Mode,
    pub regime: DecodingRegime,
    /// Required security bits for each checked protocol step.
    pub bits: u32,
    /// Maximum grinding bits per step; zero-knowledge mode receives no grinding credit.
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
pub struct KneeWeight(pub(super) OrderedFloat<f64>);
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
    pub(super) fn step(self, rate: u32, folds: u32) -> u32 {
        let rate = rate.saturating_add(folds.saturating_sub(1));
        match self {
            Self::Capped(cap) => rate.min(cap),
            _ => rate,
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
