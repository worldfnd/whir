//! Parameter selection for WHIR and Zook.
//!
//! `derive` constructs concrete configurations; `security` evaluates their bounds.
//! `search` selects rates through the same construction path.
//! Mathematical references use the pinned [ProveKit whitepaper].
//!
//! [ProveKit whitepaper]: https://github.com/worldfnd/provekit/tree/6605cf5242567e17899a26c27db829c7ce9eb478/whitepaper

pub(crate) mod config;
pub(crate) mod derive;
pub(crate) mod error;
pub(crate) mod masks;
pub(crate) mod pair;
pub(crate) mod search;
pub(crate) mod security;
pub(crate) mod spec;

#[cfg(test)]
mod parameter_regressions;
#[cfg(test)]
pub(crate) mod test_utils;

pub use config::{BasecasePlan, ProtocolConfig, RoundConfig};
pub use derive::LayoutError;
pub use error::{ChainSource, ChainTarget, DeriveError, SecurityEvent, ZookSecurityError};
pub use masks::{JointMaskConfig, RoundMasks, WitnessMaskConfig};
pub use security::{
    zook_mask_binding_bits, SecurityCheck, ZookFinalOpeningSoundness, ZookRoundSoundness,
};
pub use spec::{
    DecodingRegime, FoldingFactor, KneeWeight, Mode, ParseDecodingRegimeError, PowBudget,
    RateSchedule, SecuritySpec, TuningSpec, DEFAULT_POW_BUDGET_BITS,
};
