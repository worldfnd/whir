//! Parameter selection for HVZK-WHIR.
//!
//! Soundness and ZK bound derivations (referred to in submodule comments as
//! "the bounds doc, §N") live at
//! <https://hackmd.io/@1q1q-TiuQN6fAkxaN41u-Q/ryBoT_UA-e>.

pub(crate) mod basecase;
pub(crate) mod bounds;
pub(crate) mod branch;
pub(crate) mod build_round;
pub(crate) mod code_switch;
pub(crate) mod derive;
pub(crate) mod error;
pub(crate) mod irs_commit;
pub(crate) mod layout;
pub(crate) mod mask_proximity;
pub(crate) mod protocol_config;
pub(crate) mod regime;
pub(crate) mod solved;
pub(crate) mod spec;
pub(crate) mod sumcheck;

#[cfg(test)]
pub(crate) mod test_utils;

pub use branch::{Branch, SolveMode};
pub use error::{ChainSource, ChainTarget, DeriveError, Pow};
pub use layout::LayoutError;
pub use protocol_config::{
    BasecasePlan, MaskOracleConfig, MaskOracleInfo, ProtocolConfig, RoundConfig, RoundMode,
};
pub use solved::Solved;
pub use spec::{
    DecodingRegime, FoldingFactor, ListSize, LogInvRate, MaskCodeMessageLen, Mode, OodSampleBudget,
    ParseDecodingRegimeError, PowBudget, RoundContext, SecuritySpec, TuningSpec, ZkSpec,
    DEFAULT_POW_BUDGET_BITS,
};
