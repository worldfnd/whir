//! Parameter selection for HVZK-WHIR.
//!
//! Soundness and ZK bound derivations (referred to in submodule comments as
//! "the bounds doc, §N") live at
//! <https://hackmd.io/@1q1q-TiuQN6fAkxaN41u-Q/ryBoT_UA-e>.
//!
//! # Module map
//!
//! Selection flows `spec` → `derive` → `config`:
//!
//! - **inputs**: `spec` — the caller's `SecuritySpec` + `TuningSpec`.
//! - **pipeline**: `derive` (entry) walks the witness shape — `layout` lays out
//!   the per-round skeleton and `build_round` turns each shape into a config;
//!   `branch`, `regime`, and `adaptive` thread mode / decoding-regime / rate
//!   choices through it.
//! - **sub-protocol solvers** (one per runtime protocol, each owning its own
//!   analytic-error formula): `sumcheck`, `code_switch`, `mask_proximity`,
//!   `basecase`, `irs_commit`.
//! - **outputs**: `config` (the assembled `ProtocolConfig` + per-slot
//!   validation) and `solved` (the analytic-floor wrapper solvers record).
//! - **support**: `error` (`DeriveError`) and `bounds` (analytic primitives
//!   used throughout).
//! - **batched**: `batched` — the multi-bundle extension layered over the above.

// inputs
pub(crate) mod spec;

// derivation pipeline (spec → ProtocolConfig)
pub(crate) mod adaptive;
pub(crate) mod branch;
pub(crate) mod build_round;
pub(crate) mod derive;
pub(crate) mod layout;
pub(crate) mod regime;

// per-sub-protocol parameter solvers
pub(crate) mod basecase;
pub(crate) mod code_switch;
pub(crate) mod irs_commit;
pub(crate) mod mask_proximity;
pub(crate) mod sumcheck;

// outputs
pub(crate) mod config;
pub(crate) mod solved;

// support
pub(crate) mod bounds;
pub(crate) mod error;

// multi-bundle batching (extends the above)
pub mod batched;

#[cfg(test)]
pub(crate) mod test_utils;

// Re-exports — alphabetical by source module; see the module map above for roles.

pub use batched::{
    BatchedProtocolConfig, BatchedTuningSpec, BundleConfig, BundleSpec, MergeSchedule, RoundJoin,
};
pub use config::{MaskOracleConfig, ProtocolConfig, RoundConfig};
pub use error::{ChainSource, ChainTarget, DeriveError, Pow, RoundSlot};
pub use layout::LayoutError;
pub use spec::{
    DecodingRegime, FoldingFactor, KneeWeight, ListSize, LogInvRate, MaskCodeMessageLen, Mode,
    OodSampleBudget, ParseDecodingRegimeError, PowBudget, RateSchedule, RoundContext, SecuritySpec,
    TuningSpec, ZkSpec, DEFAULT_POW_BUDGET_BITS,
};
