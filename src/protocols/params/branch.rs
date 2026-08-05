//! Standard-vs-ZK branching used to thread mode through the build pipeline.
//!
//! [`Branch<T>`] is the shared shape: a transient choice between the standard
//! path (no payload) and the zero-knowledge path (payload `T`). Concrete
//! pipeline stages alias it with the payload they carry:
//!
//! - [`RoundBuildMode`] — input to [`super::build_round::build_round_config`].
//! - [`OodMode`] — input to OOD-bound helpers.
//! - [`SolveMode`] — input to the per-sub-protocol solvers (`sumcheck`, `code_switch`).
//!
//! Sharing one enum gives us a free [`Branch::map`] for stage-to-stage
//! payload conversions, replacing one-off `to_ood_mode`-style helpers.

use crate::protocols::params::{
    protocol_config::MaskOracleInfo,
    spec::{LogInvRate, ZkSpec},
};

/// Standard (no payload) vs. zero-knowledge (payload `T`).
#[derive(Clone, Copy, Debug)]
pub enum Branch<T> {
    Standard,
    ZeroKnowledge(T),
}

impl<T> Branch<T> {
    pub const fn is_zk(&self) -> bool {
        matches!(self, Self::ZeroKnowledge(_))
    }

    /// Transform the ZK payload, leaving `Standard` unchanged. Replaces
    /// per-stage `to_*` conversion helpers.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Branch<U> {
        match self {
            Self::Standard => Branch::Standard,
            Self::ZeroKnowledge(t) => Branch::ZeroKnowledge(f(t)),
        }
    }

    pub const fn as_ref(&self) -> Branch<&T> {
        match self {
            Self::Standard => Branch::Standard,
            Self::ZeroKnowledge(t) => Branch::ZeroKnowledge(t),
        }
    }
}

/// Payload carried by [`RoundBuildMode::ZeroKnowledge`] — references the
/// `SecuritySpec` (so its lifetime threads through) plus the planner-chosen
/// `C_zk` rate.
#[derive(Clone, Copy, Debug)]
pub struct RoundBuildPayload<'a> {
    pub zk_spec: ZkSpec<'a>,
    pub c_zk_log_inv_rate: LogInvRate,
}

/// Mode-dispatch input for [`super::build_round::build_round_config`].
pub type RoundBuildMode<'a> = Branch<RoundBuildPayload<'a>>;

/// Mode flag for the OOD security bound. Payload is the `C_zk` log-inverse
/// rate; formulas coerce to `f64` at the point of use.
pub type OodMode = Branch<LogInvRate>;

/// Solver-input mode for the per-round sumcheck and code-switch builders.
pub type SolveMode = Branch<MaskOracleInfo>;
