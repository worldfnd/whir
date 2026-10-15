//! Standard-vs-ZK branching used to thread mode through the build pipeline.
//!
//! [`Branch<T>`] is the shared shape: a transient choice between the standard
//! path (no payload) and the zero-knowledge path (payload `T`). Concrete
//! pipeline stages alias it with the payload they carry:
//!
//! - [`RoundBuildMode`] — input to [`super::build_round::build_round_config`].
//! - [`SolveMode`] — input to the per-sub-protocol solvers (`sumcheck`, `code_switch`).
//!
//! Sharing one enum gives us a free [`Branch::map`] for stage-to-stage
//! payload conversions, replacing one-off `to_ood_mode`-style helpers.

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

/// Mode used while building round configurations.
pub type RoundBuildMode = Branch<()>;

/// Mode used by the sub-protocol parameter solvers.
pub type SolveMode = Branch<()>;
