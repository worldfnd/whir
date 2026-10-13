//! Batched (multi-bundle) parameter selection.
//!
//! Mirrors the single-track params pipeline for `Vec<WitnessBundle>` inputs:
//!   - [`spec`]      — caller-facing inputs  (cf. [`super::spec`])
//!   - [`pre_merge`] — auto-compute the size-convergence folds (internal)
//!   - [`derive`]    — derivation orchestrator (cf. [`super::derive`])
//!   - [`placement`] — converge bundles to a round-0 merge
//!   - [`config`]    — derived output type + invariants (cf. [`super::config`])

mod config;
mod derive;
mod placement;
mod pre_merge;
mod spec;

#[cfg(test)]
mod tests;

pub use config::{BatchedProtocolConfig, BundleConfig, MergeSchedule, RoundJoin};
pub use spec::{BatchedTuningSpec, BundleSpec};
