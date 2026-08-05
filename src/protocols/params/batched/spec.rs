//! Caller-facing inputs for the batched protocol: bundle shapes plus the shared
//! schedule tuning. Mirrors [`super::super::spec`] for the single-track protocol.

use serde::{Deserialize, Serialize};

use crate::protocols::params::{
    error::DeriveError,
    spec::{FoldingFactor, RateSchedule},
};

/// User-facing bundle shape: `num_polys` polys of identical `length` committed under one IRS commitment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BundleSpec {
    /// Per-poly length in field elements (must be a power of two).
    pub length: usize,
    /// Number of polys committed together in this bundle.
    pub num_polys: usize,
}

/// Caller-supplied tuning for the batched protocol.
///
/// The merge topology is derived automatically from the bundle sizes:
/// equal-sized bundles merge at round 0, and a size gap is closed by one
/// auto-computed pre-merge round before a round-0 merge (see [`super::derive`]).
/// Must use [`FoldingFactor::Constant`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchedTuningSpec {
    pub bundles: Vec<BundleSpec>,
    pub starting_log_inv_rate: u32,
    /// Folding factor for the shared rounds (i.e. inner.rounds[]); also the budget
    /// for the pre-merge convergence fold.
    pub folding_factor: FoldingFactor,
    pub rate_schedule: RateSchedule,
}

/// Validate per-bundle input shape (power-of-two sizes, matching `num_polys`).
/// Returns the agreed-upon `num_polys` on success.
pub(super) fn validate_bundle_inputs(bundles: &[BundleSpec]) -> Result<usize, DeriveError> {
    if bundles.is_empty() {
        return Err(DeriveError::BatchedUnsupported {
            reason: "batched tuning must declare at least one bundle".into(),
        });
    }
    for (i, b) in bundles.iter().enumerate() {
        if !b.length.is_power_of_two() {
            return Err(DeriveError::BatchedUnsupported {
                reason: format!("bundles[{i}].length = {} is not a power of 2", b.length),
            });
        }
        if b.num_polys == 0 {
            return Err(DeriveError::BatchedUnsupported {
                reason: format!("bundles[{i}].num_polys must be >= 1"),
            });
        }
        if !b.num_polys.is_power_of_two() {
            return Err(DeriveError::BatchedUnsupported {
                reason: format!(
                    "bundles[{i}].num_polys = {} is not a power of 2",
                    b.num_polys
                ),
            });
        }
    }
    let canonical = bundles[0].num_polys;
    for (i, b) in bundles.iter().enumerate().skip(1) {
        if b.num_polys != canonical {
            return Err(DeriveError::BatchedUnsupported {
                reason: format!(
                    "bundles[{i}].num_polys = {} differs from bundles[0].num_polys = {}; \
                     mixed poly counts need depth canonicalisation (follow-up work)",
                    b.num_polys, canonical,
                ),
            });
        }
    }
    Ok(canonical)
}
