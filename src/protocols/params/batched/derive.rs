//! Orchestrates [`BatchedProtocolConfig`] derivation.
//!
//! The merge topology is fixed: every bundle converges to a single round-0 merge.
//! `derive` auto-computes one rate-pinned pre-merge round per smaller bundle (the
//! size budget is `folding_factor`), derives the shared inner config at the merge
//! point, then lets [`super::placement::place`] assemble the configs + schedule.
//!
//! The two batched knobs land in separate stages:
//!   - the **rate schedule** is forwarded untouched to [`ProtocolConfig::derive`]
//!     in [`derive_inner`] — the batch layer never reasons about rates;
//!   - the **size convergence** is owned by [`super::pre_merge::auto_pre_merge_plan`]
//!     + [`super::placement`].

use ark_ff::Field;

use crate::{
    algebra::embedding::Embedding,
    protocols::params::{
        batched::{
            config::BatchedProtocolConfig,
            placement::{merge_point_vsize, place},
            pre_merge::auto_pre_merge_plan,
            spec::{validate_bundle_inputs, BatchedTuningSpec},
        },
        config::ProtocolConfig,
        error::DeriveError,
        spec::{FoldingFactor, SecuritySpec, TuningSpec},
    },
};

impl<M: Embedding + Default> BatchedProtocolConfig<M> {
    /// Derive a [`BatchedProtocolConfig`] from a security spec and a batched tuning.
    // Mirrors `ProtocolConfig::derive`, which also takes an owned `SecuritySpec`
    // config input; the body only needs to borrow it.
    #[allow(clippy::needless_pass_by_value)]
    pub fn derive(spec: SecuritySpec, tuning: BatchedTuningSpec) -> Result<Self, DeriveError>
    where
        M::Target: Field,
    {
        let canonical_num_polys = validate_bundle_inputs(&tuning.bundles)?;

        // The pre-merge convergence folds down by the (constant) folding factor.
        let FoldingFactor::Constant(fold) = tuning.folding_factor else {
            return Err(DeriveError::BatchedUnsupported {
                reason: format!(
                    "batched protocol requires FoldingFactor::Constant; got {:?}",
                    tuning.folding_factor
                ),
            });
        };

        let bundle_lengths: Vec<usize> = tuning.bundles.iter().map(|b| b.length).collect();
        let folds = auto_pre_merge_plan(&bundle_lengths, fold)?;

        let merge_vsize = merge_point_vsize(&tuning, canonical_num_polys, &folds)?;
        let inner = derive_inner::<M>(&spec, merge_vsize, &tuning)?;
        let (bundle_configs, schedule) =
            place(&spec, &tuning, canonical_num_polys, &folds, &inner)?;

        Self::try_new(inner, schedule, bundle_configs, tuning)
    }
}

/// Derive the shared single-track inner config at the merge point. The batch layer
/// forwards `rate_schedule` here untouched and lets [`ProtocolConfig::derive`] own
/// rate planning.
fn derive_inner<M: Embedding + Default>(
    spec: &SecuritySpec,
    vector_size: usize,
    tuning: &BatchedTuningSpec,
) -> Result<ProtocolConfig<M>, DeriveError>
where
    M::Target: Field,
{
    let inner_tuning = TuningSpec {
        vector_size,
        starting_log_inv_rate: tuning.starting_log_inv_rate,
        folding_factor: tuning.folding_factor.clone(),
        rate_schedule: tuning.rate_schedule,
    };
    ProtocolConfig::<M>::derive(spec.clone(), inner_tuning)
}
