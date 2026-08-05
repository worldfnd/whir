//! Pre-merge planning: converge bundles of differing sizes onto a single merge
//! point so one selector merge can gather them. The batched analogue of
//! [`super::super::layout`], which lays out the single-track round shapes.

use crate::protocols::params::error::DeriveError;

/// Per-bundle pre-merge fold schedule: entry `b` holds bundle `b`'s convergence
/// folds (empty when it already sits at the merge point). Every bundle reaches
/// the common merge point `2^(max_log − max_fold)` in at most one pre-merge
/// round. [`super::derive`] computes these from the bundle sizes; callers no
/// longer specify folds.
pub(super) fn auto_pre_merge_plan(
    bundle_lengths: &[usize],
    max_fold: usize,
) -> Result<Vec<Vec<usize>>, DeriveError> {
    if bundle_lengths.is_empty() {
        return Err(DeriveError::BatchedUnsupported {
            reason: "auto_pre_merge_plan: at least one bundle required".into(),
        });
    }
    if max_fold == 0 {
        return Err(DeriveError::BatchedUnsupported {
            reason: "auto_pre_merge_plan: max_fold must be >= 1".into(),
        });
    }

    let mut log_sizes: Vec<usize> = Vec::with_capacity(bundle_lengths.len());
    for (i, &l) in bundle_lengths.iter().enumerate() {
        if !l.is_power_of_two() {
            return Err(DeriveError::BatchedUnsupported {
                reason: format!(
                    "auto_pre_merge_plan: bundle_lengths[{i}] = {l} is not a power of two"
                ),
            });
        }
        log_sizes.push(l.trailing_zeros() as usize);
    }
    let max_log = *log_sizes.iter().max().expect("non-empty");
    let min_log = *log_sizes.iter().min().expect("non-empty");

    // Same size across bundles: round-0 merge is strictly cheaper than per-bundle pre-merge.
    if max_log == min_log {
        return Ok(vec![Vec::new(); bundle_lengths.len()]);
    }

    let log_merge = max_log.saturating_sub(max_fold);

    let mut per_bundle: Vec<Vec<usize>> = Vec::with_capacity(bundle_lengths.len());
    for (i, &log_size) in log_sizes.iter().enumerate() {
        if log_size < log_merge {
            return Err(DeriveError::BatchedUnsupported {
                reason: format!(
                    "auto_pre_merge_plan: bundles[{i}] log2(length) = {log_size} is below \
                     merge point log = {log_merge}; bundle is too small to reach the merge \
                     point. Adjust max_fold ({max_fold}) so the merge point sits below \
                     every bundle"
                ),
            });
        }
        let fold = log_size - log_merge;
        if fold == 0 {
            per_bundle.push(Vec::new());
        } else {
            per_bundle.push(vec![fold]);
        }
    }

    Ok(per_bundle)
}
