//! Bundle placement: converge every bundle to a single round-0 merge.
//!
//! Equal-sized bundles already sit at the merge point (empty fold schedules) and
//! merge directly at round 0. A bundle that is smaller runs **one** auto-computed,
//! rate-pinned pre-merge round to fold down to the common merge point, then all
//! bundles join at round 0 under one selector merge. The fold schedules are
//! computed by [`super::pre_merge::auto_pre_merge_plan`] and the merge-point size
//! by [`merge_point_vsize`]; [`super::derive`] wires them together.

use ark_ff::Field;

use crate::{
    algebra::{embedding::Embedding, fields::FieldWithSize},
    protocols::{
        irs_commit,
        params::{
            batched::{
                config::{irs_shape_matches, BundleConfig, MergeSchedule, RoundJoin},
                spec::{BatchedTuningSpec, BundleSpec},
            },
            branch::{Branch, RoundBuildMode, RoundBuildPayload},
            build_round::build_round_config,
            config::ProtocolConfig,
            error::{grind_to_at, DeriveError, Pow, RoundSlot},
            layout::RoundShape,
            spec::{LogInvRate, SecuritySpec},
        },
        proof_of_work,
        zook::batched::selector::merge as selector_merge,
    },
};

/// The shared merge-point `vector_size` every bundle converges to after applying
/// its pre-merge folds. Errors if the folds don't land every bundle on one size.
pub(super) fn merge_point_vsize(
    tuning: &BatchedTuningSpec,
    canonical_num_polys: usize,
    folds_per_bundle: &[Vec<usize>],
) -> Result<usize, DeriveError> {
    let mut merge_vsizes: Vec<usize> = Vec::with_capacity(tuning.bundles.len());
    for (i, (bundle_spec, folds)) in tuning.bundles.iter().zip(folds_per_bundle).enumerate() {
        let log_len = bundle_spec.length.trailing_zeros() as usize;
        let total_fold: usize = folds.iter().copied().sum();
        if total_fold > log_len {
            return Err(DeriveError::BatchedUnsupported {
                reason: format!(
                    "bundles[{i}]: pre-merge folds sum {total_fold} exceeds log2(length) {log_len}",
                ),
            });
        }
        let post_length = bundle_spec.length >> total_fold;
        let post_vsize = canonical_num_polys
            .checked_mul(post_length)
            .ok_or_else(|| DeriveError::BatchedUnsupported {
                reason: format!("bundles[{i}]: num_polys × post-pre-merge length overflows usize"),
            })?;
        merge_vsizes.push(post_vsize);
    }

    let merge_vsize = merge_vsizes[0];
    for (i, &m) in merge_vsizes.iter().enumerate().skip(1) {
        if m != merge_vsize {
            return Err(DeriveError::BatchedUnsupported {
                reason: format!(
                    "bundles don't converge to a common merge point: \
                     bundle 0 → vector_size {merge_vsize}, bundle {i} → vector_size {m}"
                ),
            });
        }
    }
    Ok(merge_vsize)
}

/// Per-bundle configs (each with its pre-merge rounds) plus the single round-0
/// merge schedule produced by [`place`].
type Placement<M> = (
    Vec<BundleConfig<M>>,
    MergeSchedule<<M as Embedding>::Target>,
);

/// Build the per-bundle configs (each with its pre-merge rounds) and the single
/// round-0 merge schedule, against the inner derived at the merge point.
pub(super) fn place<M: Embedding + Default>(
    spec: &SecuritySpec,
    tuning: &BatchedTuningSpec,
    canonical_num_polys: usize,
    folds_per_bundle: &[Vec<usize>],
    inner: &ProtocolConfig<M>,
) -> Result<Placement<M>, DeriveError>
where
    M::Target: Field,
{
    if !inner.has_rounds() {
        return Err(DeriveError::BatchedUnsupported {
            reason: "merge point too small to support shared rounds (basecase-only inner); \
                     use a smaller folding factor or larger bundles"
                .into(),
        });
    }

    let mode: RoundBuildMode<'_> = spec.as_zk().map_or(Branch::Standard, |zk_spec| {
        Branch::ZeroKnowledge(RoundBuildPayload {
            zk_spec,
            c_zk_log_inv_rate: LogInvRate::new(4),
        })
    });
    let ctx = PreMergeCtx {
        spec,
        mode,
        canonical_num_polys,
        starting_log_inv_rate: tuning.starting_log_inv_rate,
        shared_first_fold: tuning.folding_factor.at_round(0) as u32,
        inner_first_source: inner
            .first_round()
            .expect("has_rounds checked above")
            .code_switch()
            .config()
            .source(),
    };

    let mut bundle_configs: Vec<BundleConfig<M>> = Vec::with_capacity(tuning.bundles.len());
    for (i, (bundle_spec, folds)) in tuning.bundles.iter().zip(folds_per_bundle).enumerate() {
        bundle_configs.push(build_pre_merge_bundle(&ctx, i, bundle_spec, folds)?);
    }

    // Everyone converges at round 0, so one selector merge gathers all bundles.
    let t = tuning.bundles.len();
    let selector = selector_merge_config_for_join::<M::Target>(
        spec,
        0,
        t,
        ctx.inner_first_source.vector_size(),
    )?;
    let joins = vec![RoundJoin {
        round_index: Some(0),
        bundle_indices: (0..t).collect(),
        t,
        selector,
    }];
    Ok((bundle_configs, MergeSchedule { joins }))
}

/// Grind a selector-merge PoW slot to close the gap to the security target
/// (pass-through PoW when the merge is trivial, i.e. `selector_dim == 0`).
fn selector_merge_config_for_join<F: Field>(
    spec: &SecuritySpec,
    join_index: usize,
    active_block_count: usize,
    message_length: usize,
) -> Result<selector_merge::Config<F>, DeriveError> {
    let preview = selector_merge::Config::<F>::new(
        active_block_count,
        message_length,
        proof_of_work::Config::none(),
    );
    let pow = if preview.selector_dim() == 0 {
        proof_of_work::Config::none()
    } else {
        let analytic = preview.analytic_error_bits(F::field_size_bits());
        grind_to_at(
            spec,
            analytic,
            Pow::BatchedSelectorMerge { index: join_index },
        )?
    };
    Ok(selector_merge::Config::<F>::new(
        active_block_count,
        message_length,
        pow,
    ))
}

/// Inputs shared across every bundle's pre-merge round body.
struct PreMergeCtx<'a, M: Embedding> {
    spec: &'a SecuritySpec,
    mode: RoundBuildMode<'a>,
    canonical_num_polys: usize,
    starting_log_inv_rate: u32,
    shared_first_fold: u32,
    inner_first_source: &'a irs_commit::Config<M>,
}

/// Build one bundle's `pre_merge_rounds` (folding it down to the merge point),
/// check the last target aligns with the shared schedule, and assemble its config.
/// Empty `folds` ⇒ no pre-merge rounds (the bundle is already at the merge point).
fn build_pre_merge_bundle<M: Embedding + Default>(
    ctx: &PreMergeCtx<'_, M>,
    bundle_idx: usize,
    bundle_spec: &BundleSpec,
    folds: &[usize],
) -> Result<BundleConfig<M>, DeriveError>
where
    M::Target: Field,
{
    let mut current_vsize = ctx.canonical_num_polys * bundle_spec.length;
    let mut pre_merge_rounds = Vec::with_capacity(folds.len());
    for (k, &fold) in folds.iter().enumerate() {
        debug_assert!(fold >= 1, "auto-computed pre-merge folds are always >= 1");
        // target_fold is the NEXT round's source fold — it determines this round's
        // target IRS shape so it aligns with the next consumer.
        let target_fold = if k + 1 < folds.len() {
            folds[k + 1] as u32
        } else {
            ctx.shared_first_fold
        };
        let shape = RoundShape {
            round_slot: RoundSlot::PreMerge(bundle_idx),
            source_vector_size: current_vsize,
            source_log_inv_rate: ctx.starting_log_inv_rate,
            source_folding_factor: fold as u32,
            target_folding_factor: target_fold,
            // Capped pin: target rate = source rate so all bundles' targets match.
            target_log_inv_rate: ctx.starting_log_inv_rate,
        };
        pre_merge_rounds.push(build_round_config::<M>(ctx.spec, &shape, ctx.mode)?);
        current_vsize >>= fold;
    }

    if let Some(last) = pre_merge_rounds.last() {
        assert_pre_merge_aligns(
            bundle_idx,
            last.code_switch().config().target(),
            ctx.inner_first_source,
        )?;
    }

    let irs_config = pre_merge_rounds.first().map_or_else(
        || ctx.inner_first_source.clone(),
        |first| first.code_switch().config().source().clone(),
    );

    Ok(BundleConfig {
        spec: *bundle_spec,
        pad_to: None,
        join_round: Some(0),
        irs_config,
        pre_merge_rounds,
    })
}

/// A bundle's last pre-merge target must structurally match the shared
/// `inner.rounds[0].source` so it can join cleanly at round 0. `last_target` is
/// `IrsConfig<Identity<M::Target>>` while `inner_first_source` is `IrsConfig<M>`,
/// so [`irs_shape_matches`] compares structural fields only.
fn assert_pre_merge_aligns<A: Embedding, B: Embedding>(
    bundle_idx: usize,
    last_target: &irs_commit::Config<A>,
    inner_first_source: &irs_commit::Config<B>,
) -> Result<(), DeriveError> {
    if irs_shape_matches(last_target, inner_first_source) {
        return Ok(());
    }
    Err(DeriveError::BatchedUnsupported {
        reason: format!(
            "bundles[{bundle_idx}] pre-merge target does not align with shared inner.rounds[0].source: \
             target(vec={}, ι={}, n={}, code={}, mask={}) vs \
             source(vec={}, ι={}, n={}, code={}, mask={})",
            last_target.vector_size(), last_target.interleaving_depth(),
            last_target.num_vectors(), last_target.codeword_length(),
            last_target.mask_length(),
            inner_first_source.vector_size(), inner_first_source.interleaving_depth(),
            inner_first_source.num_vectors(), inner_first_source.codeword_length(),
            inner_first_source.mask_length(),
        ),
    })
}
