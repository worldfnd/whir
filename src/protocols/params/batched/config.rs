//! Batched protocol configuration: wraps [`ProtocolConfig`] with per-bundle IRS configs
//! and a merge schedule for the batched zook orchestrator. Mirrors
//! [`super::super::config`]: it owns the derived output type plus every
//! static and security-target invariant check. Derivation lives in [`super::derive`].

use ark_ff::Field;
use serde::{Deserialize, Serialize};

use crate::{
    algebra::{embedding::Embedding, fields::FieldWithSize},
    bits::Bits,
    protocols::{
        irs_commit,
        params::{
            batched::spec::{BatchedTuningSpec, BundleSpec},
            config::{ProtocolConfig, RoundConfig},
            error::{DeriveError, Pow},
            spec::SecuritySpec,
        },
        proof_of_work,
        zook::batched::{bundle, selector::merge as selector_merge},
    },
};

/// Per-bundle derived configuration. Populated by the scheduler.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct BundleConfig<M: Embedding> {
    pub spec: BundleSpec,
    /// Padded length the scheduler will commit at; `None` means no padding.
    pub pad_to: Option<usize>,
    /// Round index at which this bundle joins the shared active list; `None` means basecase.
    /// For bundles with pre-merge rounds, this is the round reached after all pre-merge folds.
    pub join_round: Option<usize>,
    /// IRS commit config for this bundle (`interleaving_depth` widened to `num_polys × base`).
    pub irs_config: irs_commit::Config<M>,
    /// Per-bundle pre-merge rounds run before joining the shared schedule. Empty when the
    /// bundle's natural commit size lands on a shared schedule round entry directly.
    #[serde(default = "Vec::new")]
    pub pre_merge_rounds: Vec<RoundConfig<M>>,
}

/// One scheduled merge event: which bundles join at which round, plus the selector merge config.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RoundJoin<F: Field> {
    /// `Some(r)` for a join at the entry of WHIR round `r`; `None` for basecase.
    pub round_index: Option<usize>,
    /// Indices of joining bundles in ascending order.
    pub bundle_indices: Vec<usize>,
    /// Active block count at the merge point (`bundle_indices.len()` at the first non-empty
    /// round, `+1` thereafter for the carried block).
    pub t: usize,
    pub selector: selector_merge::Config<F>,
}

/// Merge schedule — totally ordered list of joins across rounds and (optionally) basecase.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct MergeSchedule<F: Field> {
    pub joins: Vec<RoundJoin<F>>,
}

impl<F: Field> MergeSchedule<F> {
    /// Empty schedule.
    pub const fn empty() -> Self {
        Self { joins: Vec::new() }
    }

    /// Total number of bundles referenced across all joins.
    pub fn bundle_count(&self) -> usize {
        self.joins.iter().map(|j| j.bundle_indices.len()).sum()
    }

    /// Returns the basecase join, if scheduled.
    pub fn basecase_join(&self) -> Option<&RoundJoin<F>> {
        self.joins.iter().find(|j| j.round_index.is_none())
    }

    /// Returns the join scheduled at round `r`, if any.
    pub fn join_at(&self, round: usize) -> Option<&RoundJoin<F>> {
        self.joins.iter().find(|j| j.round_index == Some(round))
    }
}

/// Top-level batched protocol configuration: an inner [`ProtocolConfig`] plus per-bundle IRS
/// configs and a merge schedule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct BatchedProtocolConfig<M: Embedding> {
    pub(super) inner: ProtocolConfig<M>,
    pub(super) schedule: MergeSchedule<M::Target>,
    pub(super) bundle_configs: Vec<BundleConfig<M>>,
    pub(super) tuning: BatchedTuningSpec,
}

impl<M: Embedding> BatchedProtocolConfig<M> {
    /// Construct from already-derived pieces and validate all static invariants.
    pub fn try_new(
        inner: ProtocolConfig<M>,
        schedule: MergeSchedule<M::Target>,
        bundle_configs: Vec<BundleConfig<M>>,
        tuning: BatchedTuningSpec,
    ) -> Result<Self, DeriveError>
    where
        M::Target: Field,
    {
        let cfg = Self::new_unchecked(inner, schedule, bundle_configs, tuning);
        cfg.validate_static_invariants()?;
        Ok(cfg)
    }

    pub(crate) const fn new_unchecked(
        inner: ProtocolConfig<M>,
        schedule: MergeSchedule<M::Target>,
        bundle_configs: Vec<BundleConfig<M>>,
        tuning: BatchedTuningSpec,
    ) -> Self {
        Self {
            inner,
            schedule,
            bundle_configs,
            tuning,
        }
    }

    /// Inner single-track config driving per-round WHIR sub-protocols.
    pub const fn inner(&self) -> &ProtocolConfig<M> {
        &self.inner
    }

    /// Merge schedule decided at derive time.
    pub const fn schedule(&self) -> &MergeSchedule<M::Target> {
        &self.schedule
    }

    /// Per-bundle derived configs, in caller's bundle order.
    pub fn bundle_configs(&self) -> &[BundleConfig<M>] {
        &self.bundle_configs
    }

    pub const fn tuning(&self) -> &BatchedTuningSpec {
        &self.tuning
    }

    /// Shape-match `other` against the source IRS of the inner round at
    /// `join_round` (0 = base `first_round`, ≥1 = ext `tail_rounds`). Field-free,
    /// so it spans the head/tail type split.
    fn join_source_matches<A: Embedding>(
        &self,
        join_round: usize,
        other: &irs_commit::Config<A>,
    ) -> bool {
        if join_round == 0 {
            self.inner.first_round().is_some_and(|first| {
                irs_shape_matches(other, first.code_switch().config().source())
            })
        } else {
            self.inner
                .tail_rounds()
                .get(join_round - 1)
                .is_some_and(|round| {
                    irs_shape_matches(other, round.code_switch().config().source())
                })
        }
    }

    #[allow(clippy::too_many_lines)]
    fn validate_static_invariants(&self) -> Result<(), DeriveError>
    where
        M::Target: Field,
    {
        self.inner.validate()?;

        if self.bundle_configs.is_empty() {
            return Err(batched_invariant("at least one bundle config is required"));
        }
        if self.tuning.bundles.len() != self.bundle_configs.len() {
            return Err(batched_invariant(format!(
                "tuning bundle count ({}) != bundle config count ({})",
                self.tuning.bundles.len(),
                self.bundle_configs.len()
            )));
        }
        for (i, (spec, cfg)) in self
            .tuning
            .bundles
            .iter()
            .zip(&self.bundle_configs)
            .enumerate()
        {
            if *spec != cfg.spec {
                return Err(batched_invariant(format!(
                    "bundle_configs[{i}].spec does not match tuning.bundles[{i}]"
                )));
            }
        }

        if self.schedule.joins.is_empty() {
            return Err(batched_invariant(
                "merge schedule must contain at least one join",
            ));
        }
        if self.schedule.basecase_join().is_some() {
            return Err(batched_invariant(
                "basecase joins are not supported by the batched prover/verifier",
            ));
        }

        let mut seen_bundles = vec![false; self.bundle_configs.len()];
        let mut previous_round = None;
        let mut carrier_exists = false;
        for (join_pos, join) in self.schedule.joins.iter().enumerate() {
            let Some(round) = join.round_index else {
                return Err(batched_invariant("basecase join encountered in schedule"));
            };
            if round >= self.inner.num_rounds() {
                return Err(batched_invariant(format!(
                    "join {join_pos} targets round {round}, but inner has {} rounds",
                    self.inner.num_rounds()
                )));
            }
            if join_pos == 0 && round != 0 {
                return Err(batched_invariant(
                    "first join must occur at round 0 so the round walk has an active block",
                ));
            }
            if let Some(prev) = previous_round {
                if round <= prev {
                    return Err(batched_invariant(format!(
                        "join rounds must be strictly increasing; join {join_pos} has {round} after {prev}"
                    )));
                }
            }
            previous_round = Some(round);

            if join.bundle_indices.is_empty() {
                return Err(batched_invariant(format!(
                    "join {join_pos} has no bundle indices"
                )));
            }
            let expected_t = join.bundle_indices.len() + usize::from(carrier_exists);
            if join.t != expected_t {
                return Err(batched_invariant(format!(
                    "join {join_pos} has t = {}, expected {expected_t}",
                    join.t
                )));
            }
            if join.selector.t() != join.t {
                return Err(batched_invariant(format!(
                    "join {join_pos} selector.t = {}, expected {}",
                    join.selector.t(),
                    join.t
                )));
            }
            let expected_len = self
                .inner
                .round_source_vector_size(round)
                .expect("round < num_rounds checked above");
            if join.selector.message_length() != expected_len {
                return Err(batched_invariant(format!(
                    "join {join_pos} selector length = {}, expected round {round} source vector_size {expected_len}",
                    join.selector.message_length()
                )));
            }

            for &bundle_idx in &join.bundle_indices {
                let Some(seen) = seen_bundles.get_mut(bundle_idx) else {
                    return Err(batched_invariant(format!(
                        "join {join_pos} references out-of-range bundle index {bundle_idx}"
                    )));
                };
                if *seen {
                    return Err(batched_invariant(format!(
                        "bundle index {bundle_idx} appears in more than one join"
                    )));
                }
                *seen = true;
                let cfg = &self.bundle_configs[bundle_idx];
                if cfg.join_round != Some(round) {
                    return Err(batched_invariant(format!(
                        "bundle_configs[{bundle_idx}].join_round = {:?}, but schedule places it at {round}",
                        cfg.join_round
                    )));
                }
            }

            carrier_exists = true;
        }
        if seen_bundles.iter().any(|seen| !seen) {
            return Err(batched_invariant(
                "merge schedule does not cover every bundle exactly once",
            ));
        }

        for (i, cfg) in self.bundle_configs.iter().enumerate() {
            let Some(join_round) = cfg.join_round else {
                return Err(batched_invariant(format!(
                    "bundle_configs[{i}] has unsupported basecase join"
                )));
            };
            if cfg.pre_merge_rounds.is_empty() {
                // A bundle that joins without pre-merge rounds carries a base
                // `M::Source` witness, so it can only match a round whose source
                // is `IrsConfig<M>` — i.e. round 0 (`first_round`). Tail rounds
                // are ext-only; joining one requires a prior code-switch.
                if !self.join_source_matches(join_round, &cfg.irs_config) {
                    return Err(batched_invariant(format!(
                        "bundle_configs[{i}].irs_config does not match join round {join_round} source"
                    )));
                }
                continue;
            }

            let first_source = cfg.pre_merge_rounds[0].code_switch().config().source();
            if !irs_shape_matches(&cfg.irs_config, first_source) {
                return Err(batched_invariant(format!(
                    "bundle_configs[{i}].irs_config does not match first pre-merge source"
                )));
            }
            for (k, window) in cfg.pre_merge_rounds.windows(2).enumerate() {
                let prev_target = window[0].code_switch().config().target();
                let next_source = window[1].code_switch().config().source();
                if !irs_shape_matches(prev_target, next_source) {
                    return Err(batched_invariant(format!(
                        "bundle_configs[{i}] pre-merge round {k} target does not match round {} source",
                        k + 1
                    )));
                }
            }
            let last_target = cfg
                .pre_merge_rounds
                .last()
                .expect("non-empty pre-merge")
                .code_switch()
                .config()
                .target();
            if !self.join_source_matches(join_round, last_target) {
                return Err(batched_invariant(format!(
                    "bundle_configs[{i}] final pre-merge target does not match join round {join_round} source"
                )));
            }
        }

        self.validate_selector_security_target_met()
    }

    /// Validate all security slots; intra-bundle γ-RLC slots depend on runtime claim counts.
    pub fn validate_security_target_met_for_claims(
        &self,
        bundle_claim_counts: &[usize],
    ) -> Result<(), DeriveError>
    where
        M::Target: Field,
    {
        self.inner.validate_security_target_met()?;
        self.validate_selector_security_target_met()?;

        if bundle_claim_counts.len() != self.bundle_configs.len() {
            return Err(DeriveError::BatchedUnsupported {
                reason: format!(
                    "bundle claim count length ({}) != bundle config length ({})",
                    bundle_claim_counts.len(),
                    self.bundle_configs.len()
                ),
            });
        }

        let spec = self.inner.security();
        let field_bits = M::Target::field_size_bits();
        for (index, &total_claims) in bundle_claim_counts.iter().enumerate() {
            if total_claims <= 1 {
                continue;
            }
            let analytic = bundle::analytic_error_bits(total_claims, field_bits);
            validate_batched_soundness_slot(
                spec,
                Pow::BatchedIntraBundleRlc { index },
                analytic,
                proof_of_work::Config::none(),
            )?;
        }
        Ok(())
    }

    fn validate_selector_security_target_met(&self) -> Result<(), DeriveError>
    where
        M::Target: Field,
    {
        let spec = self.inner.security();
        let field_bits = M::Target::field_size_bits();
        for (index, join) in self.schedule.joins.iter().enumerate() {
            if join.selector.selector_dim() == 0 {
                continue;
            }
            let analytic = join.selector.analytic_error_bits(field_bits);
            let pow = join.selector.selector().sumcheck().round_pow();
            validate_batched_soundness_slot(
                spec,
                Pow::BatchedSelectorMerge { index },
                analytic,
                pow,
            )?;
        }
        Ok(())
    }
}

fn validate_batched_soundness_slot(
    spec: &SecuritySpec,
    pow: Pow,
    analytic: Bits,
    pow_config: proof_of_work::Config,
) -> Result<(), DeriveError> {
    const EPS: f64 = 1e-3;
    let pow_bits = pow_config.difficulty();
    let max = Bits::new(f64::from(spec.pow_budget.bits()));
    if pow_bits > max {
        return Err(DeriveError::PowBudgetExceeded {
            pow,
            required: pow_bits,
            max,
        });
    }

    let target = Bits::new(f64::from(spec.target_security_bits));
    if analytic + pow_bits + Bits::new(EPS) < target {
        return Err(DeriveError::SecurityTargetNotMet {
            pow,
            analytic,
            pow_bits,
            target,
        });
    }
    Ok(())
}

fn batched_invariant(reason: impl Into<String>) -> DeriveError {
    DeriveError::BatchedUnsupported {
        reason: reason.into(),
    }
}

pub(super) const fn irs_shape_matches<A: Embedding, B: Embedding>(
    lhs: &irs_commit::Config<A>,
    rhs: &irs_commit::Config<B>,
) -> bool {
    lhs.vector_size() == rhs.vector_size()
        && lhs.interleaving_depth() == rhs.interleaving_depth()
        && lhs.num_vectors() == rhs.num_vectors()
        && lhs.codeword_length() == rhs.codeword_length()
        && lhs.mask_length() == rhs.mask_length()
}
