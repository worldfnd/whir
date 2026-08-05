//! Adaptive rate planner.
//!
//! Picks per-round `target_log_inv_rate` values by enumerating candidates,
//! scoring them on a (prover_time_proxy, proof_size_proxy) pareto knee, and
//! returning the schedule chosen by [`KneeWeight`]. Driven by
//! [`super::layout::round_layout`] when [`super::spec::RateSchedule::Adaptive`]
//! is selected.

use std::collections::HashMap;

use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        fields::FieldWithSize,
    },
    protocols::{
        irs_commit::Config as IrsConfig,
        params::{
            basecase as basecase_params,
            branch::{Branch, RoundBuildMode},
            build_round::{build_mask_oracle, solve_t_ood},
            code_switch as code_switch_params,
            error::DeriveError,
            irs_commit as irs_params,
            layout::RoundShape,
            protocol_config::MaskOracleInfo,
            spec::{KneeWeight, Mode, OodSampleBudget, RoundContext, SecuritySpec, TuningSpec},
            sumcheck as sumcheck_params,
        },
    },
};

/// Hard cap on `log_inv_rate` searched. Beyond this the codeword would blow
/// past sane NTT-friendly sizes for typical witnesses.
const ADAPTIVE_MAX_LOG_INV_RATE: u32 = 20;

/// Real per-IRS dimensions extracted from a built `IrsConfig`. Drives the
/// cost proxy with NTT-rounded codeword length and the actual decoding-regime
/// query count — no closed-form approximation.
#[derive(Clone, Copy, Debug)]
struct RoundDims {
    codeword_length: usize,
    in_domain_samples: usize,
    interleaving_depth: usize,
}

/// Fixed per-encode-call overhead in the cost-proxy's nominal units.
///
/// Each `interleaved_encode` invocation pays a setup cost (allocator,
/// roots-table extension, rayon thread fan-out) that's amortized poorly for
/// small NTTs. The production trace showed sub-ms cold-cache outliers on
/// 4k-element NTTs that the pure `codeword · log codeword · interleaving`
/// model wouldn't see. This constant biases the planner against schedules
/// that fragment work into many small encodes — same direction as proof-size
/// pressure but for a different reason.
///
/// Calibration: at ~1.4 ns / field-op (variable cost) on Apple Silicon,
/// per-call fixed overhead observed at ~10 μs ≈ 7000 "ops" worth. Round to
/// 4096 — within an order of magnitude is enough for correct pareto ordering;
/// the absolute number doesn't matter under log-scale knee normalization.
const ENCODE_FIXED_OVERHEAD: f64 = 4096.0;

/// NTT/Merkle cost proxy for one encoded IRS. Uses real solver outputs so
/// per-round NTT smoothness rounding (`12288 = 4096·3` etc.) and per-regime
/// query counts (Johnson vs Unique vs Capacity) are accurate.
///
/// Constants are nominal — pareto ordering is invariant under positive
/// rescaling of either axis, and the log-knee picker compounds that.
fn round_cost_from_dims(dims: RoundDims, field_bytes: f64, hash_bytes: f64) -> (f64, f64) {
    let codeword = dims.codeword_length as f64;
    let interleaving = dims.interleaving_depth as f64;
    let queries = dims.in_domain_samples as f64;
    let log_codeword = codeword.log2().max(1.0);

    let encode = ENCODE_FIXED_OVERHEAD + codeword * log_codeword * interleaving;
    let proof = queries * (interleaving * field_bytes + log_codeword * hash_bytes);
    (encode, proof)
}

#[derive(Clone, Copy, Debug)]
struct Cost {
    encode: f64,
    proof: f64,
}

impl Cost {
    const ZERO: Self = Self {
        encode: 0.0,
        proof: 0.0,
    };
    fn add(self, other: (f64, f64)) -> Self {
        Self {
            encode: self.encode + other.0,
            proof: self.proof + other.1,
        }
    }
    fn dominates(self, other: Self) -> bool {
        self.encode <= other.encode
            && self.proof <= other.proof
            && (self.encode < other.encode || self.proof < other.proof)
    }
}

/// Bound on how aggressive Adaptive can be at a single round, expressed as a
/// multiple of the canonical WHIR per-round step `folding − 1` (the increment
/// [`super::spec::RateSchedule::Stepping`] applies, inherited from legacy
/// in-place WHIR's stepping invariant). This lets Adaptive explore schedules
/// slightly more aggressive than the legacy step, gated by the per-candidate
/// feasibility check. Pure search-space heuristic, not a correctness bound.
const ADAPTIVE_STEP_BUDGET: u32 = 2;

/// Search per-round target rates that minimize a pareto-knee cost. The
/// skeleton is fixed (folding factors, message-length sequence); only
/// `target_log_inv_rate` per round is searched. Each candidate (round,
/// source_rate, target_rate) is checked against the per-round PoW budget via
/// the actual analytic-error formulas (memoized) — schedules whose PoW gap
/// at any round would exceed `pow_budget` are dropped before pareto.
///
/// Returns `Err(DeriveError::AdaptiveNoFeasibleSchedule)` if no candidate
/// passes the per-slot PoW check — the spec is too tight for the planner's
/// search space.
pub(super) fn plan_adaptive_rates<M: Embedding + Default>(
    spec: &SecuritySpec,
    tuning: &TuningSpec,
    knee_weight: KneeWeight,
    skeleton: &[RoundShape],
    basecase_vector_size: usize,
    mode: RoundBuildMode<'_>,
) -> Result<Vec<u32>, DeriveError> {
    let mut planner: Planner<'_, M> = Planner::new(spec, skeleton, basecase_vector_size, mode);
    let candidates = planner.search(tuning.starting_log_inv_rate);
    // No feasible candidates means even the most conservative search choice
    // (constant rate at every round) failed the per-slot PoW check. The spec
    // is fundamentally too tight — surface this as a planner-level error
    // rather than returning placeholder rates that would just fail downstream
    // validation with a less informative message.
    if candidates.is_empty() {
        return Err(DeriveError::AdaptiveNoFeasibleSchedule);
    }
    Ok(pick_knee(pareto_frontier(candidates), knee_weight.get()))
}

/// Adapter that returns `Some(())` when an analytic floor `bits` leaves a
/// gap small enough for `pow_budget` to close. Used via `?` so a too-low
/// floor short-circuits the surrounding `_dims` builder.
fn fits(bits: f64, deficit: f64) -> Option<()> {
    (bits >= deficit).then_some(())
}

/// DFS state for the adaptive rate search. Owns the loop-invariants and the
/// two memoization caches so `recurse` doesn't have to thread a dozen
/// parameters down each level.
struct Planner<'a, M> {
    spec: &'a SecuritySpec,
    skeleton: &'a [RoundShape],
    basecase_vector_size: usize,
    mode: RoundBuildMode<'a>,
    field_bytes: f64,
    hash_bytes: f64,
    /// Memoize per `(round_idx, source_rate, target_rate)`. `Some(dims)` is
    /// feasible with real IRS dimensions; `None` is infeasible (PoW budget
    /// can't close the analytic gap).
    round_cache: HashMap<(usize, u32, u32), Option<RoundDims>>,
    /// Basecase candidates only vary by rate, so cache separately.
    basecase_cache: HashMap<u32, Option<RoundDims>>,
    _m: std::marker::PhantomData<M>,
}

impl<'a, M: Embedding + Default> Planner<'a, M> {
    fn new(
        spec: &'a SecuritySpec,
        skeleton: &'a [RoundShape],
        basecase_vector_size: usize,
        mode: RoundBuildMode<'a>,
    ) -> Self {
        Self {
            spec,
            skeleton,
            basecase_vector_size,
            mode,
            field_bytes: <M::Target as FieldWithSize>::field_size_bits() / 8.0,
            hash_bytes: 32.0,
            round_cache: HashMap::new(),
            basecase_cache: HashMap::new(),
            _m: std::marker::PhantomData,
        }
    }

    fn search(&mut self, starting_log_inv_rate: u32) -> Vec<(Vec<u32>, Cost)> {
        let mut out = Vec::new();
        let mut chosen = Vec::with_capacity(self.skeleton.len());
        self.recurse(0, starting_log_inv_rate, &mut chosen, Cost::ZERO, &mut out);
        out
    }

    fn recurse(
        &mut self,
        idx: usize,
        cur_rate: u32,
        chosen: &mut Vec<u32>,
        acc: Cost,
        out: &mut Vec<(Vec<u32>, Cost)>,
    ) {
        // handle basecase
        if idx == self.skeleton.len() {
            let spec = self.spec;
            let vector_size = self.basecase_vector_size;
            let dims = *self
                .basecase_cache
                .entry(cur_rate)
                .or_insert_with(|| basecase_dims::<M>(spec, vector_size, cur_rate));
            if let Some(dims) = dims {
                let (e, p) = round_cost_from_dims(dims, self.field_bytes, self.hash_bytes);
                out.push((chosen.clone(), acc.add((e, p))));
            }
            return;
        }
        let shape = self.skeleton[idx];
        // Per-round step capped at `ADAPTIVE_STEP_BUDGET · (folding − 1)` —
        // the canonical WHIR per-round increment scaled by the search budget.
        // Hard cap on absolute rate via `ADAPTIVE_MAX_LOG_INV_RATE`. Each
        // candidate (src, tgt) is feasibility-checked below; exceeding the
        // canonical increment is allowed when the PoW budget actually fits.
        let max_step = shape
            .source_folding_factor
            .saturating_sub(1)
            .saturating_mul(ADAPTIVE_STEP_BUDGET);
        for delta in 0..=max_step {
            let next_rate = cur_rate.saturating_add(delta);
            if next_rate > ADAPTIVE_MAX_LOG_INV_RATE {
                break;
            }
            let spec = self.spec;
            let mode = self.mode;
            let dims = *self
                .round_cache
                .entry((idx, cur_rate, next_rate))
                .or_insert_with(|| try_round_dims::<M>(spec, &shape, cur_rate, next_rate, mode));
            let Some(dims) = dims else { continue };
            let (e, p) = round_cost_from_dims(dims, self.field_bytes, self.hash_bytes);
            chosen.push(next_rate);
            self.recurse(idx + 1, next_rate, chosen, acc.add((e, p)), out);
            chosen.pop();
        }
    }
}

/// Drop any (schedule, cost) dominated by another. `O(|candidates|²)` —
/// fine at planner scale.
fn pareto_frontier(candidates: Vec<(Vec<u32>, Cost)>) -> Vec<(Vec<u32>, Cost)> {
    let mut frontier: Vec<(Vec<u32>, Cost)> = Vec::new();
    'outer: for cand in candidates {
        for f in &frontier {
            if f.1.dominates(cand.1) {
                continue 'outer;
            }
        }
        frontier.retain(|f| !cand.1.dominates(f.1));
        frontier.push(cand);
    }
    frontier
}

/// Weighted log-scale pareto knee. Picks the schedule whose deficit from
/// per-axis minima — measured in log-space, weighted by `knee_weight` — is
/// smallest. Log-space normalizes the units mismatch between the
/// encode/proof proxies (ops vs bytes) and makes the picker invariant to any
/// constant rescaling of either axis.
///
/// `knee_weight ∈ [0, 1]` is the encode-axis bias (see
/// [`super::spec::KneeWeight`]).
fn pick_knee(frontier: Vec<(Vec<u32>, Cost)>, knee_weight: f64) -> Vec<u32> {
    let logs: Vec<(Vec<u32>, f64, f64)> = frontier
        .into_iter()
        .map(|(s, c)| (s, c.encode.max(1.0).log2(), c.proof.max(1.0).log2()))
        .collect();
    let min_e = logs.iter().map(|f| f.1).fold(f64::INFINITY, f64::min);
    let min_p = logs.iter().map(|f| f.2).fold(f64::INFINITY, f64::min);
    let score = |le: f64, lp: f64| {
        knee_weight * (le - min_e).powi(2) + (1.0 - knee_weight) * (lp - min_p).powi(2)
    };
    logs.into_iter()
        .min_by(|a, b| {
            score(a.1, a.2)
                .partial_cmp(&score(b.1, b.2))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("frontier non-empty: candidates non-empty implies frontier non-empty")
        .0
}

/// Build the basecase IRS at `cur_rate` and return its dimensions, or `None`
/// if its analytic floors can't be closed by `pow_budget`.
fn basecase_dims<M: Embedding + Default>(
    spec: &SecuritySpec,
    vector_size: usize,
    log_inv_rate: u32,
) -> Option<RoundDims> {
    let ctx = RoundContext {
        vector_size,
        log_inv_rate,
        folding_factor: 0,
    };
    let commit: IrsConfig<Identity<M::Target>> =
        irs_params::solve(spec, &ctx, OodSampleBudget::ZERO).ok()?;

    let max_deficit = f64::from(spec.target_security_bits) - f64::from(spec.pow_budget.bits());

    fits(
        f64::from(sumcheck_params::analytic_error_bits(
            &commit,
            Option::<MaskOracleInfo>::None,
        )),
        max_deficit,
    )?;
    if matches!(spec.mode, Mode::ZeroKnowledge) {
        fits(
            f64::from(basecase_params::analytic_error_bits(&commit)),
            max_deficit,
        )?;
    }

    Some(RoundDims {
        codeword_length: commit.codeword_length(),
        in_domain_samples: commit.in_domain_samples(),
        interleaving_depth: commit.interleaving_depth(),
    })
}

/// Build a round at this (source_rate, target_rate, mode) and return the
/// **target IRS's** dimensions (`codeword_length`, `in_domain_samples`,
/// `interleaving_depth`) if every per-slot analytic floor fits inside
/// `pow_budget`. Returns `None` if any floor is too low — even max grinding
/// can't close the gap.
///
/// The returned dims drive the cost proxy, so the planner uses NTT-rounded
/// codeword sizes and per-regime query counts — no closed-form approximation.
fn try_round_dims<M: Embedding + Default>(
    spec: &SecuritySpec,
    shape: &RoundShape,
    source_log_inv_rate: u32,
    target_log_inv_rate: u32,
    mode: RoundBuildMode<'_>,
) -> Option<RoundDims> {
    let max_deficit = f64::from(spec.target_security_bits) - f64::from(spec.pow_budget.bits());

    let src_ctx = RoundContext {
        vector_size: shape.source_vector_size,
        log_inv_rate: source_log_inv_rate,
        folding_factor: shape.source_folding_factor,
    };
    let target_log_degree = f64::from(
        shape
            .source_vector_size
            .trailing_zeros()
            .saturating_sub(shape.source_folding_factor),
    );
    let target_list_size = spec
        .decoding_regime
        .list_size_estimate(target_log_degree, f64::from(target_log_inv_rate));

    let ood_mode = mode.map(|p| p.c_zk_log_inv_rate);
    let (source, t_ood) = solve_t_ood::<M>(spec, &src_ctx, target_list_size, ood_mode, 0).ok()?;

    let target_budget = match mode {
        Branch::Standard => OodSampleBudget::ZERO,
        Branch::ZeroKnowledge(_) => OodSampleBudget::new(t_ood),
    };
    let tgt_ctx = RoundContext {
        vector_size: source.message_length(),
        log_inv_rate: target_log_inv_rate,
        folding_factor: shape.target_folding_factor,
    };
    let target_irs: IrsConfig<Identity<M::Target>> =
        irs_params::solve(spec, &tgt_ctx, target_budget).ok()?;

    let mask_info = match mode {
        Branch::Standard => None,
        Branch::ZeroKnowledge(payload) => {
            let mo = build_mask_oracle::<M>(
                payload.zk_spec,
                &src_ctx,
                &source,
                t_ood,
                payload.c_zk_log_inv_rate,
                shape.round_index,
            )
            .ok()?;
            fits(f64::from(mo.analytic_bits()), max_deficit)?;
            Some(mo.info())
        }
    };

    fits(
        f64::from(sumcheck_params::analytic_error_bits(&source, mask_info)),
        max_deficit,
    )?;
    fits(
        f64::from(code_switch_params::analytic_error_bits(
            &source,
            &target_irs,
            t_ood,
            mask_info,
        )),
        max_deficit,
    )?;

    Some(RoundDims {
        codeword_length: target_irs.codeword_length(),
        in_domain_samples: target_irs.in_domain_samples(),
        interleaving_depth: target_irs.interleaving_depth(),
    })
}
