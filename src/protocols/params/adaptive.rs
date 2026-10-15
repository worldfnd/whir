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
            basecase::{solve as solve_basecase, ZookFinalOpeningSoundness},
            branch::RoundBuildMode,
            build_round::build_round_config,
            error::DeriveError,
            irs_commit::solve as solve_irs,
            layout::RoundShape,
            mask_proximity::{solve_commit as solve_joint_commit, zook_mask_binding_bits},
            protocol_config::{embedding_degree, RoundConfig},
            spec::{KneeWeight, RoundContext, SecuritySpec, TuningSpec},
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
    mask_coefficients: usize,
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

/// Search per-round target rates that minimize a pareto-knee cost.
///
/// Folding factors and message lengths stay fixed. Standard candidates may
/// use their PoW budget; Zook candidates must meet the full analytic target,
/// including joint-mask binding and final opening. Returns
/// `AdaptiveNoFeasibleSchedule` if no candidate satisfies these bounds.
pub(super) fn plan_adaptive_rates<M: Embedding + Default>(
    spec: &SecuritySpec,
    tuning: &TuningSpec,
    knee_weight: KneeWeight,
    skeleton: &[RoundShape],
    basecase_vector_size: usize,
    mode: RoundBuildMode,
) -> Result<Vec<u32>, DeriveError> {
    let mut planner: Planner<'_, M> = Planner::new(spec, skeleton, basecase_vector_size, mode);
    let candidates = planner.search(tuning.starting_log_inv_rate);
    // Report infeasibility within this search space before selecting a schedule.
    if candidates.is_empty() {
        return Err(DeriveError::AdaptiveNoFeasibleSchedule);
    }
    Ok(pick_knee(pareto_frontier(candidates), knee_weight.get()))
}

/// DFS state for the adaptive rate search. Owns the loop-invariants and the
/// two memoization caches so `recurse` doesn't have to thread a dozen
/// parameters down each level.
struct Planner<'a, M> {
    spec: &'a SecuritySpec,
    skeleton: &'a [RoundShape],
    basecase_vector_size: usize,
    mode: RoundBuildMode,
    field_bytes: f64,
    hash_bytes: f64,
    /// Memoize per `(round_idx, source_rate, target_rate)`. `Some(dims)` is
    /// feasible with real IRS dimensions; `None` violates a size or security bound.
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
        mode: RoundBuildMode,
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
        self.recurse(
            0,
            starting_log_inv_rate,
            &mut chosen,
            Cost::ZERO,
            0,
            &mut out,
        );
        out
    }

    fn recurse(
        &mut self,
        idx: usize,
        cur_rate: u32,
        chosen: &mut Vec<u32>,
        acc: Cost,
        mask_coefficients: usize,
        out: &mut Vec<(Vec<u32>, Cost)>,
    ) {
        // handle basecase
        if idx == self.skeleton.len() {
            let spec = self.spec;
            let vector_size = self.basecase_vector_size;
            let dims = *self.basecase_cache.entry(cur_rate).or_insert_with(|| {
                basecase_dims::<M>(
                    spec,
                    vector_size,
                    cur_rate,
                    usize::from(!self.skeleton.is_empty()),
                )
            });
            if let Some(dims) = dims {
                let (e, p) = round_cost_from_dims(dims, self.field_bytes, self.hash_bytes);
                let mut cost = acc.add((e, p));
                if self.mode.is_zk() {
                    let Some(payload) = mask_coefficients.checked_add(dims.mask_coefficients)
                    else {
                        return;
                    };
                    let degree = if self.skeleton.is_empty() {
                        1
                    } else {
                        let Ok(degree) = embedding_degree::<M>() else {
                            return;
                        };
                        degree
                    };
                    let Ok(witness) = solve_irs::<Identity<M::Target>>(
                        spec,
                        &RoundContext {
                            vector_size,
                            log_inv_rate: cur_rate,
                            folding_factor: 0,
                        },
                    ) else {
                        return;
                    };
                    let Ok(masks) = solve_joint_commit::<M::Target>(spec, payload, degree) else {
                        return;
                    };
                    let Ok(binding) = zook_mask_binding_bits(&masks) else {
                        return;
                    };
                    let Ok(opening) = ZookFinalOpeningSoundness::for_codes(&witness, &masks) else {
                        return;
                    };
                    let target = f64::from(spec.target_security_bits);
                    if [binding, opening.combination, opening.queries]
                        .into_iter()
                        .any(|bits| f64::from(bits) < target)
                    {
                        return;
                    }
                    cost = cost.add(round_cost_from_dims(
                        dimensions(&masks, 0),
                        self.field_bytes,
                        self.hash_bytes,
                    ));
                }
                out.push((chosen.clone(), cost));
            }
            return;
        }
        let shape = self.skeleton[idx];
        // Per-round step capped at `ADAPTIVE_STEP_BUDGET · (folding − 1)` —
        // the canonical WHIR per-round increment scaled by the search budget.
        // Hard cap on absolute rate via `ADAPTIVE_MAX_LOG_INV_RATE`. Each
        // candidate (src, tgt) is feasibility-checked below; exceeding the
        // canonical increment is allowed when all configured bounds hold.
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
            let Some(next_masks) = mask_coefficients.checked_add(dims.mask_coefficients) else {
                chosen.pop();
                continue;
            };
            self.recurse(idx + 1, next_rate, chosen, acc.add((e, p)), next_masks, out);
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

/// Return the final commitment dimensions and its mask budget.
fn basecase_dims<M: Embedding + Default>(
    spec: &SecuritySpec,
    vector_size: usize,
    log_inv_rate: u32,
    incoming_deep: usize,
) -> Option<RoundDims> {
    let config =
        solve_basecase::<M::Target>(spec, vector_size, log_inv_rate, incoming_deep).ok()?;
    let masks = config
        .witness_mask()
        .map(|mask| mask.num_coefficients())
        .transpose()
        .ok()?
        .unwrap_or(0);
    Some(dimensions(config.commit(), masks))
}

/// Use the same round solver as ProtocolConfig::derive for candidate feasibility.
fn try_round_dims<M: Embedding + Default>(
    spec: &SecuritySpec,
    shape: &RoundShape,
    source_log_inv_rate: u32,
    target_log_inv_rate: u32,
    mode: RoundBuildMode,
) -> Option<RoundDims> {
    let shape = RoundShape {
        source_log_inv_rate,
        target_log_inv_rate,
        ..*shape
    };
    if shape.round_index == 0 {
        round_dimensions(&build_round_config::<M>(spec, &shape, mode).ok()?)
    } else {
        round_dimensions(&build_round_config::<Identity<M::Target>>(spec, &shape, mode).ok()?)
    }
}

fn round_dimensions<M: Embedding>(round: &RoundConfig<M>) -> Option<RoundDims> {
    let masks = match round.masks() {
        Some(masks) => masks
            .witness
            .num_coefficients()
            .ok()?
            .checked_add(masks.sumcheck_coefficients)?,
        None => 0,
    };
    Some(dimensions(round.code_switch().target(), masks))
}

const fn dimensions<M: Embedding>(code: &IrsConfig<M>, mask_coefficients: usize) -> RoundDims {
    RoundDims {
        codeword_length: code.codeword_length(),
        in_domain_samples: code.in_domain_samples(),
        interleaving_depth: code.interleaving_depth(),
        mask_coefficients,
    }
}
