//! Enumerate feasible rate schedules, then select a weighted Pareto knee.

use std::collections::HashMap;

use super::{
    derive::{solve_irs, solve_node},
    embedding_degree,
    masks::{solve_commit, WitnessMaskConfig},
    security::{zook_final_opening_bits, zook_mask_binding_bits},
    DeriveError, KneeWeight, RoundContext, SecuritySpec, TuningSpec,
};
use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        fields::FieldWithSize,
    },
    protocols::irs_commit::Config as IrsConfig,
};

// Search limits are tuning choices; every candidate still passes protocol bounds.
const ADAPTIVE_MAX_LOG_INV_RATE: u32 = 20;
// Setup cost per encode call, calibrated from Apple Silicon traces.
const ENCODE_FIXED_OVERHEAD: f64 = 4096.0;

// [encoding work, proof bytes]. Keep each addition in protocol order.
type Cost = [f64; 2];
type FinalCosts = (Cost, Option<Cost>);
fn code_cost<M: Embedding>(code: &IrsConfig<M>) -> Cost {
    let n = code.codeword_length() as f64;
    let columns = code.num_messages() as f64;
    let log_n = n.log2().max(1.0);
    [
        ENCODE_FIXED_OVERHEAD + n * log_n * columns,
        code.in_domain_samples() as f64
            * (columns * (M::Target::field_size_bits() / 8.0) + log_n * 32.0),
    ]
}

pub(super) fn plan_adaptive_rates<M: Embedding + Default>(
    spec: &SecuritySpec,
    tuning: &TuningSpec,
    weight: KneeWeight,
    layout: &[RoundContext],
) -> Result<Vec<u32>, DeriveError> {
    let mut planner = Planner {
        spec,
        layout,
        rounds: HashMap::new(),
        finals: HashMap::new(),
        path: Vec::new(),
        candidates: Vec::new(),
    };
    planner.visit::<M>(tuning.starting_log_inv_rate, [0.0; 2], 0);
    pick_knee(pareto_frontier(planner.candidates), weight.get())
        .ok_or(DeriveError::AdaptiveNoFeasibleSchedule)
}

struct Planner<'a> {
    spec: &'a SecuritySpec,
    layout: &'a [RoundContext],
    rounds: HashMap<(usize, u32, u32), Option<(Cost, usize)>>,
    finals: HashMap<(u32, usize), Option<FinalCosts>>,
    path: Vec<u32>,
    candidates: Vec<(Vec<u32>, Cost)>,
}
impl Planner<'_> {
    fn visit<M: Embedding + Default>(&mut self, rate: u32, cost: Cost, masks: usize) {
        let add = |a: Cost, b: Cost| [a[0] + b[0], a[1] + b[1]];
        let index = self.path.len();
        let source = RoundContext {
            log_inv_rate: rate,
            ..self.layout[index]
        };
        if index + 1 == self.layout.len() {
            let final_cost = *self.finals.entry((rate, masks)).or_insert_with(|| {
                terminal_cost::<M>(self.spec, &source, usize::from(index != 0), masks)
            });
            if let Some((basecase, masks)) = final_cost {
                let cost = add(cost, basecase);
                self.candidates.push((
                    self.path.clone(),
                    masks.map_or(cost, |mask| add(cost, mask)),
                ));
            }
            return;
        }
        // Search up to twice WHIR's canonical rate increment.
        let max_step = source.folding_factor.saturating_sub(1).saturating_mul(2);
        for delta in 0..=max_step {
            let next = rate.saturating_add(delta);
            if next > ADAPTIVE_MAX_LOG_INV_RATE {
                break;
            }
            let target = RoundContext {
                log_inv_rate: next,
                ..self.layout[index + 1]
            };
            let round = *self.rounds.entry((index, rate, next)).or_insert_with(|| {
                if index == 0 {
                    round_cost::<M>(self.spec, &source, &target, index)
                } else {
                    round_cost::<Identity<M::Target>>(self.spec, &source, &target, index)
                }
            });
            let Some((round, extra)) = round else {
                continue;
            };
            let Some(masks) = masks.checked_add(extra) else {
                continue;
            };
            self.path.push(next);
            self.visit::<M>(next, add(cost, round), masks);
            self.path.pop();
        }
    }
}

fn round_cost<M: Embedding + Default>(
    spec: &SecuritySpec,
    source: &RoundContext,
    target: &RoundContext,
    index: usize,
) -> Option<(Cost, usize)> {
    let source = solve_irs::<M>(spec, source).ok()?;
    let target = solve_irs(spec, target).ok()?;
    let node = solve_node(spec, source, Some(&target), index).ok()?;
    let masks = node.round(&target, index, spec.mode.is_zk()).masks;
    let masks = masks.map_or(Some(0), |masks| {
        masks
            .witness
            .num_coefficients()
            .ok()?
            .checked_add(masks.sumcheck_coefficients)
    })?;
    Some((code_cost(&target), masks))
}

fn terminal_cost<M: Embedding + Default>(
    spec: &SecuritySpec,
    context: &RoundContext,
    incoming: usize,
    masks: usize,
) -> Option<FinalCosts> {
    let code = solve_irs::<Identity<M::Target>>(spec, context).ok()?;
    let node = solve_node(spec, code, None, incoming).ok()?;
    let cost = code_cost(&node.code);
    if !spec.mode.is_zk() {
        return Some((cost, None));
    }
    let payload = masks.checked_add(
        WitnessMaskConfig::for_code(&node.code, incoming)
            .and_then(|mask| mask.num_coefficients())
            .ok()?,
    )?;
    let degree = if incoming == 0 {
        1
    } else {
        embedding_degree::<M>().ok()?
    };
    let masks = solve_commit::<M::Target>(spec, payload, degree).ok()?;
    let binding = zook_mask_binding_bits(&masks).ok()?;
    let (combination, queries) = zook_final_opening_bits(&node.code, &masks).ok()?;
    if [binding, combination, queries]
        .into_iter()
        .any(|bits| f64::from(bits) < f64::from(spec.target_security_bits))
    {
        return None;
    }
    let mut mask_cost = code_cost(&masks);
    mask_cost[1] += masks.vector_size() as f64 * (M::Target::field_size_bits() / 8.0);
    Some((cost, Some(mask_cost)))
}

// Preserve enumeration order and equal-cost ties.
fn pareto_frontier(candidates: Vec<(Vec<u32>, Cost)>) -> Vec<(Vec<u32>, Cost)> {
    let dominates = |a: Cost, b: Cost| a[0] <= b[0] && a[1] <= b[1] && (a[0] < b[0] || a[1] < b[1]);
    let mut frontier: Vec<(Vec<u32>, Cost)> = Vec::new();
    for (rates, cost) in candidates {
        if frontier.iter().any(|(_, other)| dominates(*other, cost)) {
            continue;
        }
        frontier.retain(|(_, other)| !dominates(cost, *other));
        frontier.push((rates, cost));
    }
    frontier
}

// Weighted distance from per-axis minima in log space; equal scores keep enumeration order.
fn pick_knee(frontier: Vec<(Vec<u32>, Cost)>, weight: f64) -> Option<Vec<u32>> {
    let logs: Vec<_> = frontier
        .into_iter()
        .map(|(rates, [encode, proof])| (rates, encode.max(1.0).log2(), proof.max(1.0).log2()))
        .collect();
    let min_e = logs.iter().map(|f| f.1).fold(f64::INFINITY, f64::min);
    let min_p = logs.iter().map(|f| f.2).fold(f64::INFINITY, f64::min);
    let score =
        |e: f64, p: f64| weight * (e - min_e).powi(2) + (1.0 - weight) * (p - min_p).powi(2);
    logs.into_iter()
        .min_by(|a, b| {
            score(a.1, a.2)
                .partial_cmp(&score(b.1, b.2))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|candidate| candidate.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        algebra::fields::Field64_2,
        protocols::params::tests::{assert_close, code},
    };

    #[test]
    fn shared_mask_tree_counts_both_columns_and_one_query_path() {
        let single = code::<Identity<Field64_2>>(128, 1, 0.25, 1);
        let shared = code::<Identity<Field64_2>>(128, 1, 0.25, 2);
        let single_cost = code_cost(&single);
        let shared_cost = code_cost(&shared);
        assert_close(shared_cost[0], 2.0 * single_cost[0] - ENCODE_FIXED_OVERHEAD);
        assert_close(
            shared_cost[1],
            single_cost[1]
                + shared.in_domain_samples() as f64 * (Field64_2::field_size_bits() / 8.0),
        );
        assert!(shared_cost[1] < 2.0 * single_cost[1]);
    }
}
