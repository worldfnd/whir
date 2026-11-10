//! Enumerate feasible schedules with cached shared protocol construction.

use std::collections::HashMap;

use super::{
    derive::{round, solve_irs, terminal},
    embedding_degree,
    masks::{checked_code, WitnessMaskConfig},
    DeriveError, Params, RoundContext,
};
use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        fields::FieldWithSize,
    },
    protocols::irs_commit::Config as IrsConfig,
};

// Setup cost per encode call, calibrated from Apple Silicon traces.
const ENCODE_FIXED_OVERHEAD: f64 = 4096.0;
type Cost = [f64; 2];
// Separate terminal and mask costs preserve the original addition order.
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
    spec: &Params,
    weight: f64,
    layout: &[RoundContext],
) -> Result<Vec<u32>, DeriveError> {
    let mut planner = Planner {
        spec,
        layout,
        rounds: HashMap::new(),
        finals: HashMap::new(),
        path: Vec::new(),
        frontier: Vec::new(),
    };
    planner.visit::<M>(spec.rate, [0.0; 2], 0);
    pick_knee(planner.frontier, weight).ok_or(DeriveError(
        "adaptive rate planner found no feasible schedule",
    ))
}

struct Planner<'a> {
    spec: &'a Params,
    layout: &'a [RoundContext],
    rounds: HashMap<(usize, u32, u32), Option<(Cost, usize)>>,
    finals: HashMap<(u32, usize), Option<FinalCosts>>,
    path: Vec<u32>,
    frontier: Vec<(Vec<u32>, Cost)>,
}
impl Planner<'_> {
    fn visit<M: Embedding + Default>(&mut self, rate: u32, cost: Cost, masks: usize) {
        let add = |a: Cost, b: Cost| [a[0] + b[0], a[1] + b[1]];
        let index = self.path.len();
        let source = RoundContext {
            rate,
            ..self.layout[index]
        };
        if index + 1 == self.layout.len() {
            let final_cost = *self.finals.entry((rate, masks)).or_insert_with(|| {
                terminal_cost::<M>(self.spec, &source, usize::from(index != 0), masks)
            });
            if let Some((basecase, masks)) = final_cost {
                let cost = add(cost, basecase);
                let cost = masks.map_or(cost, |mask| add(cost, mask));
                let dominates =
                    |a: Cost, b: Cost| a[0] <= b[0] && a[1] <= b[1] && (a[0] < b[0] || a[1] < b[1]);
                if !self
                    .frontier
                    .iter()
                    .any(|(_, other)| dominates(*other, cost))
                {
                    self.frontier.retain(|(_, other)| !dominates(cost, *other));
                    self.frontier.push((self.path.clone(), cost));
                }
            }
            return;
        }
        // Search up to twice WHIR's canonical increment, in ascending order.
        let end = rate.saturating_add(source.folds.saturating_sub(1).saturating_mul(2));
        for next in rate..=end.min(20) {
            let target = RoundContext {
                rate: next,
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
    spec: &Params,
    source: &RoundContext,
    target: &RoundContext,
    index: usize,
) -> Option<(Cost, usize)> {
    let source = solve_irs::<M>(spec, source).ok()?;
    let target = solve_irs(spec, target).ok()?;
    let round = round(spec, &source, &target, index).ok()?;
    let masks = round.masks.map_or(Some(0), |mask| {
        mask.num_coefficients()
            .ok()?
            .checked_add(2 * round.sumcheck.num_rounds() + 1)
    })?;
    Some((code_cost(&target), masks))
}

fn terminal_cost<M: Embedding + Default>(
    spec: &Params,
    context: &RoundContext,
    incoming: usize,
    masks: usize,
) -> Option<FinalCosts> {
    let code = solve_irs::<Identity<M::Target>>(spec, context).ok()?;
    let _ = terminal(spec, &code, incoming).ok()?;
    let cost = code_cost(&code);
    if !spec.zk {
        return Some((cost, None));
    }
    let payload = masks.checked_add(
        WitnessMaskConfig::for_code(&code, incoming)
            .and_then(WitnessMaskConfig::num_coefficients)
            .ok()?,
    )?;
    let degree = if incoming == 0 {
        1
    } else {
        embedding_degree::<M>().ok()?
    };
    let masks = checked_code(spec, payload, degree, &code).ok()?;
    let mut mask_cost = code_cost(&masks);
    mask_cost[1] += masks.vector_size() as f64 * (M::Target::field_size_bits() / 8.0);
    Some((cost, Some(mask_cost)))
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
        algebra::{embedding::Identity, fields::Field64_2},
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
