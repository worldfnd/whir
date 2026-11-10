//! Enumerate feasible rate schedules and select a weighted Pareto knee.
//! Candidates use the same round and basecase solvers as protocol construction.

use std::collections::HashMap;

use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        fields::FieldWithSize,
    },
    protocols::{
        irs_commit::Config as IrsConfig,
        params::{
            config::{embedding_degree, RoundConfig},
            derive::{with_basecase, with_round_config},
            error::DeriveError,
            masks::solve_commit as solve_joint_commit,
            security::{zook_mask_binding_bits, ZookFinalOpeningSoundness},
            spec::{KneeWeight, RoundContext, SecuritySpec, TuningSpec},
        },
    },
};

/// Search-space limits, not security bounds. Each candidate still requires validation.
/// The step budget multiplies WHIR's canonical increment, `folding - 1`.
const ADAPTIVE_MAX_LOG_INV_RATE: u32 = 20;
const ADAPTIVE_STEP_BUDGET: u32 = 2;

/// Nominal setup work for each encode call, calibrated from Apple Silicon traces.
/// This penalizes many small encodes beyond the variable `n * log2(n)` work.
/// Positive rescaling of either cost axis does not affect Pareto ordering.
const ENCODE_FIXED_OVERHEAD: f64 = 4096.0;

#[derive(Clone, Copy, Debug, Default)]
struct Cost {
    encode: f64,
    proof: f64,
}

/// Witness cost, then optional mask cost; separate values preserve addition order.
type FinalCosts = (Cost, Option<Cost>);

impl Cost {
    fn add(self, other: Self) -> Self {
        Self {
            encode: self.encode + other.encode,
            proof: self.proof + other.proof,
        }
    }

    fn dominates(self, other: Self) -> bool {
        self.encode <= other.encode
            && self.proof <= other.proof
            && (self.encode < other.encode || self.proof < other.proof)
    }
}

/// Use actual NTT-rounded lengths, interleaving, and decoding-regime query counts.
fn code_cost<M: Embedding>(code: &IrsConfig<M>) -> Cost {
    let codeword = code.codeword_length() as f64;
    let interleaving = code.num_messages() as f64;
    let queries = code.in_domain_samples() as f64;
    let log_codeword = codeword.log2().max(1.0);
    let field_bytes = M::Target::field_size_bits() / 8.0;
    Cost {
        encode: ENCODE_FIXED_OVERHEAD + codeword * log_codeword * interleaving,
        proof: queries * (interleaving * field_bytes + log_codeword * 32.0),
    }
}

/// Standard candidates may use PoW; Zook candidates must meet the full analytic target.
/// The context sequence includes the final commitment.
pub(super) fn plan_adaptive_rates<M: Embedding + Default>(
    spec: &SecuritySpec,
    tuning: &TuningSpec,
    knee_weight: KneeWeight,
    skeleton: &[RoundContext],
) -> Result<Vec<u32>, DeriveError> {
    let candidates = Planner {
        spec,
        skeleton,
        round_costs: HashMap::new(),
        final_costs: HashMap::new(),
    }
    .search::<M>(tuning.starting_log_inv_rate);
    if candidates.is_empty() {
        return Err(DeriveError::AdaptiveNoFeasibleSchedule);
    }
    Ok(pick_knee(pareto_frontier(candidates), knee_weight.get()))
}

struct Planner<'a> {
    spec: &'a SecuritySpec,
    skeleton: &'a [RoundContext],
    round_costs: HashMap<(usize, u32, u32), Option<(Cost, usize)>>,
    final_costs: HashMap<(u32, usize), Option<FinalCosts>>,
}

impl Planner<'_> {
    fn search<M: Embedding + Default>(mut self, starting_rate: u32) -> Vec<(Vec<u32>, Cost)> {
        let mut out = Vec::new();
        self.recurse::<M>(starting_rate, &mut Vec::new(), Cost::default(), 0, &mut out);
        out
    }

    fn recurse<M: Embedding + Default>(
        &mut self,
        rate: u32,
        chosen: &mut Vec<u32>,
        cost: Cost,
        mask_coefficients: usize,
        out: &mut Vec<(Vec<u32>, Cost)>,
    ) {
        let index = chosen.len();
        if index + 1 == self.skeleton.len() {
            if let Some((basecase, masks)) = self.finish::<M>(rate, mask_coefficients) {
                let cost = cost.add(basecase);
                out.push((chosen.clone(), masks.map_or(cost, |mask| cost.add(mask))));
            }
            return;
        }
        let source = RoundContext {
            log_inv_rate: rate,
            ..self.skeleton[index]
        };
        let max_step = source
            .folding_factor
            .saturating_sub(1)
            .saturating_mul(ADAPTIVE_STEP_BUDGET);
        for delta in 0..=max_step {
            let next_rate = rate.saturating_add(delta);
            if next_rate > ADAPTIVE_MAX_LOG_INV_RATE {
                break;
            }
            let target = RoundContext {
                log_inv_rate: next_rate,
                ..self.skeleton[index + 1]
            };
            let round = *self
                .round_costs
                .entry((index, rate, next_rate))
                .or_insert_with(|| {
                    if index == 0 {
                        with_round_config::<M, _>(self.spec, &source, &target, index, |round| {
                            round_cost(&round)
                        })
                        .ok()
                        .flatten()
                    } else {
                        with_round_config::<Identity<M::Target>, _>(
                            self.spec,
                            &source,
                            &target,
                            index,
                            |round| round_cost(&round),
                        )
                        .ok()
                        .flatten()
                    }
                });
            let Some((round_cost, masks)) = round else {
                continue;
            };
            let Some(masks) = mask_coefficients.checked_add(masks) else {
                continue;
            };
            chosen.push(next_rate);
            self.recurse::<M>(next_rate, chosen, cost.add(round_cost), masks, out);
            chosen.pop();
        }
    }

    /// Account for the final witness, joint masks, and revealed mask message.
    fn finish<M: Embedding + Default>(
        &mut self,
        rate: u32,
        mask_coefficients: usize,
    ) -> Option<FinalCosts> {
        *self
            .final_costs
            .entry((rate, mask_coefficients))
            .or_insert_with(|| {
                with_basecase::<M::Target, _>(
                    self.spec,
                    self.skeleton.last()?.vector_size,
                    rate,
                    usize::from(self.skeleton.len() > 1),
                    |basecase| {
                        let cost = code_cost(basecase.commit());
                        let Some(mask) = basecase.witness_mask() else {
                            return Some((cost, None));
                        };
                        let payload =
                            mask_coefficients.checked_add(mask.num_coefficients().ok()?)?;
                        let degree = if self.skeleton.len() == 1 {
                            1
                        } else {
                            embedding_degree::<M>().ok()?
                        };
                        let masks =
                            solve_joint_commit::<M::Target>(self.spec, payload, degree).ok()?;
                        let binding = zook_mask_binding_bits(&masks).ok()?;
                        let opening =
                            ZookFinalOpeningSoundness::for_codes(basecase.commit(), &masks).ok()?;
                        let target = f64::from(self.spec.target_security_bits);
                        if [binding, opening.combination, opening.queries]
                            .into_iter()
                            .any(|bits| f64::from(bits) < target)
                        {
                            return None;
                        }
                        let mut mask_cost = code_cost(&masks);
                        mask_cost.proof +=
                            masks.vector_size() as f64 * (M::Target::field_size_bits() / 8.0);
                        Some((cost, Some(mask_cost)))
                    },
                )
                .ok()
                .flatten()
            })
    }
}

fn round_cost<M: Embedding>(round: &RoundConfig<'_, M>) -> Option<(Cost, usize)> {
    let masks = match round.masks() {
        Some(masks) => masks
            .witness
            .num_coefficients()
            .ok()?
            .checked_add(masks.sumcheck_coefficients)?,
        None => 0,
    };
    Some((code_cost(round.code_switch().target()), masks))
}

/// Preserve candidate order and equal-cost ties while removing dominated schedules.
fn pareto_frontier(candidates: Vec<(Vec<u32>, Cost)>) -> Vec<(Vec<u32>, Cost)> {
    let mut frontier: Vec<(Vec<u32>, Cost)> = Vec::new();
    for candidate in candidates {
        if frontier.iter().any(|(_, cost)| cost.dominates(candidate.1)) {
            continue;
        }
        frontier.retain(|(_, cost)| !candidate.1.dominates(*cost));
        frontier.push(candidate);
    }
    frontier
}

/// Minimize weighted squared distance from the per-axis minima in log space.
/// Log normalization removes cost-unit differences and preserves constant rescaling.
/// `knee_weight` gives the encode-axis bias; equal scores retain enumeration order.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        algebra::fields::Field64_2,
        protocols::params::test_utils::{assert_close, johnson_code_with_vectors},
    };

    #[test]
    fn shared_mask_tree_counts_both_columns_and_one_query_path() {
        let single = johnson_code_with_vectors::<Identity<Field64_2>>(128, 1, 0.25, 1);
        let shared = johnson_code_with_vectors::<Identity<Field64_2>>(128, 1, 0.25, 2);
        let single_cost = code_cost(&single);
        let shared_cost = code_cost(&shared);
        assert_close(
            shared_cost.encode,
            2.0 * single_cost.encode - ENCODE_FIXED_OVERHEAD,
        );
        assert_close(
            shared_cost.proof,
            single_cost.proof
                + shared.in_domain_samples() as f64 * (Field64_2::field_size_bits() / 8.0),
        );
        assert!(shared_cost.proof < 2.0 * single_cost.proof);
    }
}
