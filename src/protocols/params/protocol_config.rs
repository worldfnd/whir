//! Output of [`super::derive`]: the assembled per-round and basecase configs.

use ark_ff::Field;
use serde::{Deserialize, Serialize};

use crate::{
    algebra::{embedding::Embedding, fields::FieldWithSize},
    bits::Bits,
    protocols::{
        basecase::Config as BasecaseConfig,
        code_switch::Config as CodeSwitchConfig,
        mask_proximity::Config as MaskProximityConfig,
        params::{
            basecase as basecase_params,
            bounds::usize_to_f64,
            code_switch as code_switch_params,
            error::{ChainSource, ChainTarget, DeriveError, Pow},
            mask_proximity as mask_proximity_params,
            solved::Solved,
            spec::{ListSize, MaskCodeMessageLen, OodSampleBudget, SecuritySpec, TuningSpec},
            sumcheck as sumcheck_params,
        },
        proof_of_work::Config as PowConfig,
        sumcheck::Config as SumcheckConfig,
    },
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProtocolConfig<M: Embedding> {
    security: SecuritySpec,
    tuning: TuningSpec,
    rounds: Vec<RoundConfig<M>>,
    basecase: BasecasePlan<M::Target>,
}

impl<M: Embedding> ProtocolConfig<M> {
    pub(crate) const fn new(
        security: SecuritySpec,
        tuning: TuningSpec,
        rounds: Vec<RoundConfig<M>>,
        basecase: BasecasePlan<M::Target>,
    ) -> Self {
        Self {
            security,
            tuning,
            rounds,
            basecase,
        }
    }

    pub const fn security(&self) -> &SecuritySpec {
        &self.security
    }

    pub const fn tuning(&self) -> &TuningSpec {
        &self.tuning
    }

    pub fn rounds(&self) -> &[RoundConfig<M>] {
        &self.rounds
    }

    pub const fn basecase(&self) -> &BasecaseConfig<M::Target> {
        self.basecase.config()
    }

    pub const fn basecase_plan(&self) -> &BasecasePlan<M::Target> {
        &self.basecase
    }

    /// `true` if every PoW slot's difficulty fits within `security.pow_budget`.
    pub fn check_pow_bits(&self) -> bool {
        self.validate_pow_budget().is_ok()
    }

    /// Returns `true` if every post-construction invariant holds.
    pub fn check_all_invariants(&self) -> bool {
        self.validate().is_ok()
    }

    /// Run every post-construction invariant check.
    ///
    /// Shape chaining runs first: the analytic recomputes behind the PoW
    /// checks assume per-config shape invariants hold.
    pub fn validate(&self) -> Result<(), DeriveError> {
        self.validate_round_chaining()?;
        self.validate_pow_budget()?;
        self.validate_security_target_met()?;
        Ok(())
    }

    /// Every PoW slot in the plan, in round order followed by the basecase.
    ///
    /// The single source of truth for slot enumeration: budget validation,
    /// target validation, and the per-slot test assertions all iterate this.
    /// A sub-protocol added here is automatically covered by every check.
    pub(crate) fn pow_slots(&self) -> impl Iterator<Item = PowSlot> + '_ {
        let round_slots = self.rounds.iter().flat_map(|r| {
            let mask_info = r.mask_oracle_info();
            let index = r.round_index;
            let cs = r.code_switch.config();
            let sumcheck = PowSlot {
                kind: Pow::RoundSumcheck { index },
                pow: r.sumcheck.round_pow(),
                recorded: r.sumcheck.analytic(),
                recompute: sumcheck_params::analytic_error_bits(cs.source(), mask_info),
            };
            let code_switch = PowSlot {
                kind: Pow::RoundCodeSwitch { index },
                pow: cs.pow(),
                recorded: r.code_switch.analytic(),
                recompute: code_switch_params::analytic_error_bits(
                    cs.source(),
                    cs.target(),
                    cs.out_domain_samples(),
                    mask_info,
                ),
            };
            let mask_slots = r
                .mask_oracle()
                .map(|mo| {
                    [mo.sumcheck_masks(), mo.cs_mask()].map(|mp| PowSlot {
                        kind: Pow::RoundMaskProximity { index },
                        pow: mp.pow(),
                        recorded: mp.analytic(),
                        recompute: mask_proximity_params::analytic_error_bits(
                            mp.c_zk_commit(),
                            mp.num_masks(),
                        ),
                    })
                })
                .into_iter()
                .flatten();
            [Some(sumcheck), Some(code_switch)]
                .into_iter()
                .flatten()
                .chain(mask_slots)
        });

        let basecase = self.basecase.config();
        let basecase_sumcheck = PowSlot {
            kind: Pow::BasecaseSumcheck,
            pow: basecase.sumcheck().round_pow(),
            recorded: self.basecase.sumcheck_analytic(),
            recompute: sumcheck_params::analytic_error_bits(basecase.commit(), None),
        };
        let gamma = basecase.is_zk().then(|| PowSlot {
            kind: Pow::BasecaseGammaCombination,
            pow: basecase.pow(),
            recorded: self.basecase.gamma_analytic(),
            recompute: basecase_params::analytic_error_bits(basecase.commit()),
        });

        round_slots.chain([Some(basecase_sumcheck), gamma].into_iter().flatten())
    }

    /// For each PoW slot: verify (a) the analytic-bits floor recorded at
    /// solve time still matches a fresh recompute from the config's current
    /// state, and (b) `recorded + pow.difficulty() ≥ target_security_bits`.
    ///
    /// `grind_to_at` guarantees (b) at solve time. If (a) holds, (b) holds
    /// trivially. If (a) drifts, (b) may fail — most often because a planner
    /// regression overwrote an IRS field after the solver consumed it.
    ///
    /// `EPS` matches the `assert_pow_closes_gap` slack used by the per-slot
    /// proptest helper, so validation stays consistent with test-time
    /// assertions.
    pub fn validate_security_target_met(&self) -> Result<(), DeriveError> {
        const EPS: f64 = 1e-3;
        let eps = Bits::new(EPS);
        let target = Bits::new(f64::from(self.security.target_security_bits));
        for slot in self.pow_slots() {
            if slot.recorded.abs_diff(slot.recompute) > eps {
                return Err(DeriveError::AnalyticDrift {
                    pow: slot.kind,
                    recorded: slot.recorded,
                    recompute: slot.recompute,
                });
            }
            let pow_bits = slot.pow.difficulty();
            if slot.recorded + pow_bits + eps < target {
                return Err(DeriveError::SecurityTargetNotMet {
                    pow: slot.kind,
                    analytic: slot.recorded,
                    pow_bits,
                    target,
                });
            }
        }
        Ok(())
    }

    /// PoW slot difficulty ≤ `security.pow_budget` for every slot.
    pub fn validate_pow_budget(&self) -> Result<(), DeriveError> {
        let max = Bits::new(f64::from(self.security.pow_budget.bits()));
        for slot in self.pow_slots() {
            let required = slot.pow.difficulty();
            if required > max {
                return Err(DeriveError::PowBudgetExceeded {
                    pow: slot.kind,
                    required,
                    max,
                });
            }
        }
        Ok(())
    }

    /// Cross-round shape chaining:
    /// - adjacent rounds: `round[i+1].source.vector_size == round[i].target.vector_size`
    /// - last round → basecase: `basecase.commit.vector_size == last.target.vector_size`
    /// - no rounds: `basecase.commit.vector_size == tuning.vector_size`
    pub fn validate_round_chaining(&self) -> Result<(), DeriveError> {
        for window in self.rounds.windows(2) {
            let prev = &window[0];
            let next = &window[1];
            let expected = prev.code_switch.target().vector_size();
            let found = next.code_switch.source().vector_size();
            if expected != found {
                return Err(DeriveError::RoundChainBroken {
                    from: ChainSource::Round(prev.round_index),
                    to: ChainTarget::NextRound(next.round_index),
                    expected,
                    found,
                });
            }
        }

        let basecase_vector_size = self.basecase.commit().vector_size();
        let expected = self.rounds.last().map_or(self.tuning.vector_size, |last| {
            last.code_switch.target().vector_size()
        });
        if expected != basecase_vector_size {
            let from = self
                .rounds
                .last()
                .map_or(ChainSource::Tuning, |r| ChainSource::Round(r.round_index));
            return Err(DeriveError::RoundChainBroken {
                from,
                to: ChainTarget::Basecase,
                expected,
                found: basecase_vector_size,
            });
        }

        Ok(())
    }

    /// HVZK privacy error in bits, summed across ZK rounds:
    /// `−log Σ_r (t_ood_r² + t_ood_r) / (2|F|)` (bounds doc, §5.3 + §5.7).
    pub fn privacy_error_bits(&self) -> Bits {
        let field_bits = <M::Target as FieldWithSize>::field_size_bits();
        let mut total_error = 0.0_f64;
        for r in &self.rounds {
            if let RoundMode::ZeroKnowledge { t_ood, .. } = &r.mode {
                let t = usize_to_f64(t_ood.get());
                let log_err = f64::midpoint(t * t, t).log2() - field_bits;
                total_error += 2_f64.powf(log_err);
            }
        }
        if total_error == 0.0 {
            return Bits::new(f64::from(self.security.target_security_bits));
        }
        Bits::new((-total_error.log2()).max(0.0))
    }
}

impl<M: Embedding> ProtocolConfig<M> {
    /// Analytic soundness bits (excluding PoW).
    pub fn analytic_bits(&self) -> Bits {
        let mut min_bits = f64::from(self.basecase.config().analytic_bits());
        for round in &self.rounds {
            min_bits = min_bits.min(f64::from(round.analytic_bits()));
        }
        Bits::new(min_bits.max(0.0))
    }
}

/// One PoW grind slot in a derived plan: its identity, configured grind, the
/// analytic floor recorded at solve time, and a fresh recompute of that floor
/// for drift detection.
pub struct PowSlot {
    pub kind: Pow,
    pub pow: PowConfig,
    pub recorded: Bits,
    pub recompute: Bits,
}

/// Output of [`super::basecase::solve`]: the runtime basecase config plus the
/// analytic floors its two PoW slots were ground against.
///
/// `gamma_analytic` is always computed (the Lemma 7.4 formula is well-defined
/// in both modes) but only enters validation when the basecase is ZK — the
/// γ slot does not exist otherwise.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct BasecasePlan<F: Field> {
    config: BasecaseConfig<F>,
    sumcheck_analytic: Bits,
    gamma_analytic: Bits,
}

impl<F: Field> BasecasePlan<F> {
    pub(crate) const fn new(
        config: BasecaseConfig<F>,
        sumcheck_analytic: Bits,
        gamma_analytic: Bits,
    ) -> Self {
        Self {
            config,
            sumcheck_analytic,
            gamma_analytic,
        }
    }

    pub const fn config(&self) -> &BasecaseConfig<F> {
        &self.config
    }

    pub const fn sumcheck_analytic(&self) -> Bits {
        self.sumcheck_analytic
    }

    pub const fn gamma_analytic(&self) -> Bits {
        self.gamma_analytic
    }
}

impl<F: Field> std::ops::Deref for BasecasePlan<F> {
    type Target = BasecaseConfig<F>;

    fn deref(&self) -> &BasecaseConfig<F> {
        &self.config
    }
}

#[cfg(test)]
impl<M: Embedding> ProtocolConfig<M> {
    pub(crate) const fn override_basecase_pow_for_test(&mut self, pow: PowConfig) {
        self.basecase.config.set_pow_for_test(pow);
    }

    pub(crate) fn truncate_rounds_for_test(&mut self, len: usize) {
        self.rounds.truncate(len);
    }

    pub(crate) fn corrupt_round_target_vector_size_for_test(
        &mut self,
        round_idx: usize,
        new_size: usize,
    ) {
        self.rounds[round_idx]
            .code_switch
            .config_mut_for_test()
            .target_mut_for_test()
            .set_vector_size_for_test(new_size);
    }

    pub(crate) fn corrupt_round_sumcheck_analytic_for_test(
        &mut self,
        round_idx: usize,
        new_value: Bits,
    ) {
        self.rounds[round_idx]
            .sumcheck
            .set_analytic_for_test(new_value);
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RoundConfig<M: Embedding> {
    round_index: usize,
    sumcheck: Solved<SumcheckConfig<M::Target>>,
    code_switch: Solved<CodeSwitchConfig<M>>,
    mode: RoundMode<M::Target>,
}

impl<M: Embedding> RoundConfig<M> {
    pub(crate) const fn new(
        round_index: usize,
        sumcheck: Solved<SumcheckConfig<M::Target>>,
        code_switch: Solved<CodeSwitchConfig<M>>,
        mode: RoundMode<M::Target>,
    ) -> Self {
        Self {
            round_index,
            sumcheck,
            code_switch,
            mode,
        }
    }

    pub const fn round_index(&self) -> usize {
        self.round_index
    }

    pub const fn sumcheck(&self) -> &Solved<SumcheckConfig<M::Target>> {
        &self.sumcheck
    }

    pub const fn code_switch(&self) -> &Solved<CodeSwitchConfig<M>> {
        &self.code_switch
    }

    pub const fn mode(&self) -> &RoundMode<M::Target> {
        &self.mode
    }

    /// Borrow the round's mask oracle if this is a ZK round.
    pub fn mask_oracle(&self) -> Option<&MaskOracleConfig<M::Target>> {
        match &self.mode {
            RoundMode::Standard => None,
            RoundMode::ZeroKnowledge { mask_oracle, .. } => Some(mask_oracle),
        }
    }

    /// Slim mask-oracle view derived from `mask_oracle()`.
    pub fn mask_oracle_info(&self) -> Option<MaskOracleInfo> {
        self.mask_oracle().map(MaskOracleConfig::info)
    }
}

/// Standard vs. ZK round.
///
/// The ZK payload owns the round's mask oracle, so a mode/oracle mismatch is
/// unrepresentable. Boxed to keep the `Standard` variant from paying the
/// oracle's footprint.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub enum RoundMode<F: Field> {
    Standard,
    ZeroKnowledge {
        /// Lemma 9.9 OOD-sample budget (bounds doc §5.2).
        t_ood: OodSampleBudget,
        /// Sized for this round's `k + 1` masks.
        mask_oracle: Box<MaskOracleConfig<F>>,
    },
}

impl<F: Field> RoundMode<F> {
    pub const fn is_zk(&self) -> bool {
        matches!(self, Self::ZeroKnowledge { .. })
    }
}

impl<M: Embedding> RoundConfig<M> {
    /// Round-level analytic floor: the smallest of `sumcheck`, `code_switch`,
    /// and (when present) the per-round mask-oracle proximity check.
    pub fn analytic_bits(&self) -> Bits {
        let source = &self.code_switch.source();
        let target = &self.code_switch.target();
        let mask_info = self.mask_oracle_info();

        let sumcheck_term = f64::from(sumcheck_params::analytic_error_bits(source, mask_info));
        let code_switch_term = f64::from(code_switch_params::analytic_error_bits(
            source,
            target,
            self.code_switch.out_domain_samples(),
            mask_info,
        ));
        let mask_oracle_term = self
            .mask_oracle()
            .map_or(f64::INFINITY, |mo| f64::from(mo.analytic_bits()));

        Bits::new(
            sumcheck_term
                .min(code_switch_term)
                .min(mask_oracle_term)
                .max(0.0),
        )
    }
}

/// One round's mask oracle, split across two independent C_zk trees:
///   - `sumcheck_masks`: the `k` sumcheck masks (Lemma 6.4), each of length
///     `next_pow_2(mask_length)`. Committed BEFORE sumcheck.
///   - `cs_mask`: the single `(r ‖ s)` code-switch mask (Construction 9.7),
///     length `ℓ_zk`. Committed AFTER sumcheck so its `r` part can carry the
///     folded source-IRS randomness.
///
/// Both trees share the same C_zk code rate, so `info()` exposes a single
/// shared list-size to downstream solvers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct MaskOracleConfig<F: Field> {
    sumcheck_masks: Solved<MaskProximityConfig<F>>,
    cs_mask: Solved<MaskProximityConfig<F>>,
    /// `next_pow2(r + t_ood)` for this round (Lemma 9.3).
    l_zk: MaskCodeMessageLen,
    /// Lemma 9.9 OOD-sample budget (bounds doc §5.2).
    t_ood: OodSampleBudget,
}

impl<F: Field> MaskOracleConfig<F> {
    pub(crate) const fn new(
        sumcheck_masks: Solved<MaskProximityConfig<F>>,
        cs_mask: Solved<MaskProximityConfig<F>>,
        l_zk: MaskCodeMessageLen,
        t_ood: OodSampleBudget,
    ) -> Self {
        Self {
            sumcheck_masks,
            cs_mask,
            l_zk,
            t_ood,
        }
    }

    pub const fn sumcheck_masks(&self) -> &Solved<MaskProximityConfig<F>> {
        &self.sumcheck_masks
    }

    pub const fn cs_mask(&self) -> &Solved<MaskProximityConfig<F>> {
        &self.cs_mask
    }

    pub const fn l_zk(&self) -> MaskCodeMessageLen {
        self.l_zk
    }

    pub const fn t_ood(&self) -> OodSampleBudget {
        self.t_ood
    }

    pub fn info(&self) -> MaskOracleInfo {
        // Both sub-trees use the same C_zk rate, so either's list size is the
        // shared C_zk list size.
        MaskOracleInfo {
            c_zk_list_size: ListSize::new(self.cs_mask.c_zk_commit().list_size()),
            l_zk: self.l_zk,
        }
    }
}

/// Slim mask-oracle view (C_zk's list size + ℓ_zk).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MaskOracleInfo {
    pub c_zk_list_size: ListSize,
    pub l_zk: MaskCodeMessageLen,
}

impl<F: Field> MaskOracleConfig<F> {
    /// Analytic soundness bits (excluding PoW) for this round's mask oracle.
    /// Returns the minimum of both sub-trees.
    pub fn analytic_bits(&self) -> Bits {
        Bits::new(f64::from(self.sumcheck_masks.analytic()).min(f64::from(self.cs_mask.analytic())))
    }
}
