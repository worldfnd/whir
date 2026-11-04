//! Derived commitments, mask layouts, and per-step security accounting.

use std::ops::Range;

use ark_ff::Field;
use serde::{Deserialize, Serialize};

use crate::{
    algebra::embedding::{Embedding, Identity},
    bits::Bits,
    protocols::{
        basecase::Config as BasecaseConfig,
        code_switch::Config as CodeSwitchConfig,
        irs_commit::Config as IrsConfig,
        params::{
            basecase::ZookFinalOpeningSoundness,
            code_switch::{analytic_error_bits as code_switch_error_bits, zook_query_error_bits},
            error::{ChainSource, ChainTarget, DeriveError, Pow, ZookSecurityError},
            mask_proximity::zook_mask_binding_bits,
            solved::Solved,
            spec::{Mode, SecuritySpec, TuningSpec},
            sumcheck::{
                analytic_error_bits as sumcheck_error_bits, mask_coefficients,
                revealed_sumcheck_error_bits, zook_sumcheck_error_bits,
            },
        },
        proof_of_work::Config as PowConfig,
        sumcheck::{Config as SumcheckConfig, SumcheckMode},
    },
};

/// Commitments and mask lengths selected before the prover samples randomness.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProtocolConfig<M: Embedding> {
    security: SecuritySpec,
    tuning: TuningSpec,
    padded_vector_size: usize,
    first_round: Option<RoundConfig<M>>,
    tail_rounds: Vec<RoundConfig<Identity<M::Target>>>,
    basecase: BasecasePlan<M::Target>,
    joint_masks: Option<JointMaskConfig<M::Target>>,
}

impl<M: Embedding> ProtocolConfig<M> {
    pub(crate) const fn new(
        security: SecuritySpec,
        tuning: TuningSpec,
        padded_vector_size: usize,
        first_round: Option<RoundConfig<M>>,
        tail_rounds: Vec<RoundConfig<Identity<M::Target>>>,
        basecase: BasecasePlan<M::Target>,
        joint_masks: Option<JointMaskConfig<M::Target>>,
    ) -> Self {
        Self {
            security,
            tuning,
            padded_vector_size,
            first_round,
            tail_rounds,
            basecase,
            joint_masks,
        }
    }

    pub const fn security(&self) -> &SecuritySpec {
        &self.security
    }
    pub const fn tuning(&self) -> &TuningSpec {
        &self.tuning
    }

    /// Witness length after deterministic zero padding, before masking.
    pub const fn padded_vector_size(&self) -> usize {
        self.padded_vector_size
    }

    pub const fn first_round(&self) -> Option<&RoundConfig<M>> {
        self.first_round.as_ref()
    }
    pub fn tail_rounds(&self) -> &[RoundConfig<Identity<M::Target>>] {
        &self.tail_rounds
    }
    pub const fn has_rounds(&self) -> bool {
        self.first_round.is_some()
    }
    pub const fn basecase(&self) -> &BasecasePlan<M::Target> {
        &self.basecase
    }
    pub const fn basecase_plan(&self) -> &BasecasePlan<M::Target> {
        &self.basecase
    }

    /// One commitment containing every sparse sumcheck mask and witness mask.
    pub const fn joint_masks(&self) -> Option<&JointMaskConfig<M::Target>> {
        self.joint_masks.as_ref()
    }

    pub fn num_rounds(&self) -> usize {
        usize::from(self.first_round.is_some()) + self.tail_rounds.len()
    }

    /// Mask layout at a witness commitment, including the final commitment.
    ///
    /// Index zero selects the initial witness; `num_rounds()` selects the final one.
    pub fn witness_mask(&self, index: usize) -> Option<WitnessMaskConfig> {
        assert!(index <= self.num_rounds());
        if index == self.num_rounds() {
            self.basecase.witness_mask().copied()
        } else if index == 0 {
            self.first_round
                .as_ref()
                .and_then(|round| round.masks())
                .map(|masks| masks.witness)
        } else {
            self.tail_rounds[index - 1]
                .masks()
                .map(|masks| masks.witness)
        }
    }

    fn round_views(&self) -> impl Iterator<Item = &dyn RoundMetrics> + '_ {
        self.first_round
            .iter()
            .map(|r| r as &dyn RoundMetrics)
            .chain(self.tail_rounds.iter().map(|r| r as &dyn RoundMetrics))
    }

    pub fn check_pow_bits(&self) -> bool {
        self.validate_pow_budget().is_ok()
    }
    pub fn check_all_invariants(&self) -> bool {
        self.validate().is_ok()
    }

    /// Validate dimensions, mask coverage, and every configured security bound.
    pub fn validate(&self) -> Result<(), DeriveError> {
        self.validate_round_chaining()?;
        self.validate_masks()?;
        self.validate_pow_budget()?;
        self.validate_security_target_met()
    }

    pub fn validate_round_chaining(&self) -> Result<(), DeriveError> {
        let mut expected = self.padded_vector_size;
        let mut from = ChainSource::Tuning;
        for (index, round) in self.round_views().enumerate() {
            if round.round_index() != index {
                return Err(ZookSecurityError::InvalidRoundShape.into());
            }
            if round.source_vector_size() != expected {
                return Err(DeriveError::RoundChainBroken {
                    from,
                    to: ChainTarget::NextRound(round.round_index()),
                    expected,
                    found: round.source_vector_size(),
                });
            }
            expected = round.target_vector_size();
            from = ChainSource::Round(round.round_index());
        }
        if self.basecase.commit().vector_size() != expected {
            return Err(DeriveError::RoundChainBroken {
                from,
                to: ChainTarget::Basecase,
                expected,
                found: self.basecase.commit().vector_size(),
            });
        }
        let mut previous = self
            .first_round
            .as_ref()
            .map(|round| round.code_switch.target());
        for round in &self.tail_rounds {
            if previous != Some(round.code_switch.source()) {
                return Err(ZookSecurityError::InvalidRoundShape.into());
            }
            previous = Some(round.code_switch.target());
        }
        if previous.is_some_and(|commit| commit != self.basecase.commit()) {
            return Err(ZookSecurityError::InvalidRoundShape.into());
        }
        if let BasecasePlan::ZeroKnowledge {
            commit, sumcheck, ..
        } = &self.basecase
        {
            if sumcheck.initial_size() != commit.vector_size()
                || sumcheck.num_rounds() != commit.vector_size().trailing_zeros() as usize
                || !matches!(sumcheck.mode(), SumcheckMode::Standard)
                || sumcheck.round_pow() != PowConfig::none()
            {
                return Err(ZookSecurityError::InvalidRoundShape.into());
            }
        }
        Ok(())
    }

    fn validate_masks(&self) -> Result<(), DeriveError> {
        if self.padded_vector_size < self.tuning.vector_size
            || !self.padded_vector_size.is_power_of_two()
        {
            return Err(DeriveError::InvalidMaskLayout);
        }
        let is_zk = matches!(self.security.mode, Mode::ZeroKnowledge);
        if is_zk != self.joint_masks.is_some() || is_zk != self.basecase.is_zk() {
            return Err(DeriveError::InvalidMaskLayout);
        }
        for round in self.round_views() {
            if is_zk != round.masks().is_some() {
                return Err(DeriveError::InvalidMaskLayout);
            }
            round.validate_mask()?;
        }
        if let Some(joint) = &self.joint_masks {
            let rounds: Vec<_> = self.round_views().collect();
            if joint.sumcheck_masks.len() != rounds.len()
                || joint.witness_masks.len() != rounds.len() + 1
            {
                return Err(DeriveError::InvalidMaskLayout);
            }
            let mut end = 0;
            for (round, range) in rounds.iter().zip(&joint.sumcheck_masks) {
                let masks = round.masks().ok_or(DeriveError::InvalidMaskLayout)?;
                check_range(range, &mut end, masks.sumcheck_coefficients)?;
            }
            for (round, range) in rounds.iter().zip(&joint.witness_masks) {
                let masks = round.masks().ok_or(DeriveError::InvalidMaskLayout)?;
                check_range(range, &mut end, masks.witness.num_coefficients()?)?;
            }
            let final_mask = self
                .basecase
                .witness_mask()
                .ok_or(DeriveError::InvalidMaskLayout)?;
            final_mask.validate(self.basecase.commit(), usize::from(self.has_rounds()))?;
            let final_range = joint
                .witness_masks
                .last()
                .ok_or(DeriveError::InvalidMaskLayout)?;
            check_range(final_range, &mut end, final_mask.num_coefficients()?)?;
            let blinder_len = joint
                .commit
                .in_domain_samples()
                .checked_add(1)
                .ok_or(DeriveError::SizeOverflow)?;
            check_range(&joint.blinder, &mut end, blinder_len)?;
            let degree = if self.has_rounds() {
                embedding_degree::<M>()?
            } else {
                1
            };
            check_range(&joint.basefield_blinder, &mut end, degree)?;
            if end > joint.commit.vector_size() {
                return Err(DeriveError::InvalidMaskLayout);
            }
        }
        Ok(())
    }

    pub(super) fn pow_slots(&self) -> Result<Vec<PowSlot>, DeriveError> {
        let mut slots = Vec::new();
        for round in self.round_views() {
            slots.extend(round.pow_slots()?);
        }
        match &self.basecase {
            BasecasePlan::Standard {
                config,
                sumcheck_analytic,
            } => slots.push(PowSlot {
                kind: Pow::BasecaseSumcheck,
                pow: config.sumcheck().round_pow(),
                recorded: *sumcheck_analytic,
                recompute: sumcheck_error_bits(config.commit(), false)?,
            }),
            BasecasePlan::ZeroKnowledge {
                commit, sumcheck, ..
            } => {
                let joint = self
                    .joint_masks
                    .as_ref()
                    .ok_or(DeriveError::InvalidMaskLayout)?;
                let opening = ZookFinalOpeningSoundness::for_codes(commit, &joint.commit)?;
                slots.extend([
                    PowSlot {
                        kind: Pow::BasecaseSumcheck,
                        pow: sumcheck.round_pow(),
                        recorded: sumcheck.analytic(),
                        recompute: revealed_sumcheck_error_bits::<M::Target>(),
                    },
                    PowSlot {
                        kind: Pow::JointMaskBinding,
                        pow: PowConfig::none(),
                        recorded: joint.commit.analytic(),
                        recompute: zook_mask_binding_bits(&joint.commit)?,
                    },
                    PowSlot {
                        kind: Pow::FinalOpeningCombination,
                        pow: PowConfig::none(),
                        recorded: joint.opening.combination,
                        recompute: opening.combination,
                    },
                    PowSlot {
                        kind: Pow::FinalOpeningQueries,
                        pow: PowConfig::none(),
                        recorded: joint.opening.queries,
                        recompute: opening.queries,
                    },
                ]);
            }
        }
        Ok(slots)
    }

    pub fn validate_security_target_met(&self) -> Result<(), DeriveError> {
        let eps = Bits::new(1e-9);
        let target = Bits::new(f64::from(self.security.target_security_bits));
        for slot in self.pow_slots()? {
            if slot.recorded.abs_diff(slot.recompute) > eps {
                return Err(DeriveError::AnalyticDrift {
                    pow: slot.kind,
                    recorded: slot.recorded,
                    recompute: slot.recompute,
                });
            }
            // Zook's IOP bounds receive no Fiat–Shamir or grinding credit.
            let pow_bits = if matches!(self.security.mode, Mode::Standard) {
                slot.pow.difficulty()
            } else {
                Bits::new(0.0)
            };
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

    pub fn validate_pow_budget(&self) -> Result<(), DeriveError> {
        let max = Bits::new(f64::from(self.security.pow_budget.bits()));
        for slot in self.pow_slots()? {
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

    /// Minimum recorded per-step security; excludes grinding credit.
    pub fn analytic_bits(&self) -> Bits {
        let mut bits = self.basecase.analytic_bits();
        for round in self.round_views() {
            bits = bits.min(round.analytic());
        }
        if let Some(joint) = &self.joint_masks {
            bits = bits
                .min(joint.commit.analytic())
                .min(joint.opening.combination)
                .min(joint.opening.queries);
        }
        bits
    }
}

/// Prefix mask for each contiguous coefficient block encoded by IRS.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WitnessMaskConfig {
    /// Independent encoded polynomials in this commitment.
    pub num_polynomials: usize,
    /// Random prefix length in each polynomial.
    pub coefficients_per_polynomial: usize,
}

impl WitnessMaskConfig {
    /// Cover every opened row and incoming DEEP evaluation without increasing degree.
    ///
    /// Section 5.4, “ZO0K”, “The required masks”, budgets openings per witness.
    /// IRS rows reveal every column, so each polynomial receives its own prefix.
    /// An incoming DEEP evaluation reserves one extra coefficient per polynomial.
    /// This includes the final witness: it is evaluated before the final opening.
    pub(crate) fn for_code<M: Embedding>(
        code: &IrsConfig<M>,
        incoming_deep: usize,
    ) -> Result<Self, DeriveError> {
        let config = Self {
            num_polynomials: code.interleaving_depth(),
            coefficients_per_polynomial: code
                .in_domain_samples()
                .checked_add(incoming_deep)
                .ok_or(DeriveError::SizeOverflow)?,
        };
        config.validate(code, incoming_deep)?;
        Ok(config)
    }

    /// Number of random coefficients to sample for this witness commitment.
    pub fn num_coefficients(self) -> Result<usize, DeriveError> {
        self.num_polynomials
            .checked_mul(self.coefficients_per_polynomial)
            .ok_or(DeriveError::SizeOverflow)
    }

    fn validate<M: Embedding>(
        &self,
        code: &IrsConfig<M>,
        incoming_deep: usize,
    ) -> Result<(), DeriveError> {
        let required = code
            .in_domain_samples()
            .checked_add(incoming_deep)
            .ok_or(DeriveError::SizeOverflow)?;
        if code.num_vectors() != 1
            || self.num_polynomials != code.interleaving_depth()
            || self.coefficients_per_polynomial != required
            || code.mask_length() != 0
        {
            return Err(DeriveError::InvalidMaskLayout);
        }
        if required > code.message_length() {
            return Err(DeriveError::MaskDoesNotFit {
                required,
                available: code.message_length(),
            });
        }
        Ok(())
    }
}

/// Masks consumed by one folding round.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoundMasks {
    /// Independent overlaid prefix in every source polynomial.
    pub witness: WitnessMaskConfig,
    /// One constant and two coefficients per folded variable: `2 * folds + 1`.
    pub sumcheck_coefficients: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoundMode {
    Standard,
    ZeroKnowledge(RoundMasks),
}

impl RoundMode {
    pub const fn is_zk(&self) -> bool {
        matches!(self, Self::ZeroKnowledge(_))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RoundConfig<M: Embedding> {
    round_index: usize,
    sumcheck: Solved<SumcheckConfig<M::Target>>,
    code_switch: Solved<CodeSwitchConfig<M>>,
    mode: RoundMode,
}

impl<M: Embedding> RoundConfig<M> {
    pub(crate) const fn new(
        round_index: usize,
        sumcheck: Solved<SumcheckConfig<M::Target>>,
        code_switch: Solved<CodeSwitchConfig<M>>,
        mode: RoundMode,
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
    pub const fn mode(&self) -> &RoundMode {
        &self.mode
    }
    pub const fn masks(&self) -> Option<&RoundMasks> {
        match &self.mode {
            RoundMode::Standard => None,
            RoundMode::ZeroKnowledge(masks) => Some(masks),
        }
    }
    pub fn analytic_bits(&self) -> Bits {
        self.sumcheck.analytic().min(self.code_switch.analytic())
    }
}

trait RoundMetrics {
    fn round_index(&self) -> usize;
    fn source_vector_size(&self) -> usize;
    fn target_vector_size(&self) -> usize;
    fn masks(&self) -> Option<&RoundMasks>;
    fn validate_mask(&self) -> Result<(), DeriveError>;
    fn analytic(&self) -> Bits;
    fn pow_slots(&self) -> Result<Vec<PowSlot>, DeriveError>;
}

impl<M: Embedding> RoundMetrics for RoundConfig<M> {
    fn round_index(&self) -> usize {
        self.round_index
    }
    fn source_vector_size(&self) -> usize {
        self.code_switch.source().vector_size()
    }
    fn target_vector_size(&self) -> usize {
        self.code_switch.target().vector_size()
    }
    fn masks(&self) -> Option<&RoundMasks> {
        self.masks()
    }
    fn analytic(&self) -> Bits {
        self.analytic_bits()
    }
    fn validate_mask(&self) -> Result<(), DeriveError> {
        let source = self.code_switch.source();
        let folds =
            u32::try_from(self.sumcheck.num_rounds()).map_err(|_| DeriveError::SizeOverflow)?;
        if self.sumcheck.initial_size() != source.vector_size()
            || 1_usize.checked_shl(folds) != Some(source.interleaving_depth())
            || source.message_length() != self.code_switch.target().vector_size()
        {
            return Err(ZookSecurityError::InvalidRoundShape.into());
        }
        if let Some(masks) = self.masks() {
            if self.code_switch.out_domain_samples() != 1 {
                return Err(ZookSecurityError::InvalidRoundShape.into());
            }
            masks.witness.validate(
                self.code_switch.source(),
                usize::from(self.round_index != 0),
            )?;
            let expected = mask_coefficients(self.sumcheck.num_rounds())?;
            if masks.sumcheck_coefficients != expected {
                return Err(DeriveError::InvalidMaskLayout);
            }
        }
        Ok(())
    }
    fn pow_slots(&self) -> Result<Vec<PowSlot>, DeriveError> {
        let cs = self.code_switch.config();
        let is_zk = self.mode.is_zk();
        let sumcheck = sumcheck_error_bits(cs.source(), is_zk)?;
        let mut slots = vec![
            PowSlot {
                kind: Pow::RoundSumcheck {
                    index: self.round_index,
                },
                pow: self.sumcheck.round_pow(),
                recorded: self.sumcheck.analytic(),
                recompute: sumcheck,
            },
            PowSlot {
                kind: Pow::RoundCodeSwitch {
                    index: self.round_index,
                },
                pow: cs.pow(),
                recorded: self.code_switch.analytic(),
                recompute: code_switch_error_bits(
                    cs.source(),
                    cs.target(),
                    cs.out_domain_samples(),
                    is_zk,
                )?,
            },
        ];
        if is_zk {
            slots.push(PowSlot {
                kind: Pow::RoundMaskCombination {
                    index: self.round_index,
                },
                pow: PowConfig::none(),
                recorded: self.sumcheck.analytic(),
                recompute: sumcheck,
            });
        }
        Ok(slots)
    }
}

/// Final witness commitment and the protocol used to open it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub enum BasecasePlan<F: Field> {
    Standard {
        config: BasecaseConfig<F>,
        sumcheck_analytic: Bits,
    },
    ZeroKnowledge {
        commit: IrsConfig<Identity<F>>,
        witness_mask: WitnessMaskConfig,
        /// Reduce the revealed combined witness against the implicit kernel.
        sumcheck: Solved<SumcheckConfig<F>>,
    },
}

impl<F: Field> BasecasePlan<F> {
    pub const fn commit(&self) -> &IrsConfig<Identity<F>> {
        match self {
            Self::Standard { config, .. } => config.commit(),
            Self::ZeroKnowledge { commit, .. } => commit,
        }
    }
    pub const fn is_zk(&self) -> bool {
        matches!(self, Self::ZeroKnowledge { .. })
    }
    pub const fn witness_mask(&self) -> Option<&WitnessMaskConfig> {
        match self {
            Self::Standard { .. } => None,
            Self::ZeroKnowledge { witness_mask, .. } => Some(witness_mask),
        }
    }
    const fn analytic_bits(&self) -> Bits {
        match self {
            Self::Standard {
                sumcheck_analytic, ..
            } => *sumcheck_analytic,
            Self::ZeroKnowledge { sumcheck, .. } => sumcheck.analytic(),
        }
    }
}

/// One mask commitment, with disjoint ranges in coefficient order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct JointMaskConfig<F: Field> {
    pub(crate) commit: Solved<IrsConfig<Identity<F>>>,
    pub(crate) sumcheck_masks: Vec<Range<usize>>,
    pub(crate) witness_masks: Vec<Range<usize>>,
    pub(crate) blinder: Range<usize>,
    pub(crate) basefield_blinder: Range<usize>,
    pub(crate) opening: ZookFinalOpeningSoundness,
}

impl<F: Field> JointMaskConfig<F> {
    /// Ordinary IRS commitment, with its mask-binding security bound.
    pub const fn commit(&self) -> &Solved<IrsConfig<Identity<F>>> {
        &self.commit
    }
    /// Sparse sumcheck coefficient ranges, in round order.
    pub fn sumcheck_masks(&self) -> &[Range<usize>] {
        &self.sumcheck_masks
    }
    /// Witness mask ranges, from the initial commitment through the final witness.
    pub fn witness_masks(&self) -> &[Range<usize>] {
        &self.witness_masks
    }
    /// Random coefficients covering joint-mask queries and its binding evaluation.
    pub const fn blinder(&self) -> &Range<usize> {
        &self.blinder
    }
    /// Membership-check blinder sampled in the initial witness commitment's field.
    pub const fn basefield_blinder(&self) -> &Range<usize> {
        &self.basefield_blinder
    }
    /// Security bounds for opening the final witness and accumulated mask claims.
    pub const fn opening(&self) -> ZookFinalOpeningSoundness {
        self.opening
    }
}

/// A recorded bound and a fresh recomputation at one protocol step.
pub(super) struct PowSlot {
    pub kind: Pow,
    pub pow: PowConfig,
    pub recorded: Bits,
    pub recompute: Bits,
}

fn check_range(range: &Range<usize>, end: &mut usize, length: usize) -> Result<(), DeriveError> {
    let expected_end = end.checked_add(length).ok_or(DeriveError::SizeOverflow)?;
    if range.start != *end || range.end != expected_end {
        return Err(DeriveError::InvalidMaskLayout);
    }
    *end = expected_end;
    Ok(())
}

pub(super) fn embedding_degree<M: Embedding>() -> Result<usize, DeriveError> {
    let source = M::Source::extension_degree();
    let target = M::Target::extension_degree();
    if source == 0 || !target.is_multiple_of(source) {
        return Err(DeriveError::InvalidMaskLayout);
    }
    usize::try_from(target / source).map_err(|_| DeriveError::SizeOverflow)
}

/// Per-step IOP security conditional on joint-mask binding; excludes hashing and grinding.
/// Each field is `max(0, -log2(epsilon))` for that step's error bound.
///
/// ProveKit Prover Whitepaper §6.4, “Soundness”, theorem “ZO0K soundness under
/// mask binding”.
/// Mask binding is accounted separately by `zook_mask_binding_bits`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZookRoundSoundness {
    /// Sparse-mask batching: degree-one constrained agreement for the source code.
    /// Uses the conservative degree-two bound; §6.4, sumcheck phase, mask combination.
    pub mask_combination: Bits,
    /// Each quadratic sumcheck step: §6.4, sumcheck phase.
    pub sumcheck_step: Bits,
    /// Sum of DEEP and in-domain query errors: §6.4, query phase.
    pub queries: Bits,
    /// Random aggregation of queried constraints: §6.4, aggregation step.
    pub aggregation: Bits,
}

impl ZookRoundSoundness {
    /// Evaluate a round with one DEEP challenge in the target field outside its domain.
    pub fn for_codes<M: Embedding>(
        source: &IrsConfig<M>,
        target: &IrsConfig<Identity<M::Target>>,
    ) -> Result<Self, ZookSecurityError> {
        let (queries, aggregation) = zook_query_error_bits(source, target)?;
        let agreement = zook_sumcheck_error_bits(source)?;
        Ok(Self {
            // The degree-two constraint bound also covers degree-one batching.
            mask_combination: agreement,
            sumcheck_step: agreement,
            queries,
            aggregation,
        })
    }

    /// Minimum per-step security: `min(b_mask, b_sumcheck, b_queries, b_aggregation)`.
    /// This excludes mask binding, the final opening, and any union across steps.
    pub fn minimum_bits(self) -> Bits {
        self.mask_combination
            .min(self.sumcheck_step)
            .min(self.queries)
            .min(self.aggregation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        algebra::{
            embedding::Basefield,
            fields::{Field64_2, FieldWithSize},
        },
        protocols::params::{
            mask_proximity::zook_mask_binding_bits,
            test_utils::{assert_probability_bits, johnson_code},
        },
    };

    #[test]
    fn round_uses_extension_field_and_full_folded_degree() {
        let source = johnson_code::<Basefield<Field64_2>>(256, 4, 0.25);
        let target = johnson_code::<Identity<Field64_2>>(64, 2, 0.125);
        let field = 2.0_f64.powf(Field64_2::field_size_bits());
        let soundness = ZookRoundSoundness::for_codes(&source, &target).unwrap();
        let l = source.list_size();
        assert_probability_bits(
            soundness.sumcheck_step,
            l * ((2.0 / 3.0) * l.powi(4) * source.rate() + 1.0) * source.codeword_length() as f64
                / field,
        );
        assert_eq!(soundness.mask_combination, soundness.sumcheck_step);
        let deep_error = target.list_size().powi(2) / 2.0 * target.vector_size() as f64
            / (field - target.codeword_length() as f64);
        let rho = source.rate();
        let query_error = (rho.sqrt() * 1.05).powf(source.in_domain_samples() as f64);
        assert_probability_bits(soundness.queries, deep_error + query_error);
        assert_probability_bits(
            soundness.aggregation,
            target.list_size() * source.in_domain_samples() as f64 / field,
        );
    }

    #[test]
    fn rejects_invalid_shapes() {
        let interleaved = johnson_code::<Identity<Field64_2>>(128, 2, 0.25);
        assert_eq!(
            zook_mask_binding_bits(&interleaved),
            Err(ZookSecurityError::ExpectedMaskAndBlinder)
        );
        let wrong_target = johnson_code::<Identity<Field64_2>>(32, 1, 0.25);
        assert_eq!(
            ZookRoundSoundness::for_codes(&interleaved, &wrong_target),
            Err(ZookSecurityError::InvalidRoundShape)
        );
    }
}
