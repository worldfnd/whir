//! Derived commitments, mask layouts, and per-step security accounting.

use std::ops::Range;

use ark_ff::Field;
use serde::{Deserialize, Deserializer, Serialize};

use crate::{
    algebra::embedding::{Embedding, Identity},
    protocols::{
        basecase::Config as BasecaseConfig,
        code_switch::Config as CodeSwitchConfig,
        irs_commit::Config as IrsConfig,
        params::{
            error::{ChainSource, ChainTarget, DeriveError, ZookSecurityError},
            masks::{mask_coefficients, JointMaskConfig, RoundMasks, WitnessMaskConfig},
            spec::{Mode, SecuritySpec, TuningSpec},
        },
        sumcheck::Config as SumcheckConfig,
    },
};

/// Commitments and mask lengths selected before the prover samples randomness.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
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

/// Unchecked input exists only while deserializing.
#[derive(Deserialize)]
#[serde(bound = "")]
struct RawProtocolConfig<M: Embedding> {
    security: SecuritySpec,
    tuning: TuningSpec,
    padded_vector_size: usize,
    first_round: Option<RoundConfig<M>>,
    tail_rounds: Vec<RoundConfig<Identity<M::Target>>>,
    basecase: BasecasePlan<M::Target>,
    joint_masks: Option<JointMaskConfig<M::Target>>,
}

impl<'de, M: Embedding + Default> Deserialize<'de> for ProtocolConfig<M> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = RawProtocolConfig::<M>::deserialize(deserializer)?;
        let config = Self::new(
            raw.security,
            raw.tuning,
            raw.padded_vector_size,
            raw.first_round,
            raw.tail_rounds,
            raw.basecase,
            raw.joint_masks,
        );
        // Reconstruction is the only validation path for external configurations.
        // Compare before inspecting any untrusted nested dimensions or bounds.
        let canonical = Self::derive(config.security.clone(), config.tuning.clone())
            .map_err(serde::de::Error::custom)?;
        if config != canonical {
            return Err(serde::de::Error::custom(
                "configuration differs from its derived parameters",
            ));
        }
        Ok(canonical)
    }
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

    fn round_masks(&self) -> impl Iterator<Item = Option<&RoundMasks>> {
        self.first_round
            .iter()
            .map(RoundConfig::masks)
            .chain(self.tail_rounds.iter().map(RoundConfig::masks))
    }

    /// Validate dimensions, mask coverage, and every configured security bound.
    pub fn validate(&self) -> Result<(), DeriveError> {
        self.validate_round_chaining()?;
        self.validate_masks()?;
        self.validate_security()
    }

    fn validate_round_chaining(&self) -> Result<(), DeriveError> {
        let mut expected = self.padded_vector_size;
        let mut from = ChainSource::Tuning;
        let dimensions = self
            .first_round
            .iter()
            .map(RoundConfig::dimensions)
            .chain(self.tail_rounds.iter().map(RoundConfig::dimensions));
        for (index, (round_index, source_size, target_size)) in dimensions.enumerate() {
            if round_index != index {
                return Err(ZookSecurityError::InvalidRoundShape.into());
            }
            if source_size != expected {
                return Err(DeriveError::RoundChainBroken {
                    from,
                    to: ChainTarget::NextRound(round_index),
                    expected,
                    found: source_size,
                });
            }
            expected = target_size;
            from = ChainSource::Round(round_index);
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
        for result in self
            .first_round
            .iter()
            .map(|round| round.validate_mask(is_zk))
            .chain(
                self.tail_rounds
                    .iter()
                    .map(|round| round.validate_mask(is_zk)),
            )
        {
            result?;
        }
        if let Some(joint) = &self.joint_masks {
            if joint.sumcheck_masks.len() != self.num_rounds()
                || joint.witness_masks.len() != self.num_rounds() + 1
            {
                return Err(DeriveError::InvalidMaskLayout);
            }
            let mut end = 0;
            for (masks, range) in self.round_masks().zip(&joint.sumcheck_masks) {
                let masks = masks.ok_or(DeriveError::InvalidMaskLayout)?;
                check_range(range, &mut end, masks.sumcheck_coefficients)?;
            }
            for (masks, range) in self.round_masks().zip(&joint.witness_masks) {
                let masks = masks.ok_or(DeriveError::InvalidMaskLayout)?;
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
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RoundConfig<M: Embedding> {
    round_index: usize,
    sumcheck: SumcheckConfig<M::Target>,
    code_switch: CodeSwitchConfig<M>,
    masks: Option<RoundMasks>,
}

impl<M: Embedding> RoundConfig<M> {
    pub(crate) const fn new(
        round_index: usize,
        sumcheck: SumcheckConfig<M::Target>,
        code_switch: CodeSwitchConfig<M>,
        masks: Option<RoundMasks>,
    ) -> Self {
        Self {
            round_index,
            sumcheck,
            code_switch,
            masks,
        }
    }
    pub const fn round_index(&self) -> usize {
        self.round_index
    }
    pub const fn sumcheck(&self) -> &SumcheckConfig<M::Target> {
        &self.sumcheck
    }
    pub const fn code_switch(&self) -> &CodeSwitchConfig<M> {
        &self.code_switch
    }
    pub const fn masks(&self) -> Option<&RoundMasks> {
        self.masks.as_ref()
    }

    const fn dimensions(&self) -> (usize, usize, usize) {
        (
            self.round_index,
            self.code_switch.source().vector_size(),
            self.code_switch.target().vector_size(),
        )
    }

    fn validate_mask(&self, is_zk: bool) -> Result<(), DeriveError> {
        if is_zk != self.masks().is_some() {
            return Err(DeriveError::InvalidMaskLayout);
        }
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
}

/// Final witness commitment and the protocol used to open it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub enum BasecasePlan<F: Field> {
    Standard {
        config: BasecaseConfig<F>,
    },
    ZeroKnowledge {
        commit: IrsConfig<Identity<F>>,
        witness_mask: WitnessMaskConfig,
        /// Reduce the revealed combined witness against the implicit kernel.
        sumcheck: SumcheckConfig<F>,
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
