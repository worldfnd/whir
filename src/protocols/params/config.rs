//! A canonical commitment sequence and borrowed protocol views.

use ark_ff::Field;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{
    error::DeriveError,
    masks::{mask_coefficients, JointMaskConfig, RoundMasks, WitnessMaskConfig},
    spec::{SecuritySpec, TuningSpec},
};
use crate::{
    algebra::embedding::{Embedding, Identity},
    protocols::{
        basecase::{BasecaseMode, Config as BasecaseConfig},
        code_switch::{CodeSwitchMode, Config as CodeSwitchConfig},
        irs_commit::Config as IrsConfig,
        proof_of_work::Config as PowConfig,
        sumcheck::{Config as SumcheckConfig, SumcheckMaskLen, SumcheckMode},
    },
    type_info::Typed,
};

/// The serialized request is the only external configuration input.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, bound = "")]
pub struct ProtocolSpec<M: Embedding> {
    pub embedding: Typed<M>,
    pub security: SecuritySpec,
    pub tuning: TuningSpec,
}

/// Each IRS configuration occurs once. The last target is the final commitment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolConfig<M: Embedding> {
    pub(super) request: ProtocolSpec<M>,
    pub(super) first: Option<Node<M>>,
    pub(super) targets: Vec<Node<Identity<M::Target>>>,
    pub(super) joint_masks: Option<JointMaskConfig<M::Target>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Node<M: Embedding> {
    pub code: IrsConfig<M>,
    pub sumcheck_pow: PowConfig,
    pub switch: Option<(usize, PowConfig)>,
}

impl<M: Embedding> Serialize for ProtocolConfig<M> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.request.serialize(serializer)
    }
}
impl<'de, M: Embedding + Default> Deserialize<'de> for ProtocolConfig<M> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let request = ProtocolSpec::<M>::deserialize(deserializer)?;
        if request.embedding.0 != M::default() {
            return Err(serde::de::Error::custom(
                "configuration requires the default embedding",
            ));
        }
        Self::derive(request.security, request.tuning).map_err(serde::de::Error::custom)
    }
}

impl<M: Embedding> ProtocolConfig<M> {
    pub const fn security(&self) -> &SecuritySpec {
        &self.request.security
    }
    pub const fn tuning(&self) -> &TuningSpec {
        &self.request.tuning
    }
    pub fn padded_vector_size(&self) -> usize {
        self.first.as_ref().map_or_else(
            || self.final_commit().vector_size(),
            |node| node.code.vector_size(),
        )
    }
    pub fn first_round(&self) -> Option<RoundConfig<'_, M>> {
        self.first
            .as_ref()
            .map(|node| node.round(&self.targets[0].code, 0, self.security().mode.is_zk()))
    }
    pub fn tail_rounds(
        &self,
    ) -> impl ExactSizeIterator<Item = RoundConfig<'_, Identity<M::Target>>> {
        self.targets
            .windows(2)
            .enumerate()
            .map(|(i, pair)| pair[0].round(&pair[1].code, i + 1, self.security().mode.is_zk()))
    }
    pub const fn has_rounds(&self) -> bool {
        self.first.is_some()
    }
    pub fn num_rounds(&self) -> usize {
        self.targets.len() - 1 + usize::from(self.has_rounds())
    }
    pub fn final_commit(&self) -> &IrsConfig<Identity<M::Target>> {
        &self.targets.last().expect("final commitment").code
    }
    pub fn basecase(&self) -> BasecasePlan<'_, M::Target> {
        self.targets
            .last()
            .expect("final commitment")
            .basecase(self.security().mode.is_zk(), usize::from(self.has_rounds()))
    }
    pub const fn joint_masks(&self) -> Option<&JointMaskConfig<M::Target>> {
        self.joint_masks.as_ref()
    }
    /// Mask dimensions come from the commitment that owns the corresponding witness.
    pub fn witness_mask(&self, index: usize) -> Option<WitnessMaskConfig> {
        assert!(index <= self.num_rounds());
        if !self.security().mode.is_zk() {
            return None;
        }
        let mask = if index == 0 && self.has_rounds() {
            WitnessMaskConfig::for_code(&self.first.as_ref().unwrap().code, 0)
        } else {
            let node = &self.targets[index - usize::from(self.has_rounds())];
            WitnessMaskConfig::for_code(&node.code, usize::from(index != 0))
        };
        Some(mask.expect("construction checked witness mask dimensions"))
    }
    /// Structural consistency follows from shared ownership and checked construction.
    pub fn validate(&self) -> Result<(), DeriveError> {
        self.validate_security()
    }
}

impl<M: Embedding> Node<M> {
    pub(super) fn round<'a>(
        &'a self,
        target: &'a IrsConfig<Identity<M::Target>>,
        index: usize,
        is_zk: bool,
    ) -> RoundConfig<'a, M> {
        let folds = self.code.interleaving_depth().ilog2() as usize;
        let (ood, pow) = self.switch.expect("nonterminal commitment");
        let masks = is_zk.then(|| RoundMasks {
            witness: WitnessMaskConfig::for_code(&self.code, usize::from(index != 0))
                .expect("construction checked witness mask dimensions"),
            sumcheck_coefficients: mask_coefficients(folds).expect("bounded folding factor"),
        });
        RoundConfig::new(
            index,
            SumcheckConfig::new(
                self.code.vector_size(),
                self.sumcheck_pow,
                folds,
                if is_zk {
                    SumcheckMode::ZeroKnowledge {
                        mask_length: SumcheckMaskLen::new(3),
                    }
                } else {
                    SumcheckMode::Standard
                },
            ),
            CodeSwitchConfig::new(&self.code, target, ood, CodeSwitchMode::Standard, pow),
            masks,
        )
    }
}

impl<F: Field> Node<Identity<F>> {
    pub(super) fn basecase(&self, is_zk: bool, incoming: usize) -> BasecasePlan<'_, F> {
        let sumcheck = SumcheckConfig::new(
            self.code.vector_size(),
            self.sumcheck_pow,
            self.code.vector_size().ilog2() as usize,
            SumcheckMode::Standard,
        );
        if is_zk {
            BasecasePlan::ZeroKnowledge {
                commit: &self.code,
                witness_mask: WitnessMaskConfig::for_code(&self.code, incoming)
                    .expect("construction checked final mask dimensions"),
                sumcheck,
            }
        } else {
            BasecasePlan::Standard {
                config: BasecaseConfig::new(
                    &self.code,
                    sumcheck,
                    BasecaseMode::Standard,
                    PowConfig::none(),
                ),
            }
        }
    }
}

/// A temporary view joins adjacent commitments; it owns no IRS configurations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoundConfig<'a, M: Embedding> {
    round_index: usize,
    sumcheck: SumcheckConfig<M::Target>,
    code_switch: CodeSwitchConfig<'a, M>,
    masks: Option<RoundMasks>,
}
impl<'a, M: Embedding> RoundConfig<'a, M> {
    pub(crate) const fn new(
        round_index: usize,
        sumcheck: SumcheckConfig<M::Target>,
        code_switch: CodeSwitchConfig<'a, M>,
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
    pub const fn code_switch(&self) -> &CodeSwitchConfig<'a, M> {
        &self.code_switch
    }
    pub const fn masks(&self) -> Option<&RoundMasks> {
        self.masks.as_ref()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BasecasePlan<'a, F: Field> {
    Standard {
        config: BasecaseConfig<'a, F>,
    },
    ZeroKnowledge {
        commit: &'a IrsConfig<Identity<F>>,
        witness_mask: WitnessMaskConfig,
        sumcheck: SumcheckConfig<F>,
    },
}
impl<'a, F: Field> BasecasePlan<'a, F> {
    pub const fn commit(&self) -> &'a IrsConfig<Identity<F>> {
        match self {
            Self::Standard { config } => config.commit(),
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

pub(super) fn embedding_degree<M: Embedding>() -> Result<usize, DeriveError> {
    let source = M::Source::extension_degree();
    let target = M::Target::extension_degree();
    if source == 0 || !target.is_multiple_of(source) {
        return Err(DeriveError::InvalidMaskLayout);
    }
    usize::try_from(target / source).map_err(|_| DeriveError::SizeOverflow)
}
