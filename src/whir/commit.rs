//! Witness commitment: `ProtocolConfig::commit` and `receive_commitment`, with ZK masks overlaid from `zk`.

use ark_ff::Field;
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, RngCore};
#[cfg(feature = "tracing")]
use tracing::instrument;
use zeroize::Zeroizing;

use crate::{
    algebra::embedding::Embedding,
    buffer::{Buffer, BufferMath, BufferOps},
    protocols::irs_commit::{Commitment as IrsCommitment, Witness as IrsWitness},
    transcript::{Codec, DuplexSpongeInterface, ProverState, VerificationResult, VerifierState},
    whir::{
        params::ProtocolConfig,
        zk::{
            overlay_mask, prove_binding, receive_binding, sample_masks, JointMaskWitness,
            MaskBinding,
        },
    },
};

/// Prover handle from [`ProtocolConfig::commit`]; consumed by `prove`.
#[must_use]
#[derive(Clone, Debug)]
pub struct CommittedWitness<M: Embedding> {
    pub(crate) state: CommittedState<M>,
    pub(crate) joint_masks: Option<JointMaskWitness<M::Target>>,
    pub(super) mask_binding: Option<MaskBinding<M::Target>>,
}

/// Internal: the two branches differ in their IRS witness field type.
#[derive(Clone, Debug)]
pub enum CommittedState<M: Embedding> {
    /// Plan has ≥ 1 round; committed through `rounds[0].code_switch.source`.
    /// The message stays in `M::Source`: the first round's sumcheck lifts it
    /// into `M::Target` at its first fold.
    Round {
        message: Buffer<M::Source>,
        irs_witness: IrsWitness<M::Source>,
    },
    /// Basecase-only plan; witness was lifted into `M::Target` first.
    Basecase {
        message: Buffer<M::Target>,
        irs_witness: IrsWitness<M::Target>,
    },
}

/// Verifier handle from [`ProtocolConfig::receive_commitment`]; consumed by `verify`.
#[must_use]
#[derive(Clone, Debug)]
pub struct Commitment<F: Field> {
    pub(crate) irs: IrsCommitment,
    pub(crate) joint_masks: Option<IrsCommitment>,
    pub(super) mask_binding: Option<MaskBinding<F>>,
}

impl<M: Embedding + Default> ProtocolConfig<M> {
    /// Commit all masks, then the padded witness with overlaid prefixes.
    ///
    /// Section 5.3, “Overlaid masks instead of zk-encoding”: each source polynomial
    /// is committed as `p + mask` with unchanged degree. Section 5.4, “ZO0K”,
    /// “The required masks”: every mask is sampled before folding starts.
    #[cfg_attr(feature = "tracing", instrument(skip_all, name = "whir::commit", fields(vector_size = self.params().size, num_rounds = self.num_rounds())))]
    pub fn commit<H, R>(
        &self,
        ps: &mut ProverState<H, R>,
        mut witness: Buffer<M::Source>,
    ) -> CommittedWitness<M>
    where
        Standard: Distribution<M::Source> + Distribution<M::Target>,
        H: DuplexSpongeInterface<U = u8>,
        R: RngCore + CryptoRng,
        M::Target: Codec,
    {
        assert_eq!(witness.len(), self.params().size, "whir witness length");

        witness.resize_zeroed(self.padded_vector_size());
        let mut initial_mask = Zeroizing::new(Vec::new());
        let joint_masks = self.joint_masks().map(|config| {
            let (coefficients, source_mask) =
                sample_masks::<M, _>(config, self.has_rounds(), ps.rng());
            initial_mask = source_mask;
            let opening_blinder = Buffer::random(ps.rng(), config.commit.vector_size());
            let irs_witness = config.commit.commit(ps, &[&coefficients, &opening_blinder]);
            JointMaskWitness {
                coefficients,
                opening_blinder,
                irs_witness,
            }
        });
        let mask_binding = self
            .joint_masks()
            .zip(joint_masks.as_ref())
            .map(|(config, masks)| {
                prove_binding(
                    config,
                    masks.coefficients.to_slice(),
                    self.mask_source_degree(),
                    ps,
                )
            });

        let state = if let Some(round) = self.first_round() {
            if let Some(masks) = round.masks {
                witness = overlay_mask(witness, &initial_mask, masks);
            }
            let irs_witness = round.code_switch.source().commit(ps, &[&witness]);
            CommittedState::Round {
                message: witness,
                irs_witness,
            }
        } else {
            // Basecase IRS is over `M::Target`; lift before committing.
            let embedding = M::default();
            let mut message = witness.mixed_lift(&embedding);
            if let (Some(config), Some(joint), Some(mask)) = (
                self.joint_masks(),
                joint_masks.as_ref(),
                self.witness_mask(0),
            ) {
                message = overlay_mask(
                    message,
                    &joint.coefficients.to_slice()[config.witness_masks[0].clone()],
                    mask,
                );
            }
            let irs_witness = self.final_commit().commit(ps, &[&message]);
            CommittedState::Basecase {
                message,
                irs_witness,
            }
        };
        CommittedWitness {
            state,
            joint_masks,
            mask_binding,
        }
    }

    /// Verifier mirror of [`Self::commit`].
    #[cfg_attr(feature = "tracing", instrument(skip_all, name = "whir::receive_commitment", fields(vector_size = self.params().size, num_rounds = self.num_rounds())))]
    pub fn receive_commitment<H>(
        &self,
        vs: &mut VerifierState<H>,
    ) -> VerificationResult<Commitment<M::Target>>
    where
        H: DuplexSpongeInterface<U = u8>,
        M::Target: Codec,
    {
        let joint_masks = self
            .joint_masks()
            .map(|config| config.commit.receive_commitment(vs))
            .transpose()?;
        let mask_binding = self
            .joint_masks()
            .map(|_| receive_binding(self.mask_source_degree(), vs))
            .transpose()?;
        let irs = match self.first_round() {
            Some(round) => round.code_switch.source().receive_commitment(vs)?,
            None => self.final_commit().receive_commitment(vs)?,
        };
        Ok(Commitment {
            irs,
            joint_masks,
            mask_binding,
        })
    }

    fn mask_source_degree(&self) -> u64 {
        if self.has_rounds() {
            M::Source::extension_degree()
        } else {
            M::Target::extension_degree()
        }
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::AdditiveGroup;

    use super::*;
    use crate::{
        algebra::{
            embedding::{Basefield, Identity},
            fields::{Field64, Field64_2},
            univariate_evaluate,
        },
        protocols::irs_commit::Evaluations,
        transcript::{codecs::Empty, DomainSeparator, ProverMessage, VerifierMessage},
        whir::params::{Mode, Params, RateSchedule},
    };

    type M = Basefield<Field64_2>;

    fn config<M: Embedding + Default>(mode: Mode, vector_size: usize) -> ProtocolConfig<M> {
        ProtocolConfig::derive(Params {
            bits: 80,
            mode,
            regime: crate::decoding::DecodingRegime::Johnson,
            pow: 0,
            hash: crate::hash::BLAKE3,
            size: vector_size,
            rate: 2,
            folds: [2, 2],
            schedule: RateSchedule::Stepping,
        })
        .unwrap()
    }

    fn roundtrip<M: Embedding + Default>(config: &ProtocolConfig<M>) -> CommittedWitness<M>
    where
        M::Target: Codec<[u8]>,
        M::Source: ProverMessage,
        Standard: Distribution<M::Source> + Distribution<M::Target>,
    {
        let witness = Buffer::<M::Source>::ones(config.params().size);
        let ds = DomainSeparator::protocol(&"whir-commit-test")
            .session(&"joint mask and witness commitment")
            .instance(&Empty);
        let mut ps = ProverState::new_std(&ds);
        let committed = config.commit(&mut ps, witness);
        if let Some((layout, masks)) = config.joint_masks().zip(committed.joint_masks.as_ref()) {
            let _ = layout.commit.open(&mut ps, &[&masks.irs_witness]);
        }
        match &committed.state {
            CommittedState::Round { irs_witness, .. } => {
                let _ = config
                    .first_round()
                    .unwrap()
                    .code_switch
                    .source()
                    .open(&mut ps, &[irs_witness]);
            }
            CommittedState::Basecase { irs_witness, .. } => {
                let _ = config.final_commit().open(&mut ps, &[irs_witness]);
            }
        }
        let prover_challenge: M::Target = ps.verifier_message();
        let proof = ps.proof();
        let mut vs = VerifierState::new_std(&ds, &proof);
        let received = config.receive_commitment(&mut vs).unwrap();
        assert_eq!(received.mask_binding, committed.mask_binding);
        assert_eq!(
            received.mask_binding.is_some(),
            config.joint_masks().is_some()
        );
        if let Some(binding) = &received.mask_binding {
            let coefficients = committed
                .joint_masks
                .as_ref()
                .unwrap()
                .coefficients
                .to_slice();
            assert_eq!(
                binding.evaluate(config.joint_masks().unwrap(), coefficients),
                binding.values,
            );
        }
        assert_eq!(
            received.joint_masks.is_some(),
            config.joint_masks().is_some()
        );
        if let Some(layout) = config.joint_masks() {
            let values = layout
                .commit
                .verify(&mut vs, &[received.joint_masks.as_ref().unwrap()])
                .unwrap();
            let joint = committed.joint_masks.as_ref().unwrap();
            assert_eq!(values.matrix.len(), 2 * values.points.len());
            for (&point, row) in values
                .points
                .iter()
                .zip(values.matrix.to_slice().as_chunks::<2>().0)
            {
                assert_eq!(
                    row[0],
                    univariate_evaluate(joint.coefficients.to_slice(), point)
                );
                assert_eq!(
                    row[1],
                    univariate_evaluate(joint.opening_blinder.to_slice(), point)
                );
            }
        }
        match &committed.state {
            CommittedState::Round { message, .. } => {
                let code = config.first_round().unwrap().code_switch.source();
                let values = code.verify(&mut vs, &[&received.irs]).unwrap();
                assert_openings(&values, message, code.interleaving_depth());
            }
            CommittedState::Basecase { message, .. } => {
                let values = config
                    .final_commit()
                    .verify(&mut vs, &[&received.irs])
                    .unwrap();
                assert_openings(&values, message, 1);
            }
        }
        let verifier_challenge: M::Target = vs.verifier_message();
        assert_eq!(prover_challenge, verifier_challenge);
        vs.check_eof().unwrap();
        committed
    }

    fn assert_openings<F: Field>(values: &Evaluations<F>, message: &Buffer<F>, depth: usize) {
        assert_eq!(values.matrix.len(), values.points.len() * depth);
        let block_len = message.len() / depth;
        for (&point, row) in values
            .points
            .iter()
            .zip(values.matrix.to_slice().chunks_exact(depth))
        {
            for (polynomial, &value) in message.to_slice().chunks_exact(block_len).zip(row) {
                assert_eq!(univariate_evaluate(polynomial, point), value);
            }
        }
    }

    #[test]
    fn commits_source_overlays_from_the_joint_mask() {
        let config = config::<M>(Mode::ZeroKnowledge, 4096);
        let committed = roundtrip(&config);
        let joint = committed.joint_masks.as_ref().unwrap();
        let layout = config.joint_masks().unwrap();
        let round = config.first_round().unwrap();
        let mask = round.masks.unwrap();
        let range = layout.witness_masks[0].clone();
        let CommittedState::Round {
            message,
            irs_witness,
        } = &committed.state
        else {
            panic!("expected a folding round");
        };
        assert_eq!(message.len(), config.padded_vector_size());
        assert!(irs_witness.masks.is_empty());
        assert!(joint.irs_witness.masks.is_empty());
        let block_len = round.code_switch.source().message_length();
        for (index, &value) in message.to_slice().iter().enumerate() {
            let column = index / block_len;
            let coefficient = index % block_len;
            let expected_mask = if coefficient < mask.coefficients_per_polynomial {
                joint.coefficients.to_slice()
                    [range.start + column * mask.coefficients_per_polynomial + coefficient]
            } else {
                Field64_2::ZERO
            };
            assert_eq!(M::default().map(value - Field64::ONE), expected_mask);
        }
    }

    #[test]
    fn small_witness_is_padded_then_masked_in_the_target_field() {
        let config = config::<M>(Mode::ZeroKnowledge, 8);
        assert!(!config.has_rounds());
        let committed = roundtrip(&config);
        let joint = committed.joint_masks.as_ref().unwrap();
        let layout = config.joint_masks().unwrap();
        let range = layout.witness_masks[0].clone();
        let CommittedState::Basecase {
            message,
            irs_witness,
        } = &committed.state
        else {
            panic!("expected a basecase-only commitment");
        };
        assert_eq!(message.len(), config.padded_vector_size());
        assert!(message.len() > config.params().size);
        assert!(irs_witness.masks.is_empty());
        for (index, &value) in message.to_slice().iter().enumerate() {
            let original = if index < 8 {
                Field64_2::ONE
            } else {
                Field64_2::ZERO
            };
            let mask = if index < range.len() {
                joint.coefficients.to_slice()[range.start + index]
            } else {
                Field64_2::ZERO
            };
            assert_eq!(value, original + mask);
        }
    }

    #[test]
    fn commitment_supports_identity_embedding() {
        let config = config::<Identity<Field64_2>>(Mode::ZeroKnowledge, 4096);
        let committed = roundtrip(&config);
        assert!(committed.joint_masks.is_some());
        assert!(matches!(committed.state, CommittedState::Round { .. }));
    }

    #[test]
    fn standard_commitment_preserves_the_witness() {
        for size in [8, 4096] {
            let config = config::<M>(Mode::Standard, size);
            let committed = roundtrip(&config);
            assert!(committed.joint_masks.is_none());
            match &committed.state {
                CommittedState::Round {
                    message,
                    irs_witness,
                } => {
                    assert!(message
                        .to_slice()
                        .iter()
                        .all(|value| *value == Field64::ONE));
                    assert!(irs_witness.masks.is_empty());
                }
                CommittedState::Basecase {
                    message,
                    irs_witness,
                } => {
                    assert!(message
                        .to_slice()
                        .iter()
                        .all(|value| *value == Field64_2::ONE));
                    assert!(irs_witness.masks.is_empty());
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "whir witness length")]
    fn commit_rejects_wrong_witness_size() {
        let config = config::<M>(Mode::ZeroKnowledge, 4096);
        let ds = DomainSeparator::protocol(&"whir-commit-test")
            .session(&"wrong witness size")
            .instance(&Empty);
        let mut ps = ProverState::new_std(&ds);
        let _ = config.commit(&mut ps, Buffer::zeros(4095));
    }
}
