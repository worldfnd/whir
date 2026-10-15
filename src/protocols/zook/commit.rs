//! Joint masks and degree-preserving initial witness commitments.

use std::iter::once;

use ark_ff::{AdditiveGroup, Field};
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, Rng, RngCore};
#[cfg(feature = "tracing")]
use tracing::instrument;
use zeroize::Zeroizing;

use crate::{
    algebra::{embedding::Embedding, random_vector},
    buffer::{Buffer, BufferMath, BufferOps},
    hash::Hash,
    protocols::{
        irs_commit::{Commitment as IrsCommitment, Witness as IrsWitness},
        params::{JointMaskConfig, ProtocolConfig, WitnessMaskConfig},
        zook::binding::{prove_binding, receive_binding, MaskBinding},
    },
    transcript::{
        Codec, DuplexSpongeInterface, ProverMessage, ProverState, VerificationResult, VerifierState,
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

/// Secret coefficients and their single IRS commitment, retained for mask openings.
#[derive(Clone, Debug)]
pub(crate) struct JointMaskWitness<F: Field> {
    pub(crate) coefficients: Buffer<F>,
    pub(crate) irs_witness: IrsWitness<F>,
}

impl<F: Field> Drop for JointMaskWitness<F> {
    fn drop(&mut self) {
        self.coefficients.wipe();
        self.irs_witness.masks.wipe();
        self.irs_witness.matrix.wipe();
    }
}

/// Internal: the two branches differ in their IRS witness field type.
#[derive(Clone, Debug)]
pub(crate) enum CommittedState<M: Embedding> {
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
    /// Commit all Zook masks, then the padded witness with overlaid prefixes.
    ///
    /// Section 5.3, “Overlaid masks instead of zk-encoding”: each source polynomial
    /// is committed as `p + mask` with unchanged degree. Section 5.4, “ZO0K”,
    /// “The required masks”: every mask is sampled before folding starts.
    #[cfg_attr(feature = "tracing", instrument(skip_all, name = "zook::commit", fields(vector_size = self.tuning().vector_size, num_rounds = self.num_rounds())))]
    pub fn commit<H, R>(
        &self,
        ps: &mut ProverState<H, R>,
        mut witness: Buffer<M::Source>,
    ) -> CommittedWitness<M>
    where
        Standard: Distribution<M::Source> + Distribution<M::Target>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        M::Target: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(
            witness.len(),
            self.tuning().vector_size,
            "zook witness length",
        );

        witness.resize_zeroed(self.padded_vector_size());
        let mut initial_mask = Zeroizing::new(Vec::new());
        let joint_masks = self.joint_masks().map(|config| {
            let (coefficients, source_mask) =
                sample_masks::<M, _>(config, self.has_rounds(), ps.rng());
            initial_mask = source_mask;
            let irs_witness = config.commit().commit(ps, &[&coefficients]);
            JointMaskWitness {
                coefficients,
                irs_witness,
            }
        });

        let state = if let Some(round) = self.first_round() {
            if let Some(masks) = round.masks() {
                witness = overlay_mask(witness, &initial_mask, masks.witness);
            }
            let irs_witness = round.code_switch().source().commit(ps, &[&witness]);
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
                self.basecase().witness_mask(),
            ) {
                message = overlay_mask(
                    message,
                    &joint.coefficients.to_slice()[config.witness_masks()[0].clone()],
                    *mask,
                );
            }
            let irs_witness = self.basecase().commit().commit(ps, &[&message]);
            CommittedState::Basecase {
                message,
                irs_witness,
            }
        };
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
        CommittedWitness {
            state,
            joint_masks,
            mask_binding,
        }
    }

    /// Verifier mirror of [`Self::commit`].
    #[cfg_attr(feature = "tracing", instrument(skip_all, name = "zook::receive_commitment", fields(vector_size = self.tuning().vector_size, num_rounds = self.num_rounds())))]
    pub fn receive_commitment<H>(
        &self,
        vs: &mut VerifierState<H>,
    ) -> VerificationResult<Commitment<M::Target>>
    where
        H: DuplexSpongeInterface,
        M::Target: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let joint_masks = self
            .joint_masks()
            .map(|config| config.commit().receive_commitment(vs))
            .transpose()?;
        let irs = match self.first_round() {
            Some(round) => round.code_switch().source().receive_commitment(vs)?,
            None => self.basecase().commit().receive_commitment(vs)?,
        };
        let mask_binding = self
            .joint_masks()
            .map(|config| receive_binding(config, self.mask_source_degree(), vs))
            .transpose()?;
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

/// Sample the configured coefficient ranges; unused encoding padding stays zero.
///
/// Section 5.4, “ZO0K”, “The required masks”: the initial mask and Frobenius-check
/// blinder are sampled in the source field. All other masks use the target field.
/// A basecase-only witness is lifted first, so all its masks use the target field.
fn sample_masks<M, R>(
    config: &JointMaskConfig<M::Target>,
    has_rounds: bool,
    rng: &mut R,
) -> (Buffer<M::Target>, Zeroizing<Vec<M::Source>>)
where
    M: Embedding + Default,
    R: RngCore + CryptoRng,
    Standard: Distribution<M::Source> + Distribution<M::Target>,
{
    let mut coefficients = vec![M::Target::ZERO; config.commit().vector_size()];
    for range in config
        .sumcheck_masks()
        .iter()
        .chain(config.witness_masks().iter().skip(usize::from(has_rounds)))
        .chain(once(config.blinder()))
    {
        for coefficient in &mut coefficients[range.clone()] {
            *coefficient = rng.gen();
        }
    }
    let embedding = M::default();
    let initial_mask = if has_rounds {
        let range = config.witness_masks()[0].clone();
        let mask = Zeroizing::new(random_vector::<M::Source>(&mut *rng, range.len()));
        for (coefficient, &value) in coefficients[range].iter_mut().zip(mask.iter()) {
            *coefficient = embedding.map(value);
        }
        mask
    } else {
        Zeroizing::new(Vec::new())
    };
    for coefficient in &mut coefficients[config.basefield_blinder().clone()] {
        *coefficient = if has_rounds {
            embedding.map(rng.gen())
        } else {
            rng.gen()
        };
    }
    (Buffer::from(coefficients), initial_mask)
}

/// Add an independent prefix to each contiguous IRS polynomial block.
///
/// For polynomial `j` and coefficient `i < q`, replace `p[j * m + i]` with
/// `p[j * m + i] + mask[j * q + i]`; all remaining coefficients are unchanged.
/// Here `m` is the block length and `q` the configured prefix length.
pub(super) fn overlay_mask<F: Field>(
    message: Buffer<F>,
    mask: &[F],
    config: WitnessMaskConfig,
) -> Buffer<F> {
    assert!(config.num_polynomials > 0);
    assert!(message.len().is_multiple_of(config.num_polynomials));
    assert_eq!(
        config
            .num_polynomials
            .checked_mul(config.coefficients_per_polynomial),
        Some(mask.len()),
    );
    let block_len = message.len() / config.num_polynomials;
    let prefix = config.coefficients_per_polynomial;
    assert!(prefix > 0 && prefix <= block_len);
    let mut coefficients = message.into_vec();
    for (polynomial, mask) in coefficients
        .chunks_exact_mut(block_len)
        .zip(mask.chunks_exact(prefix))
    {
        for (coefficient, &randomness) in polynomial[..prefix].iter_mut().zip(mask) {
            *coefficient += randomness;
        }
    }
    Buffer::from(coefficients)
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::{
        algebra::{
            embedding::{Basefield, Identity},
            fields::{Field64, Field64_2},
            linear_form::Evaluate,
            univariate_evaluate,
        },
        protocols::{
            irs_commit::Evaluations,
            params::{FoldingFactor, Mode, PowBudget, RateSchedule, SecuritySpec, TuningSpec},
        },
        transcript::{codecs::Empty, DomainSeparator, VerifierMessage},
    };

    type M = Basefield<Field64_2>;

    fn config<M: Embedding + Default>(mode: Mode, vector_size: usize) -> ProtocolConfig<M> {
        ProtocolConfig::derive(
            SecuritySpec::new(80)
                .with_mode(mode)
                .with_pow_budget(PowBudget::Forbidden),
            TuningSpec {
                vector_size,
                starting_log_inv_rate: 2,
                folding_factor: FoldingFactor::Constant(2),
                rate_schedule: RateSchedule::Stepping,
            },
        )
        .unwrap()
    }

    fn roundtrip<M: Embedding + Default>(config: &ProtocolConfig<M>) -> CommittedWitness<M>
    where
        M::Target: Codec<[u8]>,
        Standard: Distribution<M::Source> + Distribution<M::Target>,
    {
        let witness = Buffer::<M::Source>::ones(config.tuning().vector_size);
        let ds = DomainSeparator::protocol(&"zook-commit-test")
            .session(&"joint mask and witness commitment")
            .instance(&Empty);
        let mut ps = ProverState::new_std(&ds);
        let committed = config.commit(&mut ps, witness);
        if let Some((layout, masks)) = config.joint_masks().zip(committed.joint_masks.as_ref()) {
            let _ = layout.commit().open(&mut ps, &[&masks.irs_witness]);
        }
        match &committed.state {
            CommittedState::Round { irs_witness, .. } => {
                let _ = config
                    .first_round()
                    .unwrap()
                    .code_switch()
                    .source()
                    .open(&mut ps, &[irs_witness]);
            }
            CommittedState::Basecase { irs_witness, .. } => {
                let _ = config.basecase().commit().open(&mut ps, &[irs_witness]);
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
            for (form, &value) in binding.forms.iter().zip(&binding.values) {
                assert_eq!(form.evaluate(&Identity::new(), coefficients), value);
            }
        }
        assert_eq!(
            received.joint_masks.is_some(),
            config.joint_masks().is_some()
        );
        if let Some(layout) = config.joint_masks() {
            let values = layout
                .commit()
                .verify(&mut vs, &[received.joint_masks.as_ref().unwrap()])
                .unwrap();
            assert_openings(
                &values,
                &committed.joint_masks.as_ref().unwrap().coefficients,
                1,
            );
        }
        match &committed.state {
            CommittedState::Round { message, .. } => {
                let code = config.first_round().unwrap().code_switch().source();
                let values = code.verify(&mut vs, &[&received.irs]).unwrap();
                assert_openings(&values, message, code.interleaving_depth());
            }
            CommittedState::Basecase { message, .. } => {
                let values = config
                    .basecase()
                    .commit()
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
    fn overlay_changes_only_each_polynomial_prefix() {
        let message = Buffer::from((1..=8).map(Field64::from).collect::<Vec<_>>());
        let mask = [10, 20, 30, 40].map(Field64::from);
        let masked = overlay_mask(
            message,
            &mask,
            WitnessMaskConfig {
                num_polynomials: 2,
                coefficients_per_polynomial: 2,
            },
        );
        assert_eq!(
            masked.to_slice(),
            &[11, 22, 3, 4, 35, 46, 7, 8].map(Field64::from)
        );
    }

    #[test]
    fn commits_source_overlays_from_the_joint_mask() {
        let config = config::<M>(Mode::ZeroKnowledge, 4096);
        let committed = roundtrip(&config);
        let joint = committed.joint_masks.as_ref().unwrap();
        let layout = config.joint_masks().unwrap();
        let round = config.first_round().unwrap();
        let mask = round.masks().unwrap().witness;
        let range = layout.witness_masks()[0].clone();
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
        let block_len = round.code_switch().source().message_length();
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
        let range = layout.witness_masks()[0].clone();
        let CommittedState::Basecase {
            message,
            irs_witness,
        } = &committed.state
        else {
            panic!("expected a basecase-only commitment");
        };
        assert_eq!(message.len(), config.padded_vector_size());
        assert!(message.len() > config.tuning().vector_size);
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
    fn initial_mask_and_membership_blinder_use_the_source_field() {
        let config = config::<M>(Mode::ZeroKnowledge, 4096);
        let layout = config.joint_masks().unwrap();
        let mut rng = StdRng::seed_from_u64(17);
        let (mut coefficients, initial) = sample_masks::<M, _>(layout, true, &mut rng);
        let first = layout.witness_masks()[0].clone();
        for (&value, &source) in coefficients.to_slice()[first.clone()]
            .iter()
            .zip(initial.iter())
        {
            assert_eq!(value, M::default().map(source));
        }
        for range in [first, layout.basefield_blinder().clone()] {
            assert!(coefficients.to_slice()[range]
                .iter()
                .all(|value| value.frobenius_map(1) == *value));
        }
        for range in layout
            .sumcheck_masks()
            .iter()
            .chain(layout.witness_masks().iter().skip(1))
            .chain(once(layout.blinder()))
        {
            assert!(coefficients.to_slice()[range.clone()]
                .iter()
                .any(|value| value.frobenius_map(1) != *value));
        }
        assert!(coefficients.to_slice()[layout.basefield_blinder().end..]
            .iter()
            .all(|value| *value == Field64_2::ZERO));
        coefficients.wipe();
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
    #[should_panic(expected = "zook witness length")]
    fn commit_rejects_wrong_witness_size() {
        let config = config::<M>(Mode::ZeroKnowledge, 4096);
        let ds = DomainSeparator::protocol(&"zook-commit-test")
            .session(&"wrong witness size")
            .instance(&Empty);
        let mut ps = ProverState::new_std(&ds);
        let _ = config.commit(&mut ps, Buffer::zeros(4095));
    }
}
