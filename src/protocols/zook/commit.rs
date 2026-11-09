//! Joint masks and degree-preserving initial witness commitments.

use std::{iter::once, sync::Arc};

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
        irs_commit::{Commitment as IrsCommitment, Config as IrsConfig, Witness as IrsWitness},
        params::{FoldingFactor, JointMaskConfig, ProtocolConfig, WitnessMaskConfig},
        zook::binding::{prove_binding, receive_binding, MaskBinding},
    },
    transcript::{
        Codec, DuplexSpongeInterface, ProverMessage, ProverState, VerificationError,
        VerificationResult, VerifierState,
    },
};

/// Local provenance for opaque handles; the transcript still authenticates every root.
#[derive(Clone, Debug)]
pub(super) struct PairMember {
    config: Arc<Hash>,
    first: bool,
}

impl PairMember {
    fn second(&self, binding: Hash) -> VerificationResult<Self> {
        if !self.first || *self.config != binding {
            return Err(VerificationError);
        }
        Ok(Self {
            config: self.config.clone(),
            first: false,
        })
    }

    pub(super) fn matches(&self, second: &Self, binding: Hash) -> bool {
        self.first
            && !second.first
            && *self.config == binding
            && Arc::ptr_eq(&self.config, &second.config)
    }
}

/// Prover handle from [`ProtocolConfig::commit`]; consumed by `prove`.
#[must_use]
#[derive(Clone, Debug)]
pub struct CommittedWitness<M: Embedding> {
    pub(crate) state: CommittedState<M>,
    pub(super) second_source: Option<IrsWitness<M::Source>>,
    pub(super) pair: Option<PairMember>,
    pub(super) pending_mask: Option<Zeroizing<Vec<M::Source>>>,
    pub(crate) joint_masks: Option<JointMaskWitness<M::Target>>,
    pub(super) mask_binding: Option<MaskBinding<M::Target>>,
}

/// Joint masks and their final-opening blinder, committed as two columns in one tree.
#[derive(Clone, Debug)]
pub(crate) struct JointMaskWitness<F: Field> {
    pub(crate) coefficients: Buffer<F>,
    pub(crate) opening_blinder: Buffer<F>,
    pub(crate) irs_witness: IrsWitness<F>,
}

impl<F: Field> Drop for JointMaskWitness<F> {
    fn drop(&mut self) {
        self.coefficients.wipe();
        self.opening_blinder.wipe();
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
    pub(super) second_source: Option<IrsCommitment>,
    pub(super) pair: Option<PairMember>,
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
        witness: Buffer<M::Source>,
    ) -> CommittedWitness<M>
    where
        Standard: Distribution<M::Source> + Distribution<M::Target>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        M::Target: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        self.commit_inner(ps, witness, false)
            .expect("zook witness length")
    }

    /// Commit a staged pair with shared masks, using the ordinary per-witness configuration.
    /// Pass `None` for the first witness, then `Some(&mut first)` after application challenges.
    /// The second call consumes its precommitted mask. Open each pair only once.
    pub fn commit_pair<H, R>(
        &self,
        ps: &mut ProverState<H, R>,
        mut witness: Buffer<M::Source>,
        previous: Option<&mut CommittedWitness<M>>,
    ) -> VerificationResult<CommittedWitness<M>>
    where
        Standard: Distribution<M::Source> + Distribution<M::Target>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        M::Target: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let config = self.pair_config()?;
        let Some(first) = previous else {
            return config.commit_inner(ps, witness, true);
        };
        let source = config.pair_source()?;
        let pair = first
            .pair
            .as_ref()
            .ok_or(VerificationError)?
            .second(config.pair_binding())?;
        if witness.len() != source.vector_size() {
            return Err(VerificationError);
        }
        let mask = first.pending_mask.take().ok_or(VerificationError)?;
        if let Some(mut shape) = config.witness_mask(0) {
            shape.num_polynomials /= 2;
            witness = overlay_mask(witness, &mask, shape);
        }
        let irs_witness = source.commit(ps, &[&witness]);
        Ok(CommittedWitness {
            state: CommittedState::Round {
                message: witness,
                irs_witness,
            },
            second_source: None,
            pair: Some(pair),
            pending_mask: None,
            joint_masks: None,
            mask_binding: None,
        })
    }

    fn commit_inner<H, R>(
        &self,
        ps: &mut ProverState<H, R>,
        mut witness: Buffer<M::Source>,
        paired: bool,
    ) -> VerificationResult<CommittedWitness<M>>
    where
        Standard: Distribution<M::Source> + Distribution<M::Target>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        M::Target: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let source = paired.then(|| self.pair_source()).transpose()?;
        let parts = 1 + usize::from(paired);
        if witness.len() != self.tuning().vector_size / parts {
            return Err(VerificationError);
        }

        witness.resize_zeroed(self.padded_vector_size() / parts);
        let mut initial_mask = Zeroizing::new(Vec::new());
        let joint_masks = self.joint_masks().map(|config| {
            let (coefficients, source_mask) =
                sample_masks::<M, _>(config, self.has_rounds(), ps.rng());
            initial_mask = source_mask;
            let opening_blinder = Buffer::random(ps.rng(), config.commit().vector_size());
            let irs_witness = config
                .commit()
                .commit(ps, &[&coefficients, &opening_blinder]);
            JointMaskWitness {
                coefficients,
                opening_blinder,
                irs_witness,
            }
        });

        let state = if let Some(round) = self.first_round() {
            if let Some(masks) = round.masks() {
                let mut shape = masks.witness;
                shape.num_polynomials /= parts;
                witness = overlay_mask(witness, &initial_mask[..initial_mask.len() / parts], shape);
            }
            let irs_witness = source
                .as_ref()
                .unwrap_or_else(|| round.code_switch().source())
                .commit(ps, &[&witness]);
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
        let mut committed = CommittedWitness {
            state,
            second_source: None,
            pair: None,
            pending_mask: None,
            joint_masks,
            mask_binding,
        };
        if paired {
            committed.pair = Some(PairMember {
                config: Arc::new(self.pair_binding()),
                first: true,
            });
            let midpoint = initial_mask.len() / 2;
            committed.pending_mask = Some(Zeroizing::new(initial_mask[midpoint..].to_vec()));
        }
        Ok(committed)
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
            .map(|_| receive_binding(self.mask_source_degree(), vs))
            .transpose()?;
        Ok(Commitment {
            irs,
            second_source: None,
            pair: None,
            joint_masks,
            mask_binding,
        })
    }

    /// Receive staged roots around application messages; pass `None`, then `Some(&first)`.
    pub fn receive_pair<H>(
        &self,
        vs: &mut VerifierState<H>,
        previous: Option<&Commitment<M::Target>>,
    ) -> VerificationResult<Commitment<M::Target>>
    where
        H: DuplexSpongeInterface,
        M::Target: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let config = self.pair_config()?;
        let Some(first) = previous else {
            // Reading a root does not depend on its row width.
            let mut first = config.receive_commitment(vs)?;
            first.pair = Some(PairMember {
                config: Arc::new(config.pair_binding()),
                first: true,
            });
            return Ok(first);
        };
        let pair = first
            .pair
            .as_ref()
            .ok_or(VerificationError)?
            .second(config.pair_binding())?;
        Ok(Commitment {
            irs: config.pair_source()?.receive_commitment(vs)?,
            second_source: None,
            pair: Some(pair),
            joint_masks: None,
            mask_binding: None,
        })
    }

    pub(super) fn pair_config(&self) -> VerificationResult<Self> {
        self.validate().map_err(|_| VerificationError)?;
        if self.tuning().vector_size != self.padded_vector_size() {
            return Err(VerificationError);
        }
        let mut tuning = self.tuning().clone();
        tuning.vector_size = tuning.vector_size.checked_mul(2).ok_or(VerificationError)?;
        tuning.folding_factor = FoldingFactor::ConstantFromSecondRound {
            initial: tuning
                .folding_factor
                .at_round(0)
                .checked_add(1)
                .ok_or(VerificationError)?,
            rest: tuning.folding_factor.at_round(1),
        };
        let config =
            Self::derive(self.security().clone(), tuning).map_err(|_| VerificationError)?;
        let _ = config.pair_source()?;
        Ok(config)
    }

    pub(super) fn pair_binding(&self) -> Hash {
        let mut hash = blake3::Hasher::new();
        hash.update(b"zook-shared-pair-v1");
        ciborium::into_writer(self, &mut hash).expect("pair configuration serializes");
        Hash(*hash.finalize().as_bytes())
    }

    pub(super) fn pair_source(&self) -> VerificationResult<IrsConfig<M>> {
        self.validate().map_err(|_| VerificationError)?;
        if self.tuning().vector_size != self.padded_vector_size() {
            return Err(VerificationError);
        }
        self.first_round()
            .and_then(|round| round.code_switch().source().pair_half())
            .ok_or(VerificationError)
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
                .commit()
                .verify(&mut vs, &[received.joint_masks.as_ref().unwrap()])
                .unwrap();
            let joint = committed.joint_masks.as_ref().unwrap();
            assert_eq!(values.matrix.len(), 2 * values.points.len());
            for (&point, row) in values
                .points
                .iter()
                .zip(values.matrix.to_slice().chunks_exact(2))
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
