//! Staged witness commitments and one opening of their concatenation.

use ark_ff::Field;
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, RngCore};
use zeroize::Zeroizing;

use super::{
    commit::CommittedState,
    params::JointOpeningConfig,
    round::{reduce_prover, reduce_verifier},
    zk::{overlay_mask, prove_binding, receive_binding, sample_masks, JointMaskWitness},
    Commitment, CommittedWitness, FinalClaim, ProverClaim,
};
use crate::{
    algebra::{embedding::Embedding, linear_form::LinearForm},
    buffer::{Buffer, BufferMath, BufferOps},
    hash::Hash,
    protocols::irs_commit::{Commitment as IrsCommitment, Witness},
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState,
        VerificationError, VerificationResult, VerifierState,
    },
};

/// First witness commitment with the second witness's reserved mask.
#[must_use]
pub struct FirstWitness<'a, M: Embedding> {
    config: &'a JointOpeningConfig<M>,
    left: CommittedWitness<M>,
    reserved: Zeroizing<Vec<M::Source>>,
}

/// Both witness commitments and their shared masks, consumed by one joint opening.
#[must_use]
pub struct JointWitness<'a, M: Embedding> {
    config: &'a JointOpeningConfig<M>,
    left: CommittedWitness<M>,
    right: Witness<M::Source>,
}

/// Received first witness root and mask-binding equations.
#[must_use]
pub struct FirstCommitment<'a, M: Embedding> {
    config: &'a JointOpeningConfig<M>,
    left: Commitment<M::Target>,
}

/// Both witness roots and their shared-mask root, consumed by joint verification.
#[must_use]
pub struct JointCommitment<'a, M: Embedding> {
    config: &'a JointOpeningConfig<M>,
    left: Commitment<M::Target>,
    right: IrsCommitment,
}

/// Source-field initial masks for L and R, in that order.
type InitialMasks<M> = Zeroizing<Vec<<M as Embedding>::Source>>;

/// Commit every mask before either witness; returns them with L's and R's initial masks.
fn commit_masks<M, H, R>(
    config: &JointOpeningConfig<M>,
    ps: &mut ProverState<H, R>,
) -> Option<(JointMaskWitness<M::Target>, InitialMasks<M>)>
where
    M: Embedding + Default,
    Standard: Distribution<M::Source> + Distribution<M::Target>,
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
    M::Target: Codec<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    config.combined().joint_masks().map(|layout| {
        let (coefficients, initial) = sample_masks::<M, _>(layout, true, ps.rng());
        let opening_blinder = Buffer::random(ps.rng(), layout.commit.vector_size());
        let irs_witness = layout.commit.commit(ps, &[&coefficients, &opening_blinder]);
        let masks = JointMaskWitness {
            coefficients,
            opening_blinder,
            irs_witness,
        };
        (masks, initial)
    })
}

/// Receive the shared-mask root before either witness.
fn receive_masks<M, H>(
    config: &JointOpeningConfig<M>,
    vs: &mut VerifierState<H>,
) -> VerificationResult<Option<IrsCommitment>>
where
    M: Embedding + Default,
    H: DuplexSpongeInterface,
    M::Target: Codec<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    config
        .combined()
        .joint_masks()
        .map(|layout| layout.commit.receive_commitment(vs))
        .transpose()
}

fn commit_witness<M, H, R>(
    config: &JointOpeningConfig<M>,
    ps: &mut ProverState<H, R>,
    mut message: Buffer<M::Source>,
    mask: &[M::Source],
) -> CommittedState<M>
where
    M: Embedding + Default,
    Standard: Distribution<M::Source>,
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
    M::Target: Codec<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    assert_eq!(
        message.len(),
        config.witness_commitment().vector_size(),
        "whir witness length",
    );
    if let Some(mut shape) = config.combined().witness_mask(0) {
        shape.num_polynomials /= 2;
        message = overlay_mask(message, mask, shape);
    }
    let root = config.witness_commitment().commit(ps, &[&message]);
    CommittedState::Round {
        message,
        irs_witness: root,
    }
}

impl<M: Embedding + Default> JointOpeningConfig<M> {
    /// Commit every mask, then L with the first half of the initial masks; reserve the rest for R.
    /// Use the same transcript through both commitments and the joint opening.
    pub fn commit_first<H, R>(
        &self,
        ps: &mut ProverState<H, R>,
        witness: Buffer<M::Source>,
    ) -> FirstWitness<'_, M>
    where
        Standard: Distribution<M::Source> + Distribution<M::Target>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        M::Target: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let (masks, initial) = commit_masks(self, ps).unzip();
        let initial = initial.unwrap_or_default();
        let midpoint = initial.len() / 2;
        let mask_binding =
            self.combined()
                .joint_masks()
                .zip(masks.as_ref())
                .map(|(layout, masks)| {
                    prove_binding(
                        layout,
                        masks.coefficients.to_slice(),
                        M::Source::extension_degree(),
                        ps,
                    )
                });
        let state = commit_witness(self, ps, witness, &initial[..midpoint]);
        FirstWitness {
            config: self,
            left: CommittedWitness {
                state,
                joint_masks: masks,
                mask_binding,
            },
            reserved: Zeroizing::new(initial[midpoint..].to_vec()),
        }
    }

    /// Receive the shared-mask root, the binding equations for the full mask collection, and L's root.
    pub fn receive_first<H>(
        &self,
        vs: &mut VerifierState<H>,
    ) -> VerificationResult<FirstCommitment<'_, M>>
    where
        H: DuplexSpongeInterface,
        M::Target: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let masks = receive_masks(self, vs)?;
        let mask_binding = self
            .combined()
            .joint_masks()
            .map(|_| receive_binding(M::Source::extension_degree(), vs))
            .transpose()?;
        let irs = self.witness_commitment().receive_commitment(vs)?;
        Ok(FirstCommitment {
            config: self,
            left: Commitment {
                irs,
                joint_masks: masks,
                mask_binding,
            },
        })
    }
}

impl<'a, M: Embedding + Default> FirstWitness<'a, M> {
    /// Commit R after application challenges, consuming its precommitted mask.
    pub fn commit_second<H, R>(
        mut self,
        ps: &mut ProverState<H, R>,
        witness: Buffer<M::Source>,
    ) -> JointWitness<'a, M>
    where
        Standard: Distribution<M::Source>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        M::Target: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let CommittedState::Round {
            message: right_message,
            irs_witness: right,
        } = commit_witness(self.config, ps, witness, &self.reserved)
        else {
            unreachable!("staged commitments always have a first folding block");
        };
        let CommittedState::Round { message, .. } = &mut self.left.state else {
            unreachable!("staged commitments always have a first folding block");
        };
        *message = message.concat(&right_message);
        JointWitness {
            config: self.config,
            left: self.left,
            right,
        }
    }
}

impl<'a, M: Embedding + Default> FirstCommitment<'a, M> {
    /// Receive R's root after the application's intervening messages and challenges.
    pub fn receive_second<H>(
        self,
        vs: &mut VerifierState<H>,
    ) -> VerificationResult<JointCommitment<'a, M>>
    where
        H: DuplexSpongeInterface,
        M::Target: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let right = self.config.witness_commitment().receive_commitment(vs)?;
        Ok(JointCommitment {
            config: self.config,
            left: self.left,
            right,
        })
    }
}

fn check_claims<M: Embedding>(
    config: &JointOpeningConfig<M>,
    forms: &[&dyn LinearForm<M::Target>],
    values: &[M::Target],
) -> VerificationResult<()> {
    if forms.is_empty()
        || forms.len() != values.len()
        || forms
            .iter()
            .any(|form| form.size() != config.combined().params().size)
    {
        return Err(VerificationError);
    }
    Ok(())
}

impl<M: Embedding + Default> JointWitness<'_, M> {
    /// Open the concatenated witnesses once; callers bind public forms to their instance.
    pub fn prove<H, R>(
        self,
        ps: &mut ProverState<H, R>,
        forms: &[&dyn LinearForm<M::Target>],
        values: &[M::Target],
    ) -> ProverClaim<M::Target>
    where
        M::Target: Codec<[H::U]>,
        Standard: Distribution<M::Target>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert!(
            check_claims(self.config, forms, values).is_ok(),
            "whir joint claims",
        );
        ps.prover_message(&U64(values.len() as u64));
        ps.prover_messages(values);
        let reduced = reduce_prover(
            self.config.combined(),
            ps,
            self.left,
            forms,
            values,
            Some((self.config.witness_commitment(), self.right)),
        );
        self.config.combined().prove_final_opening(ps, reduced)
    }
}

impl<M: Embedding + Default> JointCommitment<'_, M> {
    /// Verify the joint opening; the caller must then check `FinalClaim::verify` and transcript completion.
    pub fn verify<H>(
        self,
        vs: &mut VerifierState<H>,
        forms: &[&dyn LinearForm<M::Target>],
        values: &[M::Target],
    ) -> VerificationResult<FinalClaim<M::Target>>
    where
        M::Target: Codec<[H::U]>,
        Standard: Distribution<M::Target>,
        H: DuplexSpongeInterface,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        check_claims(self.config, forms, values)?;
        if vs.prover_message::<U64>()?.0 != values.len() as u64 {
            return Err(VerificationError);
        }
        for &value in values {
            if vs.prover_message::<M::Target>()? != value {
                return Err(VerificationError);
            }
        }
        let reduced = reduce_verifier(
            self.config.combined(),
            vs,
            self.left,
            forms,
            values,
            Some((self.config.witness_commitment(), &self.right)),
        )?;
        self.config.combined().verify_final_opening(vs, reduced)
    }
}
