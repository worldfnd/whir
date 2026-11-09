//! Open two staged commitments with one precommitted mask collection.
//! Callers bind public forms before opening and finish with `FinalClaim::verify`.
//! Each pair is single-use, including any cloned handles.

use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, RngCore};
use zeroize::Zeroize;

use super::{
    commit::CommittedState, Commitment, CommittedWitness, FinalClaim, ProtocolConfig, ProverClaim,
};
use crate::{
    algebra::{embedding::Embedding, linear_form::LinearForm},
    buffer::BufferOps,
    hash::Hash,
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState,
        VerificationError, VerificationResult, VerifierState,
    },
};

impl<M: Embedding + Default> ProtocolConfig<M> {
    /// Check support before either staged commitment changes the transcript.
    pub fn validate_pair(&self) -> VerificationResult<()> {
        self.pair_config().map(|_| ())
    }

    fn pair_claims(
        &self,
        forms: &[&dyn LinearForm<M::Target>],
        values: &[M::Target],
    ) -> VerificationResult<Self> {
        let config = self.pair_config()?;
        if forms.is_empty()
            || forms.len() != values.len()
            || forms
                .iter()
                .any(|form| form.size() != config.tuning().vector_size)
        {
            return Err(VerificationError);
        }
        Ok(config)
    }

    /// Open `[left | right]` from the two staged `commit_pair` calls.
    /// Both original roots are authenticated before the ordinary continuation.
    pub fn prove_pair<H, R>(
        &self,
        ps: &mut ProverState<H, R>,
        committed: [CommittedWitness<M>; 2],
        forms: &[&dyn LinearForm<M::Target>],
        values: &[M::Target],
    ) -> VerificationResult<ProverClaim<M::Target>>
    where
        M::Target: Default + Zeroize + Codec<[H::U]>,
        Standard: Distribution<M::Target>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let config = self.pair_claims(forms, values)?;
        let [mut left, right] = committed;
        let binding = config.pair_binding();
        let masked = config.joint_masks().is_some();
        if !left
            .pair
            .as_ref()
            .zip(right.pair.as_ref())
            .is_some_and(|(left, right)| left.matches(right, binding))
            || left.pending_mask.is_some()
            || right.pending_mask.is_some()
            || left.second_source.is_some()
            || right.second_source.is_some()
            || right.joint_masks.is_some()
            || right.mask_binding.is_some()
            || left.joint_masks.is_some() != masked
            || left.mask_binding.is_some() != masked
        {
            return Err(VerificationError);
        }
        let CommittedState::Round {
            message: right_message,
            irs_witness: right_root,
        } = right.state
        else {
            return Err(VerificationError);
        };
        let CommittedState::Round { message, .. } = &mut left.state else {
            return Err(VerificationError);
        };
        if message.len() != self.tuning().vector_size || right_message.len() != message.len() {
            return Err(VerificationError);
        }
        *message = message.concat(&right_message);
        left.second_source = Some(right_root);
        left.pair = None;
        ps.prover_message(&binding);
        ps.prover_message(&U64(values.len() as u64));
        ps.prover_messages(values);
        Ok(config.prove(ps, left, forms, values))
    }

    /// Verify staged commitments; the result retains the leading selector.
    /// The caller must finish with `FinalClaim::verify(forms)` and transcript completion.
    pub fn verify_pair<H>(
        &self,
        vs: &mut VerifierState<H>,
        commitments: [Commitment<M::Target>; 2],
        forms: &[&dyn LinearForm<M::Target>],
        values: &[M::Target],
    ) -> VerificationResult<FinalClaim<M::Target>>
    where
        M::Target: Default + Codec<[H::U]>,
        Standard: Distribution<M::Target>,
        H: DuplexSpongeInterface,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let config = self.pair_claims(forms, values)?;
        let [mut left, right] = commitments;
        let binding = config.pair_binding();
        let masked = config.joint_masks().is_some();
        if !left
            .pair
            .as_ref()
            .zip(right.pair.as_ref())
            .is_some_and(|(left, right)| left.matches(right, binding))
            || left.second_source.is_some()
            || right.second_source.is_some()
            || right.joint_masks.is_some()
            || right.mask_binding.is_some()
            || left.joint_masks.is_some() != masked
            || left.mask_binding.is_some() != masked
        {
            return Err(VerificationError);
        }
        if vs.prover_message::<Hash>()? != binding
            || vs.prover_message::<U64>()?.0 != values.len() as u64
        {
            return Err(VerificationError);
        }
        for &value in values {
            if vs.prover_message::<M::Target>()? != value {
                return Err(VerificationError);
            }
        }
        left.second_source = Some(right.irs);
        left.pair = None;
        config.verify(vs, left, forms, values)
    }
}
