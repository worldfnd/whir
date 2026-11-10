//! Open two independently committed witnesses with one leading selector coordinate.
//!
//! Both handles must use the same configuration and be opened only once. The first
//! block authenticates both source roots; all later blocks share one witness.
//! Callers must bind public forms to the transcript before opening; the pair
//! header binds the configuration and claimed values, not form contents.

use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, RngCore};
use zeroize::Zeroize;

use super::{
    commit::CommittedState,
    round::{reduce_prover, reduce_verifier},
    Commitment, CommittedWitness, FinalClaim, ProtocolConfig, ProverClaim,
};
use crate::{
    algebra::{embedding::Embedding, linear_form::LinearForm},
    buffer::BufferOps,
    hash::Hash,
    protocols::{irs_commit::Config as IrsConfig, params::RoundConfig},
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState,
        VerificationError, VerificationResult, VerifierState,
    },
};

impl<M: Embedding + Default> ProtocolConfig<M> {
    /// Check pair support before committing. Requires a power-of-two witness
    /// and a nonempty first folding block with a code switch.
    /// Bounds retain the configured per-step target; the pair adds a selector
    /// step and a second initial mask-binding event to the protocol's union bound.
    pub fn validate_pair(&self) -> VerificationResult<()> {
        self.pair_round(&self.pair_source()?).map(|_| ())
    }

    fn pair_opening_round<'a>(
        &'a self,
        source: &'a IrsConfig<M>,
        forms: &[&dyn LinearForm<M::Target>],
        values: &[M::Target],
    ) -> VerificationResult<RoundConfig<'a, M>> {
        let first = self.pair_round(source)?;
        if forms.is_empty()
            || forms.len() != values.len()
            || forms
                .iter()
                .any(|form| form.size() != first.sumcheck.initial_size())
        {
            return Err(VerificationError);
        }
        Ok(first)
    }

    /// Prove forms on `[left | right]`; the result point starts with the selector.
    /// Handles must come from independent commitments under this configuration.
    /// Bind the public forms to the transcript before opening either handle.
    /// Invalid shapes and repeated mask roots fail before changing the transcript.
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
        let source = self.pair_source()?;
        let first = self.pair_opening_round(&source, forms, values)?;
        let source = self
            .first_round()
            .ok_or(VerificationError)?
            .code_switch
            .source();
        let binding = self.commitment_config_binding();
        if committed.iter().any(|w| {
            let source_matches = matches!(&w.state,
                CommittedState::Round { message, irs_witness }
                    if message.len() == source.vector_size()
                        && irs_witness.matrix.len() == source.size()
                        && irs_witness.masks.len() == source.mask_length() * source.num_messages()
                        && irs_witness.matrix_witness.num_nodes() == source.matrix_commit().merkle_tree.num_nodes());
            let masks_match = match (self.joint_masks(), &w.joint_masks, &w.mask_binding) {
                (Some(layout), Some(masks), Some(_)) => {
                    masks.coefficients.len() == layout.commit.vector_size()
                        && masks.opening_blinder.len() == layout.commit.vector_size()
                        && masks.irs_witness.matrix.len() == layout.commit.size()
                        && masks.irs_witness.matrix_witness.num_nodes() == layout.commit.matrix_commit().merkle_tree.num_nodes()
                }
                (None, None, None) => true,
                _ => false,
            };
            w.config_binding != binding || !source_matches || !masks_match
        }) {
            return Err(VerificationError);
        }
        if let (Some(left), Some(right)) = (&committed[0].joint_masks, &committed[1].joint_masks) {
            if left.irs_witness.matrix_witness.commitment()
                == right.irs_witness.matrix_witness.commitment()
            {
                return Err(VerificationError);
            }
        }
        ps.prover_message(&binding);
        ps.prover_message(&U64(values.len() as u64));
        ps.prover_messages(values);
        let [left, right] = committed;
        let reduced = reduce_prover(self, ps, left, Some(right), Some(&first), forms, values);
        Ok(self.prove_final_opening(ps, reduced, first.sumcheck.initial_size()))
    }

    /// Verify a pair of independent, single-use commitments under this configuration.
    /// The public forms must already be bound to the transcript.
    /// The caller must finish by calling `FinalClaim::verify(forms)`.
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
        let source = self.pair_source()?;
        let first = self.pair_opening_round(&source, forms, values)?;
        let expected_binding = self.commitment_config_binding();
        if commitments
            .iter()
            .any(|c| c.config_binding != expected_binding)
        {
            return Err(VerificationError);
        }
        match (&commitments[0].joint_masks, &commitments[1].joint_masks) {
            (Some(left), Some(right))
                if self.joint_masks().is_some()
                    && left != right
                    && commitments.iter().all(|c| c.mask_binding.is_some()) => {}
            (None, None)
                if self.joint_masks().is_none()
                    && commitments.iter().all(|c| c.mask_binding.is_none()) => {}
            _ => return Err(VerificationError),
        }
        let binding: Hash = vs.prover_message()?;
        let count: U64 = vs.prover_message()?;
        if binding != expected_binding || count.0 != values.len() as u64 {
            return Err(VerificationError);
        }
        for &value in values {
            if vs.prover_message::<M::Target>()? != value {
                return Err(VerificationError);
            }
        }
        let [left, right] = commitments;
        let reduced = reduce_verifier(self, vs, left, Some(right), Some(&first), forms, values)?;
        self.verify_final_opening(vs, reduced, first.sumcheck.initial_size())
    }
}
