//! Open two existing commitments with one leading selector coordinate.
//!
//! Both commitments use this source configuration and the same transcript.
//! Forms describe `[left | right]` and are fixed before the opening begins.
//! Claim values are bound before sampling the shared batching challenge.

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
    /// Check pair support before committing. The source must have a code-switch round.
    pub fn validate_pair(&self) -> VerificationResult<()> {
        self.pair_config().map(|_| ())
    }

    fn pair_opening_config(
        &self,
        forms: &[&dyn LinearForm<M::Target>],
        values: &[M::Target],
    ) -> VerificationResult<Self> {
        let pair = self.pair_config()?;
        if forms.is_empty()
            || forms.len() != values.len()
            || forms
                .iter()
                .any(|form| form.size() != pair.tuning().vector_size)
        {
            return Err(VerificationError);
        }
        Ok(pair)
    }

    fn pair_binding(&self) -> Hash {
        let mut digest = blake3::Hasher::new();
        digest.update(b"zook-pair-v1");
        digest.update(&serde_json::to_vec(self).expect("source configuration serializes"));
        Hash(*digest.finalize().as_bytes())
    }

    /// Prove forms over `[left | right]`. The result point starts with the selector.
    /// Invalid parameters, claims, or handles fail before the transcript changes.
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
        let pair = self.pair_opening_config(forms, values)?;
        let source = self
            .first_round()
            .expect("validated first round")
            .code_switch()
            .source();
        if committed.iter().any(|w| {
            !matches!(&w.state,
            CommittedState::Round { message, irs_witness }
                if message.len() == source.vector_size()
                    && irs_witness.matrix.len() == source.size()
                    && irs_witness.masks.len() == source.mask_length() * source.num_messages()
                    && irs_witness.matrix_witness.num_nodes()
                        == source.matrix_commit().merkle_tree.num_nodes())
        }) {
            return Err(VerificationError);
        }
        ps.prover_message(&self.pair_binding());
        ps.prover_message(&U64(values.len() as u64));
        ps.prover_messages(values);
        let [left, right] = committed;
        Ok(pair.prove_with_second(ps, left, Some((source, right)), forms, values))
    }

    /// Verify the pair opening. The caller must finish with `FinalClaim::verify(forms)`.
    pub fn verify_pair<H>(
        &self,
        vs: &mut VerifierState<H>,
        commitments: [Commitment; 2],
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
        let pair = self.pair_opening_config(forms, values)?;
        let binding: Hash = vs.prover_message()?;
        let count: U64 = vs.prover_message()?;
        if binding != self.pair_binding() || count.0 != values.len() as u64 {
            return Err(VerificationError);
        }
        for expected in values {
            if vs.prover_message::<M::Target>()? != *expected {
                return Err(VerificationError);
            }
        }
        let [left, right] = commitments;
        let source = self
            .first_round()
            .expect("validated first round")
            .code_switch()
            .source();
        pair.verify_with_second(vs, left, Some((source, &right)), forms, values)
    }
}
