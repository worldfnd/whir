//! Independent-commitment pair openings remain unsupported.
//! Use `commit_pair` with ordinary `prove` for a shared-mask pair opening.

use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, RngCore};
use zeroize::Zeroize;

use super::{Commitment, CommittedWitness, FinalClaim, ProtocolConfig, ProverClaim};
use crate::{
    algebra::{embedding::Embedding, linear_form::LinearForm},
    hash::Hash,
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState,
        VerificationError, VerificationResult, VerifierState,
    },
};

impl<M: Embedding + Default> ProtocolConfig<M> {
    /// Pair openings of independently committed witnesses remain unsupported.
    /// Returns an error for every configuration.
    pub const fn validate_pair(&self) -> VerificationResult<()> {
        Err(VerificationError)
    }

    /// Pair openings of independently committed witnesses remain unsupported.
    /// Returns an error without changing the transcript.
    pub fn prove_pair<H, R>(
        &self,
        _ps: &mut ProverState<H, R>,
        _committed: [CommittedWitness<M>; 2],
        _forms: &[&dyn LinearForm<M::Target>],
        _values: &[M::Target],
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
        Err(VerificationError)
    }

    /// Pair openings of independently committed witnesses remain unsupported.
    /// Returns an error without consuming or changing the transcript.
    pub fn verify_pair<H>(
        &self,
        _vs: &mut VerifierState<H>,
        _commitments: [Commitment<M::Target>; 2],
        _forms: &[&dyn LinearForm<M::Target>],
        _values: &[M::Target],
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
        Err(VerificationError)
    }
}
