//! Wrapper around Spongefish to add out-of-band hint messages.
//!
//! We need these for the Merkle tree proofs as doing them in-transcript
//! would roughly double the verifier cost.

pub mod codecs;
mod mock_sponge;
mod rng;

#[cfg(debug_assertions)]
use std::any::type_name;
use std::fmt::Debug;

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256, Sha3_512};
use spongefish::{derive_session_id, LengthPrefixed, NargReader, PrivateRng, SessionId};
pub use spongefish::{DefaultHash, DuplexSpongeInit, DuplexSpongeInterface, VerificationError};

#[cfg(test)]
pub use self::mock_sponge::MockSponge;
pub use self::{
    codecs::{Codec, Encoding, FromNarg, FromUniform, ProverMessage},
    rng::ProverRng,
};

pub type VerificationResult<T> = Result<T, VerificationError>;

#[macro_export]
macro_rules! verify {
    ($cond:expr) => {
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !$cond {
            #[cfg(feature = "verifier_panics")]
            panic!("Verification failed: {}", stringify!($cond));

            #[cfg(not(feature = "verifier_panics"))]
            return Err(spongefish::VerificationError);
        };
    };
}

#[derive(Clone, Copy, Debug)]
pub struct DomainSeparator<'a, I> {
    pub protocol_id: [u8; 64],
    pub session_id: [u8; 32],
    instance: &'a I,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub enum Interaction {
    ProverMessage(String),
    VerifierMessage(String),
    Hint(String),
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub struct Proof {
    pub narg_string: Vec<u8>,
    pub hints: Vec<u8>,

    #[cfg(debug_assertions)]
    pub pattern: Vec<Interaction>,
}

/// The prover's transcript.
///
/// `H` is the transcript sponge. `R` is the sponge behind the prover's
/// private RNG.
pub struct ProverState<H = DefaultHash, R = DefaultHash>
where
    H: DuplexSpongeInterface<U = u8>,
    R: DuplexSpongeInit<U = u8>,
{
    inner: spongefish::ProverState<H, R>,
    hints: Vec<u8>,

    #[cfg(debug_assertions)]
    pattern: Vec<Interaction>,
}

pub struct VerifierState<'a, H = DefaultHash>
where
    H: DuplexSpongeInterface<U = u8>,
{
    inner: spongefish::VerifierState<'a, H>,
    hints: NargReader<'a>,

    #[cfg(debug_assertions)]
    pattern: &'a [Interaction],
}

pub trait VerifierMessage {
    fn verifier_message<T>(&mut self) -> T
    where
        T: FromUniform;

    fn verifier_message_vec<T>(&mut self, count: usize) -> Vec<T>
    where
        T: FromUniform,
    {
        (0..count).map(|_| self.verifier_message()).collect()
    }
}

impl DomainSeparator<'static, ()> {
    pub fn protocol<C: Serialize>(config: &C) -> Self {
        const INSTANCE: &() = &();
        let mut hash = Sha3_512::new();
        ciborium::into_writer(config, &mut hash).expect("Computing protocol hash failed");
        let protocol_id: [u8; 64] = hash.finalize().into();
        Self {
            protocol_id,
            session_id: [0; 32],
            instance: INSTANCE,
        }
    }

    #[must_use]
    pub fn session<S: Serialize>(self, session: &S) -> Self {
        let mut hash = Sha3_256::new();
        ciborium::into_writer(session, &mut hash).expect("Computing session hash failed");
        let session_id: [u8; 32] = hash.finalize().into();
        Self { session_id, ..self }
    }

    pub const fn instance<I>(self, instance: &I) -> DomainSeparator<'_, I> {
        DomainSeparator {
            protocol_id: self.protocol_id,
            session_id: self.session_id,
            instance,
        }
    }
}

impl<I: Encoding> DomainSeparator<'_, I> {
    /// The spongefish session identifier: `DeriveSessionID(protocol_id ‖ session_id)`.
    fn spongefish_session_id<H: DuplexSpongeInit<U = u8>>(&self) -> SessionId {
        let mut tag = [0; 96];
        tag[..64].copy_from_slice(&self.protocol_id);
        tag[64..].copy_from_slice(&self.session_id);
        derive_session_id::<H>(&tag)
    }
}

impl<H, R> ProverState<H, R>
where
    H: DuplexSpongeInit<U = u8>,
    R: DuplexSpongeInit<U = u8>,
{
    /// Construct a new prover state.
    ///
    /// The private RNG is seeded from the operating system's entropy source.
    pub fn new<I: Encoding>(ds: &DomainSeparator<'_, I>) -> Self {
        // The `u32` length prefix keeps the absorbed instance prefix-free and non-empty.
        let instance = ds.instance.encode();
        Self::from_inner(spongefish::ProverState::from_parts(
            &ds.spongefish_session_id::<H>(),
            &LengthPrefixed(instance.as_ref()),
            PrivateRng::from_os_entropy(),
        ))
    }
}

impl ProverState {
    /// Construct a new prover state with the default duplex hash function.
    pub fn new_std<I: Encoding>(ds: &DomainSeparator<'_, I>) -> Self {
        Self::new(ds)
    }
}

#[cfg(test)]
impl<H: DuplexSpongeInterface<U = u8>> ProverState<H> {
    /// Construct a prover state around a prepared sponge.
    ///
    /// Skips the session identifier and the instance.
    pub fn from_sponge(sponge: H) -> Self {
        Self::from_inner(spongefish::ProverState::from(sponge))
    }
}

impl<H, R> ProverState<H, R>
where
    H: DuplexSpongeInterface<U = u8>,
    R: DuplexSpongeInit<U = u8>,
{
    const fn from_inner(inner: spongefish::ProverState<H, R>) -> Self {
        Self {
            inner,
            hints: Vec::new(),

            #[cfg(debug_assertions)]
            pattern: Vec::new(),
        }
    }

    /// Access the prover's private RNG.
    pub const fn rng(&mut self) -> ProverRng<'_, R> {
        ProverRng(self.inner.rng())
    }

    #[cfg_attr(test, track_caller)]
    pub fn prover_message<T>(&mut self, message: &T)
    where
        T: Encoding + ?Sized,
    {
        #[cfg(debug_assertions)]
        self.push(Interaction::ProverMessage(type_name::<T>().to_owned()));
        let encoded = message.encode();
        self.inner.prover_message_with(
            encoded.as_ref(),
            |bytes| bytes,
            |bytes, narg_string| narg_string.extend_from_slice(bytes),
        );
    }

    #[cfg_attr(test, track_caller)]
    pub fn prover_messages<T>(&mut self, messages: &[T])
    where
        T: Encoding,
    {
        for message in messages {
            self.prover_message(message);
        }
    }

    #[cfg_attr(test, track_caller)]
    pub fn prover_hint<T>(&mut self, hint: &T)
    where
        T: Encoding,
    {
        #[cfg(debug_assertions)]
        self.push(Interaction::Hint(type_name::<T>().to_owned()));
        self.hints.extend_from_slice(hint.encode().as_ref());
    }

    #[cfg_attr(test, track_caller)]
    pub fn prover_hints<T>(&mut self, hints: &[T])
    where
        T: Encoding,
    {
        for hint in hints {
            self.prover_hint(hint);
        }
    }

    pub fn proof(self) -> Proof {
        Proof {
            narg_string: self.inner.into_narg_string(),
            hints: self.hints,

            #[cfg(debug_assertions)]
            pattern: self.pattern,
        }
    }

    #[cfg(debug_assertions)]
    fn push(&mut self, interaction: Interaction) {
        self.pattern.push(interaction);
    }
}

impl<H, R> VerifierMessage for ProverState<H, R>
where
    H: DuplexSpongeInterface<U = u8>,
    R: DuplexSpongeInit<U = u8>,
{
    #[cfg_attr(test, track_caller)]
    fn verifier_message<T>(&mut self) -> T
    where
        T: FromUniform,
    {
        #[cfg(debug_assertions)]
        self.push(Interaction::VerifierMessage(type_name::<T>().to_owned()));
        self.inner
            .verifier_message_as(T::UNIFORM_LEN, T::from_uniform)
    }
}

impl<'a, H> VerifierState<'a, H>
where
    H: DuplexSpongeInit<U = u8>,
{
    pub fn new<I: Encoding>(ds: &DomainSeparator<'_, I>, proof: &'a Proof) -> Self {
        let instance = ds.instance.encode();
        Self::from_inner(
            spongefish::VerifierState::new(
                &ds.spongefish_session_id::<H>(),
                &LengthPrefixed(instance.as_ref()),
                &proof.narg_string,
            ),
            proof,
        )
    }
}

impl<'a> VerifierState<'a> {
    /// Construct a new verifier state with the default duplex hash function.
    pub fn new_std<I: Encoding>(ds: &DomainSeparator<'_, I>, proof: &'a Proof) -> Self {
        Self::new(ds, proof)
    }
}

impl<'a, H> VerifierState<'a, H>
where
    H: DuplexSpongeInterface<U = u8>,
{
    /// Construct a verifier state around a prepared sponge.
    ///
    /// Skips the session identifier and the instance.
    #[cfg(test)]
    pub fn from_sponge(proof: &'a Proof, sponge: H) -> Self {
        Self::from_inner(
            spongefish::VerifierState::from_parts(sponge, &proof.narg_string),
            proof,
        )
    }

    const fn from_inner(inner: spongefish::VerifierState<'a, H>, proof: &'a Proof) -> Self {
        Self {
            inner,
            hints: NargReader::new(proof.hints.as_slice()),
            #[cfg(debug_assertions)]
            pattern: proof.pattern.as_slice(),
        }
    }

    pub const fn as_spongefish(&mut self) -> &mut spongefish::VerifierState<'a, H> {
        &mut self.inner
    }

    #[cfg_attr(debug_assertions, track_caller)]
    pub fn check_eof(self) -> VerificationResult<()> {
        #[cfg(debug_assertions)]
        assert_eq!(self.pattern, []);
        verify!(self.inner.check_eof().is_ok());
        verify!(self.hints.is_empty());
        Ok(())
    }

    #[cfg_attr(test, track_caller)]
    pub fn prover_message<T>(&mut self) -> VerificationResult<T>
    where
        T: ProverMessage,
    {
        #[cfg(debug_assertions)]
        self.pop_pattern(&Interaction::ProverMessage(type_name::<T>().to_owned()));
        self.inner
            .prover_message_with(|reader| reader.read_with(T::from_narg), T::encode)
    }

    #[cfg_attr(test, track_caller)]
    pub fn prover_messages_vec<T>(&mut self, len: usize) -> VerificationResult<Vec<T>>
    where
        T: ProverMessage,
    {
        (0..len).map(|_| self.prover_message()).collect()
    }

    #[cfg_attr(test, track_caller)]
    pub fn prover_hint<T>(&mut self) -> VerificationResult<T>
    where
        T: FromNarg,
    {
        #[cfg(debug_assertions)]
        self.pop_pattern(&Interaction::Hint(type_name::<T>().to_owned()));
        self.hints.read_with(T::from_narg)
    }

    #[cfg_attr(test, track_caller)]
    pub fn prover_hints_vec<T>(&mut self, len: usize) -> VerificationResult<Vec<T>>
    where
        T: FromNarg,
    {
        (0..len).map(|_| self.prover_hint()).collect()
    }

    #[cfg(debug_assertions)]
    #[track_caller]
    fn pop_pattern(&mut self, interaction: &Interaction) {
        assert_ne!(self.pattern, []);
        let (expected, tail) = self.pattern.split_first().unwrap();
        assert_eq!(
            interaction, expected,
            "Transcript error: Expected interaction {expected:?} got {interaction:?}"
        );
        self.pattern = tail;
    }
}

impl<H> VerifierMessage for VerifierState<'_, H>
where
    H: DuplexSpongeInterface<U = u8>,
{
    #[cfg_attr(test, track_caller)]
    fn verifier_message<T>(&mut self) -> T
    where
        T: FromUniform,
    {
        #[cfg(debug_assertions)]
        self.pop_pattern(&Interaction::VerifierMessage(type_name::<T>().to_owned()));
        self.inner
            .verifier_message_as(T::UNIFORM_LEN, T::from_uniform)
    }
}

#[cfg(test)]
mod tests {
    use std::panic::catch_unwind;

    use super::{
        codecs::{Empty, U64},
        *,
    };
    use crate::algebra::fields::Field64;

    /// Sends a field element and a hint, then samples a challenge.
    fn prove<I: Encoding>(ds: &DomainSeparator<'_, I>, message: Field64) -> (Proof, Field64) {
        let mut ps = ProverState::new_std(ds);
        ps.prover_message(&message);
        ps.prover_hint(&U64(7));
        let challenge = ps.verifier_message();
        (ps.proof(), challenge)
    }

    fn verify<I: Encoding>(
        ds: &DomainSeparator<'_, I>,
        proof: &Proof,
    ) -> VerificationResult<(Field64, Field64)> {
        let mut vs = VerifierState::new_std(ds, proof);
        let message = vs.prover_message()?;
        let U64(hint) = vs.prover_hint()?;
        verify!(hint == 7);
        let challenge = vs.verifier_message();
        vs.check_eof()?;
        Ok((message, challenge))
    }

    /// Rejection is an error, or a panic under `verifier_panics`.
    fn rejects(ds: &DomainSeparator<'_, Empty>, proof: &Proof) -> bool {
        catch_unwind(|| verify(ds, proof)).map_or(true, |result| result.is_err())
    }

    #[test]
    fn round_trip_with_empty_instance() {
        let ds = DomainSeparator::protocol(&"transcript-test").instance(&Empty);
        let message = Field64::from(5);
        let (proof, challenge) = prove(&ds, message);
        assert_eq!(verify(&ds, &proof).unwrap(), (message, challenge));
    }

    #[test]
    fn challenge_binds_protocol_session_and_instance() {
        let base = DomainSeparator::protocol(&"transcript-test");
        let message = Field64::from(5);
        let (_, challenge) = prove(&base.instance(&Empty), message);
        let (_, other_protocol) = prove(
            &DomainSeparator::protocol(&"other").instance(&Empty),
            message,
        );
        let (_, other_session) = prove(&base.session(&"other").instance(&Empty), message);
        let (_, other_instance) = prove(&base.instance(&U64(0)), message);
        for other in [other_protocol, other_session, other_instance] {
            assert_ne!(other, challenge);
        }
    }

    #[test]
    fn rejects_trailing_bytes() {
        let ds = DomainSeparator::protocol(&"transcript-test").instance(&Empty);
        let (proof, _) = prove(&ds, Field64::from(5));

        let mut trailing_narg = proof.clone();
        trailing_narg.narg_string.push(0);
        assert!(rejects(&ds, &trailing_narg));

        let mut trailing_hint = proof;
        trailing_hint.hints.push(0);
        assert!(rejects(&ds, &trailing_hint));
    }
}
