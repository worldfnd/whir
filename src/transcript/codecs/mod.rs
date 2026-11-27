//! Transcript codecs over the byte alphabet.
//!
//! Spongefish ships no codecs for arkworks types, and the orphan rule stops
//! whir from implementing spongefish's codec traits for them. These traits
//! carry the same contracts as spongefish's `Encoding`, `FromNarg` and
//! `FromUniform`, restricted to byte sponges.

mod ark;

use spongefish::NargReader;
use static_assertions::assert_impl_all;
use zerocopy::{FromBytes, IntoBytes, KnownLayout};

use super::VerificationResult;

/// Map a prover message into bytes.
///
/// The map must be injective and prefix-free. The same bytes are absorbed
/// into the sponge and appended to the NARG string.
pub trait Encoding {
    type Encoded: AsRef<[u8]>;

    fn encode(&self) -> Self::Encoded;
}

/// Parse a prover message from the NARG string.
///
/// Must return the pre-image of [`Encoding::encode`], or an error.
pub trait FromNarg: Sized {
    fn from_narg(reader: &NargReader<'_>) -> VerificationResult<Self>;
}

/// Map uniform bytes to a verifier message.
///
/// For uniform input, the output must be uniform or statistically close to it.
pub trait FromUniform {
    /// Number of uniform bytes [`FromUniform::from_uniform`] takes.
    const UNIFORM_LEN: usize;

    fn from_uniform(uniform: &[u8]) -> Self;
}

/// A value the prover can send: in the transcript or as a hint.
pub trait ProverMessage: Encoding + FromNarg {}

impl<T: Encoding + FromNarg> ProverMessage for T {}

/// A value that can be both a prover message and a verifier message.
pub trait Codec: ProverMessage + FromUniform {}

impl<T: ProverMessage + FromUniform> Codec for T {}

/// An empty object. Like `()` with a `Codec`.
pub struct Empty;

impl Encoding for Empty {
    type Encoded = [u8; 0];

    fn encode(&self) -> Self::Encoded {
        []
    }
}

impl FromNarg for Empty {
    fn from_narg(_reader: &NargReader<'_>) -> VerificationResult<Self> {
        Ok(Self)
    }
}

impl FromUniform for Empty {
    const UNIFORM_LEN: usize = 0;

    fn from_uniform(_uniform: &[u8]) -> Self {
        Self
    }
}

assert_impl_all!(Empty: Codec);

/// A `u64` as 8 little-endian bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, FromBytes, KnownLayout, IntoBytes)]
#[repr(transparent)]
pub struct U64(pub u64);

impl Encoding for U64 {
    type Encoded = [u8; 8];

    fn encode(&self) -> Self::Encoded {
        self.0.to_le_bytes()
    }
}

impl FromNarg for U64 {
    fn from_narg(reader: &NargReader<'_>) -> VerificationResult<Self> {
        reader.take_array().map(u64::from_le_bytes).map(Self)
    }
}

impl FromUniform for U64 {
    const UNIFORM_LEN: usize = 8;

    fn from_uniform(uniform: &[u8]) -> Self {
        Self(u64::from_le_bytes(<[u8; 8]>::from_uniform(uniform)))
    }
}

assert_impl_all!(U64: Codec);

impl Encoding for u8 {
    type Encoded = [Self; 1];

    fn encode(&self) -> Self::Encoded {
        [*self]
    }
}

impl FromNarg for u8 {
    fn from_narg(reader: &NargReader<'_>) -> VerificationResult<Self> {
        reader.take_array().map(|[byte]: [Self; 1]| byte)
    }
}

impl FromUniform for u8 {
    const UNIFORM_LEN: usize = 1;

    fn from_uniform(uniform: &[u8]) -> Self {
        uniform[0]
    }
}

assert_impl_all!(u8: Codec);

impl<const N: usize> Encoding for [u8; N] {
    type Encoded = Self;

    fn encode(&self) -> Self::Encoded {
        *self
    }
}

impl<const N: usize> FromNarg for [u8; N] {
    fn from_narg(reader: &NargReader<'_>) -> VerificationResult<Self> {
        reader.take_array()
    }
}

impl<const N: usize> FromUniform for [u8; N] {
    const UNIFORM_LEN: usize = N;

    fn from_uniform(uniform: &[u8]) -> Self {
        let mut bytes = [0; N];
        bytes.copy_from_slice(uniform);
        bytes
    }
}

assert_impl_all!([u8; 32]: Codec);
