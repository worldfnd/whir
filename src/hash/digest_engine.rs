//! SHA2, SHA3, and Keccak hash engines built from any Rust-Crypto `Digest` with 32-byte output.

use std::{borrow::Cow, marker::PhantomData};

use const_oid::ObjectIdentifier;
use digest::{const_oid::AssociatedOid, consts::U32, Digest};
use hex_literal::hex;
use zerocopy::IntoBytes;

use super::{Hash, HashEngine, HASH_COUNTER};
use crate::{engines::EngineId, utils::zip_strict};

pub const SHA2: EngineId = EngineId::new(hex!(
    "018eaa247cb8957ab1e9fdac885450403c5e7bda1acaaa504e4cc8f2f76bb076"
));
pub const SHA3: EngineId = EngineId::new(hex!(
    "aa802dcdabad8ea8a1430919893b96c30e31ff5734b385999108aa202d27dc12"
));
pub const KECCAK: EngineId = EngineId::new(hex!(
    "ddd248964e320312a66775aee8e16c88c927734be59aca09b7af6deb0ad00e8c"
));

pub type Sha2 = DigestEngine<sha2::Sha256>;
pub type Sha3 = DigestEngine<sha3::Sha3_256>;
pub type Keccak = DigestEngine<sha3::Keccak256>;

/// Wrapper around a `Digest` to implement the `Hasher` trait.
///
/// This allows using any hash in the Rust-Crypto ecosystem,
/// though performance may vary.
#[derive(Clone, Copy, Debug)]
pub struct DigestEngine<D: Digest> {
    name: &'static str,
    oid: Option<ObjectIdentifier>,
    _digest: PhantomData<D>,
}

impl<D> DigestEngine<D>
where
    D: Digest<OutputSize = U32> + Send + Sync,
{
    pub fn from_name(name: &'static str) -> Self {
        assert_eq!(<D as Digest>::output_size(), size_of::<Hash>());
        Self {
            name,
            oid: None,
            _digest: PhantomData,
        }
    }

    pub fn from_name_oid(name: &'static str, oid: ObjectIdentifier) -> Self {
        assert_eq!(<D as Digest>::output_size(), size_of::<Hash>());
        Self {
            name,
            oid: Some(oid),
            _digest: PhantomData,
        }
    }

    pub fn from_name_assoc_oid(name: &'static str) -> Self
    where
        D: AssociatedOid,
    {
        Self::from_name_oid(name, D::OID)
    }
}

impl Default for Sha2 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha2 {
    pub fn new() -> Self {
        Self::from_name_assoc_oid("sha2")
    }
}

impl Default for Sha3 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha3 {
    pub fn new() -> Self {
        Self::from_name_assoc_oid("sha3")
    }
}

impl Default for Keccak {
    fn default() -> Self {
        Self::new()
    }
}

impl Keccak {
    pub fn new() -> Self {
        Self::from_name("keccak")
    }
}

impl<D> HashEngine for DigestEngine<D>
where
    D: Digest<OutputSize = U32> + Send + Sync,
{
    fn name(&self) -> Cow<'_, str> {
        self.name.into()
    }

    fn oid(&self) -> Option<ObjectIdentifier> {
        self.oid
    }

    fn supports_size(&self, _size: usize) -> bool {
        true
    }

    fn preferred_batch_size(&self) -> usize {
        1
    }

    fn hash_many(&self, size: usize, input: &[u8], output: &mut [Hash]) {
        assert_eq!(
            input.len(),
            size * output.len(),
            "Input length ({}) should be size * output.len() = {size} * {}",
            input.len(),
            output.len()
        );

        if size == 0 {
            output.fill(Hash(D::digest([]).into()));
        } else {
            for (input, out) in zip_strict(input.chunks_exact(size), output.iter_mut()) {
                let input = input.as_bytes();
                let hash = D::digest(input);
                out.as_mut_bytes().copy_from_slice(hash.as_ref());
            }
            HASH_COUNTER.add(input.len() / size);
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::engines::Engine;

    #[test]
    fn test_protocol_ids() {
        assert_eq!(Sha2::new().engine_id(), SHA2);
        assert_eq!(Sha3::new().engine_id(), SHA3);
        assert_eq!(Keccak::new().engine_id(), KECCAK);
    }
}
