//! Codecs for arkworks fields.

use ark_ff::{BigInteger, Field, Fp, Fp2, Fp2Config, Fp3, Fp3Config, FpConfig, PrimeField};
use spongefish::NargReader;

use super::{Encoding, FromNarg, FromUniform};
use crate::transcript::{VerificationError, VerificationResult};

/// Bytes per base-field limb in the NARG encoding: `⌈log2(p) / 8⌉`.
const fn limb_len(modulus_bits: u32) -> usize {
    modulus_bits.div_ceil(8) as usize
}

/// Uniform bytes per field element: each limb takes 32 bytes more than its
/// encoding, so reducing mod p has bias at most 2^-256.
const fn uniform_len(modulus_bits: u32, degree: usize) -> usize {
    degree * (limb_len(modulus_bits) + 32)
}

/// Each base-field limb as `⌈log2(p) / 8⌉` little-endian bytes.
fn encode_field<F: Field>(value: &F) -> Vec<u8> {
    let limb = limb_len(F::BasePrimeField::MODULUS_BIT_SIZE);
    let mut bytes = Vec::with_capacity(limb * F::extension_degree() as usize);
    for element in value.to_base_prime_field_elements() {
        let start = bytes.len();
        bytes.extend_from_slice(&element.into_bigint().to_bytes_le());
        bytes.resize(start + limb, 0);
    }
    bytes
}

fn field_from_narg<F: Field>(reader: &NargReader<'_>) -> VerificationResult<F> {
    let limb = limb_len(F::BasePrimeField::MODULUS_BIT_SIZE);
    let bytes = reader.take(limb * F::extension_degree() as usize)?;
    let limbs = bytes
        .chunks_exact(limb)
        .map(F::BasePrimeField::from_le_bytes_mod_order);
    F::from_base_prime_field_elems(limbs).ok_or(VerificationError)
}

fn field_from_uniform<F: Field>(uniform: &[u8]) -> F {
    let chunk = limb_len(F::BasePrimeField::MODULUS_BIT_SIZE) + 32;
    let limbs = uniform
        .chunks_exact(chunk)
        .map(F::BasePrimeField::from_le_bytes_mod_order);
    F::from_base_prime_field_elems(limbs).expect("uniform length matches the extension degree")
}

macro_rules! impl_field_codecs {
    (impl [$($generics:tt)*] for $type:ty, modulus_bits = $bits:expr, degree = $degree:expr) => {
        impl<$($generics)*> Encoding for $type {
            type Encoded = Vec<u8>;

            fn encode(&self) -> Self::Encoded {
                encode_field(self)
            }
        }

        impl<$($generics)*> FromNarg for $type {
            fn from_narg(reader: &NargReader<'_>) -> VerificationResult<Self> {
                field_from_narg(reader)
            }
        }

        impl<$($generics)*> FromUniform for $type {
            const UNIFORM_LEN: usize = uniform_len($bits, $degree);

            fn from_uniform(uniform: &[u8]) -> Self {
                field_from_uniform(uniform)
            }
        }
    };
}

impl_field_codecs!(
    impl [C: FpConfig<N>, const N: usize] for Fp<C, N>,
    modulus_bits = <Self as PrimeField>::MODULUS_BIT_SIZE,
    degree = 1
);
impl_field_codecs!(
    impl [C: Fp2Config] for Fp2<C>,
    modulus_bits = <C::Fp as PrimeField>::MODULUS_BIT_SIZE,
    degree = 2
);
impl_field_codecs!(
    impl [C: Fp3Config] for Fp3<C>,
    modulus_bits = <C::Fp as PrimeField>::MODULUS_BIT_SIZE,
    degree = 3
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        algebra::fields::{Field128, Field192, Field256, Field64, Field64_2, Field64_3},
        transcript::Codec,
    };

    fn field_codecs<F: Field + Codec>() {
        let limb = limb_len(F::BasePrimeField::MODULUS_BIT_SIZE);
        let degree = F::extension_degree() as usize;
        assert_eq!(F::UNIFORM_LEN, degree * (limb + 32));

        let value = F::rand(&mut ark_std::test_rng());
        let encoded = value.encode();
        assert_eq!(encoded.as_ref().len(), degree * limb);

        let reader = NargReader::new(encoded.as_ref());
        assert_eq!(reader.read_with(F::from_narg).unwrap(), value);
        assert!(reader.is_empty());

        let truncated = NargReader::new(&encoded.as_ref()[1..]);
        assert!(truncated.read_with(F::from_narg).is_err());
    }

    #[test]
    fn field_codecs_round_trip() {
        field_codecs::<Field64>();
        field_codecs::<Field128>();
        field_codecs::<Field192>();
        field_codecs::<Field256>();
        field_codecs::<Field64_2>();
        field_codecs::<Field64_3>();
    }
}
