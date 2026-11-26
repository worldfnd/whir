//! Workaround for `ark_ff` lacking Serde support.
//!
//! See <https://github.com/arkworks-rs/algebra/pull/506>

use serde::{de::Error as _, ser::Error as _, Deserialize as _, Deserializer, Serializer};

use crate::utils::zip_strict;

/// Serialize using ark_serialize
pub mod canonical {
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

    use super::*;

    pub fn serialize<T, S>(obj: &T, serializer: S) -> Result<S::Ok, S::Error>
    where
        T: CanonicalSerialize,
        S: Serializer,
    {
        let mut buf = Vec::with_capacity(obj.compressed_size());
        obj.serialize_compressed(&mut buf)
            .map_err(|e| S::Error::custom(format!("Failed to serialize: {e}")))?;
        super::bytes::serialize(&buf, serializer)
    }

    pub fn deserialize<'de, T, D>(deserializer: D) -> Result<T, D::Error>
    where
        T: CanonicalDeserialize,
        D: Deserializer<'de>,
    {
        let bytes = super::bytes::deserialize(deserializer)?;
        let mut reader = &*bytes;
        let obj = T::deserialize_compressed(&mut reader)
            .map_err(|e| D::Error::custom(format!("while deserializing: {e}")))?;
        if !reader.is_empty() {
            return Err(D::Error::custom("while deserializing: trailing bytes"));
        }

        Ok(obj)
    }
}

pub mod bytes {
    use super::*;

    pub fn serialize<S>(value: &[u8], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if serializer.is_human_readable() {
            let hex = hex::encode(value);
            serializer.serialize_str(&hex)
        } else {
            serializer.serialize_bytes(value)
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            let hex = String::deserialize(deserializer)?;
            hex::decode(hex)
                .map_err(|e| D::Error::custom(format!("while deserializing bytes: {e}")))
        } else {
            <Vec<u8>>::deserialize(deserializer)
        }
    }
}

pub mod bigint {
    use std::iter::repeat;

    use ark_ff::BigInteger;
    use zerocopy::IntoBytes;

    use super::*;

    pub fn serialize<T, S>(obj: &T, serializer: S) -> Result<S::Ok, S::Error>
    where
        T: BigInteger,
        S: Serializer,
    {
        // Encode as big endian bytes without leading zeros.
        let value = obj.to_bytes_be();
        let leading_zeros = value.iter().take_while(|&&b| b == 0).count();
        let value = &value[leading_zeros..];
        bytes::serialize(value, serializer)
    }

    pub fn deserialize<'de, T, D>(deserializer: D) -> Result<T, D::Error>
    where
        T: BigInteger,
        D: Deserializer<'de>,
    {
        let bytes = super::bytes::deserialize(deserializer)?;
        if bytes[0] == 0 {
            return Err(D::Error::custom("Bigint has leading zeros."));
        }
        if bytes.len() > T::NUM_LIMBS * 8 {
            return Err(D::Error::custom("Value exceeds target size."));
        }
        let le_bytes = bytes.into_iter().rev().chain(repeat(0_u8));

        let mut result = T::default();
        #[cfg(target_endian = "little")]
        let bytes = result.as_mut().as_mut_bytes();
        let bytes_len = bytes.len();
        for (dst, src) in zip_strict(bytes.iter_mut(), le_bytes.take(bytes_len)) {
            *dst = src;
        }
        Ok(result)
    }
}

pub mod field {
    use ark_ff::{Field, PrimeField};
    use serde::{ser::SerializeSeq, Deserialize, Serialize};

    use super::*;

    struct Wrapper<F: Field>(F);

    impl<F: Field> Serialize for Wrapper<F> {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            serialize(&self.0, serializer)
        }
    }

    impl<'de, F: Field> Deserialize<'de> for Wrapper<F> {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            deserialize(deserializer).map(Wrapper)
        }
    }

    pub fn serialize<T, S>(obj: &T, serializer: S) -> Result<S::Ok, S::Error>
    where
        T: Field,
        S: Serializer,
    {
        if T::extension_degree() == 1 {
            // For prime fields encode the element directly
            let base = obj.to_base_prime_field_elements().next().unwrap();
            super::bigint::serialize(&base.into_bigint(), serializer)
        } else {
            // For extension fields encode a sequence of base field elements.
            let mut seq = serializer.serialize_seq(Some(T::extension_degree() as usize))?;
            for coeff in obj.to_base_prime_field_elements() {
                seq.serialize_element(&Wrapper(coeff))?;
            }
            seq.end()
        }
    }

    pub fn deserialize<'de, T, D>(deserializer: D) -> Result<T, D::Error>
    where
        T: Field,
        D: Deserializer<'de>,
    {
        if T::extension_degree() == 1 {
            let bigint = super::bigint::deserialize(deserializer)?;
            let base = T::BasePrimeField::from_bigint(bigint)
                .ok_or_else(|| D::Error::custom("Prime field element not reduced."))?;
            Ok(T::from_base_prime_field(base))
        } else {
            let coeffs = <Vec<Wrapper<T::BasePrimeField>>>::deserialize(deserializer)?;
            let num_coeffs = coeffs.len();
            let coeffs = coeffs.into_iter().map(|c| c.0);
            T::from_base_prime_field_elems(coeffs).ok_or_else(|| {
                D::Error::custom(format!(
                    "Incorrect number of elements {num_coeffs} for extension degree {}",
                    T::extension_degree()
                ))
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::{BigInteger, Field, PrimeField, UniformRand};
    use serde::{Deserialize, Serialize};

    use crate::{
        algebra::fields::{Field256, Field64, Field64_3},
        test_utils::test_serde,
    };

    #[test]
    fn test_bigint() {
        #[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
        struct Wrapper<T: BigInteger>(#[serde(with = "crate::ark_serde::bigint")] T);

        test_serde(&Wrapper(Field256::MODULUS));
        test_serde(&Wrapper(Field64::MODULUS));
    }

    #[test]
    fn test_field() {
        #[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
        struct Wrapper<T: Field>(#[serde(with = "crate::ark_serde::field")] T);

        let mut rng = ark_std::rand::thread_rng();

        test_serde(&Wrapper(Field256::rand(&mut rng)));
        test_serde(&Wrapper(Field64::rand(&mut rng)));
        test_serde(&Wrapper(Field64_3::rand(&mut rng)));
    }
}
