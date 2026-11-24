//! Arkworks (`ark_ff`), `ark_serialize`, and `spongefish` compatibility boilerplate
//! for [`Goldilocks<u64>`].
//!
//! All arithmetic delegates to the core methods in `super` (`goldilocks.rs`), and
//! boundaries requiring unique representation lazily canonicalize via
//! [`Goldilocks::reduce_fully`].

use core::{
    cmp::Ordering,
    fmt::{self, Debug, Display, Formatter},
    hash::{Hash, Hasher},
    iter::{self, Product, Sum},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    str::FromStr,
};

use ark_ff::{
    AdditiveGroup, BigInt, FftField, Field, LegendreSymbol, One, PrimeField, SqrtPrecomputation,
    Zero,
};
use ark_serialize::{
    buffer_byte_size, CanonicalDeserialize, CanonicalDeserializeWithFlags, CanonicalSerialize,
    CanonicalSerializeWithFlags, Compress, EmptyFlags, Flags, SerializationError, Valid, Validate,
};
use ark_std::{
    io::{Read, Write},
    rand::{
        distributions::{Distribution, Standard},
        Rng,
    },
};
use num_bigint::BigUint;
use zeroize::Zeroize;

use super::{Goldilocks, GoldilocksReduce, MODULUS};

// ── Comparison, formatting, hashing, zeroize (lazy canonicalization) ────────

impl PartialEq for Goldilocks<u64> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.reduce_fully() == other.reduce_fully()
    }
}

impl Eq for Goldilocks<u64> {}

impl Hash for Goldilocks<u64> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.reduce_fully().hash(state);
    }
}

impl Ord for Goldilocks<u64> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.reduce_fully().cmp(&other.reduce_fully())
    }
}

impl PartialOrd for Goldilocks<u64> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Debug for Goldilocks<u64> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.reduce_fully())
    }
}

impl Display for Goldilocks<u64> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.reduce_fully())
    }
}

impl Zeroize for Goldilocks<u64> {
    fn zeroize(&mut self) {
        self.0.zeroize();
    }
}

impl FromStr for Goldilocks<u64> {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        let (neg, digits) = s.strip_prefix('-').map_or((false, s), |d| (true, d));
        if digits.is_empty() {
            return Err(());
        }
        let val = BigUint::from_str(digits).map_err(|_| ())?;
        let elem = Self::from(val);
        Ok(if neg { -elem } else { elem })
    }
}

// ── Operators (all val/ref combinations required by ark_ff::Field) ──────────

impl Neg for Goldilocks<u64> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        self.neg_impl()
    }
}

macro_rules! impl_op_variants {
    ($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident, $body:expr) => {
        impl $trait<Self> for Goldilocks<u64> {
            type Output = Self;

            #[inline]
            fn $method(self, rhs: Self) -> Self {
                let f: fn(Self, Self) -> Self = $body;
                f(self, rhs)
            }
        }

        impl $trait<&Self> for Goldilocks<u64> {
            type Output = Self;

            #[inline]
            fn $method(self, rhs: &Self) -> Self {
                self.$method(*rhs)
            }
        }

        impl $trait<&mut Self> for Goldilocks<u64> {
            type Output = Self;

            #[inline]
            fn $method(self, rhs: &mut Self) -> Self {
                self.$method(*rhs)
            }
        }

        impl $trait<Goldilocks<u64>> for &Goldilocks<u64> {
            type Output = Goldilocks<u64>;

            #[inline]
            fn $method(self, rhs: Goldilocks<u64>) -> Goldilocks<u64> {
                (*self).$method(rhs)
            }
        }

        impl $trait<&Goldilocks<u64>> for &Goldilocks<u64> {
            type Output = Goldilocks<u64>;

            #[inline]
            fn $method(self, rhs: &Goldilocks<u64>) -> Goldilocks<u64> {
                (*self).$method(*rhs)
            }
        }

        impl $assign_trait<Self> for Goldilocks<u64> {
            #[inline]
            fn $assign_method(&mut self, rhs: Self) {
                *self = (*self).$method(rhs);
            }
        }

        impl $assign_trait<&Self> for Goldilocks<u64> {
            #[inline]
            fn $assign_method(&mut self, rhs: &Self) {
                *self = (*self).$method(*rhs);
            }
        }

        impl $assign_trait<&mut Self> for Goldilocks<u64> {
            #[inline]
            fn $assign_method(&mut self, rhs: &mut Self) {
                *self = (*self).$method(*rhs);
            }
        }
    };
}

impl_op_variants!(Add, add, AddAssign, add_assign, Goldilocks::add_impl);
impl_op_variants!(Sub, sub, SubAssign, sub_assign, Goldilocks::sub_impl);
impl_op_variants!(Mul, mul, MulAssign, mul_assign, Goldilocks::mul_impl);
impl_op_variants!(Div, div, DivAssign, div_assign, |a, b| {
    a.mul_impl(b.inv_impl().expect("division by zero"))
});

impl Sum for Goldilocks<u64> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, Add::add)
    }
}

impl<'a> Sum<&'a Self> for Goldilocks<u64> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.copied().sum()
    }
}

impl Product for Goldilocks<u64> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, Mul::mul)
    }
}

impl<'a> Product<&'a Self> for Goldilocks<u64> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.copied().product()
    }
}

// ── Integer & BigInt conversions required by ark_ff::PrimeField ─────────────

impl From<bool> for Goldilocks<u64> {
    #[inline]
    fn from(v: bool) -> Self {
        Self(u64::from(v))
    }
}

macro_rules! impl_from_unsigned {
    ($($t:ty),*) => {
        $(
            impl From<$t> for Goldilocks<u64> {
                #[inline]
                fn from(v: $t) -> Self {
                    Self(u64::from(v))
                }
            }
        )*
    };
}

impl_from_unsigned!(u8, u16, u32, u64);

impl From<u128> for Goldilocks<u64> {
    #[inline]
    fn from(v: u128) -> Self {
        v.reduce_u64()
    }
}

macro_rules! impl_from_signed {
    ($($t:ty),*) => {
        $(
            impl From<$t> for Goldilocks<u64> {
                #[inline]
                fn from(v: $t) -> Self {
                    let abs = Self::from(v.unsigned_abs());
                    if v < 0 { -abs } else { abs }
                }
            }
        )*
    };
}

impl_from_signed!(i8, i16, i32, i64, i128);

impl From<BigInt<1>> for Goldilocks<u64> {
    #[inline]
    fn from(v: BigInt<1>) -> Self {
        Self(v.0[0] % MODULUS)
    }
}

impl From<Goldilocks<u64>> for BigInt<1> {
    #[inline]
    fn from(v: Goldilocks<u64>) -> Self {
        Self([v.reduce_fully()])
    }
}

impl From<BigUint> for Goldilocks<u64> {
    fn from(v: BigUint) -> Self {
        let rem = v % BigUint::from(MODULUS);
        Self(rem.iter_u64_digits().next().unwrap_or(0))
    }
}

impl From<Goldilocks<u64>> for BigUint {
    fn from(v: Goldilocks<u64>) -> Self {
        Self::from(v.reduce_fully())
    }
}

// ── ark_ff traits (Zero, One, AdditiveGroup, Field, PrimeField, FftField) ───

impl Zero for Goldilocks<u64> {
    #[inline]
    fn zero() -> Self {
        Self(0)
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.reduce_fully() == 0
    }
}

impl One for Goldilocks<u64> {
    #[inline]
    fn one() -> Self {
        Self(1)
    }

    #[inline]
    fn is_one(&self) -> bool {
        self.reduce_fully() == 1
    }
}

impl AdditiveGroup for Goldilocks<u64> {
    type Scalar = Self;
    const ZERO: Self = Self(0);

    #[inline]
    fn double_in_place(&mut self) -> &mut Self {
        *self = self.add_impl(*self);
        self
    }

    #[inline]
    fn neg_in_place(&mut self) -> &mut Self {
        *self = self.neg_impl();
        self
    }
}

impl Field for Goldilocks<u64> {
    type BasePrimeField = Self;

    const ONE: Self = Self(1);
    const NEG_ONE: Self = Self(MODULUS - 1);

    const SQRT_PRECOMP: Option<SqrtPrecomputation<Self>> =
        Some(SqrtPrecomputation::TonelliShanks {
            two_adicity: Self::TWO_ADICITY,
            quadratic_nonresidue_to_trace: Self::TWO_ADIC_ROOT_OF_UNITY,
            trace_of_modulus_minus_one_div_two: &[0x7fff_ffff],
        });

    fn extension_degree() -> u64 {
        1
    }

    fn from_base_prime_field(elem: Self::BasePrimeField) -> Self {
        elem
    }

    fn to_base_prime_field_elements(&self) -> impl Iterator<Item = Self::BasePrimeField> {
        iter::once(*self)
    }

    fn from_base_prime_field_elems(
        elems: impl IntoIterator<Item = Self::BasePrimeField>,
    ) -> Option<Self> {
        let mut iter = elems.into_iter();
        let first = iter.next()?;
        iter.next().is_none().then_some(first)
    }

    #[inline]
    fn characteristic() -> &'static [u64] {
        &[MODULUS]
    }

    fn from_random_bytes_with_flags<F: Flags>(bytes: &[u8]) -> Option<(Self, F)> {
        if F::BIT_SIZE > 8 {
            return None;
        }
        let output_byte_size = buffer_byte_size(64 + F::BIT_SIZE);
        let mut buf = [0u8; 9];
        let copy_len = bytes.len().min(output_byte_size);
        buf[..copy_len].copy_from_slice(&bytes[..copy_len]);
        let flags = if F::BIT_SIZE == 0 {
            F::from_u8(0)?
        } else {
            let mask = u8::MAX.checked_shl(8 - F::BIT_SIZE as u32).unwrap_or(0);
            F::from_u8(buf[output_byte_size - 1] & mask)?
        };
        let val = u64::from_le_bytes(buf[..8].try_into().ok()?);
        Self::from_bigint(BigInt([val])).map(|elem| (elem, flags))
    }

    #[inline]
    fn sum_of_products<const T: usize>(a: &[Self; T], b: &[Self; T]) -> Self {
        Self::sum_of_products_impl(a, b)
    }

    #[inline]
    fn square(&self) -> Self {
        self.square_impl()
    }

    #[inline]
    fn square_in_place(&mut self) -> &mut Self {
        *self = self.square_impl();
        self
    }

    #[inline]
    fn inverse(&self) -> Option<Self> {
        self.inv_impl()
    }

    #[inline]
    fn inverse_in_place(&mut self) -> Option<&mut Self> {
        self.inv_impl().map(|inv| {
            *self = inv;
            self
        })
    }

    #[inline]
    fn frobenius_map_in_place(&mut self, _: usize) {}

    #[inline]
    fn legendre(&self) -> LegendreSymbol {
        let s = self.pow(Self::MODULUS_MINUS_ONE_DIV_TWO);
        if s.is_zero() {
            LegendreSymbol::Zero
        } else if s.is_one() {
            LegendreSymbol::QuadraticResidue
        } else {
            LegendreSymbol::QuadraticNonResidue
        }
    }

    #[inline]
    fn mul_by_base_prime_field(&self, elem: &Self::BasePrimeField) -> Self {
        self.mul_impl(*elem)
    }
}

impl PrimeField for Goldilocks<u64> {
    type BigInt = BigInt<1>;

    const MODULUS: BigInt<1> = BigInt([MODULUS]);
    const MODULUS_MINUS_ONE_DIV_TWO: BigInt<1> = BigInt([(MODULUS - 1) / 2]);
    const MODULUS_BIT_SIZE: u32 = 64;
    const TRACE: BigInt<1> = BigInt([(MODULUS - 1) >> 32]);
    const TRACE_MINUS_ONE_DIV_TWO: BigInt<1> = BigInt([0x7fff_ffff]);

    #[inline]
    fn from_bigint(repr: BigInt<1>) -> Option<Self> {
        (repr.0[0] < MODULUS).then_some(Self(repr.0[0]))
    }

    #[inline]
    fn into_bigint(self) -> BigInt<1> {
        BigInt([self.reduce_fully()])
    }
}

impl FftField for Goldilocks<u64> {
    const GENERATOR: Self = Self(7);
    const TWO_ADICITY: u32 = 32;
    const TWO_ADIC_ROOT_OF_UNITY: Self = Self(0x1856_29dc_da58_878c);
    const SMALL_SUBGROUP_BASE: Option<u32> = Some(15);
    const SMALL_SUBGROUP_BASE_ADICITY: Option<u32> = Some(1);
    const LARGE_SUBGROUP_ROOT_OF_UNITY: Option<Self> = Some(Self(0x1104_ccab_9493_baeb));
}

impl Distribution<Goldilocks<u64>> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Goldilocks<u64> {
        loop {
            let v: u64 = rng.gen();
            if v < MODULUS {
                return Goldilocks(v);
            }
        }
    }
}

// ── ark_serialize boilerplate ───────────────────────────────────────────────

impl Valid for Goldilocks<u64> {
    const TRIVIAL_CHECK: bool = true;

    #[inline]
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalSerializeWithFlags for Goldilocks<u64> {
    fn serialize_with_flags<W: Write, F: Flags>(
        &self,
        mut writer: W,
        flags: F,
    ) -> Result<(), SerializationError> {
        if F::BIT_SIZE > 8 {
            return Err(SerializationError::NotEnoughSpace);
        }
        let mut buf = [0u8; 9];
        buf[..8].copy_from_slice(&self.reduce_fully().to_le_bytes());
        let len = self.serialized_size_with_flags::<F>();
        buf[len - 1] |= flags.u8_bitmask();
        writer.write_all(&buf[..len])?;
        Ok(())
    }

    #[inline]
    fn serialized_size_with_flags<F: Flags>(&self) -> usize {
        buffer_byte_size(64 + F::BIT_SIZE)
    }
}

impl CanonicalSerialize for Goldilocks<u64> {
    #[inline]
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        self.serialize_with_flags(writer, EmptyFlags)
    }

    #[inline]
    fn serialized_size(&self, _compress: Compress) -> usize {
        8
    }
}

impl CanonicalDeserializeWithFlags for Goldilocks<u64> {
    fn deserialize_with_flags<R: Read, F: Flags>(
        mut reader: R,
    ) -> Result<(Self, F), SerializationError> {
        if F::BIT_SIZE > 8 {
            return Err(SerializationError::NotEnoughSpace);
        }
        let len = buffer_byte_size(64 + F::BIT_SIZE);
        let mut buf = [0u8; 9];
        reader.read_exact(&mut buf[..len])?;
        let flags = F::from_u8_remove_flags(&mut buf[len - 1])
            .ok_or(SerializationError::UnexpectedFlags)?;
        if len == 9 && buf[8] != 0 {
            return Err(SerializationError::InvalidData);
        }
        let val = u64::from_le_bytes(buf[..8].try_into().unwrap());
        Self::from_bigint(BigInt([val]))
            .map(|elem| (elem, flags))
            .ok_or(SerializationError::InvalidData)
    }
}

impl CanonicalDeserialize for Goldilocks<u64> {
    #[inline]
    fn deserialize_with_mode<R: Read>(
        reader: R,
        _compress: Compress,
        _validate: Validate,
    ) -> Result<Self, SerializationError> {
        Self::deserialize_with_flags::<R, EmptyFlags>(reader).map(|(elem, _)| elem)
    }
}

// ── spongefish transcript boilerplate ───────────────────────────────────────

impl spongefish::Unit for Goldilocks<u64> {
    const ZERO: Self = Self(0);
}

impl spongefish::Encoding<[u8]> for Goldilocks<u64> {
    #[inline]
    fn encode(&self) -> impl AsRef<[u8]> {
        self.reduce_fully().to_le_bytes()
    }
}

/// Stack-allocated 40-byte decoding buffer (8-byte modulus + 32 bytes extra randomness).
pub struct GoldilocksDecodingBuf([u8; 40]);

impl Default for GoldilocksDecodingBuf {
    #[inline]
    fn default() -> Self {
        Self([0u8; 40])
    }
}

impl AsMut<[u8]> for GoldilocksDecodingBuf {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl spongefish::Decoding<[u8]> for Goldilocks<u64> {
    type Repr = GoldilocksDecodingBuf;

    #[inline]
    fn decode(repr: Self::Repr) -> Self {
        Self::from_le_bytes_mod_order(&repr.0)
    }
}

impl spongefish::NargDeserialize for Goldilocks<u64> {
    fn deserialize_from_narg(buf: &mut &[u8]) -> spongefish::VerificationResult<Self> {
        if buf.len() < 8 {
            return Err(spongefish::VerificationError);
        }
        let (head, tail) = buf.split_at(8);
        *buf = tail;
        Ok(Self::from_le_bytes_mod_order(head))
    }
}
