use std::marker::PhantomData;

use ark_ff::{
    BigInt, Field, Fp, Fp128, Fp192, Fp2, Fp256, Fp2Config, Fp3, Fp3Config, Fp64, FpConfig,
    MontBackend, MontConfig, PrimeField, SqrtPrecomputation,
};
use serde::{Deserialize, Serialize};
use zerocopy::IntoBytes;

use crate::type_info::TypeInfo;

pub trait FieldWithSize {
    fn field_size_bits() -> f64;
}

impl<F> FieldWithSize for F
where
    F: Field,
{
    fn field_size_bits() -> f64 {
        // Compute modulus as f64
        const BASE264: f64 = 18_446_744_073_709_551_616_f64;
        let modulus = F::BasePrimeField::MODULUS;
        let limbs_le = modulus.as_ref();
        let mut modulus = 0.0_f64;
        for limb in limbs_le.iter().rev() {
            modulus *= BASE264;
            modulus += *limb as f64;
        }
        modulus.log2() * F::extension_degree() as f64
    }
}

/// Type information for a finite field.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FieldInfo {
    /// Field characteristic (aka prime or modulus) in big-endian without leading zeros.
    #[serde(with = "crate::ark_serde::bytes")]
    characteristic: Vec<u8>,

    /// Extension degree of the field.
    extension_degree: usize,
}

impl<F: Field> TypeInfo for F {
    type Info = FieldInfo;

    fn type_info() -> Self::Info {
        // Get the bytes of the characteristic in little-endian order.
        #[cfg(not(target_endian = "little"))]
        compile_error!("This crate requires a little-endian target.");
        let characteristic = F::characteristic().as_bytes();
        // Convert to big-endian vec without leading zeros.
        let characteristic = characteristic
            .iter()
            .copied()
            .rev()
            .skip_while(|&b| b == 0)
            .collect();
        FieldInfo {
            characteristic,
            extension_degree: F::extension_degree() as usize,
        }
    }
}

#[derive(MontConfig)]
#[modulus = "21888242871839275222246405745257275088548364400416034343698204186575808495617"]
#[generator = "5"]
#[small_subgroup_base = "3"]
#[small_subgroup_power = "2"]
pub struct BN254Config;
pub type Field256 = Fp256<MontBackend<BN254Config, 4>>;

#[derive(MontConfig)]
#[modulus = "3801539170989320091464968600173246866371124347557388484609"]
#[generator = "3"]
pub struct FConfig192;
pub type Field192 = Fp192<MontBackend<FConfig192, 3>>;

#[derive(MontConfig)]
#[modulus = "340282366920938463463374557953744961537"]
#[generator = "3"]
pub struct FrConfig128;
pub type Field128 = Fp128<MontBackend<FrConfig128, 2>>;

/// The Goldilocks prime, `p = 2^64 - 2^32 + 1`.
const GOLDILOCKS_P: u64 = 0xFFFF_FFFF_0000_0001;

/// `ε = 2^32 - 1`. Because `2^64 ≡ ε` and `2^96 ≡ -1 (mod p)`, a 128-bit value
/// folds down to 64 bits using only shifts, masks and one multiply by `ε`.
const GOLDILOCKS_EPSILON: u64 = 0xFFFF_FFFF;

/// The Goldilocks field with specialized Solinas arithmetic.
///
/// Elements are kept in canonical form (`[0, p)`, no Montgomery factor):
/// products are reduced with the folds above instead of Montgomery REDC, and
/// `from_bigint` / `into_bigint` become no-ops.
pub struct GoldilocksConfig;
pub type Field64 = Fp64<GoldilocksConfig>;

/// `Field64` from a canonical integer (must be `< p`).
const fn gold(x: u64) -> Field64 {
    assert!(x < GOLDILOCKS_P);
    Fp(BigInt::new([x]), PhantomData)
}

/// The canonical value of `a`.
#[inline]
const fn gold_raw(a: Field64) -> u64 {
    a.0 .0[0]
}

/// Bring a value `< 2p` into the canonical range `[0, p)`.
#[inline]
const fn gold_canonical(x: u64) -> u64 {
    if x >= GOLDILOCKS_P {
        x - GOLDILOCKS_P
    } else {
        x
    }
}

#[inline]
const fn gold_add(a: u64, b: u64) -> u64 {
    // On carry the wrapped sum is missing 2^64 ≡ ε; adding it back cannot
    // overflow since a, b < p. The result is < 2p either way.
    let (sum, carry) = a.overflowing_add(b);
    gold_canonical(sum + GOLDILOCKS_EPSILON * carry as u64)
}

#[inline]
const fn gold_sub(a: u64, b: u64) -> u64 {
    // On borrow the wrapped difference is `a - b + 2^64`; subtracting ε turns
    // the excess 2^64 into p, landing in `[0, p)` directly.
    let (diff, borrow) = a.overflowing_sub(b);
    diff - GOLDILOCKS_EPSILON * borrow as u64
}

#[inline]
const fn gold_neg(a: u64) -> u64 {
    if a == 0 {
        0
    } else {
        GOLDILOCKS_P - a
    }
}

/// Reduce a 128-bit product. Splitting `x = lo + 2^64 mid + 2^96 hi` (with
/// `mid`, `hi` 32 bits each) gives `x ≡ lo + ε mid - hi (mod p)`.
#[inline]
const fn gold_reduce128(x: u128) -> u64 {
    let lo = x as u64;
    let mid = (x >> 64) as u64 & GOLDILOCKS_EPSILON;
    let hi = (x >> 96) as u64;

    // lo - hi: a borrow's -2^64 is compensated by -ε (cannot underflow twice).
    let (t, borrow) = lo.overflowing_sub(hi);
    let t = t - GOLDILOCKS_EPSILON * borrow as u64;
    // + ε mid (< p): a carry's 2^64 folds to ε, like in `gold_add`.
    let (sum, carry) = t.overflowing_add(mid * GOLDILOCKS_EPSILON);
    gold_canonical(sum + GOLDILOCKS_EPSILON * carry as u64)
}

#[inline]
const fn gold_mul(a: u64, b: u64) -> u64 {
    gold_reduce128(a as u128 * b as u128)
}

/// `x^(2^n)`.
const fn gold_sqn(mut x: u64, n: u32) -> u64 {
    let mut i = 0;
    while i < n {
        x = gold_mul(x, x);
        i += 1;
    }
    x
}

/// `a^(p-2)` (Fermat). The exponent `2^64 - 2^32 - 1` is all-ones except bit
/// 32, so build `a^(2^31 - 1)` by doubling chunks, then shift it over itself:
/// 63 squarings and 9 multiplications instead of ~125 for square-and-multiply.
const fn gold_inv(a: u64) -> u64 {
    let t2 = gold_mul(gold_sqn(a, 1), a); // a^(2^2 - 1)
    let t3 = gold_mul(gold_sqn(t2, 1), a); // a^(2^3 - 1)
    let t6 = gold_mul(gold_sqn(t3, 3), t3); // a^(2^6 - 1)
    let t12 = gold_mul(gold_sqn(t6, 6), t6); // a^(2^12 - 1)
    let t24 = gold_mul(gold_sqn(t12, 12), t12); // a^(2^24 - 1)
    let t30 = gold_mul(gold_sqn(t24, 6), t6); // a^(2^30 - 1)
    let t31 = gold_mul(gold_sqn(t30, 1), a); // a^(2^31 - 1)
    let t63 = gold_mul(gold_sqn(t31, 32), t31); // a^(2^63 - 2^32 + 2^31 - 1)
    gold_mul(gold_sqn(t63, 1), a) // a^(2^64 - 2^32 - 1) = a^(p - 2)
}

impl FpConfig<1> for GoldilocksConfig {
    const MODULUS: BigInt<1> = BigInt::new([GOLDILOCKS_P]);
    const GENERATOR: Field64 = gold(7);
    const ZERO: Field64 = gold(0);
    const ONE: Field64 = gold(1);
    const NEG_ONE: Field64 = gold(GOLDILOCKS_P - 1);
    // p - 1 = 2^32 * 3 * 5 * 17 * 257 * 65537. Alongside the 2^32 tower this
    // exposes the order 3 * 2^32 subgroup that the radix-2/3 NTT builds on.
    const TWO_ADICITY: u32 = 32;
    // 7^((p-1) / 2^32)
    const TWO_ADIC_ROOT_OF_UNITY: Field64 = gold(1_753_635_133_440_165_772);
    const SMALL_SUBGROUP_BASE: Option<u32> = Some(3);
    const SMALL_SUBGROUP_BASE_ADICITY: Option<u32> = Some(1);
    // 7^((p-1) / (3 * 2^32))
    const LARGE_SUBGROUP_ROOT_OF_UNITY: Option<Field64> = Some(gold(14_159_254_819_154_955_796));
    const SQRT_PRECOMP: Option<SqrtPrecomputation<Field64>> =
        Some(SqrtPrecomputation::TonelliShanks {
            two_adicity: 32,
            // 7^t for the odd trace t = (p-1) / 2^32; coincides with
            // TWO_ADIC_ROOT_OF_UNITY.
            quadratic_nonresidue_to_trace: gold(1_753_635_133_440_165_772),
            // (t - 1) / 2
            trace_of_modulus_minus_one_div_two: &[2_147_483_647],
        });

    fn add_assign(a: &mut Field64, b: &Field64) {
        a.0 .0[0] = gold_add(gold_raw(*a), gold_raw(*b));
    }

    fn sub_assign(a: &mut Field64, b: &Field64) {
        a.0 .0[0] = gold_sub(gold_raw(*a), gold_raw(*b));
    }

    fn double_in_place(a: &mut Field64) {
        a.0 .0[0] = gold_add(gold_raw(*a), gold_raw(*a));
    }

    fn neg_in_place(a: &mut Field64) {
        a.0 .0[0] = gold_neg(gold_raw(*a));
    }

    fn mul_assign(a: &mut Field64, b: &Field64) {
        a.0 .0[0] = gold_mul(gold_raw(*a), gold_raw(*b));
    }

    fn square_in_place(a: &mut Field64) {
        let v = gold_raw(*a);
        a.0 .0[0] = gold_mul(v, v);
    }

    fn sum_of_products<const T: usize>(a: &[Field64; T], b: &[Field64; T]) -> Field64 {
        let mut acc = 0;
        for (a, b) in a.iter().zip(b) {
            acc = gold_add(acc, gold_mul(gold_raw(*a), gold_raw(*b)));
        }
        gold(acc)
    }

    fn inverse(a: &Field64) -> Option<Field64> {
        let v = gold_raw(*a);
        (v != 0).then(|| gold(gold_inv(v)))
    }

    fn from_bigint(other: BigInt<1>) -> Option<Field64> {
        (other.0[0] < GOLDILOCKS_P).then(|| gold(other.0[0]))
    }

    fn into_bigint(other: Field64) -> BigInt<1> {
        other.0
    }
}

pub type Field64_2 = Fp2<F2Config64>;
pub struct F2Config64;
impl Fp2Config for F2Config64 {
    type Fp = Field64;

    const NONRESIDUE: Self::Fp = gold(7);

    const FROBENIUS_COEFF_FP2_C1: &'static [Self::Fp] = &[
        // Fq(7)**(((q^0) - 1) / 2)
        gold(1),
        // Fq(7)**(((q^1) - 1) / 2)
        gold(18_446_744_069_414_584_320),
    ];
}

pub type Field64_3 = Fp3<F3Config64>;
pub struct F3Config64;

impl Fp3Config for F3Config64 {
    type Fp = Field64;

    const NONRESIDUE: Self::Fp = gold(2);

    const FROBENIUS_COEFF_FP3_C1: &'static [Self::Fp] = &[
        gold(1),
        // Fq(2)^(((q^1) - 1) / 3)
        gold(4_294_967_295),
        // Fq(2)^(((q^2) - 1) / 3)
        gold(18_446_744_065_119_617_025),
    ];

    const FROBENIUS_COEFF_FP3_C2: &'static [Self::Fp] = &[
        gold(1),
        // Fq(2)^(((2q^1) - 2) / 3)
        gold(18_446_744_065_119_617_025),
        // Fq(2)^(((2q^2) - 2) / 3)
        gold(4_294_967_295),
    ];

    // (q^3 - 1) = 2^32 * T where T = 1461501636310055817916238417282618014431694553085
    const TWO_ADICITY: u32 = 32;

    // 11^T
    const QUADRATIC_NONRESIDUE_TO_T: Fp3<Self> =
        Fp3::new(gold(5_944_137_876_247_729_999), gold(0), gold(0));

    // T - 1 / 2
    #[allow(clippy::unreadable_literal)]
    const TRACE_MINUS_ONE_DIV_TWO: &'static [u64] =
        &[0x80000002fffffffe, 0x80000002fffffffc, 0x7ffffffe];
}

#[cfg(test)]
mod tests {
    use static_assertions::const_assert_eq;

    use super::*;
    use crate::{
        algebra::fields::{Field256, Field64_3},
        type_info::Type,
    };

    const_assert_eq!(size_of::<Type<Field256>>(), 0);

    #[test]
    #[allow(clippy::unreadable_literal)]
    fn test_type_info_field64_3() {
        let type_info = Field64_3::type_info();
        assert_eq!(
            type_info.characteristic,
            18446744069414584321_u64.to_be_bytes().as_slice()
        );
        assert_eq!(type_info.extension_degree, 3);
    }

    #[test]
    fn test_json_goldilocks_3() {
        let field_config = Type::<Field64_3>::new();
        let json = serde_json::to_string(&field_config).unwrap();
        assert_eq!(
            json,
            "{\"characteristic\":\"ffffffff00000001\",\"extension_degree\":3}"
        );
    }
}

/// Differential tests: the hand-rolled `Field64` must agree with a generic
/// Montgomery implementation of the same field on every operation.
#[cfg(test)]
mod goldilocks_tests {
    use ark_ff::{AdditiveGroup, FftField};
    use proptest::prelude::*;

    use super::*;

    /// Montgomery-backed Goldilocks, the reference implementation.
    #[derive(MontConfig)]
    #[modulus = "18446744069414584321"]
    #[generator = "7"]
    #[small_subgroup_base = "15"]
    #[small_subgroup_power = "1"]
    struct FConfig64;
    type Field64Mont = Fp64<MontBackend<FConfig64, 1>>;

    fn both(x: u64) -> (Field64, Field64Mont) {
        (Field64::from(x), Field64Mont::from(x))
    }

    fn eq(a: Field64, b: Field64Mont) -> bool {
        a.into_bigint().0[0] == b.into_bigint().0[0]
    }

    #[test]
    fn constants_match() {
        assert!(eq(Field64::GENERATOR, Field64Mont::from(7u64)));
        let r = Field64::TWO_ADIC_ROOT_OF_UNITY;
        assert_eq!(r.pow([1u64 << 32]), Field64::ONE);
        assert_ne!(r.pow([1u64 << 31]), Field64::ONE);
        // The NTT engine relies on the auto-detected subgroup of order 3 * 2^32.
        assert_eq!(Field64::SMALL_SUBGROUP_BASE, Some(3));
        assert_eq!(Field64::SMALL_SUBGROUP_BASE_ADICITY, Some(1));
        let l = Field64::LARGE_SUBGROUP_ROOT_OF_UNITY.unwrap();
        assert_eq!(l.pow([(1u64 << 32) * 3]), Field64::ONE);
        assert_ne!(l.pow([1u64 << 32]), Field64::ONE);
        assert_eq!(gold(7), Field64::from(7u64));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4096))]

        #[test]
        fn cross_check_ops(a in 0u64..GOLDILOCKS_P, b in 0u64..GOLDILOCKS_P) {
            let (pa, ma) = both(a);
            let (pb, mb) = both(b);

            prop_assert!(eq(pa + pb, ma + mb), "add");
            prop_assert!(eq(pa - pb, ma - mb), "sub");
            prop_assert!(eq(pb - pa, mb - ma), "sub rev");
            prop_assert!(eq(pa * pb, ma * mb), "mul");
            prop_assert!(eq(pa.square(), ma.square()), "square");
            prop_assert!(eq(pa.double(), ma.double()), "double");
            prop_assert!(eq(-pa, -ma), "neg");
            prop_assert!(eq(pa.pow([b]), ma.pow([b])), "pow");

            prop_assert_eq!(pa.inverse().is_some(), ma.inverse().is_some());
            if let (Some(pi), Some(mi)) = (pa.inverse(), ma.inverse()) {
                prop_assert!(eq(pi, mi), "inverse");
                prop_assert_eq!(pa * pi, Field64::ONE);
            }
        }

        #[test]
        fn roundtrip_bigint(a in 0u64..GOLDILOCKS_P) {
            let f = Field64::from(a);
            prop_assert_eq!(f.into_bigint().0[0], a);
            prop_assert_eq!(Field64::from_bigint(f.into_bigint()).unwrap(), f);
        }
    }

    /// Values at the edges of the reduction's case analysis, which uniform
    /// sampling essentially never hits.
    #[test]
    fn boundary_values() {
        let edges = [
            0,
            1,
            2,
            GOLDILOCKS_EPSILON - 1,
            GOLDILOCKS_EPSILON,
            GOLDILOCKS_EPSILON + 1,
            1 << 63,
            GOLDILOCKS_P - 2,
            GOLDILOCKS_P - 1,
        ];
        for x in edges {
            let (px, mx) = both(x);
            assert!(eq(-px, -mx), "neg {x}");
            assert_eq!(px.inverse().is_some(), mx.inverse().is_some());
            if let (Some(pi), Some(mi)) = (px.inverse(), mx.inverse()) {
                assert!(eq(pi, mi), "inverse {x}");
            }
            for y in edges {
                let (py, my) = both(y);
                assert!(eq(px + py, mx + my), "add {x} {y}");
                assert!(eq(px - py, mx - my), "sub {x} {y}");
                assert!(eq(px * py, mx * my), "mul {x} {y}");
            }
        }
    }
}
