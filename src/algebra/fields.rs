use ark_ff::{
    AdditiveGroup, BigInt, Field, Fp128, Fp192, Fp2, Fp256, Fp2Config, Fp3, Fp3Config, Fp64,
    MontBackend, MontConfig, MontFp, PrimeField,
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

pub struct FConfig64;
pub type Field64 = Fp64<MontBackend<FConfig64, 1>>;

/// Goldilocks modulus `p = 2^64 - 2^32 + 1`.
const GOLDILOCKS: u64 = 0xffff_ffff_0000_0001;

/// Montgomery reduction `x * 2^-64 mod p` for `x < p * 2^64`.
///
/// `p^-1 mod 2^64 = 1 + 2^32`, so the Montgomery quotient needs only a shift and an add.
#[inline]
#[allow(clippy::cast_possible_truncation)] // Splitting a 128-bit product into its 64-bit halves.
fn goldilocks_redc(x: u128) -> u64 {
    let (lo, hi) = (x as u64, (x >> 64) as u64);
    let m = lo.wrapping_add(lo << 32);
    let mp_hi = ((u128::from(m) * u128::from(GOLDILOCKS)) >> 64) as u64;
    let (result, borrow) = hi.overflowing_sub(mp_hi);
    if borrow {
        result.wrapping_add(GOLDILOCKS)
    } else {
        result
    }
}

// Same constants as `#[derive(MontConfig)]` with modulus 18446744069414584321, generator 7,
// small subgroup base 15 and power 1; only the arithmetic is specialised to Goldilocks.
impl MontConfig<1> for FConfig64 {
    const MODULUS: BigInt<1> = BigInt([GOLDILOCKS]);
    const GENERATOR: Field64 = MontFp!("7");
    const TWO_ADIC_ROOT_OF_UNITY: Field64 = MontFp!("1753635133440165772");
    const SMALL_SUBGROUP_BASE: Option<u32> = Some(15);
    const SMALL_SUBGROUP_BASE_ADICITY: Option<u32> = Some(1);
    const LARGE_SUBGROUP_ROOT_OF_UNITY: Option<Field64> = Some(MontFp!("1226330035855801067"));

    #[inline]
    fn add_assign(a: &mut Field64, b: &Field64) {
        // Inputs are canonical, so a + b < 2p; 2^64 = 2^32 - 1 (mod p).
        let (sum, carry) = (a.0).0[0].overflowing_add((b.0).0[0]);
        (a.0).0[0] = if carry {
            sum.wrapping_add(0xffff_ffff)
        } else if sum >= GOLDILOCKS {
            sum - GOLDILOCKS
        } else {
            sum
        };
    }

    #[inline]
    fn sub_assign(a: &mut Field64, b: &Field64) {
        let (difference, borrow) = (a.0).0[0].overflowing_sub((b.0).0[0]);
        (a.0).0[0] = if borrow {
            difference.wrapping_sub(0xffff_ffff)
        } else {
            difference
        };
    }

    #[inline]
    fn double_in_place(a: &mut Field64) {
        let b = *a;
        Self::add_assign(a, &b);
    }

    #[inline]
    fn mul_assign(a: &mut Field64, b: &Field64) {
        (a.0).0[0] = goldilocks_redc(u128::from((a.0).0[0]) * u128::from((b.0).0[0]));
    }

    #[inline]
    fn square_in_place(a: &mut Field64) {
        let x = u128::from((a.0).0[0]);
        (a.0).0[0] = goldilocks_redc(x * x);
    }

    fn sum_of_products<const M: usize>(a: &[Field64; M], b: &[Field64; M]) -> Field64 {
        let mut acc = Field64::ZERO;
        for (x, y) in a.iter().zip(b) {
            acc += *x * y;
        }
        acc
    }
}

pub type Field64_2 = Fp2<F2Config64>;
pub struct F2Config64;
impl Fp2Config for F2Config64 {
    type Fp = Field64;

    const NONRESIDUE: Self::Fp = MontFp!("7");

    const FROBENIUS_COEFF_FP2_C1: &'static [Self::Fp] = &[
        // Fq(7)**(((q^0) - 1) / 2)
        MontFp!("1"),
        // Fq(7)**(((q^1) - 1) / 2)
        MontFp!("18446744069414584320"),
    ];
}

pub type Field64_3 = Fp3<F3Config64>;
pub struct F3Config64;

impl Fp3Config for F3Config64 {
    type Fp = Field64;

    const NONRESIDUE: Self::Fp = MontFp!("2");

    const FROBENIUS_COEFF_FP3_C1: &'static [Self::Fp] = &[
        MontFp!("1"),
        // Fq(2)^(((q^1) - 1) / 3)
        MontFp!("4294967295"),
        // Fq(2)^(((q^2) - 1) / 3)
        MontFp!("18446744065119617025"),
    ];

    const FROBENIUS_COEFF_FP3_C2: &'static [Self::Fp] = &[
        MontFp!("1"),
        // Fq(2)^(((2q^1) - 2) / 3)
        MontFp!("18446744065119617025"),
        // Fq(2)^(((2q^2) - 2) / 3)
        MontFp!("4294967295"),
    ];

    // (q^3 - 1) = 2^32 * T where T = 1461501636310055817916238417282618014431694553085
    const TWO_ADICITY: u32 = 32;

    // 11^T
    const QUADRATIC_NONRESIDUE_TO_T: Fp3<Self> =
        Fp3::new(MontFp!("5944137876247729999"), MontFp!("0"), MontFp!("0"));

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
    fn goldilocks_arithmetic_matches_reference() {
        use ark_std::{
            rand::{rngs::StdRng, Rng, SeedableRng},
            UniformRand,
        };
        const P: u64 = 0xffff_ffff_0000_0001;
        let reduce = |x: u128| u64::try_from(x % u128::from(P)).unwrap();
        let mut rng = StdRng::seed_from_u64(77);
        let edges = [
            0,
            1,
            2,
            0xffff_ffff,
            0x1_0000_0000,
            0xffff_ffff_0000_0000 - 1,
            0xffff_ffff_0000_0000,
            P / 2,
            P - 2,
            P - 1,
        ];
        let mut values: Vec<Field64> = edges.iter().map(|&v| Field64::from(v)).collect();
        values.extend((0..2000).map(|_| Field64::rand(&mut rng)));
        values.extend((0..2000).map(|_| Field64::from(rng.gen::<u64>() % P)));
        for a in &values[..400] {
            let ca = u128::from(a.into_bigint().0[0]);
            let mut square = *a;
            square.square_in_place();
            assert_eq!(square.into_bigint().0[0], reduce(ca * ca));
            assert_eq!(a.double().into_bigint().0[0], reduce(2 * ca));
            for b in &values {
                let cb = u128::from(b.into_bigint().0[0]);
                assert_eq!((*a * b).into_bigint().0[0], reduce(ca * cb));
                assert_eq!((*a + b).into_bigint().0[0], reduce(ca + cb));
                assert_eq!((*a - b).into_bigint().0[0], reduce(ca + u128::from(P) - cb));
            }
        }
        let root = <FConfig64 as MontConfig<1>>::TWO_ADIC_ROOT_OF_UNITY;
        assert_eq!(root.pow([1u64 << 32]), Field64::ONE);
        assert_ne!(root.pow([1u64 << 31]), Field64::ONE);
        let x = Field64::rand(&mut rng);
        assert_eq!(x * x.inverse().unwrap(), Field64::ONE);
        assert_eq!(
            Field64::sum_of_products(&[x, root], &[root, x]),
            x * root + root * x
        );
    }

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
