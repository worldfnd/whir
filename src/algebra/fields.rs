use ark_ff::{
    define_field, Field, Fp128, Fp192, Fp2, Fp256, Fp2Config, Fp3, Fp3Config, MontBackend,
    MontConfig, PrimeField,
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

define_field!(
    name = Field64,
    modulus = "18446744069414584321",
    generator = "7"
);

const fn gold(x: u64) -> Field64 {
    Field64Config::from_u128(x as u128)
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

/// Differential tests: the `SmallFp`-backed `Field64` must agree with a
/// generic Montgomery implementation of the same field on every operation.
#[cfg(test)]
mod goldilocks_tests {
    use ark_ff::{AdditiveGroup, FftField, Fp64, MontBackend, PrimeField};
    use proptest::prelude::*;

    use super::*;

    const GOLDILOCKS_P: u64 = 0xFFFF_FFFF_0000_0001;

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
}
