use ark_ff::{define_field, Field, Fp2, Fp2Config, Fp3, Fp3Config, PrimeField};
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

define_field!(
    name = Field256,
    modulus = "21888242871839275222246405745257275088548364400416034343698204186575808495617",
    generator = "5"
);

define_field!(
    name = Field192,
    modulus = "3801539170989320091464968600173246866371124347557388484609",
    generator = "3"
);

define_field!(
    name = Field128,
    modulus = "340282366920938463463374557953744961537",
    generator = "3"
);

define_field!(
    name = Field64,
    modulus = "18446744069414584321",
    generator = "7"
);

pub type Field64_2 = Fp2<F2Config64>;
pub struct F2Config64;
impl Fp2Config for F2Config64 {
    type Fp = Field64;

    const NONRESIDUE: Self::Fp = Field64Config::from_u128(7);

    const FROBENIUS_COEFF_FP2_C1: &'static [Self::Fp] = &[
        // Fq(7)**(((q^0) - 1) / 2)
        Field64Config::from_u128(1),
        // Fq(7)**(((q^1) - 1) / 2)
        Field64Config::from_u128(18_446_744_069_414_584_320),
    ];
}

pub type Field64_3 = Fp3<F3Config64>;
pub struct F3Config64;

impl Fp3Config for F3Config64 {
    type Fp = Field64;

    const NONRESIDUE: Self::Fp = Field64Config::from_u128(2);

    const FROBENIUS_COEFF_FP3_C1: &'static [Self::Fp] = &[
        Field64Config::from_u128(1),
        // Fq(2)^(((q^1) - 1) / 3)
        Field64Config::from_u128(4_294_967_295),
        // Fq(2)^(((q^2) - 1) / 3)
        Field64Config::from_u128(18_446_744_065_119_617_025),
    ];

    const FROBENIUS_COEFF_FP3_C2: &'static [Self::Fp] = &[
        Field64Config::from_u128(1),
        // Fq(2)^(((2q^1) - 2) / 3)
        Field64Config::from_u128(18_446_744_065_119_617_025),
        // Fq(2)^(((2q^2) - 2) / 3)
        Field64Config::from_u128(4_294_967_295),
    ];

    // (q^3 - 1) = 2^32 * T where T = 1461501636310055817916238417282618014431694553085
    const TWO_ADICITY: u32 = 32;

    // 11^T
    const QUADRATIC_NONRESIDUE_TO_T: Fp3<Self> = Fp3::new(
        Field64Config::from_u128(5_944_137_876_247_729_999),
        Field64Config::from_u128(0),
        Field64Config::from_u128(0),
    );

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

    #[test]
    fn test_fp2_encoding_roundtrip() {
        use spongefish::{Encoding, NargDeserialize};

        // Check Fp2 encoding→NargDeserialize roundtrip
        let val = Field64_2::new(Field64::from(42u64), Field64::from(7u64));
        let encoded = val.encode();
        let bytes = encoded.as_ref();

        let mut slice: &[u8] = bytes;
        let decoded = Field64_2::deserialize_from_narg(&mut slice).expect("NargDeserialize failed");
        assert!(slice.is_empty(), "Not all bytes consumed");
        assert_eq!(
            val, decoded,
            "Fp2 roundtrip failed: original={val:?}, decoded={decoded:?}",
        );
    }
}
