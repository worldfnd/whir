//! Mask-proximity (Construction 7.2) builder + Lemma 7.4 γ-combination bound.
//! ZK-only.

use ark_ff::Field;

use crate::{
    algebra::{embedding::Identity, fields::FieldWithSize},
    bits::Bits,
    protocols::{
        irs_commit::Config as IrsConfig,
        mask_proximity::Config as MaskProximityConfig,
        params::{
            bounds::usize_to_f64,
            error::{grind_to_at, DeriveError, Pow},
            solved::Solved,
            spec::SecuritySpec,
        },
    },
};

/// `c_zk.num_vectors` must equal `2 * num_masks` (originals + fresh).
pub fn solve<F: Field>(
    spec: &SecuritySpec,
    c_zk: IrsConfig<Identity<F>>,
    num_masks: usize,
    round_index: usize,
) -> Result<Solved<MaskProximityConfig<F>>, DeriveError> {
    let analytic = analytic_error_bits(&c_zk, num_masks);
    let pow = grind_to_at(
        spec,
        analytic,
        Pow::RoundMaskProximity { index: round_index },
    )?;
    Ok(Solved::new(
        MaskProximityConfig::new(c_zk, num_masks, pow),
        analytic,
    ))
}

/// γ-combination soundness (Lemma 7.4):
/// `log|F| − log(num_masks · (deg − 1))`, with `deg = c_zk.masked_message_length()`.
pub fn analytic_error_bits<F: Field>(c_zk: &IrsConfig<Identity<F>>, num_masks: usize) -> Bits {
    let field_bits = F::field_size_bits();
    let deg = c_zk.masked_message_length();
    if deg <= 1 || num_masks == 0 {
        return Bits::new(field_bits.max(0.0));
    }
    let log_combined = usize_to_f64(num_masks * deg.saturating_sub(1)).log2();
    Bits::new((field_bits - log_combined).max(0.0))
}

impl<F: Field> MaskProximityConfig<F> {
    /// Analytic soundness bits (excluding PoW) for the Lemma 7.4 γ-combination.
    pub fn analytic_bits(&self) -> Bits {
        analytic_error_bits(self.c_zk_commit(), self.num_masks())
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::{
        algebra::fields::Field64,
        hash,
        protocols::{
            irs_commit::{IrsMode, IrsParams},
            params::{
                spec::{DecodingRegime, Mode},
                test_utils::{
                    arb_zk_spec, assert_close, assert_pow_closes_gap, build_test_c_zk,
                    deterministic_spec, TEST_TARGET_RANGE,
                },
            },
        },
    };

    const FIXTURE_L_ZK: usize = 8;
    const FIXTURE_NUM_MASKS: usize = 3;
    const FIXTURE_LOG_INV_RATE: u32 = 1;

    #[test]
    fn analytic_error_formula() {
        let spec = deterministic_spec(Mode::ZeroKnowledge);
        let c_zk = build_test_c_zk(&spec, FIXTURE_L_ZK, FIXTURE_LOG_INV_RATE, FIXTURE_NUM_MASKS);

        let got = f64::from(analytic_error_bits(&c_zk, FIXTURE_NUM_MASKS));

        let field_bits = <Field64 as FieldWithSize>::field_size_bits();
        let deg = c_zk.masked_message_length();
        let log_combined = ((FIXTURE_NUM_MASKS * (deg - 1)) as f64).log2();
        let expected = (field_bits - log_combined).max(0.0);

        assert_close(got, expected);
    }

    #[test]
    fn analytic_error_saturates_when_no_masks() {
        let spec = deterministic_spec(Mode::ZeroKnowledge);
        let c_zk = build_test_c_zk(&spec, 2, 1, 1);
        let bits = f64::from(analytic_error_bits(&c_zk, 0));
        let field_bits = <Field64 as FieldWithSize>::field_size_bits();
        assert_close(bits, field_bits.max(0.0));
    }

    proptest! {
        #[test]
        fn solve_assembles(
            spec in arb_zk_spec(TEST_TARGET_RANGE),
            log_inv_rate in 1u32..=3,
            num_masks in 1usize..=8,
            l_zk_log in 1u32..=5,
        ) {
            let c_zk = build_test_c_zk(&spec, 1usize << l_zk_log, log_inv_rate, num_masks);
            let config = solve(&spec, c_zk, num_masks, 0).unwrap();
            prop_assert_eq!(config.num_masks(), num_masks);
            prop_assert_eq!(config.c_zk_commit().num_vectors(), 2 * num_masks);
            prop_assert_eq!(config.c_zk_commit().interleaving_depth(), 1);
        }

        #[test]
        fn pow_closes_gap_to_target(
            spec in arb_zk_spec(TEST_TARGET_RANGE),
            log_inv_rate in 1u32..=3,
            num_masks in 1usize..=8,
            l_zk_log in 1u32..=5,
        ) {
            let c_zk = build_test_c_zk(&spec, 1usize << l_zk_log, log_inv_rate, num_masks);
            let analytic = analytic_error_bits(&c_zk, num_masks);
            let config = solve(&spec, c_zk, num_masks, 0).unwrap();
            assert_pow_closes_gap(&spec, analytic, &config.pow());
        }
    }

    #[test]
    #[should_panic(expected = "c_zk.num_vectors must be 2 * num_masks")]
    fn solve_rejects_mismatched_num_vectors() {
        let spec = deterministic_spec(Mode::ZeroKnowledge);
        let c_zk = build_test_c_zk(&spec, 2, 1, 2);
        let _ = solve(&spec, c_zk, 3, 0);
    }

    #[test]
    #[should_panic(expected = "interleaving_depth = 1")]
    fn solve_rejects_non_unit_interleaving() {
        const SECURITY_TARGET_BITS: f64 = 80.0;
        const NUM_VECTORS: usize = 2;
        const VECTOR_SIZE: usize = 8;
        const NON_UNIT_INTERLEAVING_DEPTH: usize = 2;
        const RATE: f64 = 0.5;
        const NUM_MASKS: usize = 1;

        let spec = deterministic_spec(Mode::ZeroKnowledge);
        let c_zk = IrsConfig::<Identity<Field64>>::new(IrsParams {
            security_target: SECURITY_TARGET_BITS,
            decoding_regime: DecodingRegime::Johnson,
            hash_id: hash::BLAKE3,
            num_vectors: NUM_VECTORS,
            vector_size: VECTOR_SIZE,
            interleaving_depth: NON_UNIT_INTERLEAVING_DEPTH,
            rate: RATE,
            mode: IrsMode::Standard,
        });
        let _ = solve(&spec, c_zk, NUM_MASKS, 0);
    }
}
