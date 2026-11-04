//! Regime-agnostic analytic primitives shared across the params solvers.
//!
//! Regime-specific math (Unique / Johnson / Capacity branches) lives on
//! [`super::regime::DecodingRegimeParams`].

use std::f64::consts::LN_2;

use crate::{bits::Bits, protocols::params::error::ZookSecurityError};

/// `ρ = 2^-log_inv_rate`. Centralized so the rate formula lives in one place.
pub(super) fn rate(log_inv_rate: f64) -> f64 {
    2_f64.powf(-log_inv_rate)
}

/// Lossy `usize → f64` for analytic-error formulas. Named so individual call
/// sites can stay terse and intent-tagged.
pub(super) const fn usize_to_f64(x: usize) -> f64 {
    x as f64
}

/// log2 of the per-OOD-sample Schwartz–Zippel error: `(k-1)/|F|`.
pub fn ood_per_sample_log2(message_length: usize, field_bits: f64) -> f64 {
    ((message_length - 1) as f64).log2() - field_bits
}

/// `log2(Q - n) = log2(Q) + log2(1 - n/Q)`, where `Q = 2^field_bits`
/// is the field size and `n = domain_size`.
///
/// ProveKit Prover Whitepaper §6.4, “Soundness”, theorem “ZO0K soundness under
/// mask binding”, query phase: the DEEP denominator is `|F \ D| = Q - n`.
pub(super) fn out_of_domain_bits(
    field_bits: f64,
    domain_size: usize,
) -> Result<f64, ZookSecurityError> {
    let fraction = 2.0_f64.powf(usize_to_f64(domain_size).log2() - field_bits);
    if fraction >= 1.0 {
        return Err(ZookSecurityError::EmptyOutOfDomain);
    }
    Ok(field_bits + (-fraction).ln_1p() / LN_2)
}

/// `log2(2^a + 2^b)` for log2 error bounds `a` and `b`.
/// Adding probabilities of equal size costs one bit of security.
pub(super) fn log2_sum(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let larger = a.max(b);
    larger + 2.0_f64.powf(a.min(b) - larger).ln_1p() / LN_2
}

/// Security bits `max(0, -log2(epsilon))` from a log2 error bound.
pub(super) fn error_bits(log_error: f64) -> Bits {
    Bits::new((-log_error).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::params::test_utils::assert_close;

    /// OOD per-sample Schwartz–Zippel: `log₂((k−1) / |F|) = log₂(k−1) − field_bits`.
    #[test]
    fn ood_per_sample_log2_formula() {
        // `k = 129` so `k − 1 = 128 = 2^7` for exact `log2`.
        const K: usize = 129;
        const FIELD_BITS: f64 = 64.0;

        let got = ood_per_sample_log2(K, FIELD_BITS);
        let expected = ((K - 1) as f64).log2() - FIELD_BITS;
        assert_close(got, expected);
        // (k−1)/|F| < 1 for sane parameters ⇒ log is negative.
        assert!(got < 0.0);
    }

    #[test]
    fn log_arithmetic_handles_tiny_errors_and_small_challenge_domains() {
        let zero_error = log2_sum(f64::NEG_INFINITY, f64::NEG_INFINITY);
        assert!(zero_error.is_infinite() && zero_error.is_sign_negative());
        assert_close(log2_sum(f64::NEG_INFINITY, -80.0), -80.0);
        assert!((log2_sum(-2000.0, -2000.0) + 1999.0).abs() < 1e-12);
        assert!((out_of_domain_bits(8.0, 1).unwrap() - 255.0_f64.log2()).abs() < 1e-12);
        assert!((out_of_domain_bits(8.0, 128).unwrap() - 7.0).abs() < 1e-12);
        assert_eq!(
            out_of_domain_bits(8.0, 256),
            Err(ZookSecurityError::EmptyOutOfDomain)
        );
    }
}
