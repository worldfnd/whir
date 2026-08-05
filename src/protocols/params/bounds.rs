//! Regime-agnostic analytic primitives shared across the params solvers.
//!
//! Regime-specific math (Unique / Johnson / Capacity branches) lives on
//! [`super::regime::DecodingRegimeParams`].

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
}
