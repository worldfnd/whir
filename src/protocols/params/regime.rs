//! Reed–Solomon decoding regime — materialized per-round parameters and the
//! analytic helpers that depend on them.
//!
//! # References
//!
//! - Johnson proximity-gap error follows the BCHKS25 improvement
//!   (`O(n/η^5)`, m=10 at canonical slack) over BCIKS '20.
//! - Capacity bound follows STIR Conjecture 5.6: `(1 − ρ − η, d/(ρ·η))`-list
//!   decodability for RS codes.
//!
//! - `BCHKS25`: Ben-Sasson, Carmon, Haböck, Kopparty, Saraf, *On Proximity
//!   Gaps for Reed–Solomon Codes*, IACR ePrint 2025/2055 (Theorem 1.5).
//!   <https://eprint.iacr.org/2025/2055>
//! - `BCIKS '20`: Ben-Sasson, Carmon, Ishai, Kopparty, Saraf, *Proximity Gaps
//!   for Reed–Solomon Codes*, FOCS 2020, IACR ePrint 2020/654.
//!   <https://eprint.iacr.org/2020/654>
//! - `STIR`: Arnon, Chiesa, Fenzi, Yogev, *STIR: Reed–Solomon Proximity Testing
//!   with Fewer Queries*, CRYPTO 2024, IACR ePrint 2024/390 (Conjecture 5.6).
//!   <https://eprint.iacr.org/2024/390>

use std::f64::consts::LOG2_10;

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use crate::protocols::params::{
    bounds::{rate, usize_to_f64},
    spec::DecodingRegime,
};

/// Materialized decoding-regime parameters at a known rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecodingRegimeParams {
    Unique,
    Johnson { slack: OrderedFloat<f64> },
    Capacity { slack: OrderedFloat<f64> },
}

impl DecodingRegimeParams {
    /// Materialize spec policy at a known rate. Canonical slacks: `√ρ/20` for
    /// Johnson, `ρ/20` for Capacity.
    // TODO: Optimize picking η.
    pub fn from_policy(policy: DecodingRegime, rate: f64) -> Self {
        match policy {
            DecodingRegime::Unique => Self::Unique,
            DecodingRegime::Johnson => Self::johnson_canonical(rate),
            DecodingRegime::Capacity => Self::capacity_canonical(rate),
        }
    }

    /// Johnson regime with the canonical `η = √ρ / 20` slack.
    pub fn johnson_canonical(rate: f64) -> Self {
        Self::Johnson {
            slack: OrderedFloat(rate.sqrt() / 20.0),
        }
    }

    /// Capacity regime with the canonical `η = ρ / 20` slack.
    pub fn capacity_canonical(rate: f64) -> Self {
        Self::Capacity {
            slack: OrderedFloat(rate / 20.0),
        }
    }

    pub const fn is_unique(self) -> bool {
        matches!(self, Self::Unique)
    }

    /// `log₂ |Λ(C, δ)|`.
    pub fn list_size_log2(self, log_degree: f64, log_inv_rate: f64) -> f64 {
        match self {
            Self::Unique => 0.0,
            // Johnson: |Λ| = 1 / (2 η √ρ).
            Self::Johnson { slack } => -1.0 - slack.into_inner().log2() + 0.5 * log_inv_rate,
            // Capacity (STIR Conj 5.6): |Λ| = d / (ρ · η).
            Self::Capacity { slack } => log_degree + log_inv_rate - slack.into_inner().log2(),
        }
    }

    /// `|Λ(C, δ)|`.
    pub fn list_size(self, log_degree: f64, log_inv_rate: f64) -> f64 {
        2_f64.powf(self.list_size_log2(log_degree, log_inv_rate))
    }

    /// `log₂(1 − δ)`.
    pub fn one_minus_distance_log2(self, log_inv_rate: f64) -> f64 {
        let one_minus_delta = match self {
            Self::Unique => f64::midpoint(1.0, rate(log_inv_rate)),
            Self::Johnson { slack } => rate(log_inv_rate).sqrt() + slack.into_inner(),
            Self::Capacity { slack } => rate(log_inv_rate) + slack.into_inner(),
        };
        one_minus_delta.log2()
    }

    /// Bits of security delivered by `ood_samples` OOD challenges (STIR Lemma 4.5):
    /// `ood · (|F| − log d) − 2·log|Λ| + 1`. Returns `0` under `Unique`.
    pub fn ood_security_bits(
        self,
        log_degree: f64,
        log_inv_rate: f64,
        field_bits: f64,
        ood_samples: usize,
    ) -> f64 {
        if self.is_unique() {
            return 0.0;
        }
        let log_list = self.list_size_log2(log_degree, log_inv_rate);
        let ood = usize_to_f64(ood_samples);
        ood * (field_bits - log_degree) - 2.0 * log_list + 1.0
    }

    /// `log₂ ε_mca(C, δ)` for the per-step proximity-gaps error.
    ///
    /// - Unique: `(k − 1) / |F|`, log = `log k − |F|` (with `+ log ρ⁻¹`).
    /// - Johnson: BCHKS25 Theorem 1.5 at canonical `η = √ρ/20`, `m = 10`:
    ///   `ε ≈ (2·10.5⁵/3) · n · ρ^{−3/2} / |F|`.
    /// - Capacity: STIR Conj 5.6, `ε ≈ d / (η · ρ²) / |F|`.
    pub fn eps_mca_log2(self, log_inv_rate: f64, message_length: usize, field_bits: f64) -> f64 {
        let log_k = usize_to_f64(message_length).log2();
        let error = match self {
            Self::Unique => log_k + log_inv_rate,
            Self::Johnson { slack } => {
                // η lower bound from BCHKS25; below it the closed form understates
                // ε_mca. Hard assert (not debug): the variants are `pub`, so an
                // external caller can pick η, and an over-optimistic soundness
                // bound must not survive into release builds. Cold path, two
                // float ops.
                assert!(
                    slack.into_inner().log2() >= -(0.5 * log_inv_rate + LOG2_10 + 1.0) - 1e-6,
                    "Johnson slack η below BCHKS25 lower bound; ε_mca would be over-optimistic",
                );
                // BCHKS25 with m = 10: log_2(2·10.5⁵/3) + log n + 1.5·log ρ⁻¹.
                // Substituting n = k/ρ (codeword length) gives the `log_k`
                // (message length) + 2.5·log ρ⁻¹ form below.
                let bchks25_const = (2.0 * 10.5_f64.powi(5) / 3.0).log2();
                bchks25_const + log_k + 2.5 * log_inv_rate
            }
            Self::Capacity { slack } => {
                // η lower bound from STIR Conj 5.6; see the Johnson arm for why
                // this is a hard assert rather than debug_assert.
                assert!(
                    slack.into_inner().log2() >= -(log_inv_rate + LOG2_10 + 1.0) - 1e-6,
                    "Capacity slack η below STIR Conj 5.6 lower bound; ε_mca would be over-optimistic",
                );
                // d / (η · ρ²) at canonical η = ρ/20: log d + log 20 + 3·log ρ⁻¹.
                log_k + 3.0 * log_inv_rate + LOG2_10 + 1.0
            }
        };
        error - field_bits
    }
}

impl DecodingRegime {
    /// `|Λ|` at canonical slack, before an IRS config exists.
    pub fn list_size_estimate(self, log_degree: f64, log_inv_rate: f64) -> f64 {
        DecodingRegimeParams::from_policy(self, rate(log_inv_rate))
            .list_size(log_degree, log_inv_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::params::test_utils::assert_close;

    const TIGHT_EPS: f64 = 1e-12;

    fn johnson(slack: f64) -> DecodingRegimeParams {
        DecodingRegimeParams::Johnson {
            slack: OrderedFloat(slack),
        }
    }

    fn capacity(slack: f64) -> DecodingRegimeParams {
        DecodingRegimeParams::Capacity {
            slack: OrderedFloat(slack),
        }
    }

    /// Johnson list size: `|Λ| = 1 / (2η√ρ)`, log₂ form.
    #[test]
    fn list_size_log2_johnson_formula() {
        let got = johnson(0.1).list_size_log2(/* log_degree */ 4.0, 2.0);
        let expected = -1.0 - 0.1_f64.log2() + 0.5 * 2.0;
        assert_close(got, expected);
    }

    /// Capacity list size: `|Λ| = d / (ρ · η)`, log₂ form.
    #[test]
    fn list_size_log2_capacity_formula() {
        let got = capacity(0.125).list_size_log2(4.0, 2.0);
        let expected = 4.0 + 2.0 - 0.125_f64.log2();
        assert_close(got, expected);
    }

    /// Unique-decoding regime gives `|Λ| = 1`, i.e. log = 0.
    #[test]
    fn list_size_log2_unique_decoding_is_zero() {
        assert_close(DecodingRegimeParams::Unique.list_size_log2(4.0, 2.0), 0.0);
    }

    /// `η = √ρ/20` ⇒ `|Λ| = 10/ρ` ⇒ `list_size_estimate(_, b) = 10 · 2^b`.
    #[test]
    fn johnson_list_size_closed_form() {
        for b in [1.0, 2.0, 3.0, 5.0] {
            let got = DecodingRegime::Johnson.list_size_estimate(/* log_degree */ 4.0, b);
            let expected = 10.0 * 2_f64.powf(b);
            assert!(
                (got - expected).abs() / expected < TIGHT_EPS,
                "log_inv_rate={b}: got {got} vs {expected}",
            );
        }
    }

    /// `η = ρ/20` ⇒ `|Λ| = 20 · d / ρ²`.
    #[test]
    fn capacity_list_size_closed_form() {
        for (log_d, b) in [(4.0, 1.0), (6.0, 2.0), (8.0, 3.0)] {
            let got = DecodingRegime::Capacity.list_size_estimate(log_d, b);
            let expected = 20.0 * 2_f64.powf(log_d) * 2_f64.powf(2.0 * b);
            assert!(
                (got - expected).abs() / expected < TIGHT_EPS,
                "log_d={log_d}, log_inv_rate={b}: got {got} vs {expected}",
            );
        }
    }

    #[test]
    fn johnson_list_size_matches_config_list_size() {
        use crate::{
            algebra::{embedding::Identity, fields::Field64},
            hash,
            protocols::irs_commit::{Config, IrsMode, IrsParams},
        };
        const PLACEHOLDER_SECURITY_TARGET_BITS: f64 = 80.0;
        const PLACEHOLDER_NUM_VECTORS: usize = 2;
        const PLACEHOLDER_VECTOR_SIZE: usize = 8;
        const PLACEHOLDER_INTERLEAVING_DEPTH: usize = 1;
        const LOG_INV_RATE: u32 = 2;

        let config: Config<Identity<Field64>> = Config::new(IrsParams {
            security_target: PLACEHOLDER_SECURITY_TARGET_BITS,
            decoding_regime: DecodingRegime::Johnson,
            hash_id: hash::BLAKE3,
            num_vectors: PLACEHOLDER_NUM_VECTORS,
            vector_size: PLACEHOLDER_VECTOR_SIZE,
            interleaving_depth: PLACEHOLDER_INTERLEAVING_DEPTH,
            rate: 2_f64.powf(-f64::from(LOG_INV_RATE)),
            mode: IrsMode::Standard,
        });
        let log_degree = (config.masked_message_length() as f64).log2();
        let got = DecodingRegime::Johnson.list_size_estimate(log_degree, f64::from(LOG_INV_RATE));
        let expected = config.list_size();
        assert!(
            (got - expected).abs() / expected < TIGHT_EPS,
            "regime helper ({got}) vs Config::list_size ({expected})",
        );
    }

    /// `1 − δ` in unique-decoding mode: midpoint of 1 and ρ.
    #[test]
    fn one_minus_distance_log2_unique() {
        let log_inv_rate = 2.0;
        let got = DecodingRegimeParams::Unique.one_minus_distance_log2(log_inv_rate);
        let rho = 2_f64.powf(-log_inv_rate);
        let expected = f64::midpoint(1.0, rho).log2();
        assert_close(got, expected);
    }

    /// `1 − δ` in Johnson regime: `√ρ + η`.
    #[test]
    fn one_minus_distance_log2_johnson() {
        let log_inv_rate = 2.0;
        let eta = 0.1;
        let got = johnson(eta).one_minus_distance_log2(log_inv_rate);
        let rho = 2_f64.powf(-log_inv_rate);
        let expected = (rho.sqrt() + eta).log2();
        assert_close(got, expected);
    }

    /// `1 − δ` in Capacity regime: `ρ + η`.
    #[test]
    fn one_minus_distance_log2_capacity() {
        let log_inv_rate = 2.0;
        let eta = 0.05;
        let got = capacity(eta).one_minus_distance_log2(log_inv_rate);
        let rho = 2_f64.powf(-log_inv_rate);
        let expected = (rho + eta).log2();
        assert_close(got, expected);
    }

    /// `ood_security_bits = t · (|F| − log d) − 2·log|Λ| + 1`. Returns 0
    /// under Unique.
    #[test]
    fn ood_security_bits_formula() {
        const LOG_DEGREE: f64 = 6.0;
        const LOG_INV_RATE: f64 = 2.0;
        const FIELD_BITS: f64 = 64.0;
        const OOD: usize = 3;

        // Unique → 0 (no soundness from OOD).
        let unique = DecodingRegimeParams::Unique.ood_security_bits(
            LOG_DEGREE,
            LOG_INV_RATE,
            FIELD_BITS,
            OOD,
        );
        assert_close(unique, 0.0);

        let slack = 2_f64.powf(-LOG_INV_RATE).sqrt() / 20.0;
        let got = johnson(slack).ood_security_bits(LOG_DEGREE, LOG_INV_RATE, FIELD_BITS, OOD);
        let log_list = johnson(slack).list_size_log2(LOG_DEGREE, LOG_INV_RATE);
        let expected = (OOD as f64) * (FIELD_BITS - LOG_DEGREE) - 2.0 * log_list + 1.0;
        assert_close(got, expected);
    }

    const MCA_MESSAGE_LENGTH: usize = 16;
    const MCA_LOG_INV_RATE: f64 = 2.0;
    const MCA_FIELD_BITS: f64 = 64.0;

    /// MCA error, unique-decoding branch: `log k + log_inv_rate − field_bits`.
    #[test]
    fn eps_mca_log2_unique_decoding_formula() {
        let got = DecodingRegimeParams::Unique.eps_mca_log2(
            MCA_LOG_INV_RATE,
            MCA_MESSAGE_LENGTH,
            MCA_FIELD_BITS,
        );
        let expected = (MCA_MESSAGE_LENGTH as f64).log2() + MCA_LOG_INV_RATE - MCA_FIELD_BITS;
        assert_close(got, expected);
    }

    /// MCA error, Johnson (BCHKS25): `log₂(2·10.5⁵/3) + log k + 2.5·log_inv_rate − field_bits`.
    #[test]
    fn eps_mca_log2_johnson_formula() {
        let canonical_slack = 2_f64.powf(-MCA_LOG_INV_RATE).sqrt() / 20.0;

        let got = johnson(canonical_slack).eps_mca_log2(
            MCA_LOG_INV_RATE,
            MCA_MESSAGE_LENGTH,
            MCA_FIELD_BITS,
        );
        let bchks25_const = (2.0 * 10.5_f64.powi(5) / 3.0).log2();
        let expected = bchks25_const + (MCA_MESSAGE_LENGTH as f64).log2() + 2.5 * MCA_LOG_INV_RATE
            - MCA_FIELD_BITS;
        assert_close(got, expected);
    }

    /// MCA error, Capacity (STIR Conj 5.6 at canonical η):
    /// `log k + 3·log_inv_rate + log₂10 + 1 − field_bits`.
    #[test]
    fn eps_mca_log2_capacity_formula() {
        let canonical_slack = 2_f64.powf(-MCA_LOG_INV_RATE) / 20.0;

        let got = capacity(canonical_slack).eps_mca_log2(
            MCA_LOG_INV_RATE,
            MCA_MESSAGE_LENGTH,
            MCA_FIELD_BITS,
        );
        let expected = (MCA_MESSAGE_LENGTH as f64).log2() + 3.0 * MCA_LOG_INV_RATE + LOG2_10 + 1.0
            - MCA_FIELD_BITS;
        assert_close(got, expected);
    }

    // --- Theorem anchors -----------------------------------------------------
    //
    // The `*_formula` tests above re-evaluate the same expression as the
    // implementation, so a wrong constant or exponent updates both sides and
    // the test stays green. The tests below instead assert against literals
    // hand-derived from the theorem statements (independently of the collapsed
    // implementation form), so a transcription error in `eps_mca_log2` is
    // caught. Fixed inputs: k = 16 (log k = 4), ρ = 1/4 (log ρ⁻¹ = 2), |F| = 2⁶⁴.

    /// Unique: ε ≈ k/|F| · ρ⁻¹ ⇒ log₂ε = log₂16 + 2 − 64 = 4 + 2 − 64 = −58.
    #[test]
    fn eps_mca_log2_unique_theorem_anchor() {
        let got = DecodingRegimeParams::Unique.eps_mca_log2(
            MCA_LOG_INV_RATE,
            MCA_MESSAGE_LENGTH,
            MCA_FIELD_BITS,
        );
        assert_close(got, -58.0);
    }

    /// Johnson (BCHKS25 Theorem 1.5, η = √ρ/20, m = 10):
    /// `ε = (2·10.5⁵/3) · n · ρ^{−3/2} / |F|` with codeword length n = k/ρ = 64.
    /// log₂ε = log₂(2·10.5⁵/3) + log₂64 + log₂(ρ^{−3/2}) − 64
    ///       = 16.376624613… + 6 + 3 − 64 = −38.623375386…
    #[test]
    fn eps_mca_log2_johnson_theorem_anchor() {
        let canonical_slack = 2_f64.powf(-MCA_LOG_INV_RATE).sqrt() / 20.0;
        let got = johnson(canonical_slack).eps_mca_log2(
            MCA_LOG_INV_RATE,
            MCA_MESSAGE_LENGTH,
            MCA_FIELD_BITS,
        );
        assert_close(got, -38.623_375_386_827_36);
    }

    /// Capacity (STIR Conj 5.6, η = ρ/20): `ε = d / (η · ρ²) / |F|` with d = k = 16,
    /// η = ρ/20 = 1/80, ρ² = 1/16.
    /// log₂ε = log₂(16 / ((1/80)·(1/16))) − 64 = log₂(16·1280) − 64
    ///       = log₂20480 − 64 = 14.321928094… − 64 = −49.678071905…
    #[test]
    fn eps_mca_log2_capacity_theorem_anchor() {
        let canonical_slack = 2_f64.powf(-MCA_LOG_INV_RATE) / 20.0;
        let got = capacity(canonical_slack).eps_mca_log2(
            MCA_LOG_INV_RATE,
            MCA_MESSAGE_LENGTH,
            MCA_FIELD_BITS,
        );
        assert_close(got, -49.678_071_905_112_64);
    }

    /// `from_policy` dispatches to the canonical constructor for each regime.
    #[test]
    fn from_policy_matches_canonical() {
        assert_eq!(
            DecodingRegimeParams::from_policy(DecodingRegime::Unique, 0.25),
            DecodingRegimeParams::Unique,
        );
        assert_eq!(
            DecodingRegimeParams::from_policy(DecodingRegime::Johnson, 0.25),
            DecodingRegimeParams::johnson_canonical(0.25),
        );
        assert_eq!(
            DecodingRegimeParams::from_policy(DecodingRegime::Capacity, 0.25),
            DecodingRegimeParams::capacity_canonical(0.25),
        );
    }
}
