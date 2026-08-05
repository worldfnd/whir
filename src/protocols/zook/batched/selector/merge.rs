//! Selector-merge: reduce `t` active blocks sharing one IRS commit config to a single virtual block.
//!
//! `t == 1` is a no-op (no transcript writes), preserving byte-identical proofs with the single-track path.

use ark_ff::Field;
use ark_std::rand::{CryptoRng, RngCore};
#[cfg(feature = "tracing")]
use tracing::instrument;

use super::sumcheck;
use crate::{
    algebra::{dot, eq_weights},
    bits::Bits,
    protocols::proof_of_work,
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverState, VerificationResult,
        VerifierMessage, VerifierState,
    },
};

/// Selector-merge configuration: a [`sumcheck::Config`] plus active-block count `t`.
#[must_use]
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct Config<F: Field> {
    selector: sumcheck::Config<F>,
    t: usize,
}

impl<F: Field> Config<F> {
    /// Build a selector-merge config for `t ≥ 1` active blocks of per-block message length `L`.
    /// `t = 1` is a no-op merge.
    pub fn new(t: usize, message_length: usize, round_pow: proof_of_work::Config) -> Self {
        assert!(t >= 1);
        let selector = sumcheck::Config::new(t, message_length, round_pow);
        Self { selector, t }
    }

    /// Active block count `t`.
    pub const fn t(&self) -> usize {
        self.t
    }

    /// Per-block message length `L`.
    pub const fn message_length(&self) -> usize {
        self.selector.message_length()
    }

    /// Number of selector variables `ℓ = ⌈log₂ t⌉`. Zero when `t = 1`.
    pub const fn selector_dim(&self) -> usize {
        self.selector.selector_dim()
    }

    /// Selector-table size `H = 2^ℓ`.
    pub const fn padded_size(&self) -> usize {
        self.selector.padded_size()
    }

    pub const fn selector(&self) -> &sumcheck::Config<F> {
        &self.selector
    }

    /// Selector-sumcheck soundness in bits.
    pub fn analytic_error_bits(&self, field_bits: f64) -> Bits {
        self.selector.analytic_error_bits(field_bits)
    }
}

/// Prover-side output of the merge.
#[must_use]
#[derive(Debug, Clone)]
pub struct Witness<F: Field> {
    pub merged_message: Vec<F>,
    pub merged_covector: Vec<F>,
    pub sum: F,
    pub delta: Vec<F>,
    pub eta: F,
    pub theta: Vec<F>,
}

/// Verifier-side output of the merge: post-merge sum plus public selector challenges.
#[must_use]
#[derive(Debug, Clone)]
pub struct Opening<F: Field> {
    pub sum: F,
    pub delta: Vec<F>,
    pub eta: F,
    pub theta: Vec<F>,
}

impl<F: Field> Config<F> {
    /// Selector-merge prover.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn prove<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        messages: &[Vec<F>],
        covectors: &[Vec<F>],
        sums: &[F],
    ) -> Witness<F>
    where
        H: DuplexSpongeInterface,
        R: CryptoRng + RngCore,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
    {
        assert_eq!(messages.len(), self.t);
        assert_eq!(covectors.len(), self.t);
        assert_eq!(sums.len(), self.t);
        let l = self.message_length();
        for (b, m) in messages.iter().enumerate() {
            assert_eq!(m.len(), l, "messages[{b}] length");
        }
        for (b, c) in covectors.iter().enumerate() {
            assert_eq!(c.len(), l, "covectors[{b}] length");
        }
        debug_assert!(
            sums.iter()
                .zip(messages)
                .zip(covectors)
                .all(|((&s, m), c)| s == dot(m, c)),
            "selector_merge::prove: sums[b] must equal ⟨messages[b], covectors[b]⟩",
        );

        if self.t == 1 {
            return Witness {
                merged_message: messages[0].clone(),
                merged_covector: covectors[0].clone(),
                sum: sums[0],
                delta: Vec::new(),
                eta: F::ONE,
                theta: vec![F::ONE],
            };
        }

        let ell = self.selector_dim();
        let h = self.padded_size();

        let beta: Vec<F> = prover_state.verifier_message_vec(ell);
        let eq_at_beta = eq_weights(&beta);
        debug_assert_eq!(eq_at_beta.len(), h);
        let initial_sum: F = (0..self.t).map(|b| eq_at_beta[b] * sums[b]).sum();

        // eq carries eq_ℓ(b, β) for ALL b ∈ [0, H) including padded entries: the selector
        // sumcheck folds them as multilinears and zeroing them would corrupt the off-cube
        // extension. m and v are zero for padded entries.
        let total = h * l;
        let mut eq_table = vec![F::ZERO; total];
        let mut m_table = vec![F::ZERO; total];
        let mut v_table = vec![F::ZERO; total];
        for b in 0..h {
            let eq_b = eq_at_beta[b];
            let base = b * l;
            for slot in &mut eq_table[base..base + l] {
                *slot = eq_b;
            }
            if b < self.t {
                m_table[base..base + l].copy_from_slice(&messages[b]);
                v_table[base..base + l].copy_from_slice(&covectors[b]);
            }
        }

        let opening = self.selector.prove(
            prover_state,
            initial_sum,
            &mut eq_table,
            &mut m_table,
            &mut v_table,
        );
        debug_assert_eq!(eq_table.len(), l);
        debug_assert_eq!(m_table.len(), l);
        debug_assert_eq!(v_table.len(), l);

        let eta = opening.eta;
        let merged_covector: Vec<F> = v_table.iter().map(|&x| eta * x).collect();
        let merged_message = m_table;
        let sum = opening.final_scalar;

        let mut theta = eq_weights(&opening.delta);
        theta.truncate(self.t);

        Witness {
            merged_message,
            merged_covector,
            sum,
            delta: opening.delta,
            eta,
            theta,
        }
    }

    /// Selector-merge verifier. Mirrors [`Self::prove`].
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn verify<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
        sums: &[F],
    ) -> VerificationResult<Opening<F>>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
    {
        assert_eq!(sums.len(), self.t);

        if self.t == 1 {
            return Ok(Opening {
                sum: sums[0],
                delta: Vec::new(),
                eta: F::ONE,
                theta: vec![F::ONE],
            });
        }

        let ell = self.selector_dim();
        let beta: Vec<F> = verifier_state.verifier_message_vec(ell);
        let eq_at_beta = eq_weights(&beta);
        let initial_sum: F = (0..self.t).map(|b| eq_at_beta[b] * sums[b]).sum();

        let opening = self.selector.verify(verifier_state, initial_sum)?;
        debug_assert_eq!(opening.delta.len(), ell);

        let eta: F = opening
            .delta
            .iter()
            .zip(&beta)
            .map(|(&d, &b)| (F::ONE - d) * (F::ONE - b) + d * b)
            .product();

        let mut theta = eq_weights(&opening.delta);
        theta.truncate(self.t);

        Ok(Opening {
            sum: opening.final_scalar,
            delta: opening.delta,
            eta,
            theta,
        })
    }
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::{
        algebra::{fields::Field64, random_vector},
        transcript::{codecs::Empty, DomainSeparator, ProverState, VerifierState},
    };

    type F = Field64;

    fn build_blocks(t: usize, l: usize, seed: u64) -> (Vec<Vec<F>>, Vec<Vec<F>>, Vec<F>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let messages: Vec<Vec<F>> = (0..t).map(|_| random_vector(&mut rng, l)).collect();
        let covectors: Vec<Vec<F>> = (0..t).map(|_| random_vector(&mut rng, l)).collect();
        let sums: Vec<F> = messages
            .iter()
            .zip(&covectors)
            .map(|(m, c)| dot(m, c))
            .collect();
        (messages, covectors, sums)
    }

    fn roundtrip(t: usize, l: usize, seed: u64) -> Witness<F> {
        let (messages, covectors, sums) = build_blocks(t, l, seed);
        let cfg = Config::<F>::new(t, l, proof_of_work::Config::none());

        let ds = DomainSeparator::protocol(&"selector-merge-test")
            .session(&format!("t={t} l={l} seed={seed}"))
            .instance(&Empty);

        let mut ps = ProverState::new_std(&ds);
        let prover_witness = cfg.prove(&mut ps, &messages, &covectors, &sums);
        let proof = ps.proof();

        let mut vs = VerifierState::new_std(&ds, &proof);
        let verifier_opening = cfg.verify(&mut vs, &sums).unwrap();
        vs.check_eof().unwrap();

        assert_eq!(prover_witness.sum, verifier_opening.sum);
        assert_eq!(prover_witness.delta, verifier_opening.delta);
        assert_eq!(prover_witness.eta, verifier_opening.eta);
        assert_eq!(prover_witness.theta, verifier_opening.theta);

        assert_eq!(
            dot(
                &prover_witness.merged_message,
                &prover_witness.merged_covector
            ),
            prover_witness.sum,
        );

        prover_witness
    }

    #[test]
    fn single_block_is_noop_byte_identical_proof() {
        let (messages, covectors, sums) = build_blocks(1, 4, 0);
        let cfg = Config::<F>::new(1, 4, proof_of_work::Config::none());

        let ds = DomainSeparator::protocol(&"selector-merge-noop-test")
            .session(&"noop".to_string())
            .instance(&Empty);

        let mut ps_merge = ProverState::new_std(&ds);
        let w = cfg.prove(&mut ps_merge, &messages, &covectors, &sums);
        let proof_with_merge = ps_merge.proof();

        let ps_empty: ProverState = ProverState::new_std(&ds);
        let proof_empty = ps_empty.proof();

        assert_eq!(
            proof_with_merge, proof_empty,
            "single-block merge must write nothing to the transcript",
        );

        assert_eq!(w.merged_message, messages[0]);
        assert_eq!(w.merged_covector, covectors[0]);
        assert_eq!(w.sum, sums[0]);
        assert_eq!(w.delta, Vec::<F>::new());
        assert_eq!(w.eta, F::ONE);
        assert_eq!(w.theta, vec![F::ONE]);
    }

    #[test]
    fn two_blocks() {
        let _ = roundtrip(2, 4, 1);
        let _ = roundtrip(2, 8, 2);
    }

    #[test]
    fn four_blocks() {
        let _ = roundtrip(4, 8, 3);
    }

    #[test]
    fn padded_three_blocks() {
        // t = 3 → ℓ = 2 → H = 4: the 4th selector entry is padded.
        let _ = roundtrip(3, 8, 4);
    }

    #[test]
    fn merged_message_is_theta_weighted_combination() {
        let (messages, covectors, sums) = build_blocks(3, 8, 5);
        let cfg = Config::<F>::new(3, 8, proof_of_work::Config::none());

        let ds = DomainSeparator::protocol(&"selector-merge-theta-test")
            .session(&"theta-check".to_string())
            .instance(&Empty);

        let mut ps = ProverState::new_std(&ds);
        let w = cfg.prove(&mut ps, &messages, &covectors, &sums);

        for i in 0..8 {
            let expected_m: F = (0..3).map(|b| w.theta[b] * messages[b][i]).sum();
            assert_eq!(w.merged_message[i], expected_m);
            let expected_v: F = (0..3).map(|b| w.theta[b] * covectors[b][i]).sum();
            assert_eq!(w.merged_covector[i], w.eta * expected_v);
        }
    }

    // Exercises the `prove` sum-consistency `debug_assert!`, which is compiled
    // out in release builds, so the panic it checks only exists with
    // `debug_assertions` on.
    #[cfg(debug_assertions)]
    #[test]
    fn altered_sum_rejected() {
        let (messages, covectors, mut sums) = build_blocks(3, 4, 11);
        sums[1] += F::ONE;
        let cfg = Config::<F>::new(3, 4, proof_of_work::Config::none());

        let ds = DomainSeparator::protocol(&"selector-merge-bad-test")
            .session(&"bad-sum".to_string())
            .instance(&Empty);

        let mut ps = ProverState::new_std(&ds);
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = cfg.prove(&mut ps, &messages, &covectors, &sums);
        }))
        .expect_err("debug assert should panic on inconsistent sums");
    }
}
