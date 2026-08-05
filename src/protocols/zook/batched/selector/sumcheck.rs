//! Degree-3 selector sumcheck (`target/batching.md` §8).
//!
//! Reduces a claim of the form
//!
//! ```text
//!   S^{sel,0} = Σ_b eq_ℓ(b, β) · ⟨M(b), V(b)⟩
//! ```
//!
//! to a single inner-product claim `⟨m^★, C^★⟩ = S^★` where:
//!
//! - `δ ∈ 𝔽^ℓ` are the ℓ selector challenges sampled by the verifier,
//! - `η = eq_ℓ(δ, β)`,
//! - `m^★ = M(δ)` (the merged virtual message, length `L`),
//! - `C^★ = η · V(δ)` (the merged virtual covector, length `L`),
//! - `S^★ = η · ⟨M(δ), V(δ)⟩` is the final selector scalar.
//!
//! The round polynomial `P_q(X) = eq_q(X, β_q) · ⟨M_q(X), V_q(X)⟩` is cubic
//! in each selector variable (one degree from `eq`, one each from `M` and
//! `V`). We carry three flat tables `(eq, m, v)` of length `H · L` where
//! `H = 2^ℓ`; the high bits index the selector axis, the low bits index the
//! message axis. The `eq` table is constant along the message axis at start
//! and stays so after each fold (the fold treats each `(b, ·)` group of `L`
//! entries uniformly, since they share the same selector index `b`).
//!
//! The selector sumcheck has **no committed witness** to hide — all three
//! tables are transcript-derived scalars on the verifier side — so it runs
//! in [`SumcheckMode::Standard`] even when the surrounding Zook protocol is
//! ZK. The only soundness slot is `4·ℓ/|𝔽|` (spec §13): `ℓ/|𝔽|` from the
//! `β` cancellation step plus `3·ℓ/|𝔽|` from the degree-3 sumcheck error.

use std::num::NonZeroUsize;

use ark_ff::Field;
use ark_std::rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::{
    algebra::sumcheck::{compute_round_poly_degree3, fold},
    bits::Bits,
    protocols::{
        proof_of_work,
        sumcheck::{self, RoundPolyOracle, SumcheckMode, SumcheckOpening},
    },
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverState, VerificationResult,
        VerifierState,
    },
};

/// Selector-sumcheck configuration.
///
/// `t` is the number of active blocks (before selector-table padding to the
/// next power of two `H = 2^ℓ`). `message_length` is the length of each
/// per-block message vector — i.e., the trailing dimension `L` of the
/// `(H · L)` flat tables.
#[must_use]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Config<F: Field> {
    sumcheck: sumcheck::Config<F>,
    selector_dim: usize,
    message_length: usize,
}

impl<F: Field> Config<F> {
    /// Build a selector-sumcheck config with `t` active blocks (so
    /// `ℓ = ⌈log₂ t⌉` selector rounds) and per-block message length `L`.
    ///
    /// Pass `t = 1` to construct a zero-round config (the selector merge
    /// omits the sub-protocol entirely when only one block is active).
    pub fn new(t: usize, message_length: usize, round_pow: proof_of_work::Config) -> Self {
        assert!(t >= 1, "selector sumcheck requires t >= 1");
        assert!(
            message_length >= 1,
            "selector sumcheck requires message_length >= 1"
        );
        let selector_dim = if t > 1 {
            t.next_power_of_two().trailing_zeros() as usize
        } else {
            0
        };
        let h = 1usize << selector_dim;
        let initial_size = h * message_length;
        let sumcheck = sumcheck::Config::new(
            initial_size,
            round_pow,
            selector_dim,
            SumcheckMode::Standard,
            NonZeroUsize::new(3).expect("3 is non-zero"),
        );
        Self {
            sumcheck,
            selector_dim,
            message_length,
        }
    }

    /// Number of selector variables `ℓ = ⌈log₂ t⌉`. Zero if `t == 1`.
    pub const fn selector_dim(&self) -> usize {
        self.selector_dim
    }

    /// Per-block message length `L`.
    pub const fn message_length(&self) -> usize {
        self.message_length
    }

    /// Selector-table size `H = 2^ℓ`.
    pub const fn padded_size(&self) -> usize {
        1usize << self.selector_dim
    }

    pub const fn sumcheck(&self) -> &sumcheck::Config<F> {
        &self.sumcheck
    }

    /// Selector-sumcheck soundness in bits (spec §13): `4·ℓ/|𝔽|` summed
    /// across the `β` cancellation step and the degree-3 sumcheck error,
    /// expressed as a `log₂ |𝔽| − log₂(4ℓ)` lower bound on bits of security.
    ///
    /// Returns `field_bits` when `ℓ == 0` (no selector rounds, no soundness
    /// slot). Callers that compose security should skip the slot entirely
    /// when `selector_dim == 0`; this finite sentinel avoids constructing an
    /// infinite [`Bits`] value.
    pub fn analytic_error_bits(&self, field_bits: f64) -> Bits {
        if self.selector_dim == 0 {
            return Bits::new(field_bits);
        }
        let log_factor = (4.0 * (self.selector_dim as f64)).log2();
        Bits::new(field_bits - log_factor)
    }
}

/// Output of the selector sumcheck (shared by prover and verifier).
#[must_use]
#[derive(Debug, Clone)]
pub struct Opening<F: Field> {
    /// `δ ∈ 𝔽^ℓ` — selector challenges sampled across the ℓ rounds.
    pub delta: Vec<F>,
    /// `η = eq_ℓ(δ, β)` — the selector scale used to form `C^★ = η · V(δ)`.
    pub eta: F,
    /// `S^★ = η · ⟨M(δ), V(δ)⟩` — the final selector scalar.
    pub final_scalar: F,
}

impl<F: Field> Config<F> {
    /// Selector-sumcheck prover.
    ///
    /// - `claimed_sum = S^{sel,0} = Σ_b eq_ℓ(b, β) · ⟨M(b), V(b)⟩`
    /// - `eq_table[b · L + i] = eq_ℓ(b, β)` for `b ∈ {0,1}^ℓ`, `i ∈ [0, L)`
    ///   (constant along the message axis).
    /// - `m_table[b · L + i] = M(b)[i]`, `v_table[b · L + i] = V(b)[i]`.
    ///
    /// The unused selector entries (`b ∈ [t, H)`) must already be zero in
    /// all three tables — the caller pads the active list to the next power
    /// of two before constructing the tables.
    ///
    /// On return, the three tables are folded **in place** to length `L`:
    /// `m_table` holds `m^★ = M(δ)`, `v_table` holds `V(δ)`, and every entry
    /// of `eq_table` holds `η = eq_ℓ(δ, β)`. The caller forms
    /// `C^★ = η · V(δ)` and `S^★ = opening.final_scalar`.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn prove<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        mut claimed_sum: F,
        eq_table: &mut Vec<F>,
        m_table: &mut Vec<F>,
        v_table: &mut Vec<F>,
    ) -> Opening<F>
    where
        H: DuplexSpongeInterface,
        R: CryptoRng + RngCore,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
    {
        let h = self.padded_size();
        let l = self.message_length;
        let expected_len = h.checked_mul(l).expect("selector tables overflow");
        assert_eq!(eq_table.len(), expected_len);
        assert_eq!(m_table.len(), expected_len);
        assert_eq!(v_table.len(), expected_len);

        if self.selector_dim == 0 {
            // No rounds — single-block selector merge is a no-op. The
            // "merged" tables are the inputs unchanged; η = 1, the final
            // scalar is the claimed sum.
            return Opening {
                delta: Vec::new(),
                eta: F::ONE,
                final_scalar: claimed_sum,
            };
        }

        let oracle = SelectorOracle {
            eq: eq_table,
            m: m_table,
            v: v_table,
        };
        let SumcheckOpening {
            round_challenges,
            mask_rlc: _,
        } = self
            .sumcheck
            .prove_with_oracle(prover_state, &mut claimed_sum, &[], oracle);

        // After `prove_with_oracle`, the oracle has folded all three tables
        // by the final challenge via `finalize`. They are now length `L`
        // each, holding `M(δ)`, `V(δ)`, and `eq(δ, β)` (constant) respectively.
        debug_assert_eq!(eq_table.len(), l);
        debug_assert_eq!(m_table.len(), l);
        debug_assert_eq!(v_table.len(), l);
        let eta = eq_table[0];
        Opening {
            delta: round_challenges,
            eta,
            final_scalar: claimed_sum,
        }
    }

    /// Selector-sumcheck verifier. Mirrors [`Self::prove`].
    ///
    /// The verifier never materialises the `(eq, m, v)` tables — they are
    /// vector-valued and live in the prover's memory. Instead it follows
    /// the sumcheck transcript to derive `δ` and the final scalar `S^★`.
    /// `η` is computed by the caller (it equals `eq_ℓ(δ, β)` where `β` was
    /// sampled before the selector sumcheck started; the caller still owns
    /// `β`).
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn verify<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
        mut claimed_sum: F,
    ) -> VerificationResult<Opening<F>>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
    {
        if self.selector_dim == 0 {
            return Ok(Opening {
                delta: Vec::new(),
                eta: F::ONE,
                final_scalar: claimed_sum,
            });
        }

        let SumcheckOpening {
            round_challenges,
            mask_rlc: _,
        } = self.sumcheck.verify(verifier_state, &mut claimed_sum)?;

        // η is the caller's job — the verifier here only returns δ and the
        // sumcheck-reduced final scalar. The caller multiplies/divides by η
        // as the surrounding protocol requires.
        Ok(Opening {
            delta: round_challenges,
            eta: F::ZERO, // sentinel; the caller must overwrite from its β.
            final_scalar: claimed_sum,
        })
    }
}

/// Selector sumcheck oracle: holds three flat tables and folds them per round.
///
/// References (rather than owned vectors) so the caller retains access to
/// the post-loop folded state — see [`Config::prove`] for how the merged
/// `M(δ)`, `V(δ)`, `eq(δ, β)` are read back out.
struct SelectorOracle<'a, F: Field> {
    eq: &'a mut Vec<F>,
    m: &'a mut Vec<F>,
    v: &'a mut Vec<F>,
}

impl<F: Field> RoundPolyOracle<F> for SelectorOracle<'_, F> {
    fn degree(&self) -> usize {
        3
    }

    fn fold_and_compute(&mut self, prev_challenge: Option<F>) -> Vec<F> {
        if let Some(w) = prev_challenge {
            fold(self.eq, w);
            fold(self.m, w);
            fold(self.v, w);
        }
        let (c0, c2, c3) = compute_round_poly_degree3(self.eq, self.m, self.v);
        vec![c0, c2, c3]
    }

    fn finalize(&mut self, final_challenge: F) {
        fold(self.eq, final_challenge);
        fold(self.m, final_challenge);
        fold(self.v, final_challenge);
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::AdditiveGroup;
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::{
        algebra::{dot, eq_weights, fields::Field64, random_vector},
        transcript::{codecs::Empty, DomainSeparator},
    };

    type F = Field64;

    /// Build the three input tables from t per-block messages, t per-block
    /// covectors, and a β point. Pads selector entries beyond `t` to zero.
    fn build_tables(
        messages: &[Vec<F>],
        covectors: &[Vec<F>],
        beta: &[F],
    ) -> (Vec<F>, Vec<F>, Vec<F>, F) {
        let t = messages.len();
        assert_eq!(covectors.len(), t);
        let l = messages[0].len();
        for m in messages {
            assert_eq!(m.len(), l);
        }
        for v in covectors {
            assert_eq!(v.len(), l);
        }
        let ell = if t > 1 {
            t.next_power_of_two().trailing_zeros() as usize
        } else {
            0
        };
        let h = 1usize << ell;
        let eq_weights_at_beta = eq_weights(beta);
        assert_eq!(eq_weights_at_beta.len(), h);

        let mut eq = vec![F::ZERO; h * l];
        let mut m_flat = vec![F::ZERO; h * l];
        let mut v_flat = vec![F::ZERO; h * l];
        let mut claimed_sum = F::ZERO;
        // `eq` must carry the true `eq_ℓ(b, β)` for ALL b ∈ [0, H), not
        // only b ∈ [0, t). Spec §8.1 zeros `m` and `v` at padded entries;
        // it does NOT zero `eq`. The polynomial P(b) = eq(b,β) · ⟨0,0⟩ = 0
        // at padded entries anyway, but the multilinear extensions of
        // eq, m, v that the sumcheck operates on must match their true
        // cube values — folding to a non-trivial multilinear if `eq` is
        // wrong off-cube.
        for b in 0..h {
            let eq_b = eq_weights_at_beta[b];
            for i in 0..l {
                eq[b * l + i] = eq_b;
            }
            if b < t {
                let ip = dot(&messages[b], &covectors[b]);
                claimed_sum += eq_b * ip;
                for i in 0..l {
                    m_flat[b * l + i] = messages[b][i];
                    v_flat[b * l + i] = covectors[b][i];
                }
            }
        }
        (eq, m_flat, v_flat, claimed_sum)
    }

    fn roundtrip(t: usize, l: usize, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let messages: Vec<Vec<F>> = (0..t).map(|_| random_vector(&mut rng, l)).collect();
        let covectors: Vec<Vec<F>> = (0..t).map(|_| random_vector(&mut rng, l)).collect();
        let ell = if t > 1 {
            t.next_power_of_two().trailing_zeros() as usize
        } else {
            0
        };
        let beta: Vec<F> = random_vector(&mut rng, ell);
        let (mut eq, mut m_flat, mut v_flat, claimed_sum) =
            build_tables(&messages, &covectors, &beta);

        let cfg = Config::<F>::new(t, l, proof_of_work::Config::none());

        let ds = DomainSeparator::protocol(&"selector-sumcheck-test")
            .session(&format!("roundtrip t={t} l={l} seed={seed}"))
            .instance(&Empty);

        let mut ps = crate::transcript::ProverState::new_std(&ds);
        let prove_opening = cfg.prove(&mut ps, claimed_sum, &mut eq, &mut m_flat, &mut v_flat);
        let proof = ps.proof();

        // Verifier mirror.
        let mut vs = crate::transcript::VerifierState::new_std(&ds, &proof);
        let mut verify_opening = cfg.verify(&mut vs, claimed_sum).unwrap();
        // The verifier returns a sentinel η; it's the caller's responsibility
        // to fill it in from β. For the equality check we compute it the same
        // way the surrounding protocol would.
        verify_opening.eta = if ell == 0 {
            F::ONE
        } else {
            // η = Π_q eq_1(δ_q, β_q).
            let mut prod = F::ONE;
            for (&d, &b) in verify_opening.delta.iter().zip(&beta) {
                prod *= (F::ONE - d) * (F::ONE - b) + d * b;
            }
            prod
        };
        vs.check_eof().unwrap();

        assert_eq!(prove_opening.delta, verify_opening.delta);
        assert_eq!(prove_opening.final_scalar, verify_opening.final_scalar);
        assert_eq!(prove_opening.eta, verify_opening.eta);

        // The final scalar must equal η · ⟨M(δ), V(δ)⟩ — the merged
        // virtual inner product the surrounding protocol expects.
        let merged_m = &m_flat; // length L after folding
        let v_at_delta = &v_flat; // length L after folding
        assert_eq!(merged_m.len(), l);
        assert_eq!(v_at_delta.len(), l);
        let merged_ip = dot(merged_m, v_at_delta);
        assert_eq!(prove_opening.final_scalar, prove_opening.eta * merged_ip);

        // And M(δ) should match the naive linear combination of per-block
        // messages weighted by eq_weights(δ).
        if ell > 0 {
            let theta = eq_weights(&prove_opening.delta);
            for i in 0..l {
                let expected: F = (0..t).map(|b| theta[b] * messages[b][i]).sum();
                assert_eq!(merged_m[i], expected, "M(δ)[{i}] mismatch");
            }
            // η · V(δ) should equal η times the same linear combination of
            // per-block covectors.
            for i in 0..l {
                let expected: F = (0..t).map(|b| theta[b] * covectors[b][i]).sum();
                assert_eq!(v_at_delta[i], expected, "V(δ)[{i}] mismatch");
            }
        }
    }

    #[test]
    fn single_block_is_noop() {
        // t == 1: no selector rounds, the sumcheck is a no-op and the
        // tables come back unchanged.
        roundtrip(1, 4, 0);
    }

    #[test]
    fn two_blocks() {
        roundtrip(2, 4, 1);
        roundtrip(2, 8, 2);
    }

    #[test]
    fn four_blocks() {
        roundtrip(4, 8, 3);
    }

    #[test]
    fn padded_three_blocks() {
        // t = 3 → ℓ = 2 → H = 4. The 4th selector entry is implicit zero
        // padding (spec §8.1).
        roundtrip(3, 8, 4);
    }
}
