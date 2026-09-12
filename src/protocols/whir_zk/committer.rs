use ark_ff::FftField;
use spongefish::{Codec, DuplexSpongeInterface};
use zeroize::{Zeroize, ZeroizeOnDrop};

use super::{utils::ProtocolDims, Config};
use crate::{
    hash::Hash,
    protocols::irs_commit,
    transcript::{ProverMessage, ProverState},
};

/// Secret blinding randomness that must be scrubbed from memory on drop.
///
/// Wrapping these fields in a separate struct lets `Witness` remain
/// destructurable (no custom `Drop`) while guaranteeing zeroization
/// of the secret data regardless of how the witness is consumed.
#[derive(Zeroize, ZeroizeOnDrop)]
pub(super) struct BlindingSecrets<F: FftField> {
    /// Per-witness masking polynomials mskᵢ (ℓ-variate, 2^ℓ coefficients).
    pub(super) masking_polys: Vec<Vec<F>>,
    /// Blinding polynomials ĝ₀..ĝ_ν (ℓ-variate, 2^ℓ coefficients each).
    pub(super) g_polys: Vec<Vec<F>>,
    /// Interleaved blinding vectors [M₀, ..., M_{n-1}, ĝ₁, ..., ĝ_ν] as committed.
    pub(super) blinding_vectors: Vec<Vec<F>>,
}

/// Prover-side witness produced by Step 1 (Commitment).
///
/// Contains the IRS-commit witnesses for both WHIR instances, plus the raw
/// polynomial data needed by Steps 2-7.
///
/// Secret blinding randomness is held in `BlindingSecrets` which implements
/// `ZeroizeOnDrop`, ensuring it is scrubbed from memory even if the witness
/// is destructured for early field-level drops.
#[must_use]
#[allow(clippy::struct_field_names)]
pub struct Witness<F: FftField> {
    /// IRS-commit witness for [[f̂]] (first WHIR instance).
    pub(super) f_hat_witness: irs_commit::Witness<F, F>,
    /// IRS-commit witness for [[M]], [[ĝ₁]]..[[ĝ_ν]] (second WHIR instance).
    pub(super) blinding_poly_witness: irs_commit::Witness<F, F>,
    /// f̂ᵢ = fᵢ + mskᵢ(Φ₀) for each of the n witness polynomials.
    pub(super) f_hat_polys: Vec<Vec<F>>,
    /// Secret blinding randomness (zeroized on drop).
    pub(super) secrets: BlindingSecrets<F>,
}

impl<F: FftField> Config<F> {
    /// **Step 1 — Commitment**.
    ///
    /// For n witness polynomials f₁..fₙ:
    ///   1a. Sample n random ℓ-variate masking polynomials mskᵢ
    ///   1b. Compute f̂ᵢ = fᵢ + mskᵢ(Φ₀) and commit [[f̂]]
    ///   1c. Sample ν + 1 random ℓ-variate blinding polynomials ĝ₀..ĝ_ν
    ///   1d. Build committed vectors: n M-polynomials Mᵢ(ȳ,t) = ĝ₀(ȳ) + t·mskᵢ(ȳ)
    ///       and ν embedded ĝ-polynomials, then commit [[M]], [[ĝ₁]]..[[ĝ_ν]]
    pub fn commit<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        polynomials: &[&[F]],
    ) -> Witness<F>
    where
        H: DuplexSpongeInterface,
        R: ark_std::rand::RngCore + ark_std::rand::CryptoRng,
        F: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let num_polys = polynomials.len();
        assert!(!polynomials.is_empty(), "must have at least one polynomial");
        let expected_len = polynomials[0].len();
        assert!(
            expected_len.is_power_of_two(),
            "polynomial length must be a power of 2"
        );
        for (i, poly) in polynomials.iter().enumerate() {
            assert_eq!(
                poly.len(),
                expected_len,
                "polynomials[{i}] has length {}, expected {expected_len}",
                poly.len()
            );
        }
        let dims = ProtocolDims::new(self, num_polys);
        let ell_variate_size = 1usize << dims.ell;
        let shift = dims.mu - dims.ell; // Φ₀ extracts the top ℓ bits: Φ₀(b) = b >> shift

        // Step 1a-1b: Sample n masking polynomials, compute f̂ᵢ = fᵢ + mskᵢ(Φ₀(x̄))
        // msk is fresh per witness to preserve ZK for each fᵢ.
        let mut masking_polys = Vec::with_capacity(num_polys);
        let mut f_hat_polys = Vec::with_capacity(num_polys);
        for poly in polynomials {
            let masking_poly: Vec<F> = (0..ell_variate_size)
                .map(|_| F::rand(prover_state.rng()))
                .collect();
            let f_hat_poly: Vec<F> = poly
                .iter()
                .enumerate()
                .map(|(idx, &coeff)| coeff + masking_poly[idx >> shift])
                .collect();
            masking_polys.push(masking_poly);
            f_hat_polys.push(f_hat_poly);
        }

        // Step 1b: Commit [[f̂]] via first WHIR instance.
        let f_hat_refs: Vec<&[F]> = f_hat_polys.iter().map(|p| p.as_slice()).collect();
        let mut f_hat_witness = self.blinded_polynomial.commit(prover_state, &f_hat_refs);

        // Drop the encoded codeword; will be re-encoded immediately before each
        // open in prove_blinded_polynomial (Steps 4 and 6). This keeps the
        // ~codeword_length × interleaving_depth field elements out of the
        // resident set during the prepare_and_sumcheck rounds where global peak
        // hits.
        f_hat_witness.matrix = Vec::new();

        // Step 1c: Sample ν + 1 random ℓ-variate blinding polynomials ĝ₀..ĝ_ν.
        let num_blinding_polys = dims.num_g_polys();
        let mut g_polys = Vec::with_capacity(num_blinding_polys);
        for _ in 0..num_blinding_polys {
            g_polys.push(
                (0..ell_variate_size)
                    .map(|_| F::rand(prover_state.rng()))
                    .collect::<Vec<_>>(),
            );
        }

        // Step 1d: Build committed vectors for second WHIR instance.
        // Mᵢ(ȳ, t) = ĝ₀(ȳ) + t·mskᵢ(ȳ), stored as interleaved [g₀[k], mskᵢ[k]].
        let interleaved_blinding_vectors: Vec<Vec<F>> = masking_polys
            .iter()
            .map(|msk| {
                g_polys[0]
                    .iter()
                    .zip(msk.iter())
                    .flat_map(|(&g, &m)| [g, m])
                    .collect()
            })
            .collect();

        // Assemble blinding vectors: [M₀, ..., M_{n-1}, ĝ₁, ..., ĝ_ν].
        // ĝⱼ are ℓ-variate but committed as (ℓ+1)-variate with t-coefficient = 0,
        // stored as interleaved [gⱼ[k], 0] (coefficient for t is zero).
        let mut blinding_vectors: Vec<Vec<F>> = interleaved_blinding_vectors;
        for g in &g_polys[1..] {
            blinding_vectors.push(g.iter().flat_map(|&c| [c, F::ZERO]).collect());
        }
        let blinding_refs: Vec<&[F]> = blinding_vectors.iter().map(|v| v.as_slice()).collect();

        let mut blinding_poly_witness = self
            .blinding_polynomial
            .commit(prover_state, &blinding_refs);

        // The encoded codeword is only needed when [[M, ĝ]] is opened in
        // Step 7. Until then it is dead weight (held resident through all of
        // prove_blinded_polynomial, where global peak hits). Drop the matrix
        // here; the prover re-encodes from `secrets.blinding_vectors` just
        // before calling `blinding_polynomial.prove`.
        blinding_poly_witness.matrix = Vec::new();

        Witness {
            f_hat_witness,
            blinding_poly_witness,
            f_hat_polys,
            secrets: BlindingSecrets {
                masking_polys,
                g_polys,
                blinding_vectors,
            },
        }
    }
}
