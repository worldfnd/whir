use ark_ff::FftField;

use super::Config;
use crate::algebra::{geometric_accumulate, geometric_sequence};

/// Derived protocol dimensions for a single zkWHIR 2.0 execution.
///
/// Computed once from the config and reused across prover/verifier steps
/// to avoid recomputing (and passing individually) the same derived values.
#[derive(Clone, Copy, Debug)]
pub(super) struct ProtocolDims {
    pub(super) mu: usize,
    pub(super) ell: usize,
    pub(super) rem: usize,
    /// ν blinding polynomials excluding g₀; total g-polynomials = ν + 1 = `num_g_polys()`.
    pub(super) nu: usize,
    pub(super) size: usize,
    pub(super) num_vectors: usize,
    pub(super) num_blinding_vecs: usize,
}

impl ProtocolDims {
    pub(super) fn new<F: FftField>(config: &Config<F>, num_vectors: usize) -> Self {
        let mu = config.blinded_polynomial.initial_num_variables();
        let ell = config
            .blinding_polynomial
            .initial_num_variables()
            .checked_sub(1)
            .filter(|&e| e > 0)
            .expect("blinding polynomial must have at least 2 variables (ell >= 1)");
        let rem = mu % ell;
        let num_blinding_vecs = config.blinding_polynomial.initial_committer.num_vectors;
        let nu = num_blinding_vecs
            .checked_sub(num_vectors)
            .expect("blinding polynomial must commit more vectors than witness count");
        let size = 1 << mu;
        Self {
            mu,
            ell,
            rem,
            nu,
            size,
            num_vectors,
            num_blinding_vecs,
        }
    }

    /// Number of blinding g-polynomials: ν + 1.
    pub(super) const fn num_g_polys(&self) -> usize {
        self.nu + 1
    }

    /// Convenience wrapper for [`phi_i_bits`] using this instance's dimensions.
    pub(super) const fn phi_i_bits(&self, hypercube_idx: usize, phi_index: usize) -> usize {
        phi_i_bits(hypercube_idx, phi_index, self.mu, self.ell, self.rem)
    }
}

/// Extract the ℓ-bit sub-index from a μ-bit hypercube index `b` for the Φ_i projection.
///
/// Implements the Φ_i morphisms:
///   Φ₀(x̄) = (x₁, ..., x_ℓ)
///   Φᵢ(x̄) = (x_{(i-1)·ℓ+rem+1}, ..., x_{i·ℓ+rem})  for i ≥ 1
///
/// This is the integer-index (bit-pattern) version: extracts the same ℓ-bit
/// window that the multivariate Φᵢ would select.
///
/// Index convention (big-endian):
///   index = x_0 · 2^{μ-1} + x_1 · 2^{μ-2} + ... + x_{μ-1} · 2^0
///
/// The result is: `(b >> (μ - start - ℓ)) & ((1 << ℓ) - 1)`
const fn phi_i_bits(
    hypercube_idx: usize,
    phi_index: usize,
    mu: usize,
    ell: usize,
    rem: usize,
) -> usize {
    let start = if phi_index == 0 {
        0
    } else {
        (phi_index - 1) * ell + rem
    };
    assert!(start + ell <= mu, "phi_i_bits: window exceeds mu");
    let shift = mu - start - ell;
    (hypercube_idx >> shift) & ((1 << ell) - 1)
}

/// Compute the discrete logarithm of `target` w.r.t. `gen` in a cyclic group
/// of order `2^log_order`, using the Pohlig-Hellman algorithm.
///
/// `gen_inv` must equal `gen⁻¹`. Accepting it as a parameter lets callers
/// precompute the inverse once when computing multiple discrete logs with the
/// same generator.
///
/// Returns `i` such that `target == gen^i`, where `0 ≤ i < 2^log_order`.
/// Panics if `target` is not in `⟨gen⟩`.
///
/// Complexity: O(log_order²) field multiplications — vs O(2^log_order) for linear scan.
pub(super) fn discrete_log_pow2<F: FftField>(
    target: F,
    gen: F,
    gen_inv: F,
    log_order: u32,
) -> usize {
    debug_assert_eq!(gen * gen_inv, F::ONE, "gen_inv must be the inverse of gen");
    let mut result = 0usize;
    let mut current = target;
    let mut gen_inv_power = gen_inv; // gen^{-2^bit} accumulator

    for bit in 0..log_order {
        // current^{2^{log_order - bit - 1}} == 1  ⟺  bit `bit` of the index is 0
        let mut test = current;
        for _ in 0..(log_order - bit - 1) {
            test.square_in_place();
        }

        if test != F::ONE {
            result |= 1 << bit;
            current *= gen_inv_power;
        }

        gen_inv_power.square_in_place();
    }

    assert_eq!(
        gen.pow([result as u64]),
        target,
        "discrete log verification failed: target not in ⟨gen⟩ of order 2^{log_order}"
    );
    result
}

/// Build the μ-variate evaluation point `fold_args(r̄, z)`.
///
/// fold_args(r̄; z) := (r₁, ..., r_s, z^{2⁰}, z^{2¹}, ..., z^{2^{μ-s-1}})
///
/// Result: `(r_0, ..., r_{s-1}, z^{2^{k-1}}, z^{2^{k-2}}, ..., z^2, z)`
/// where `s = |r̄|` and `k = μ − s`.
///
/// The z-derived coordinates use descending powers (big-endian convention)
/// to match the codebase's `UnivariateEvaluation::mle_evaluate` squaring ladder.
pub(super) fn build_fold_args<F: FftField>(r_bar: &[F], z: F, mu: usize) -> Vec<F> {
    let num_folded_vars = r_bar.len();
    let num_z_vars = mu - num_folded_vars;
    let mut point = Vec::with_capacity(mu);
    point.extend(r_bar);

    // Squaring ladder: z, z², z⁴, ..., z^{2^{num_z_vars-1}}
    let mut z_pow = z;
    let mut z_pows = Vec::with_capacity(num_z_vars);
    for _ in 0..num_z_vars {
        z_pows.push(z_pow);
        z_pow.square_in_place();
    }
    // Reverse to descending order: z^{2^{num_z_vars-1}}, ..., z², z
    point.extend(z_pows.iter().rev());
    point
}

/// Build batched eq tables for the blinding proof (Step 7).
///
/// Implements the weight polynomial:
///   wᵢ(z, ȳ) = z · Σⱼ τⱼ · eq(Φᵢ(P[j]), ȳ)
///
/// `beq_i[k] = Σ_j τ^{j+1} · Σ_{c,m} eq(r̄, c) · z_j^m · δ(Φ_i(c·M+m), k)`
#[cfg_attr(feature = "tracing", tracing::instrument(skip_all, fields(num_points = lambda_z_points.len(), mu = dims.mu, ell = dims.ell, num_g_polys = dims.num_g_polys())))]
pub(super) fn build_beq_tables<F: FftField>(
    lambda_z_points: &[F],
    eq_weights: &[F],
    tau: F,
    dims: ProtocolDims,
) -> Vec<Vec<F>> {
    let mu = dims.mu;
    let ell = dims.ell;
    let rem = dims.rem;
    let num_g_polys = dims.num_g_polys();
    let half_size = 1usize << ell;
    assert!(
        eq_weights.len().is_power_of_two(),
        "eq_weights length must be a power of 2, got {}",
        eq_weights.len()
    );
    let num_folding_vars = eq_weights.len().trailing_zeros() as usize;
    assert!(
        num_folding_vars <= ell,
        "folding factor num_folding_vars={num_folding_vars} must not exceed ell={ell} (would underflow m_cap in Φ₀ window)"
    );
    let num_m_bits = mu - num_folding_vars; // number of m-bits (log2 of sub-polynomial length M)

    // Precompute τ powers: [τ, τ², ..., τ^num_points]
    let tau_powers_full = geometric_sequence(tau, lambda_z_points.len() + 1);
    let tau_powers = &tau_powers_full[1..];

    // Precompute squaring ladders z^{2^0}, z^{2^1}, ..., z^{2^{num_m_bits-1}} for each z-point
    let z_pows_all: Vec<Vec<F>> = lambda_z_points
        .iter()
        .map(|z| {
            let mut z_pows = Vec::with_capacity(num_m_bits);
            let mut z_pow = *z;
            for _ in 0..num_m_bits {
                z_pows.push(z_pow);
                z_pow.square_in_place();
            }
            z_pows
        })
        .collect();

    let num_points = lambda_z_points.len();
    let mut tables = vec![vec![F::ZERO; half_size]; num_g_polys];

    for (i, table) in tables.iter_mut().enumerate() {
        let start_i = if i == 0 { 0 } else { (i - 1) * ell + rem };

        // Bit-window decomposition relative to c|m boundary at position num_folding_vars
        let a_below = mu - start_i - ell; // free m-bits below window (= shift_i)
        let a_above = start_i.saturating_sub(num_folding_vars); // free m-bits above
        let m_cap = num_m_bits - a_below - a_above; // captured m-bits in window
        let c_cap = ell - m_cap; // captured c-bits in window (low c_cap bits of c)

        // Partial eq marginalization: eq_partial[k_c] = Σ_{c: c & mask = k_c} eq[c]
        let eq_partial = if c_cap > 0 {
            let mut eq_partial = vec![F::ZERO; 1 << c_cap];
            let c_mask = (1 << c_cap) - 1;
            for (c_idx, &weight) in eq_weights.iter().enumerate() {
                eq_partial[c_idx & c_mask] += weight;
            }
            eq_partial
        } else {
            vec![F::ONE] // Σ eq_weights = 1
        };

        // Build scalars w_j = tp_j · geo_below_j · geo_above_j
        // and bases base_j = z_j^{2^{a_below}} for geometric_accumulate
        let m_cap_size = 1usize << m_cap;
        let mut scalars = Vec::with_capacity(num_points);
        let mut bases = Vec::with_capacity(num_points);

        for (j, &tp) in tau_powers.iter().enumerate() {
            let z_pows = &z_pows_all[j];

            // Free-bit product below: Π_{j=0}^{a_below-1} (1 + z^{2^j})
            let mut geo_below = F::ONE;
            for &zp in z_pows.iter().take(a_below) {
                geo_below *= F::ONE + zp;
            }

            // Free-bit product above: Π_{j=0}^{a_above-1} (1 + z^{2^{a_below+m_cap+j}})
            let mut geo_above = F::ONE;
            for &zp in z_pows.iter().skip(a_below + m_cap).take(a_above) {
                geo_above *= F::ONE + zp;
            }

            scalars.push(tp * geo_below * geo_above);
            bases.push(if a_below < num_m_bits {
                z_pows[a_below]
            } else {
                F::ONE
            });
        }

        // m_inner[k_m] = Σ_j w_j · base_j^{k_m}
        let mut m_inner = vec![F::ZERO; m_cap_size];
        geometric_accumulate(&mut m_inner, scalars, &bases);

        // Assemble: tables[i][k_c · 2^m_cap + k_m] = eq_partial[k_c] · m_inner[k_m]
        if c_cap > 0 {
            for (k_c, &ep) in eq_partial.iter().enumerate() {
                for (k_m, &mi) in m_inner.iter().enumerate() {
                    table[k_c * m_cap_size + k_m] = ep * mi;
                }
            }
        } else {
            *table = m_inner;
        }
    }

    tables
}

/// Combine blinding-proof weights for initial OOD evaluations and later
/// evaluations of the polynomial folded by the initial sumcheck.
pub(super) fn build_beq_tables_with_initial_ood<F: FftField>(
    z_points: &[F],
    initial_ood_count: usize,
    eq_weights: &[F],
    tau: F,
    dims: ProtocolDims,
) -> Vec<Vec<F>> {
    let (initial_points, folded_points) = z_points.split_at(initial_ood_count);
    let mut tables = build_beq_tables(initial_points, &[F::ONE], tau, dims);
    let folded_tables = build_beq_tables(folded_points, eq_weights, tau, dims);
    let scale = tau.pow([initial_ood_count as u64]);
    for (table, folded) in tables.iter_mut().zip(folded_tables) {
        for (entry, folded_entry) in table.iter_mut().zip(folded) {
            *entry += scale * folded_entry;
        }
    }
    tables
}

/// RS-fold coefficient vectors for the blinding polynomials.
///
/// Produced by [`compute_rs_fold_blinding_coeffs`]; consumed when evaluating
/// m̃(r̄, z, ρ) and g̃ᵢ(r̄, z) at OOD/STIR/Γ points.
#[derive(Debug)]
pub(super) struct RsFoldCoeffs<F> {
    pub(super) masking_coeffs_all: Vec<Vec<F>>,
    pub(super) g_i_coeffs: Vec<Vec<F>>,
}

/// Precompute RS-fold coefficient vectors for the blinding polynomials (Steps 5-6).
///
/// Used to evaluate m̃(r̄, z, ρ) and g̃ᵢ(r̄, z) at OOD/STIR/Γ points.
///
/// After the initial sumcheck folds `s` variables with randomness `r̄`, the original
/// 2^μ-coefficient polynomial is viewed as `k = 2^s` sub-polynomials of length `M = 2^(μ-s)`.
/// The RS-fold is:
///
/// ```text
/// fold(r̄, poly)(z) = Σ_m [ Σ_j eq(r̄, j) · poly[j·M + m] ] · z^m
/// ```
///
/// For blinding polynomials, the 2^μ evaluation table is the lift of an ℓ-variate table
/// via Φ_i projections: `lifted[b] = table[Φ_i_bits(b)]`.
///
/// With multi-polynomial batching (n witness polynomials, batching coefficients α):
/// - `masking_coeffs_all[0][m] = Σ_j eq · [ĝ₀[Φ₀(j·M+m)] + (-ρ)·msk₀[Φ₀(j·M+m)]]`
/// - `masking_coeffs_all[i][m] = (-ρ·αⁱ) · Σ_j eq · mskᵢ[Φ₀(j·M+m)]`  for i = 1..n-1
/// - `g_i_coeffs[i][m] = Σ_j eq · ĝ_{i+1}[Φ_{i+1}(j·M+m)]`  for i = 0..ν-1
///
/// Returns [`RsFoldCoeffs`] where each inner vector has length M.
#[cfg_attr(feature = "tracing", tracing::instrument(skip_all, fields(mu = dims.mu, ell = dims.ell, num_g_polys = g_polys.len())))]
pub(super) fn compute_rs_fold_blinding_coeffs<F: FftField>(
    eq_weights: &[F],
    g_polys: &[Vec<F>],
    masking_polys: &[Vec<F>],
    alpha_coeffs: &[F],
    rho: F,
    dims: ProtocolDims,
) -> RsFoldCoeffs<F> {
    let mu = dims.mu;
    assert!(
        eq_weights.len().is_power_of_two(),
        "eq_weights length must be a power of 2, got {}",
        eq_weights.len()
    );
    let num_folding_vars = eq_weights.len().trailing_zeros() as usize;
    let num_sub_polys = 1usize << num_folding_vars;
    let sub_poly_len = 1usize << (mu - num_folding_vars);
    let num_g_polys = g_polys.len();
    let num_masking = masking_polys.len();
    let neg_rho = -rho;

    // Accumulate g₀ fold coeffs, per-masking fold coeffs, and g_i fold coeffs
    #[allow(clippy::needless_range_loop)]
    let accumulate_j = |g0_fold_accumulator: &mut Vec<F>,
                        masking_fold_accumulators: &mut Vec<Vec<F>>,
                        g_polys_fold_accumulator: &mut Vec<Vec<F>>,
                        j: usize| {
        let eq_j = eq_weights[j];
        for sub_idx in 0..sub_poly_len {
            let full_idx = j * sub_poly_len + sub_idx;
            let phi_0_idx = dims.phi_i_bits(full_idx, 0);

            g0_fold_accumulator[sub_idx] += eq_j * g_polys[0][phi_0_idx];
            for (i, msk) in masking_polys.iter().enumerate() {
                masking_fold_accumulators[i][sub_idx] += eq_j * msk[phi_0_idx];
            }

            for (gi_idx, g_poly) in g_polys[1..].iter().enumerate() {
                let phi_i_idx = dims.phi_i_bits(full_idx, gi_idx + 1);
                g_polys_fold_accumulator[gi_idx][sub_idx] += eq_j * g_poly[phi_i_idx];
            }
        }
    };

    // Assemble masking_coeffs_all from raw g₀ fold and masking folds
    let assemble =
        |g0_fold: Vec<F>, masking_folds: Vec<Vec<F>>, g_i_coeffs: Vec<Vec<F>>| -> RsFoldCoeffs<F> {
            let mut masking_coeffs_all = Vec::with_capacity(num_masking);
            // m₀ = g₀_fold + (-ρ) · msk₀_fold
            let m0: Vec<F> = g0_fold
                .iter()
                .zip(masking_folds[0].iter())
                .map(|(&g, &msk)| g + neg_rho * msk)
                .collect();
            masking_coeffs_all.push(m0);
            // mᵢ = (-ρ·αⁱ) · mskᵢ_fold for i ≥ 1
            for i in 1..num_masking {
                let scale = neg_rho * alpha_coeffs[i];
                let mi: Vec<F> = masking_folds[i].iter().map(|&v| scale * v).collect();
                masking_coeffs_all.push(mi);
            }
            RsFoldCoeffs {
                masking_coeffs_all,
                g_i_coeffs,
            }
        };

    let mut g0_fold = vec![F::ZERO; sub_poly_len];
    let mut masking_folds = vec![vec![F::ZERO; sub_poly_len]; num_masking];
    let mut g_i_coeffs = vec![vec![F::ZERO; sub_poly_len]; num_g_polys - 1];
    for j in 0..num_sub_polys {
        accumulate_j(&mut g0_fold, &mut masking_folds, &mut g_i_coeffs, j);
    }

    assemble(g0_fold, masking_folds, g_i_coeffs)
}

/// Build weight covectors for Step 7's batched blinding proof.
///
/// Constructs `n + ν` covectors used identically by both prover and verifier:
///   - `w_0`:      `beq_0[k]` for g₀, `(-ρ)·beq_0[k]` for msk₀
///   - `w_i`:      `(-ρ·αⁱ)·beq_0[k]` for mskᵢ  (1 ≤ i < num_vectors)
///   - `w_{n+j}`:  `beq_{j+1}[k]` for ĝ_{j+1}    (0 ≤ j < ν)
pub(super) fn build_weight_covectors<F: FftField>(
    beq_tables: &[Vec<F>],
    rho: F,
    alpha_coeffs: &[F],
    dims: ProtocolDims,
) -> Vec<Vec<F>> {
    let num_vectors = dims.num_vectors;
    let num_blinding_vecs = dims.num_blinding_vecs;
    let full_size = 1usize << (dims.ell + 1);

    let mut weight_covectors: Vec<Vec<F>> = Vec::with_capacity(num_blinding_vecs);

    // w_0: first M-polynomial weight (includes g₀ and msk₀)
    {
        let mut w0 = vec![F::ZERO; full_size];
        let neg_rho = -rho;
        for (chunk, &beq) in w0.chunks_exact_mut(2).zip(&beq_tables[0]) {
            chunk[0] = beq;
            chunk[1] = neg_rho * beq;
        }
        weight_covectors.push(w0);
    }

    // w_i (1 ≤ i < n): additional M-polynomial weights (masking only, no g₀)
    for &alpha in &alpha_coeffs[1..num_vectors] {
        let mut wi = vec![F::ZERO; full_size];
        let scale = -rho * alpha;
        for (chunk, &beq) in wi.chunks_exact_mut(2).zip(&beq_tables[0]) {
            chunk[1] = scale * beq;
        }
        weight_covectors.push(wi);
    }

    // w_{n+j-1} (1 ≤ j ≤ ν): ĝ_j weights
    for beq_table in beq_tables.iter().skip(1) {
        let mut wj = vec![F::ZERO; full_size];
        for (chunk, &beq) in wj.chunks_exact_mut(2).zip(beq_table) {
            chunk[0] = beq;
        }
        weight_covectors.push(wj);
    }

    weight_covectors
}

/// Map gamma points (elements of Ω₁) to their corresponding indices in the
/// initial codeword [[f̂]].
///
/// Each γ ∈ Ω₁ is a power of the round-0 generator. The discrete log gives
/// the index within the round-0 domain, and multiplying by `stride = |Ω₀|/|Ω₁|`
/// recovers the position in the initial codeword.
///
/// Used identically by both prover (to open [[f̂]]) and verifier (to verify openings).
pub(super) fn gamma_to_f_hat_indices<F: FftField>(
    gamma_points: &[F],
    config: &super::Config<F>,
) -> Vec<usize> {
    assert!(
        !config.blinded_polynomial.round_configs.is_empty(),
        "zkWHIR 2.0 requires at least one WHIR round"
    );
    let initial_codeword_len = config.blinded_polynomial.initial_committer.codeword_length;
    let round0_codeword_len = config.blinded_polynomial.round_configs[0]
        .irs_committer
        .codeword_length;
    let stride = initial_codeword_len / round0_codeword_len;
    let gen_h = config.blinded_polynomial.round_configs[0]
        .irs_committer
        .generator();
    let gen_h_inv = gen_h.inverse().expect("generator must be invertible");
    let log_round0_len = round0_codeword_len.trailing_zeros();

    gamma_points
        .iter()
        .map(|&gamma| discrete_log_pow2(gamma, gen_h, gen_h_inv, log_round0_len) * stride)
        .collect()
}

/// Compute eq_weights from r_bar. Shared helper to avoid redundant computation.
pub(super) fn compute_eq_weights<F: FftField>(r_bar: &[F]) -> Vec<F> {
    let len = 1usize << r_bar.len();
    let mut buf = vec![F::ONE; len];
    for (i, &r) in r_bar.iter().enumerate() {
        let half = 1 << i;
        for j in (0..half).rev() {
            buf[2 * j + 1] = buf[j] * r;
            buf[2 * j] = buf[j] - buf[2 * j + 1];
        }
    }
    buf
}

/// Claims about the blinding polynomials at initial OOD, later OOD, STIR, and Γ points.
#[derive(Debug)]
pub(super) struct LambdaAccumulator<F> {
    z_points: Vec<F>,
    m_evals: Vec<Vec<F>>,
    g_evals: Vec<Vec<F>>,
}

impl<F> LambdaAccumulator<F> {
    pub(super) const fn new() -> Self {
        Self {
            z_points: Vec::new(),
            m_evals: Vec::new(),
            g_evals: Vec::new(),
        }
    }

    pub(super) fn z_points(&self) -> &[F] {
        &self.z_points
    }

    pub(super) fn push(&mut self, z: F, m: Vec<F>, g: Vec<F>) {
        assert!(
            self.m_evals.is_empty() || m.len() == self.m_evals[0].len(),
            "m_evals length mismatch: expected {}, got {}",
            self.m_evals.first().map_or(0, Vec::len),
            m.len()
        );
        assert!(
            self.g_evals.is_empty() || g.len() == self.g_evals[0].len(),
            "g_evals length mismatch: expected {}, got {}",
            self.g_evals.first().map_or(0, Vec::len),
            g.len()
        );
        self.z_points.push(z);
        self.m_evals.push(m);
        self.g_evals.push(g);
    }

    #[must_use]
    pub(super) const fn len(&self) -> usize {
        self.z_points.len()
    }

    /// Retrieve the claim for blinding vector `vec_idx` at Lambda entry `lambda_idx`.
    ///
    /// Vectors `0..num_vectors` index into `m_evals`; vectors `num_vectors..` index
    /// into `g_evals` (shifted by `num_vectors`).
    pub(super) fn claim(&self, lambda_idx: usize, vec_idx: usize, num_vectors: usize) -> F
    where
        F: Copy,
    {
        if vec_idx < num_vectors {
            self.m_evals[lambda_idx][vec_idx]
        } else {
            self.g_evals[lambda_idx][vec_idx - num_vectors]
        }
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::{AdditiveGroup, FftField, Field};
    use proptest::prelude::*;

    use super::{
        build_beq_tables, build_beq_tables_with_initial_ood, build_weight_covectors,
        discrete_log_pow2, phi_i_bits, ProtocolDims,
    };
    use crate::algebra::{dot, fields::Field64, univariate_evaluate};

    #[test]
    fn initial_ood_and_folded_blinding_weights_match_direct_evaluations() {
        for (mu, ell) in [(4, 2), (5, 2), (7, 3)] {
            let nu = mu / ell;
            let dims = ProtocolDims {
                mu,
                ell,
                rem: mu % ell,
                nu,
                size: 1 << mu,
                num_vectors: 2,
                num_blinding_vecs: 2 + nu,
            };
            let points = [Field64::from(2), Field64::from(5), Field64::from(11)];
            let r = Field64::from(3);
            let eq_weights = [Field64::ONE - r, r];
            let tau = Field64::from(7);
            let tables = build_beq_tables_with_initial_ood(&points, 2, &eq_weights, tau, dims);

            for (i, table) in tables.iter().enumerate() {
                let poly: Vec<_> = (0..1 << dims.ell)
                    .map(|k| Field64::from((i * (1 << ell) + k + 1) as u64))
                    .collect();
                let lifted: Vec<_> = (0..dims.size)
                    .map(|k| poly[dims.phi_i_bits(k, i)])
                    .collect();
                let half = dims.size / eq_weights.len();
                let folded: Vec<_> = (0..half)
                    .map(|k| {
                        eq_weights
                            .iter()
                            .enumerate()
                            .map(|(c, &weight)| weight * lifted[c * half + k])
                            .sum()
                    })
                    .collect();
                let expected = tau * univariate_evaluate(&lifted, points[0])
                    + tau.square() * univariate_evaluate(&lifted, points[1])
                    + tau.square() * tau * univariate_evaluate(&folded, points[2]);
                assert_eq!(dot(table, &poly), expected);
            }
        }
    }

    #[test]
    fn initial_ood_claim_matches_masked_witness_decomposition() {
        let dims = ProtocolDims {
            mu: 4,
            ell: 2,
            rem: 0,
            nu: 2,
            size: 16,
            num_vectors: 2,
            num_blinding_vecs: 4,
        };
        let witness_polys: Vec<Vec<Field64>> = (0..2)
            .map(|i| {
                (0..dims.size)
                    .map(|k| Field64::from((i * 16 + k + 1) as u64))
                    .collect()
            })
            .collect();
        let masking_polys: Vec<Vec<Field64>> = (0u64..2)
            .map(|i| (0u64..4).map(|k| Field64::from(i * 4 + k + 2)).collect())
            .collect();
        let g_polys: Vec<Vec<Field64>> = (0u64..3)
            .map(|i| (0u64..4).map(|k| Field64::from(i * 4 + k + 10)).collect())
            .collect();
        let alpha = Field64::from(5);
        let beta = Field64::from(3);
        let rho = Field64::from(7);
        let z = Field64::from(11);

        let f_hat_evals: Vec<_> = witness_polys
            .iter()
            .zip(&masking_polys)
            .map(|(poly, mask)| {
                let f_hat: Vec<_> = (0..dims.size)
                    .map(|k| poly[k] + mask[dims.phi_i_bits(k, 0)])
                    .collect();
                univariate_evaluate(&f_hat, z)
            })
            .collect();
        let f_zk: Vec<_> = (0..dims.size)
            .map(|k| {
                rho * (witness_polys[0][k] + alpha * witness_polys[1][k])
                    + g_polys[0][dims.phi_i_bits(k, 0)]
                    + beta * g_polys[1][dims.phi_i_bits(k, 1)]
                    + beta.square() * g_polys[2][dims.phi_i_bits(k, 2)]
            })
            .collect();
        let blinding_vectors: Vec<Vec<_>> = masking_polys
            .iter()
            .map(|mask| {
                g_polys[0]
                    .iter()
                    .zip(mask)
                    .flat_map(|(&g, &m)| [g, m])
                    .collect()
            })
            .chain(
                g_polys[1..]
                    .iter()
                    .map(|g| g.iter().flat_map(|&value| [value, Field64::ZERO]).collect()),
            )
            .collect();
        let tables = build_beq_tables(&[z], &[Field64::ONE], Field64::ONE, dims);
        let weights = build_weight_covectors(&tables, rho, &[Field64::ONE, alpha], dims);
        let claims: Vec<_> = weights
            .iter()
            .zip(&blinding_vectors)
            .map(|(weight, vector)| dot(weight, vector))
            .collect();
        let reconstructed = rho * (f_hat_evals[0] + alpha * f_hat_evals[1])
            + claims[0]
            + claims[1]
            + beta * claims[2]
            + beta.square() * claims[3];

        assert_eq!(reconstructed, univariate_evaluate(&f_zk, z));
    }

    // ---------------------------------------------------------------
    // phi_i_bits unit tests
    // ---------------------------------------------------------------

    #[test]
    fn phi_i_bits_phi0_extracts_top_ell_bits() {
        // mu=8, ell=3, rem=2 (8 % 3 = 2)
        // Φ₀ extracts bits [0..3) = top 3 bits
        let mu = 8;
        let ell = 3;
        let rem = mu % ell; // 2

        // b = 0b_101_01010 = 0xAA = 170
        // top 3 bits: 101 = 5
        assert_eq!(phi_i_bits(0b1010_1010, 0, mu, ell, rem), 0b101);
        // b = 0b_111_00000 = 224
        assert_eq!(phi_i_bits(0b1110_0000, 0, mu, ell, rem), 0b111);
        // b = 0b_000_11111 = 31
        assert_eq!(phi_i_bits(0b0001_1111, 0, mu, ell, rem), 0b000);
    }

    #[test]
    fn phi_i_bits_phi1_extracts_after_rem() {
        // mu=8, ell=3, rem=2
        // Φ₁: start = (1-1)*3 + 2 = 2, extracts bits [2..5)
        let mu = 8;
        let ell = 3;
        let rem = 2;

        // b = 0b_10_110_010 = 178
        // bits [2..5) = 110 = 6
        assert_eq!(phi_i_bits(0b1011_0010, 1, mu, ell, rem), 0b110);
    }

    #[test]
    fn phi_i_bits_phi2_extracts_next_window() {
        // mu=8, ell=3, rem=2
        // Φ₂: start = (2-1)*3 + 2 = 5, extracts bits [5..8)
        let mu = 8;
        let ell = 3;
        let rem = 2;

        // b = 0b_10110_101 = 181
        // bits [5..8) = 101 = 5
        assert_eq!(phi_i_bits(0b1011_0101, 2, mu, ell, rem), 0b101);
    }

    #[test]
    fn phi_i_bits_rem_zero() {
        // mu=6, ell=3, rem=0 (6 % 3 = 0)
        // Φ₀: start=0, extracts bits [0..3) = top 3
        // Φ₁: start=0+0=0, wait: (1-1)*3+0 = 0 ... same as Φ₀
        // Actually with rem=0, Φ₁ start = (1-1)*3 + 0 = 0, same window.
        // Let's use mu=6, ell=2, rem=0
        let mu = 6;
        let ell = 2;
        let rem = 0; // 6 % 2 = 0

        // Φ₀: start=0, bits [0..2) = top 2
        // Φ₁: start=(1-1)*2+0=0, bits [0..2) = same as Φ₀
        // Φ₂: start=(2-1)*2+0=2, bits [2..4)
        // Φ₃: start=(3-1)*2+0=4, bits [4..6)

        // b = 0b_11_00_10 = 50
        assert_eq!(phi_i_bits(0b11_00_10, 0, mu, ell, rem), 0b11);
        assert_eq!(phi_i_bits(0b11_00_10, 2, mu, ell, rem), 0b00);
        assert_eq!(phi_i_bits(0b11_00_10, 3, mu, ell, rem), 0b10);
    }

    #[test]
    fn phi_i_bits_single_bit_ell() {
        // ell=1: each Φ extracts a single bit
        // mu=4, ell=1, rem=0
        // b = 0b_1011, big-endian positions: [1, 0, 1, 1]
        //
        // Φ₀: start=0                → bit position 0 (MSB) = 1
        // Φ₁: start=(0)*1+0=0        → bit position 0 (same as Φ₀) = 1
        // Φ₂: start=(1)*1+0=1        → bit position 1 = 0
        // Φ₃: start=(2)*1+0=2        → bit position 2 = 1
        // Φ₄: start=(3)*1+0=3        → bit position 3 (LSB) = 1
        let mu = 4;
        let ell = 1;
        let rem = 0;

        assert_eq!(phi_i_bits(0b1011, 0, mu, ell, rem), 1);
        assert_eq!(phi_i_bits(0b1011, 1, mu, ell, rem), 1); // overlaps with Φ₀ when rem=0
        assert_eq!(phi_i_bits(0b1011, 2, mu, ell, rem), 0);
        assert_eq!(phi_i_bits(0b1011, 3, mu, ell, rem), 1);
        assert_eq!(phi_i_bits(0b1011, 4, mu, ell, rem), 1);
    }

    #[test]
    fn phi_i_bits_boundary_all_ones() {
        let mu = 6;
        let ell = 3;
        let rem = 0;

        // b = 0b_111_111 = 63
        assert_eq!(phi_i_bits(0b111_111, 0, mu, ell, rem), 0b111);
        assert_eq!(phi_i_bits(0b111_111, 2, mu, ell, rem), 0b111);
    }

    #[test]
    fn phi_i_bits_boundary_all_zeros() {
        let mu = 6;
        let ell = 3;
        let rem = 0;

        assert_eq!(phi_i_bits(0, 0, mu, ell, rem), 0);
        assert_eq!(phi_i_bits(0, 2, mu, ell, rem), 0);
    }

    // ---------------------------------------------------------------
    // discrete_log_pow2 unit tests
    // ---------------------------------------------------------------

    /// Test helper: calls discrete_log_pow2 with an auto-computed inverse.
    fn dlog(target: Field64, gen: Field64, log_order: u32) -> usize {
        let gen_inv = gen.inverse().expect("generator must be invertible");
        discrete_log_pow2(target, gen, gen_inv, log_order)
    }

    /// Get a generator of the multiplicative subgroup of order 2^k in Field64.
    fn subgroup_gen(log_order: u32) -> Field64 {
        // Field64: p = 2^64 - 2^32 + 1, two-adic valuation = 32
        // TWO_ADIC_ROOT_OF_UNITY has order 2^32
        let max_log = Field64::TWO_ADICITY;
        assert!(log_order <= max_log);
        let mut gen = Field64::TWO_ADIC_ROOT_OF_UNITY;
        // Raise to 2^(max_log - log_order) to get order 2^log_order
        for _ in 0..(max_log - log_order) {
            gen.square_in_place();
        }
        gen
    }

    #[test]
    fn dlog_identity_is_zero() {
        for log_order in 1..=8 {
            let gen = subgroup_gen(log_order);
            assert_eq!(
                dlog(Field64::ONE, gen, log_order),
                0,
                "dlog(1, gen, {log_order}) should be 0"
            );
        }
    }

    #[test]
    fn dlog_generator_is_one() {
        for log_order in 1..=8 {
            let gen = subgroup_gen(log_order);
            assert_eq!(
                dlog(gen, gen, log_order),
                1,
                "dlog(gen, gen, {log_order}) should be 1"
            );
        }
    }

    #[test]
    fn dlog_known_powers() {
        let log_order = 4; // group of order 16
        let gen = subgroup_gen(log_order);

        for i in 0..16usize {
            let target = gen.pow([i as u64]);
            assert_eq!(
                dlog(target, gen, log_order),
                i,
                "dlog(gen^{i}, gen, {log_order}) should be {i}"
            );
        }
    }

    #[test]
    fn dlog_order_1() {
        // Group of order 2^1 = 2: elements are {1, -1}
        let gen = subgroup_gen(1);
        assert_eq!(dlog(Field64::ONE, gen, 1), 0);
        assert_eq!(dlog(gen, gen, 1), 1);
    }

    #[test]
    fn dlog_larger_group() {
        let log_order = 10; // group of order 1024
        let gen = subgroup_gen(log_order);

        // Check a handful of indices including edge cases
        for &i in &[0, 1, 2, 511, 512, 1023] {
            let target = gen.pow([i as u64]);
            assert_eq!(dlog(target, gen, log_order), i, "failed for i={i}");
        }
    }

    // ---------------------------------------------------------------
    // Property-based tests
    // ---------------------------------------------------------------

    proptest! {
        /// Roundtrip: gen^{dlog(gen^i, gen, n)} == gen^i for all i in [0, 2^n).
        #[test]
        fn dlog_roundtrip(log_order in 1u32..=12, idx in 0u32..4096) {
            let order = 1u32 << log_order;
            let i = (idx % order) as usize;
            let gen = subgroup_gen(log_order);
            let target = gen.pow([i as u64]);
            let result = dlog(target, gen, log_order);
            prop_assert_eq!(result, i, "dlog roundtrip failed for log_order={}, i={}", log_order, i);
        }

        /// phi_i_bits output is always < 2^ell (within the valid range).
        #[test]
        fn phi_i_bits_in_range(
            mu in 4usize..=16,
            ell in 1usize..=4,
            hypercube_idx_raw in 0usize..65536,
        ) {
            prop_assume!(ell <= mu);
            let rem = mu % ell;
            let num_phis = 1 + (mu - rem) / ell; // Φ₀ + ⌊(mu-rem)/ell⌋
            let max_idx = 1usize << mu;
            let hypercube_idx = hypercube_idx_raw % max_idx;

            for phi_index in 0..num_phis {
                // Ensure the window fits: start + ell <= mu
                let start = if phi_index == 0 { 0 } else { (phi_index - 1) * ell + rem };
                if start + ell > mu {
                    break;
                }
                let result = phi_i_bits(hypercube_idx, phi_index, mu, ell, rem);
                prop_assert!(
                    result < (1 << ell),
                    "phi_i_bits({}, {}, {}, {}, {}) = {} >= 2^{}",
                    hypercube_idx, phi_index, mu, ell, rem, result, ell
                );
            }
        }

        /// For rem=0 and ell dividing mu evenly, the Φ windows partition the mu bits.
        /// Concatenating all windows should reconstruct the original index.
        #[test]
        fn phi_i_bits_partition_no_remainder(
            mu_factor in 1usize..=4,
            ell in 2usize..=4,
            hypercube_idx_raw in 0usize..65536,
        ) {
            let mu = mu_factor * ell;
            prop_assume!(mu <= 16);
            let rem = 0;
            let num_phis = mu / ell; // exactly mu/ell windows when rem=0
            let max_idx = 1usize << mu;
            let b = hypercube_idx_raw % max_idx;

            // Φ₀ extracts top ell bits, Φ₁..Φ_{num_phis-1} extract subsequent windows.
            // But note: when rem=0, Φ₀ and Φ₁ have the SAME start (both 0).
            // So Φ₀ through Φ_{num_phis} with phi_index 0, 2, 3, ..., num_phis
            // actually: with rem=0, Φ₁ start = (0)*ell + 0 = 0 (same as Φ₀)
            // So the distinct windows are Φ₀, Φ₂, Φ₃, ..., Φ_{num_phis}
            // which is Φ₀ (start=0), Φ₂ (start=ell), Φ₃ (start=2*ell), ...

            // Reconstruct from non-overlapping windows
            let mut reconstructed = 0usize;
            // Φ₀: start=0
            let phi0 = phi_i_bits(b, 0, mu, ell, rem);
            reconstructed |= phi0 << (mu - ell);

            // Φ_{i+1}: start = i*ell for i >= 1
            for i in 1..num_phis {
                let phi_idx = i + 1; // Φ₂, Φ₃, ...
                let start = (phi_idx - 1) * ell; // + rem = 0
                if start + ell > mu {
                    break;
                }
                let bits = phi_i_bits(b, phi_idx, mu, ell, rem);
                let shift = mu - start - ell;
                reconstructed |= bits << shift;
            }

            prop_assert_eq!(
                reconstructed, b,
                "partition reconstruction failed for b={:#b}, mu={}, ell={}", b, mu, ell
            );
        }
    }
}
