//! NTT and related algorithms.

mod cooley_tukey;
mod matrix;

#[cfg(test)]
pub(crate) mod test_utils;

mod transpose;
mod utils;
mod wavelet;

use std::{
    fmt::Debug,
    marker::PhantomData,
    sync::{Arc, LazyLock},
};

use ark_ff::{FftField, Field};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use static_assertions::assert_obj_safe;
#[cfg(feature = "tracing")]
use tracing::instrument;

use self::{cooley_tukey::NttEngine, matrix::MatrixMut};
pub use self::{
    cooley_tukey::{generator, intt, intt_batch, ntt, ntt_batch, PartialNttPlan},
    transpose::transpose,
    wavelet::{inverse_wavelet_transform, wavelet_transform},
};
use crate::{
    algebra::fields,
    type_map::{self, TypeMap},
};

pub static NTT: LazyLock<TypeMap<NttFamily>> = LazyLock::new(|| {
    let map = TypeMap::new();
    map.insert(Arc::new(ArkNtt::<fields::Field64>::default()) as Arc<dyn ReedSolomon<_>>);
    map.insert(Arc::new(ArkNtt::<fields::Field128>::default()) as Arc<dyn ReedSolomon<_>>);
    map.insert(Arc::new(ArkNtt::<fields::Field192>::default()) as Arc<dyn ReedSolomon<_>>);
    map.insert(Arc::new(ArkNtt::<fields::Field256>::default()) as Arc<dyn ReedSolomon<_>>);
    map.insert(Arc::new(ArkNtt::<fields::Field64_2>::default()) as Arc<dyn ReedSolomon<_>>);
    map.insert(Arc::new(ArkNtt::<fields::Field64_3>::default()) as Arc<dyn ReedSolomon<_>>);
    map.insert(
        Arc::new(ArkNtt::<<fields::Field64_2 as Field>::BasePrimeField>::default())
            as Arc<dyn ReedSolomon<_>>,
    );
    map.insert(
        Arc::new(ArkNtt::<<fields::Field64_3 as Field>::BasePrimeField>::default())
            as Arc<dyn ReedSolomon<_>>,
    );
    map
});

#[derive(Default)]
pub struct NttFamily;

impl type_map::Family for NttFamily {
    type Dyn<F: 'static> = dyn ReedSolomon<F>;
}

/// Trait for a Reed-Solomon encoder implementation for a given field `F`.
pub trait ReedSolomon<F>: Debug + Send + Sync {
    fn interleaved_encode(
        &self,
        interleaved_coeffs: &[&[F]],
        codeword_length: usize,
        interleaving_depth: usize,
    ) -> Vec<F>;
}

assert_obj_safe!(ReedSolomon<crate::algebra::fields::Field256>);

#[derive(Clone, Copy, Debug, Default)]
pub struct ArkNtt<F: FftField>(PhantomData<F>);

impl<F: FftField> ReedSolomon<F> for ArkNtt<F> {
    fn interleaved_encode(
        &self,
        interleaved_coeffs: &[&[F]],
        codeword_length: usize,
        interleaving_depth: usize,
    ) -> Vec<F> {
        ark_ntt(interleaved_coeffs, codeword_length, interleaving_depth)
    }
}

pub fn interleaved_rs_encode<F: 'static>(
    interleaved_coeffs: &[&[F]],
    codeword_length: usize,
    interleaving_depth: usize,
) -> Vec<F> {
    let engine = NTT.get::<F>().expect("Unsupported field");
    engine.interleaved_encode(interleaved_coeffs, codeword_length, interleaving_depth)
}

/// Partial Reed-Solomon encode that materialises only the rows at `indices`.
///
/// Equivalent to taking [`interleaved_rs_encode`]'s output (a row-major
/// `(codeword_length, num_polys * interleaving_depth)` matrix) and
/// extracting the rows whose row index is in `indices`. Output layout is
/// row-major `(indices.len(), num_polys * interleaving_depth)`, byte-exact
/// against the full encode.
///
/// Uses an output-pruned NTT (see [`PartialNttPlan`]) so peak memory and
/// flop count are both proportional to `indices.len()`, not
/// `codeword_length`. The pruning plan is built once for the index set and
/// reused across every polynomial × interleaving slot.
#[cfg_attr(feature = "tracing", instrument(level = "debug", skip(coeffs, indices), fields(size = coeffs.len(), k = indices.len())))]
pub fn partial_interleaved_rs_encode<F: FftField>(
    coeffs: &[&[F]],
    codeword_length: usize,
    interleaving_depth: usize,
    indices: &[usize],
) -> Vec<F> {
    if coeffs.is_empty() || indices.is_empty() {
        return Vec::new();
    }
    let poly_size = coeffs[0].len();
    for poly in coeffs {
        assert_eq!(poly.len(), poly_size);
    }
    assert!(poly_size.is_multiple_of(interleaving_depth));
    let message_length = poly_size / interleaving_depth;
    assert!(codeword_length.is_multiple_of(message_length));

    let num_polys = coeffs.len();
    let num_cols = num_polys * interleaving_depth;
    let k = indices.len();

    let engine = NttEngine::<F>::new_from_cache();
    let plan = PartialNttPlan::new(codeword_length, indices);

    // Build the submatrix in batch-major layout (`(num_cols, k)`): each
    // contiguous k-chunk is one NTT's outputs. Batches are independent, so
    // populate in parallel across (poly_idx, slot_idx). Final transpose
    // converts to the row-major `(k, num_cols)` layout that
    // `irs_commit::open_inner_from_coeffs` expects.
    let mut batch_major = vec![F::ZERO; num_cols * k];

    #[cfg(feature = "parallel")]
    {
        batch_major
            .par_chunks_exact_mut(k)
            .enumerate()
            .for_each(|(col, dst)| {
                let poly_idx = col / interleaving_depth;
                let slot_idx = col % interleaving_depth;
                let block = &coeffs[poly_idx]
                    [slot_idx * message_length..(slot_idx + 1) * message_length];
                engine.ntt_partial_with_plan_into(block, &plan, dst, 1);
            });
    }
    #[cfg(not(feature = "parallel"))]
    for (col, dst) in batch_major.chunks_exact_mut(k).enumerate() {
        let poly_idx = col / interleaving_depth;
        let slot_idx = col % interleaving_depth;
        let block =
            &coeffs[poly_idx][slot_idx * message_length..(slot_idx + 1) * message_length];
        engine.ntt_partial_with_plan_into(block, &plan, dst, 1);
    }

    transpose(&mut batch_major, num_cols, k);
    batch_major
}

///
/// RS encode coefficients grouped in `interleaving_depth` contiguous blocks
/// at the rate 1/`expansion`, then interleave the evaluations per point.
///
/// This function computes the RS-code for each interleaved message and
/// outputs the interleaved alphabets in the same order as the input.
///
#[cfg_attr(feature = "tracing", instrument(level = "debug", skip(coeffs), fields(size = coeffs.len())))]
fn ark_ntt<F: FftField>(
    coeffs: &[&[F]],
    codeword_length: usize,
    interleaving_depth: usize,
) -> Vec<F> {
    if coeffs.is_empty() {
        return Vec::new();
    }
    let poly_size = coeffs[0].len();
    for poly in coeffs {
        assert_eq!(poly.len(), poly_size);
    }
    assert!(poly_size.is_multiple_of(interleaving_depth));
    let message_length = poly_size / interleaving_depth;
    let per_poly_size = codeword_length * interleaving_depth;
    let total_size = per_poly_size * coeffs.len();

    assert!(codeword_length.is_multiple_of(message_length));

    // Lay out coefficients in contiguous blocks and zero-pad each block.
    let mut result = vec![F::ZERO; total_size];
    for (poly_index, poly) in coeffs.iter().enumerate() {
        for (block_index, block) in poly.chunks_exact(message_length).enumerate() {
            let dst = poly_index * per_poly_size + block_index * codeword_length;
            result[dst..dst + message_length].copy_from_slice(block);
        }
    }

    // NTT each block, then transpose to row-major order with vectorss
    // stacked horizontally.
    ntt_batch(&mut result, codeword_length);
    transpose(
        &mut result,
        coeffs.len() * interleaving_depth,
        codeword_length,
    );
    result
}

#[cfg(test)]
mod tests {
    use ark_ff::Field;

    use super::*;
    use crate::algebra::{fields::Field64, ntt::cooley_tukey::NttEngine};

    #[test]
    fn test_expand_from_coeff_size_2() {
        let engine = NttEngine::<Field64>::new_from_fftfield();

        let c0 = Field64::from(1);
        let c1 = Field64::from(2);
        let coeffs = vec![c0, c1];
        let expansion = 2;

        let omega = engine.root(4);

        // Expansion of the coefficient vector
        //
        // The expansion factor is 2, so we extend the original coefficients as follows:
        //
        //   f0 = c0
        //   f1 = c1
        //   f2 = c0 * ω⁰ = c0
        //   f3 = c1 * ω¹ = c1 * ω
        //
        // Using c0 = 1, c1 = 2, and ω as the generator:

        let f0 = c0;
        let f1 = c1;
        let f2 = c0 * omega.pow([0]);
        let f3 = c1 * omega.pow([1]);

        // Compute the expected NTT
        //
        // The NTT for a size-2 batch follows:
        //
        //   F(0) = f0 + f1
        //   F(1) = f0 - f1
        //
        // We apply this to both pairs (f0, f1) and (f2, f3):
        //
        //   F(0) = f0 + f1
        //   F(1) = f0 - f1
        //
        //   F(2) = f2 + f3
        //   F(3) = f2 - f3
        //
        // Now using the omega-based approach:

        let expected_f0 = f0 + f1;
        let expected_f1 = f0 - f1;
        let expected_f2 = f2 + f3;
        let expected_f3 = f2 - f3;

        // The expected NTT result should be in transposed order:
        let expected_values_transposed = vec![expected_f0, expected_f2, expected_f1, expected_f3];

        let computed_values = test_utils::expand_from_coeff(&coeffs, expansion);
        assert_eq!(computed_values, expected_values_transposed);
    }

    #[test]
    fn test_expand_from_coeff_size_4() {
        let engine = NttEngine::<Field64>::new_from_fftfield();

        let c0 = Field64::from(1);
        let c1 = Field64::from(2);
        let c2 = Field64::from(3);
        let c3 = Field64::from(4);
        let coeffs = vec![c0, c1, c2, c3];
        let expansion = 4;

        let omega = engine.root(16);

        // Manual expansion of the coefficient vector
        //
        // The expansion factor is 4, so we extend the original coefficients into 16 values:
        //
        //   f0  = c0
        //   f1  = c1
        //   f2  = c2
        //   f3  = c3
        //   f4  = c0 * ω⁰  = c0
        //   f5  = c1 * ω¹  = c1 * ω
        //   f6  = c2 * ω²  = c2 * ω²
        //   f7  = c3 * ω³  = c3 * ω³
        //   f8  = c0 * ω⁰  = c0
        //   f9  = c1 * ω²  = c1 * ω²
        //   f10 = c2 * ω⁴  = c2 * ω⁴
        //   f11 = c3 * ω⁶  = c3 * ω⁶
        //   f12 = c0 * ω⁰  = c0
        //   f13 = c1 * ω³  = c1 * ω³
        //   f14 = c2 * ω⁶  = c2 * ω⁶
        //   f15 = c3 * ω⁹  = c3 * ω⁹
        //
        // With c0 = 1, c1 = 2, c2 = 3, c3 = 4, and ω as the generator:

        let f0 = c0;
        let f1 = c1;
        let f2 = c2;
        let f3 = c3;

        let f4 = c0 * omega.pow([1]).pow([0]);
        let f5 = c1 * omega.pow([1]).pow([1]);
        let f6 = c2 * omega.pow([1]).pow([2]);
        let f7 = c3 * omega.pow([1]).pow([3]);

        let f8 = c0 * omega.pow([2]).pow([0]);
        let f9 = c1 * omega.pow([2]).pow([1]);
        let f10 = c2 * omega.pow([2]).pow([2]);
        let f11 = c3 * omega.pow([2]).pow([3]);

        let f12 = c0 * omega.pow([3]).pow([0]);
        let f13 = c1 * omega.pow([3]).pow([1]);
        let f14 = c2 * omega.pow([3]).pow([2]);
        let f15 = c3 * omega.pow([3]).pow([3]);

        // Compute the expected NTT manually using omega powers
        //
        // We process the values in **four chunks of four elements**, following the radix-2
        // butterfly structure.

        let omega = engine.root(4);

        let omega1 = omega; // ω
        let omega2 = omega * omega; // ω²
        let omega3 = omega * omega2; // ω³
        let omega4 = omega * omega3; // ω⁴

        // Chunk 1 (f0 to f3)
        let expected_f0 = f0 + f1 + f2 + f3;
        let expected_f1 = f0 + f1 * omega1 + f2 * omega2 + f3 * omega3;
        let expected_f2 = f0 + f1 * omega2 + f2 * omega4 + f3 * omega2;
        let expected_f3 = f0 + f1 * omega3 + f2 * omega2 + f3 * omega1;

        // Chunk 2 (f4 to f7)
        let expected_f4 = f4 + f5 + f6 + f7;
        let expected_f5 = f4 + f5 * omega1 + f6 * omega2 + f7 * omega3;
        let expected_f6 = f4 + f5 * omega2 + f6 * omega4 + f7 * omega2;
        let expected_f7 = f4 + f5 * omega3 + f6 * omega2 + f7 * omega1;

        // Chunk 3 (f8 to f11)
        let expected_f8 = f8 + f9 + f10 + f11;
        let expected_f9 = f8 + f9 * omega1 + f10 * omega2 + f11 * omega3;
        let expected_f10 = f8 + f9 * omega2 + f10 * omega4 + f11 * omega2;
        let expected_f11 = f8 + f9 * omega3 + f10 * omega2 + f11 * omega1;

        // Chunk 4 (f12 to f15)
        let expected_f12 = f12 + f13 + f14 + f15;
        let expected_f13 = f12 + f13 * omega1 + f14 * omega2 + f15 * omega3;
        let expected_f14 = f12 + f13 * omega2 + f14 * omega4 + f15 * omega2;
        let expected_f15 = f12 + f13 * omega3 + f14 * omega2 + f15 * omega1;

        // Ensure correct NTT ordering
        let expected_values_transposed = vec![
            expected_f0,
            expected_f4,
            expected_f8,
            expected_f12,
            expected_f1,
            expected_f5,
            expected_f9,
            expected_f13,
            expected_f2,
            expected_f6,
            expected_f10,
            expected_f14,
            expected_f3,
            expected_f7,
            expected_f11,
            expected_f15,
        ];

        let computed_values = test_utils::expand_from_coeff(&coeffs, expansion);
        assert_eq!(computed_values, expected_values_transposed);
    }

    #[test]
    fn test_interleaved_rs_encode() {
        use ark_std::UniformRand;

        let mut rng = ark_std::test_rng();
        let count = 1 << 20;
        let expansion = 4;
        let folding_factor = 6;
        let codeword_length = (count >> folding_factor) * expansion;

        let poly: Vec<_> = (0..count).map(|_| Field64::rand(&mut rng)).collect();

        // Compute things the old way
        let block_size = count / (1 << folding_factor);
        let mut blocks = Vec::with_capacity(1 << folding_factor);
        for block in poly.chunks_exact(block_size) {
            blocks.push(test_utils::expand_from_coeff(block, expansion));
        }
        let evals_len = block_size * expansion;
        let mut expected = Vec::with_capacity(count * expansion);
        for i in 0..evals_len {
            for block in &blocks {
                expected.push(block[i]);
            }
        }

        // Compute things the new way
        let interleaved_ntt =
            interleaved_rs_encode(&[poly.as_slice()], codeword_length, 1 << folding_factor);
        assert_eq!(expected, interleaved_ntt);
    }

    #[test]
    fn test_partial_interleaved_rs_encode_matches_full() {
        use ark_std::{rand::Rng, UniformRand};

        let mut rng = ark_std::test_rng();

        // Span several (num_polys, interleaving_depth, M, N) shapes covering
        // the regimes that actually appear in whir_zk (single witness with
        // depth 8, multi-witness with depth 1, M = N/4 blowup).
        let cases = [
            (1usize, 1usize, 64usize, 256usize),
            (1, 8, 16, 64),
            (2, 4, 32, 128),
            (1, 8, 1 << 10, 1 << 12),
        ];

        for (num_polys, interleaving_depth, message_length, codeword_length) in cases {
            let poly_size = message_length * interleaving_depth;
            let polys: Vec<Vec<Field64>> = (0..num_polys)
                .map(|_| (0..poly_size).map(|_| Field64::rand(&mut rng)).collect())
                .collect();
            let poly_slices: Vec<&[Field64]> = polys.iter().map(Vec::as_slice).collect();

            let full = interleaved_rs_encode(&poly_slices, codeword_length, interleaving_depth);
            let num_cols = num_polys * interleaving_depth;
            assert_eq!(full.len(), codeword_length * num_cols);

            // Random subset including 0, last, and a sprinkling in between.
            let k = rng.gen_range(1..=codeword_length.min(16));
            let mut perm: Vec<usize> = (0..codeword_length).collect();
            for i in (1..codeword_length).rev() {
                perm.swap(i, rng.gen_range(0..=i));
            }
            let indices: Vec<usize> = perm.into_iter().take(k).collect();

            let partial = partial_interleaved_rs_encode(
                &poly_slices,
                codeword_length,
                interleaving_depth,
                &indices,
            );
            assert_eq!(partial.len(), k * num_cols);

            for (row, &idx) in indices.iter().enumerate() {
                let full_row = &full[idx * num_cols..(idx + 1) * num_cols];
                let partial_row = &partial[row * num_cols..(row + 1) * num_cols];
                assert_eq!(
                    partial_row, full_row,
                    "shape=({num_polys},{interleaving_depth},{message_length},{codeword_length}) row idx={idx}"
                );
            }
        }
    }
}
