//! NTT and related algorithms.

mod cooley_tukey;
mod matrix;
mod transpose;
mod utils;
mod wavelet;

use std::{
    fmt::Debug,
    sync::{Arc, LazyLock},
};

use ark_ff::Field;
use static_assertions::assert_obj_safe;

use self::matrix::MatrixMut;
pub use self::{
    cooley_tukey::NttEngine,
    transpose::transpose,
    wavelet::{inverse_wavelet_transform, wavelet_transform},
};
use crate::{
    algebra::fields,
    buffer::{Buffer, DefaultRs},
    type_map::{self, TypeMap},
};

pub static NTT: LazyLock<TypeMap<NttFamily>> = LazyLock::new(|| {
    let map = TypeMap::new();
    map.insert(
        Arc::new(DefaultRs::<fields::Field64>::new_from_fftfield()) as Arc<dyn ReedSolomon<_>>
    );
    map.insert(
        Arc::new(DefaultRs::<fields::Field128>::new_from_fftfield()) as Arc<dyn ReedSolomon<_>>
    );
    map.insert(
        Arc::new(DefaultRs::<fields::Field192>::new_from_fftfield()) as Arc<dyn ReedSolomon<_>>
    );
    map.insert(
        Arc::new(DefaultRs::<fields::Field256>::new_from_fftfield()) as Arc<dyn ReedSolomon<_>>
    );
    map.insert(
        Arc::new(DefaultRs::<fields::Field64_2>::new_from_fftfield()) as Arc<dyn ReedSolomon<_>>,
    );
    map.insert(
        Arc::new(DefaultRs::<fields::Field64_3>::new_from_fftfield()) as Arc<dyn ReedSolomon<_>>,
    );
    map.insert(Arc::new(
        DefaultRs::<<fields::Field64_2 as Field>::BasePrimeField>::new_from_fftfield(),
    ) as Arc<dyn ReedSolomon<_>>);
    map.insert(Arc::new(
        DefaultRs::<<fields::Field64_3 as Field>::BasePrimeField>::new_from_fftfield(),
    ) as Arc<dyn ReedSolomon<_>>);
    map
});

#[derive(Default)]
pub struct NttFamily;

impl type_map::Family for NttFamily {
    type Dyn<F: 'static> = dyn ReedSolomon<F>;
}

/// Reed-Solomon encoder for a given field `F`.
///
/// Pure-NTT abstraction: encodes polynomials, knows nothing about how callers
/// structure those polynomials (whir's IRS, for example, concatenates a
/// message and a mask into a single polynomial before calling this trait —
/// that split lives entirely on the caller side).
pub trait ReedSolomon<F>: Debug + Send + Sync {
    /// Smallest supported codeword length `>= size`, or `None` if `size`
    /// exceeds the engine's maximum order. The returned length is always
    /// NTT-smooth for this engine.
    fn next_order(&self, size: usize) -> Option<usize>;

    /// Generator of the multiplicative subgroup of order `codeword_length`.
    fn generator(&self, codeword_length: usize) -> F;

    /// Evaluation points for the requested codeword positions.
    ///
    /// `result[i]` is the field point at which `codeword[indices[i]]` lives.
    /// `poly_length` is the length of the polynomial whose codeword is being
    /// queried — some engines (e.g. cooley_tukey) derive their internal coset
    /// structure from it, so the same codeword index can map to different
    /// points depending on `poly_length`.
    ///
    /// # Panics
    ///
    /// Panics if any index is `>= codeword_length` or `codeword_length` is
    /// not supported.
    fn evaluation_points(
        &self,
        poly_length: usize,
        codeword_length: usize,
        indices: &[usize],
    ) -> Vec<F>;

    /// Batch-encode polynomials in parallel.
    ///
    /// All `polys[i]` must have the same length. Output is a flat buffer of
    /// `polys.len() * codeword_length` elements in row-major
    /// `(eval_index, poly)` layout: `result[i * polys.len() + j]` is poly
    /// `j`'s value at the `i`-th evaluation point.
    ///
    /// `codeword_length` must be NTT-smooth for this engine and at least the
    /// polynomial length.
    fn interleaved_encode(&self, polys: &[&[F]], codeword_length: usize) -> Buffer<F>;
}

assert_obj_safe!(ReedSolomon<crate::algebra::fields::Field256>);

pub fn next_order<F: 'static>(size: usize) -> Option<usize> {
    NTT.get::<F>()
        .expect("Unsupported NTT field.")
        .next_order(size)
}

pub fn evaluation_points<F: 'static>(
    poly_length: usize,
    codeword_length: usize,
    indices: &[usize],
) -> Vec<F> {
    NTT.get::<F>()
        .expect("Unsupported NTT field.")
        .evaluation_points(poly_length, codeword_length, indices)
}

pub fn interleaved_rs_encode<F: 'static>(polys: &[&[F]], codeword_length: usize) -> Buffer<F> {
    NTT.get::<F>()
        .expect("Unsupported NTT field.")
        .interleaved_encode(polys, codeword_length)
}

pub fn generator<F: 'static>(codeword_length: usize) -> F {
    NTT.get::<F>()
        .expect("Unsupported NTT field.")
        .generator(codeword_length)
}

#[cfg(test)]
mod tests {
    use std::iter;

    use ark_std::rand::{
        distributions::Standard, prelude::Distribution, rngs::StdRng, SeedableRng,
    };
    use proptest::{collection, prelude::Just, proptest, sample::select, strategy::Strategy};

    use super::*;
    use crate::{
        algebra::{random_vector, univariate_evaluate},
        buffer::BufferOps,
        utils::zip_strict,
    };

    fn valid_codeword_lengths<F: 'static>(size: usize, count: usize) -> Vec<usize> {
        let ntt = NTT.get::<F>().expect("No NTT engine for field.");
        iter::successors(ntt.next_order(size), |size| ntt.next_order(*size + 1))
            .take(count)
            .collect()
    }

    fn test<F: Field>(ntt: &dyn ReedSolomon<F>)
    where
        Standard: Distribution<F>,
    {
        let cases = (
            0_usize..10,
            0_usize..(1 << 10),
            0_usize..(1 << 10),
            1_usize..=32,
        )
            .prop_flat_map(|(num_messages, message_length, mask_length, sample_size)| {
                let valid_codeword_lengths =
                    valid_codeword_lengths::<F>(message_length + mask_length, 6);
                select(valid_codeword_lengths).prop_flat_map(move |codeword_length| {
                    let sample_size = sample_size.min(codeword_length.max(1));
                    (
                        Just(num_messages),
                        Just(message_length),
                        Just(mask_length),
                        Just(codeword_length),
                        collection::vec(0..codeword_length, sample_size),
                    )
                })
            });
        proptest!(|(
            seed: u64,
            (num_messages, message_length, mask_length, codeword_length, sampled_indices) in cases
        )| {
            let mut rng = StdRng::seed_from_u64(seed);
            let messages = (0..num_messages)
                .map(|_| random_vector(&mut rng, message_length))
                .collect::<Vec<_>>();
            let masks: Vec<Vec<F>> = (0..num_messages)
                .map(|_| random_vector(&mut rng, mask_length))
                .collect();
            // Build each polynomial as `message || mask`. The engine takes
            // unified polynomial slices; the message/mask split is purely a
            // caller-side concept.
            let polys: Vec<Vec<F>> = (0..num_messages)
                .map(|i| messages[i].iter().chain(masks[i].iter()).copied().collect())
                .collect();
            let poly_refs: Vec<&[F]> = polys.iter().map(Vec::as_slice).collect();
            let codeword = ntt.interleaved_encode(&poly_refs, codeword_length);

            // Output must be the right size.
            assert_eq!(codeword.len(), codeword_length * num_messages);

            // Output values are polynomial evaluations in the evaluation points.
            let mut evaluation_points = ntt.evaluation_points(message_length + mask_length, codeword_length, &sampled_indices);
            let codeword = codeword.to_slice();
            for (&index, &evaluation_point) in zip_strict(&sampled_indices, &evaluation_points) {
                let evaluations = &codeword[index * num_messages.. (index + 1) * num_messages];
                for ((message, mask), value) in zip_strict(zip_strict(&messages, &masks), evaluations) {
                    assert_eq!(*value,
                        univariate_evaluate(message, evaluation_point)
                        + evaluation_point.pow([message_length as u64])
                        * univariate_evaluate(mask, evaluation_point));
                }
            }

            // Evaluation points are unique.
            let mut sample_indices = sampled_indices;
            sample_indices.sort_unstable();
            sample_indices.dedup();
            evaluation_points.sort_unstable();
            evaluation_points.dedup();
            assert_eq!(sample_indices.len(), evaluation_points.len());
        });
    }

    #[test]
    fn test_field64_1() {
        test::<fields::Field64>(NTT.get().unwrap().as_ref());
    }
}
