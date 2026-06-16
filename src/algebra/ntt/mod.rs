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

use ark_ff::{Field, FftField};
use static_assertions::assert_obj_safe;

use self::matrix::MatrixMut;
pub use self::cooley_tukey::{NttEngine, RsDomain};
// The CPU encoder is only the active backend (and only constructible) on non-Metal builds.
#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub use self::cooley_tukey::CpuRs;
pub use self::{
    transpose::transpose,
    wavelet::{inverse_wavelet_transform, wavelet_transform},
};
use crate::{
    algebra::{buffer::ActiveBuffer, fields},
    type_map::{self, TypeMap},
};

pub static NTT: LazyLock<TypeMap<NttFamily>> = LazyLock::new(|| {
    let map = TypeMap::new();
    fn register<F: FftField>(map: &TypeMap<NttFamily>) {
        // Both backends share the same `RsDomain` coset convention and differ only in how
        // `interleaved_encode` runs: the CPU encoder needs the full NTT engine, while the
        // Metal encoder only needs the shared domain (the NTT itself runs on the GPU).
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let encoder = crate::algebra::metal_buffer::MetalRs::new(Arc::new(
            RsDomain::<F>::from_fftfield(),
        ));
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        let encoder = CpuRs::new(Arc::new(NttEngine::<F>::new_from_fftfield()));
        map.insert(Arc::new(encoder) as Arc<dyn ReedSolomon<F>>);
    }
    register::<fields::Field64>(&map);
    register::<fields::Field128>(&map);
    register::<fields::Field192>(&map);
    register::<fields::Field256>(&map);
    register::<fields::Field64_2>(&map);
    register::<fields::Field64_3>(&map);
    register::<<fields::Field64_2 as Field>::BasePrimeField>(&map);
    register::<<fields::Field64_3 as Field>::BasePrimeField>(&map);
    map
});

#[derive(Default)]
pub struct NttFamily;

impl type_map::Family for NttFamily {
    type Dyn<F: 'static> = dyn ReedSolomon<F>;
}

/// Trait for a Reed-Solomon encoder implementation for a given field `F`.
///
/// The scalar methods (`next_order`, `generator`, `evaluation_points`) describe the
/// field/domain structure and are backend independent. `interleaved_encode` produces a
/// codeword over the active backend buffer ([`ActiveBuffer`]).
pub trait ReedSolomon<F>: Debug + Send + Sync {
    /// Returns the next supported order equal or larger than `size`.
    ///
    /// The result will be an NTT-smooth number suitable for `codeword_length`.
    ///
    /// Returns `None` if `size` exceeds the largest supported order.
    fn next_order(&self, size: usize) -> Option<usize>;

    fn generator(&self, codeword_length: usize) -> F;

    /// Returns the `index`th evaluation point.
    ///
    /// `masked_message_length`: the total message length including any mask values.
    ///
    /// # Panics
    ///
    /// Panics if any of the indices are `>= codeword_length` or `order` is not supported.
    fn evaluation_points(
        &self,
        masked_message_length: usize,
        codeword_length: usize,
        indices: &[usize],
    ) -> Vec<F>;

    /// Compute a masked interleaved Reed-Solomon encoding.
    ///
    /// `vectors` are `num_vectors` buffers, each holding `interleaving_depth` messages of
    /// `message_length` elements laid out contiguously. `masks` is a
    /// `num_vectors * interleaving_depth` × `mask_length` matrix of blinding coefficients.
    /// `codeword_length` must be an NTT-smooth number >= `message_length + mask_length`.
    /// Returns a `codeword_length × (num_vectors * interleaving_depth)` matrix.
    ///
    /// Each output value is the univariate polynomial evaluation in the evaluation point
    /// corresponding with the index of a coefficient list formed by concatenating message and mask.
    fn interleaved_encode(
        &self,
        vectors: &[&ActiveBuffer<F>],
        masks: &ActiveBuffer<F>,
        message_length: usize,
        interleaving_depth: usize,
        codeword_length: usize,
    ) -> ActiveBuffer<F>;
}

assert_obj_safe!(ReedSolomon<crate::algebra::fields::Field256>);

pub fn next_order<F: 'static>(size: usize) -> Option<usize> {
    NTT.get::<F>()
        .expect("Unsupported NTT field.")
        .next_order(size)
}

pub fn evaluation_points<F: 'static>(
    masked_message_length: usize,
    codeword_length: usize,
    indices: &[usize],
) -> Vec<F> {
    NTT.get::<F>()
        .expect("Unsupported NTT field.")
        .evaluation_points(masked_message_length, codeword_length, indices)
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
        utils::{chunks_exact_or_empty, zip_strict},
    };

    fn valid_codeword_lengths<F: FftField>(size: usize, count: usize) -> Vec<usize> {
        let domain = RsDomain::<F>::from_fftfield();
        iter::successors(domain.next_order(size), |size| domain.next_order(*size + 1))
            .take(count)
            .collect()
    }

    fn test<F: FftField>(engine: &NttEngine<F>)
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
            let masks = random_vector(&mut rng, mask_length * num_messages);
            let message_refs = messages.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
            let codeword = engine.interleaved_encode_slices(
                &message_refs,
                &masks,
                codeword_length,
            );

            // Output must be the right size.
            assert_eq!(codeword.len(), codeword_length * num_messages);

            // Output values are polynomial evaluations in the evaluation points.
            let mut evaluation_points = engine.domain().evaluation_points(message_length + mask_length, codeword_length, &sampled_indices);
            for (&index, &evaluation_point) in zip_strict(&sampled_indices, &evaluation_points) {
                let evaluations = &codeword[index * num_messages.. (index + 1) * num_messages];
                let masks = chunks_exact_or_empty(&masks, mask_length, num_messages);
                for ((message, mask), value) in zip_strict(zip_strict(&messages, masks), evaluations) {
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
        let engine = NttEngine::<fields::Field64>::new_from_fftfield();
        test::<fields::Field64>(&engine);
    }
}
