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
    buffer::{Buffer, BufferOps, DefaultRs},
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

/// One coefficient segment from every logical polynomial.
///
/// Each buffer stores `rows_per_buffer` consecutive polynomial rows. Every
/// row contains `row_width` coefficients. Buffer order defines polynomial
/// order, and segment order defines coefficient order.
pub struct PolynomialSegment<'a, F> {
    buffers: &'a [&'a Buffer<F>],
    rows_per_buffer: usize,
    row_width: usize,
}

impl<'a, F: Copy> PolynomialSegment<'a, F> {
    /// Create a segment from a contiguous slice of buffer references.
    ///
    /// # Panics
    ///
    /// Panics if `rows_per_buffer * row_width` overflows or does not equal
    /// every buffer length.
    pub fn new(buffers: &'a [&'a Buffer<F>], rows_per_buffer: usize, row_width: usize) -> Self {
        let buffer_length = rows_per_buffer
            .checked_mul(row_width)
            .expect("Polynomial segment length overflow.");
        assert!(
            buffers.iter().all(|buffer| buffer.len() == buffer_length),
            "Polynomial segment buffer has the wrong length."
        );
        Self {
            buffers,
            rows_per_buffer,
            row_width,
        }
    }

    /// Create a segment and infer its row width from the buffer length.
    ///
    /// Empty buffer lists and zero-row buffers have an inferred width of zero.
    ///
    /// # Panics
    ///
    /// Panics if the first buffer length is not divisible by
    /// `rows_per_buffer`, or if buffer lengths differ.
    pub fn from_rows(buffers: &'a [&'a Buffer<F>], rows_per_buffer: usize) -> Self {
        let buffer_length = buffers.first().map_or(0, |buffer| buffer.len());
        let row_width = if rows_per_buffer == 0 {
            assert_eq!(
                buffer_length, 0,
                "A zero-row polynomial segment must contain empty buffers."
            );
            0
        } else {
            assert!(
                buffer_length.is_multiple_of(rows_per_buffer),
                "Polynomial segment buffer length is not divisible by its row count."
            );
            buffer_length / rows_per_buffer
        };
        Self::new(buffers, rows_per_buffer, row_width)
    }
}

impl<'a, F> PolynomialSegment<'a, F> {
    /// Number of physical buffers in this segment.
    pub const fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    /// Physical buffer at `index`.
    pub const fn buffer(&self, index: usize) -> &'a Buffer<F> {
        self.buffers[index]
    }

    /// Number of consecutive polynomial rows in every physical buffer.
    pub const fn rows_per_buffer(&self) -> usize {
        self.rows_per_buffer
    }

    /// Number of coefficients contributed to every polynomial.
    pub const fn row_width(&self) -> usize {
        self.row_width
    }

    /// Number of logical polynomials covered by this segment.
    pub const fn polynomial_count(&self) -> usize {
        self.buffer_count()
            .checked_mul(self.rows_per_buffer())
            .expect("Polynomial segment row count overflow.")
    }
}

/// Complete logical polynomials stored across backend-resident buffers.
///
/// Every segment covers all polynomials. Concatenating one row from each
/// segment produces one complete polynomial.
pub struct Polynomials<'a, F> {
    segments: &'a [PolynomialSegment<'a, F>],
    polynomial_count: usize,
    polynomial_length: usize,
}

impl<'a, F> Polynomials<'a, F> {
    /// Create a validated view from coefficient segments.
    ///
    /// The first segment determines the polynomial count. An empty segment
    /// list represents no polynomials. Zero-width edge segments are removed.
    ///
    /// # Panics
    ///
    /// Panics if any segment has a different polynomial count or the total
    /// polynomial length overflows.
    pub fn from_segments(segments: &'a [PolynomialSegment<'a, F>]) -> Self {
        let polynomial_count = segments
            .first()
            .map_or(0, PolynomialSegment::polynomial_count);
        let mut polynomial_length = 0usize;
        for segment in segments {
            assert_eq!(
                segment.polynomial_count(),
                polynomial_count,
                "Polynomial segment has the wrong row count."
            );
            polynomial_length = polynomial_length
                .checked_add(segment.row_width())
                .expect("Polynomial length overflow.");
        }

        let first = segments
            .iter()
            .position(|segment| segment.row_width() != 0)
            .unwrap_or(segments.len());
        let last = segments
            .iter()
            .rposition(|segment| segment.row_width() != 0)
            .map_or(first, |last| last + 1);
        let segments = &segments[first..last];

        Self {
            segments,
            polynomial_count,
            polynomial_length,
        }
    }

    /// Number of logical polynomials.
    pub const fn len(&self) -> usize {
        self.polynomial_count
    }

    /// Whether this view contains no logical polynomials.
    pub const fn is_empty(&self) -> bool {
        self.polynomial_count == 0
    }

    /// Number of coefficients in every logical polynomial.
    pub const fn polynomial_length(&self) -> usize {
        self.polynomial_length
    }

    /// Ordered coefficient segments for every logical polynomial.
    pub const fn segments(&self) -> &'a [PolynomialSegment<'a, F>] {
        self.segments
    }
}

/// Reed-Solomon encoder for a given field `F`.
///
/// Pure-NTT abstraction: encodes complete logical polynomials and knows
/// nothing about how callers construct them. The polynomial view remains
/// buffer-native, so each backend can materialize its preferred layout without
/// forcing a host transfer.
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
    /// Output is a flat buffer in row-major `(eval_index, polynomial)` layout.
    ///
    /// `codeword_length` must be NTT-smooth for this engine and at least the
    /// polynomial length.
    fn interleaved_encode(
        &self,
        polynomials: Polynomials<'_, F>,
        codeword_length: usize,
    ) -> Buffer<F>;
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

pub fn interleaved_rs_encode<F: 'static>(
    polynomials: Polynomials<'_, F>,
    codeword_length: usize,
) -> Buffer<F> {
    NTT.get::<F>()
        .expect("Unsupported NTT field.")
        .interleaved_encode(polynomials, codeword_length)
}

pub fn generator<F: 'static>(codeword_length: usize) -> F {
    NTT.get::<F>()
        .expect("Unsupported NTT field.")
        .generator(codeword_length)
}

#[cfg(test)]
mod tests {
    use std::iter;

    use ark_ff::AdditiveGroup;
    use ark_std::rand::{
        distributions::Standard, prelude::Distribution, rngs::StdRng, SeedableRng,
    };
    use proptest::{collection, prelude::Just, proptest, sample::select, strategy::Strategy};

    use super::*;
    use crate::{
        algebra::univariate_evaluate,
        buffer::{BufferMath, BufferOps},
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
                .map(|_| Buffer::random(&mut rng, message_length))
                .collect::<Vec<_>>();
            let masks = Buffer::random(&mut rng, mask_length * num_messages);
            let message_refs = messages.iter().collect::<Vec<_>>();
            let mask_refs = [&masks];
            let segments = [
                PolynomialSegment::new(&message_refs, 1, message_length),
                PolynomialSegment::new(&mask_refs, num_messages, mask_length),
            ];
            let polynomials = Polynomials::from_segments(&segments);
            let codeword = ntt.interleaved_encode(polynomials, codeword_length);

            // Output must be the right size.
            assert_eq!(codeword.len(), codeword_length * num_messages);

            // Output values are polynomial evaluations in the evaluation points.
            let mut evaluation_points = ntt.evaluation_points(message_length + mask_length, codeword_length, &sampled_indices);
            let codeword = codeword.to_slice();
            for (&index, &evaluation_point) in zip_strict(&sampled_indices, &evaluation_points) {
                let evaluations = &codeword[index * num_messages.. (index + 1) * num_messages];
                for (poly_index, (message, value)) in zip_strict(&messages, evaluations).enumerate() {
                    let mask = &masks.to_slice()[poly_index * mask_length..(poly_index + 1) * mask_length];
                    assert_eq!(*value,
                        univariate_evaluate(message.to_slice(), evaluation_point)
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

    #[test]
    fn segmented_polynomials_match_contiguous_polynomials() {
        type F = fields::Field64;

        let prefixes = [
            Buffer::from(vec![F::from(1), F::from(2)]),
            Buffer::from(vec![F::from(3), F::from(4)]),
        ];
        let prefix_refs = [&prefixes[0], &prefixes[1]];
        let suffix = Buffer::from(vec![F::from(5), F::from(6)]);
        let suffix_refs = [&suffix];
        let endings = Buffer::from(vec![F::from(7), F::from(8)]);
        let ending_refs = [&endings];
        let empty = Buffer::from(Vec::<F>::new());
        let empty_refs = [&empty];
        let segmented = [
            PolynomialSegment::new(&prefix_refs, 1, 2),
            PolynomialSegment::new(&suffix_refs, 2, 1),
            PolynomialSegment::new(&ending_refs, 2, 1),
            PolynomialSegment::new(&empty_refs, 2, 0),
        ];

        let contiguous_buffers = [
            Buffer::from(vec![F::from(1), F::from(2), F::from(5), F::from(7)]),
            Buffer::from(vec![F::from(3), F::from(4), F::from(6), F::from(8)]),
        ];
        let contiguous_refs = [&contiguous_buffers[0], &contiguous_buffers[1]];
        let contiguous = [PolynomialSegment::from_rows(&contiguous_refs, 1)];
        let ntt = NTT.get::<F>().unwrap();
        let codeword_length = ntt.next_order(6).unwrap();
        let segmented =
            ntt.interleaved_encode(Polynomials::from_segments(&segmented), codeword_length);
        let contiguous =
            ntt.interleaved_encode(Polynomials::from_segments(&contiguous), codeword_length);
        assert_eq!(segmented.to_slice(), contiguous.to_slice());
    }

    #[test]
    fn empty_segments_encode_zero_length_polynomials() {
        type F = fields::Field64;

        let buffers = [Buffer::from(Vec::<F>::new())];
        let buffer_refs = [&buffers[0]];
        let segments = [PolynomialSegment::from_rows(&buffer_refs, 2)];
        let ntt = NTT.get::<F>().unwrap();
        let codeword = ntt.interleaved_encode(Polynomials::from_segments(&segments), 4);
        assert_eq!(codeword.to_slice(), vec![F::ZERO; 8]);
    }

    #[test]
    #[should_panic]
    fn empty_batch_validates_polynomial_length() {
        type F = fields::Field64;

        let buffers: [&Buffer<F>; 0] = [];
        let segments = [PolynomialSegment::new(&buffers, 0, 5)];
        let ntt = NTT.get::<F>().unwrap();
        let _ = ntt.interleaved_encode(Polynomials::from_segments(&segments), 4);
    }

    #[test]
    #[should_panic(expected = "Polynomial segment buffer has the wrong length.")]
    fn polynomial_segment_rejects_wrong_buffer_length() {
        type F = fields::Field64;

        let buffers = [Buffer::from(vec![F::ZERO; 3])];
        let buffer_refs = [&buffers[0]];
        let _ = PolynomialSegment::new(&buffer_refs, 2, 2);
    }

    #[test]
    #[should_panic(expected = "Polynomial segment has the wrong row count.")]
    fn polynomials_reject_wrong_segment_row_count() {
        type F = fields::Field64;

        let buffers = [Buffer::from(vec![F::ZERO; 2])];
        let buffer_refs = [&buffers[0]];
        let segments = [
            PolynomialSegment::new(&buffer_refs, 1, 2),
            PolynomialSegment::new(&buffer_refs, 2, 1),
        ];
        let _ = Polynomials::from_segments(&segments);
    }
}
