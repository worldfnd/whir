use std::{any::Any, mem, ops::RangeBounds};

use ark_ff::Field;
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, Rng, RngCore};

use crate::{
    algebra::{
        dot,
        embedding::Embedding,
        linear_form::{Covector, LinearForm, UnivariateEvaluation},
        mixed_dot, mixed_multilinear_extend, mixed_scalar_mul_add, mixed_univariate_evaluate,
        sumcheck::{compute_sumcheck_polynomial, fold},
    },
    engines::EngineId,
    hash::{self, Hash},
    protocols::{matrix_commit::Encodable, merkle_tree},
};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub use super::metal_buffer::{MetalBuffer, MetalSlice, MetalSliceMut};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub type ActiveBuffer<T> = MetalBuffer<T>;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub type ActiveSlice<'a, T> = MetalSlice<'a, T>;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub type ActiveSliceMut<'a, T> = MetalSliceMut<'a, T>;

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub type ActiveBuffer<T> = CpuBuffer<T>;
#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub type ActiveSlice<'a, T> = CpuSlice<'a, T>;
#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub type ActiveSliceMut<'a, T> = CpuSliceMut<'a, T>;

pub trait BufferOps<T> {
    fn from_vec(source: Vec<T>) -> Self;
    fn from_slice(source: &[T]) -> Self;
    fn as_slice(&self) -> &[T];
    fn num_rows(&self, num_cols: usize) -> usize {
        self.len() / num_cols
    }
    fn read_rows(&self, num_cols: usize, indices: &[usize]) -> Vec<T>;
    fn at_index(&self, index: usize) -> Option<T>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn extend(&self, other: &Self) -> Self;
    fn merklize(
        &self,
        num_cols: usize,
        hash_engine_id: EngineId,
        merkle_config: &merkle_tree::Config,
    ) -> (ActiveBuffer<Hash>, Hash)
    where
        T: Encodable + Send + Sync;
}

/// Read-only operations over a contiguous run of field elements — the buffer
/// analogue of `&[F]`.
///
/// Implemented by the owned buffers ([`CpuBuffer`]/`MetalBuffer`, as the
/// full-range case) and by the borrowed read views ([`CpuSlice`]/`MetalSlice`).
/// Every op here works identically on a whole buffer or any sub-range obtained
/// via [`Self::slice`]. On `MetalBuffer` a view aliases the parent's GPU
/// allocation (byte-offset binding), so no data is copied.
pub trait BufferRead<F: Field> {
    /// A same-backend owned buffer over another field, produced by the
    /// mixed-field ops (e.g. `mixed_dot` against a target-field buffer). For an
    /// owned buffer this is the buffer itself over `T`; views report the owning
    /// buffer type.
    type TargetBuffer<T: Field>: Buffer<T>;

    /// Read-only view type produced by slicing this buffer/view.
    type Slice<'a>: BufferRead<F>
    where
        Self: 'a,
        F: 'a;

    /// Number of elements in this view.
    fn read_len(&self) -> usize;
    fn read_is_empty(&self) -> bool {
        self.read_len() == 0
    }

    /// Inner product with another view of the same length.
    fn dot(&self, other: &Self) -> F
    where
        Self: Sized;

    /// Sumcheck round coefficients `(c0, c2)` for `dot(self, other)`.
    fn sumcheck_polynomial(&self, other: &Self) -> (F, F)
    where
        Self: Sized;

    /// Multilinear extension evaluated at a target-field point.
    fn mixed_extend<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        embedding: &M,
        point: &[M::Target],
    ) -> M::Target;

    /// Inner product with a target-field buffer.
    fn mixed_dot<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        embedding: &M,
        other: &Self::TargetBuffer<T>,
    ) -> M::Target;

    /// Univariate evaluation at a target-field point.
    fn mixed_univariate_evaluate<M: Embedding<Source = F>>(
        &self,
        embedding: &M,
        point: M::Target,
    ) -> M::Target;

    /// `accumulator += weight * self`, lifted into the target field.
    fn mixed_scalar_mul_add_to<M: Embedding<Source = F>>(
        &self,
        embedding: &M,
        accumulator: &mut Self::TargetBuffer<M::Target>,
        weight: M::Target,
    );

    /// Borrow `self[range]` as a read-only view (analogous to `&v[range]`).
    fn slice(&self, range: impl RangeBounds<usize>) -> Self::Slice<'_>;
}

/// In-place, length-preserving operations over a contiguous run of field
/// elements — the buffer analogue of `&mut [F]`.
///
/// Implemented by the owned buffers (full-range) and the borrowed mutable views
/// ([`CpuSliceMut`]/`MetalSliceMut`). A sub-range obtained via [`Self::slice_mut`]
/// is the same view type as the whole buffer, so the ops compose just like
/// slicing a `Vec`.
///
/// Constructors (`zeros`, `random`, …) and length-changing operations (`fold`,
/// `zero_pad`) live on [`Buffer`], the owned-buffer trait, because a borrowed
/// view cannot construct or resize its backing storage.
pub trait BufferWrite<F: Field>: BufferRead<F> {
    /// Mutable view type produced by slicing this buffer/view.
    type SliceMut<'a>: BufferWrite<F>
    where
        Self: 'a,
        F: 'a;

    /// In-place scalar multiplication: `self[i] *= weight`.
    fn scale(&mut self, weight: F);

    /// Accumulate `Σ_j scalars[j] · evaluators[j].point^i` into entry `i`.
    fn accumulate_univariate_evaluations(
        &mut self,
        evaluators: &[UnivariateEvaluation<F>],
        scalars: &[F],
    );

    /// Borrow `self[range]` as a mutable, zero-copy view (analogous to
    /// `&mut v[range]`).
    fn slice_mut(&mut self, range: impl RangeBounds<usize>) -> Self::SliceMut<'_>;

    /// Split into two disjoint mutable views `[0, mid)` and `[mid, len)`.
    fn split_at_mut(&mut self, mid: usize) -> (Self::SliceMut<'_>, Self::SliceMut<'_>);
}

/// Owned field-buffer operations — the `Vec` half of the field API.
///
/// `BufferOps<T>` stays generic over any element type, so hashes and digests can
/// use it. `BufferRead`/`BufferWrite` are the slice-like field ops implemented
/// by owners and views. This trait is only for owned field buffers: operations
/// here construct storage or change its length, so borrowed views cannot
/// implement them.
pub trait Buffer<F: Field>: BufferOps<F> + BufferWrite<F> + Clone {
    fn zeros(length: usize) -> Self;

    /// Geometric sequence `[1, base, base², …, base^(length-1)]`.
    fn geometric_sequence(base: F, length: usize) -> Self;

    fn random<R>(rng: &mut R, length: usize) -> Self
    where
        R: RngCore + CryptoRng,
        Standard: Distribution<F>;

    fn zero_pad(&mut self);
    fn fold(&mut self, weight: F);

    fn fold_pair(&mut self, other: &mut Self, weight: F) {
        self.fold(weight);
        other.fold(weight);
    }

    fn fold_pair_sumcheck_polynomial(&mut self, other: &mut Self, weight: F) -> (F, F) {
        self.fold_pair(other, weight);
        self.sumcheck_polynomial(other)
    }

    fn linear_forms_rlc(
        size: usize,
        linear_forms: &mut [Box<dyn LinearForm<F>>],
        rlc_coeffs: &[F],
    ) -> Self;

    fn mixed_linear_combination<M: Embedding<Source = F>>(
        embedding: &M,
        vectors: &[&Self],
        coeffs: &[M::Target],
    ) -> Self::TargetBuffer<M::Target>;
}

/// Resolve any range expression (`a..b`, `..b`, `a..`, `..`) against `len`,
/// returning `(start, end)` and bounds-checking like slice indexing.
pub(crate) fn resolve_range(range: impl RangeBounds<usize>, len: usize) -> (usize, usize) {
    use std::ops::Bound::{Excluded, Included, Unbounded};
    let start = match range.start_bound() {
        Included(&s) => s,
        Excluded(&s) => s + 1,
        Unbounded => 0,
    };
    let end = match range.end_bound() {
        Included(&e) => e + 1,
        Excluded(&e) => e,
        Unbounded => len,
    };
    assert!(
        start <= end && end <= len,
        "slice range {start}..{end} out of bounds for length {len}"
    );
    (start, end)
}

/// Read-only view into a [`CpuBuffer`], backed by a borrowed sub-slice.
pub struct CpuSlice<'a, F> {
    data: &'a [F],
}

/// Mutable view into a [`CpuBuffer`], backed by a borrowed sub-slice.
pub struct CpuSliceMut<'a, F> {
    data: &'a mut [F],
}

impl<F> CpuSlice<'_, F> {
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub fn as_slice(&self) -> &[F] {
        self.data
    }
}

impl<F> CpuSliceMut<'_, F> {
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub fn as_slice(&self) -> &[F] {
        self.data
    }
}

impl<F: Field> BufferRead<F> for CpuSlice<'_, F> {
    type TargetBuffer<T: Field> = CpuBuffer<T>;
    type Slice<'a>
        = CpuSlice<'a, F>
    where
        Self: 'a,
        F: 'a;

    fn read_len(&self) -> usize {
        self.data.len()
    }
    fn dot(&self, other: &Self) -> F {
        crate::algebra::dot(self.data, other.data)
    }
    fn sumcheck_polynomial(&self, other: &Self) -> (F, F) {
        compute_sumcheck_polynomial(self.data, other.data)
    }
    fn mixed_extend<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        embedding: &M,
        point: &[M::Target],
    ) -> M::Target {
        mixed_multilinear_extend(embedding, self.data, point)
    }
    fn mixed_dot<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        embedding: &M,
        other: &CpuBuffer<T>,
    ) -> M::Target {
        mixed_dot(embedding, other.as_slice(), self.data)
    }
    fn mixed_univariate_evaluate<M: Embedding<Source = F>>(
        &self,
        embedding: &M,
        point: M::Target,
    ) -> M::Target {
        mixed_univariate_evaluate(embedding, self.data, point)
    }
    fn mixed_scalar_mul_add_to<M: Embedding<Source = F>>(
        &self,
        embedding: &M,
        accumulator: &mut CpuBuffer<M::Target>,
        weight: M::Target,
    ) {
        mixed_scalar_mul_add(embedding, &mut accumulator.data, weight, self.data);
    }

    fn slice(&self, range: impl RangeBounds<usize>) -> CpuSlice<'_, F> {
        let (start, end) = resolve_range(range, self.data.len());
        CpuSlice {
            data: &self.data[start..end],
        }
    }
}

impl<F: Field> BufferRead<F> for CpuSliceMut<'_, F> {
    type TargetBuffer<T: Field> = CpuBuffer<T>;
    type Slice<'a>
        = CpuSlice<'a, F>
    where
        Self: 'a,
        F: 'a;

    fn read_len(&self) -> usize {
        self.data.len()
    }
    fn dot(&self, other: &Self) -> F {
        crate::algebra::dot(self.data, other.data)
    }
    fn sumcheck_polynomial(&self, other: &Self) -> (F, F) {
        compute_sumcheck_polynomial(self.data, other.data)
    }
    fn mixed_extend<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        embedding: &M,
        point: &[M::Target],
    ) -> M::Target {
        mixed_multilinear_extend(embedding, self.data, point)
    }
    fn mixed_dot<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        embedding: &M,
        other: &CpuBuffer<T>,
    ) -> M::Target {
        mixed_dot(embedding, other.as_slice(), self.data)
    }
    fn mixed_univariate_evaluate<M: Embedding<Source = F>>(
        &self,
        embedding: &M,
        point: M::Target,
    ) -> M::Target {
        mixed_univariate_evaluate(embedding, self.data, point)
    }
    fn mixed_scalar_mul_add_to<M: Embedding<Source = F>>(
        &self,
        embedding: &M,
        accumulator: &mut CpuBuffer<M::Target>,
        weight: M::Target,
    ) {
        mixed_scalar_mul_add(embedding, &mut accumulator.data, weight, self.data);
    }

    fn slice(&self, range: impl RangeBounds<usize>) -> CpuSlice<'_, F> {
        let (start, end) = resolve_range(range, self.data.len());
        CpuSlice {
            data: &self.data[start..end],
        }
    }
}

impl<F: Field> BufferWrite<F> for CpuSliceMut<'_, F> {
    type SliceMut<'a>
        = CpuSliceMut<'a, F>
    where
        Self: 'a,
        F: 'a;

    fn scale(&mut self, weight: F) {
        crate::algebra::scalar_mul(self.data, weight);
    }

    fn accumulate_univariate_evaluations(
        &mut self,
        evaluators: &[UnivariateEvaluation<F>],
        scalars: &[F],
    ) {
        UnivariateEvaluation::accumulate_many(evaluators, self.data, scalars);
    }

    fn slice_mut(&mut self, range: impl RangeBounds<usize>) -> CpuSliceMut<'_, F> {
        let (start, end) = resolve_range(range, self.data.len());
        CpuSliceMut {
            data: &mut self.data[start..end],
        }
    }

    fn split_at_mut(&mut self, mid: usize) -> (CpuSliceMut<'_, F>, CpuSliceMut<'_, F>) {
        let (lo, hi) = self.data.split_at_mut(mid);
        (CpuSliceMut { data: lo }, CpuSliceMut { data: hi })
    }
}

#[derive(
    Clone,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Debug,
    Default,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct CpuBuffer<T> {
    data: Vec<T>,
}

impl<T: Clone> CpuBuffer<T> {
    pub fn from_vec(source: Vec<T>) -> Self {
        Self { data: source }
    }

    pub fn from_slice(source: &[T]) -> Self {
        Self {
            data: Vec::from(source),
        }
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }
}

impl<T: Copy> BufferOps<T> for CpuBuffer<T> {
    fn at_index(&self, index: usize) -> Option<T> {
        if index >= self.len() {
            None
        } else {
            Some(self.data[index])
        }
    }

    fn as_slice(&self) -> &[T] {
        &self.data
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn read_rows(&self, num_cols: usize, indices: &[usize]) -> Vec<T> {
        let mut result = Vec::with_capacity(indices.len() * num_cols);
        for i in indices {
            result.extend_from_slice(&self.data[i * num_cols..(i + 1) * num_cols]);
        }
        result
    }

    fn extend(&self, other: &Self) -> Self {
        let mut data = Vec::with_capacity(self.data.len() + other.data.len());
        data.extend_from_slice(&self.data);
        data.extend_from_slice(&other.data);
        Self { data }
    }

    fn from_vec(source: Vec<T>) -> Self {
        Self::from_vec(source)
    }

    fn from_slice(source: &[T]) -> Self {
        Self::from_slice(source)
    }

    fn merklize(
        &self,
        num_cols: usize,
        hash_engine_id: EngineId,
        merkle_config: &merkle_tree::Config,
    ) -> (ActiveBuffer<Hash>, Hash)
    where
        T: Encodable + Send + Sync,
    {
        let _ = num_cols; // CPU leaf hashing derives the column count from the row count.
        let engine = hash::ENGINES
            .retrieve(hash_engine_id)
            .expect("Failed to retrieve hash engine");
        #[cfg(feature = "tracing")]
        tracing::Span::current().record("engine", engine.name().as_ref());

        // Hash each row into a leaf, then build the full Merkle node array.
        let mut leaves = vec![Hash::default(); merkle_config.num_leaves];
        crate::protocols::matrix_commit::hash_rows(&*engine, self.as_slice(), &mut leaves);
        let nodes = merkle_config.build_nodes(leaves);
        let root = nodes[nodes.len() - 1];

        (ActiveBuffer::<Hash>::from_vec(nodes), root)
    }
}

impl<F: Field> BufferRead<F> for CpuBuffer<F> {
    type TargetBuffer<T: Field> = CpuBuffer<T>;
    type Slice<'a>
        = CpuSlice<'a, F>
    where
        Self: 'a,
        F: 'a;

    fn read_len(&self) -> usize {
        self.data.len()
    }

    fn dot(&self, other: &Self) -> F {
        dot(&self.data, &other.data)
    }

    fn sumcheck_polynomial(&self, other: &Self) -> (F, F) {
        compute_sumcheck_polynomial(&self.data, other.as_slice())
    }

    fn mixed_extend<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        embedding: &M,
        point: &[M::Target],
    ) -> M::Target {
        mixed_multilinear_extend(embedding, &self.data, point)
    }

    fn mixed_dot<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        embedding: &M,
        other: &CpuBuffer<T>,
    ) -> M::Target {
        mixed_dot(embedding, other.as_slice(), self.as_slice())
    }

    fn mixed_univariate_evaluate<M: Embedding<Source = F>>(
        &self,
        embedding: &M,
        point: M::Target,
    ) -> M::Target {
        mixed_univariate_evaluate(embedding, &self.data, point)
    }

    fn mixed_scalar_mul_add_to<M: Embedding<Source = F>>(
        &self,
        embedding: &M,
        accumulator: &mut CpuBuffer<M::Target>,
        weight: M::Target,
    ) {
        mixed_scalar_mul_add(embedding, &mut accumulator.data, weight, self.as_slice());
    }

    fn slice(&self, range: impl RangeBounds<usize>) -> CpuSlice<'_, F> {
        let (start, end) = resolve_range(range, self.data.len());
        CpuSlice {
            data: &self.data[start..end],
        }
    }
}

impl<F: Field> BufferWrite<F> for CpuBuffer<F> {
    type SliceMut<'a>
        = CpuSliceMut<'a, F>
    where
        Self: 'a,
        F: 'a;

    fn scale(&mut self, weight: F) {
        crate::algebra::scalar_mul(&mut self.data, weight);
    }

    fn accumulate_univariate_evaluations(
        &mut self,
        evaluators: &[UnivariateEvaluation<F>],
        scalars: &[F],
    ) {
        UnivariateEvaluation::accumulate_many(evaluators, &mut self.data, scalars);
    }

    fn slice_mut(&mut self, range: impl RangeBounds<usize>) -> CpuSliceMut<'_, F> {
        let (start, end) = resolve_range(range, self.data.len());
        CpuSliceMut {
            data: &mut self.data[start..end],
        }
    }

    fn split_at_mut(&mut self, mid: usize) -> (CpuSliceMut<'_, F>, CpuSliceMut<'_, F>) {
        let (lo, hi) = self.data.split_at_mut(mid);
        (CpuSliceMut { data: lo }, CpuSliceMut { data: hi })
    }
}

impl<F: Field> Buffer<F> for CpuBuffer<F> {
    fn zeros(length: usize) -> Self {
        Self {
            data: vec![F::ZERO; length],
        }
    }

    fn geometric_sequence(base: F, length: usize) -> Self {
        Self::from_vec(crate::algebra::geometric_sequence(base, length))
    }

    fn random<R>(rng: &mut R, length: usize) -> Self
    where
        R: RngCore + CryptoRng,
        Standard: Distribution<F>,
    {
        Self {
            data: (0..length).map(|_| rng.gen()).collect(),
        }
    }

    fn linear_forms_rlc(
        size: usize,
        linear_forms: &mut [Box<dyn LinearForm<F>>],
        rlc_coeffs: &[F],
    ) -> Self {
        assert_eq!(linear_forms.len(), rlc_coeffs.len());
        let mut covector = vec![F::ZERO; size];
        if let Some((first, linear_forms)) = linear_forms.split_first_mut() {
            debug_assert_eq!(rlc_coeffs[0], F::ONE);
            if let Some(covector_form) =
                (first.as_mut() as &mut dyn Any).downcast_mut::<Covector<F>>()
            {
                mem::swap(&mut covector, &mut covector_form.vector);
            } else {
                first.accumulate(&mut covector, F::ONE);
            }
            for (rlc_coeff, linear_form) in rlc_coeffs[1..].iter().zip(linear_forms) {
                linear_form.accumulate(&mut covector, *rlc_coeff);
            }
        }
        Self { data: covector }
    }

    fn zero_pad(&mut self) {
        if !self.is_empty() {
            self.data.resize(self.len().next_power_of_two(), F::ZERO);
        }
    }

    fn fold(&mut self, weight: F) {
        fold(&mut self.data, weight);
    }

    fn fold_pair(&mut self, other: &mut Self, weight: F) {
        self.fold(weight);
        other.fold(weight);
    }

    fn fold_pair_sumcheck_polynomial(&mut self, other: &mut Self, weight: F) -> (F, F) {
        self.fold_pair(other, weight);
        self.sumcheck_polynomial(other)
    }

    fn mixed_linear_combination<M: Embedding<Source = F>>(
        embedding: &M,
        vectors: &[&Self],
        coeffs: &[M::Target],
    ) -> Self::TargetBuffer<M::Target> {
        assert_eq!(vectors.len(), coeffs.len());
        let Some((first, vectors)) = vectors.split_first() else {
            return CpuBuffer::from_vec(Vec::new());
        };
        debug_assert_eq!(coeffs[0], M::Target::ONE);
        let mut accumulator = crate::algebra::lift(embedding, first.as_slice());
        for (coeff, vector) in coeffs[1..].iter().zip(vectors) {
            mixed_scalar_mul_add(embedding, &mut accumulator, *coeff, vector.as_slice());
        }
        CpuBuffer::from_vec(accumulator)
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::AdditiveGroup;

    use super::*;
    use crate::algebra::{
        fields::Field64, geometric_accumulate, geometric_sequence as geometric_sequence_vec,
    };

    type F = Field64;

    #[test]
    fn geometric_sequence_matches_free_function() {
        let base = F::from(7u64);
        for length in 0..6 {
            let buffer = CpuBuffer::<F>::geometric_sequence(base, length);
            assert_eq!(
                BufferOps::as_slice(&buffer),
                geometric_sequence_vec(base, length).as_slice(),
                "geometric_sequence mismatch at length {length}"
            );
        }
    }

    #[test]
    fn scale_multiplies_in_place() {
        let values = vec![F::from(1u64), F::from(2u64), F::from(3u64), F::from(4u64)];
        let weight = F::from(5u64);
        let mut buffer = CpuBuffer::from_vec(values.clone());
        buffer.scale(weight);
        let expected: Vec<F> = values.iter().map(|&v| v * weight).collect();
        assert_eq!(BufferOps::as_slice(&buffer), expected.as_slice());
    }

    #[test]
    fn slice_accumulate_matches_restricted_geometric_accumulate() {
        let len = 8usize;
        let points = vec![F::from(3u64), F::from(5u64)];
        let scalars = vec![F::from(2u64), F::from(9u64)];

        // Try a prefix slice (offset 0) and an interior slice (offset > 0).
        for (offset, window_len) in [(0usize, 5usize), (2usize, 4usize)] {
            let mut buffer = CpuBuffer::from_vec(vec![F::ZERO; len]);
            let evaluators: Vec<_> = points
                .iter()
                .map(|&point| UnivariateEvaluation::new(point, window_len))
                .collect();
            buffer
                .slice_mut(offset..offset + window_len)
                .accumulate_univariate_evaluations(&evaluators, &scalars);

            // Reference: accumulate Σ_j scalars[j]·points[j]^i into the same
            // sub-slice of a plain vector.
            let mut reference = vec![F::ZERO; len];
            geometric_accumulate(
                &mut reference[offset..offset + window_len],
                scalars.clone(),
                &points,
            );

            assert_eq!(
                BufferOps::as_slice(&buffer),
                reference.as_slice(),
                "slice accumulate mismatch at offset {offset}, len {window_len}"
            );
        }
    }

    #[test]
    fn slice_scale_multiplies_only_the_range() {
        let values = vec![F::from(1u64), F::from(2u64), F::from(3u64), F::from(4u64)];
        let weight = F::from(5u64);
        let mut buffer = CpuBuffer::from_vec(values.clone());
        buffer.slice_mut(1..3).scale(weight);
        let expected = vec![values[0], values[1] * weight, values[2] * weight, values[3]];
        assert_eq!(BufferOps::as_slice(&buffer), expected.as_slice());
    }
}
