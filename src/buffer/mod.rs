//! Backend-agnostic buffers for protocol data.
//!
//! Protocol code uses [`ActiveBuffer`], [`ActiveSlice`], and [`ActiveSliceMut`]
//! to select the CPU or GPU backend at compile time. Owned buffers manage
//! storage; slices are zero-copy views into a contiguous range.
//!
//! The trait split follows the ownership model. [`BufferOps`] is generic over
//! any element type and is used for field elements, [`struct@Hash`] digests, and
//! Merkle nodes. [`BufferRead`] and [`BufferWrite`] provide field operations for
//! owned buffers and views. [`Buffer`] contains owned-only field operations such
//! as construction, folding, and zero-padding.
//!
//! [`DefaultRs`] selects the Reed-Solomon encoder for the active backend.

pub mod cpu;
use std::ops::RangeBounds;

use ark_ff::Field;
use ark_std::rand::{
    distributions::{Distribution, Standard},
    CryptoRng, RngCore,
};

use crate::{
    algebra::{
        embedding::Embedding,
        linear_form::{LinearForm, UnivariateEvaluation},
    },
    engines::EngineId,
    hash::Hash,
    protocols::{matrix_commit::Encodable, merkle_tree},
};

pub use cpu::{CpuBuffer, CpuSlice, CpuSliceMut};

pub type ActiveBuffer<T> = CpuBuffer<T>;
pub type ActiveSlice<'a, T> = CpuSlice<'a, T>;
pub type ActiveSliceMut<'a, T> = CpuSliceMut<'a, T>;
pub type DefaultRs<T> = crate::algebra::ntt::NttEngine<T>;

/// Owned buffer operations over any element type.
///
/// This trait is not field-specific, so it also covers hash buffers and Merkle
/// node layers. Field arithmetic lives on [`BufferRead`] and [`BufferWrite`].
pub trait BufferOps<T> {
    /// Same-backend buffer type used for Merkle tree nodes.
    type Nodes: BufferOps<Hash>;

    fn from_vec(source: Vec<T>) -> Self;
    fn from_slice(source: &[T]) -> Self;
    fn as_slice(&self) -> &[T];
    /// Number of rows when the buffer is laid out with `num_cols` columns.
    fn num_rows(&self, num_cols: usize) -> usize {
        self.len() / num_cols
    }
    /// Gather full rows `indices[i] * num_cols .. (indices[i] + 1) * num_cols`.
    fn read_rows(&self, num_cols: usize, indices: &[usize]) -> Vec<T>;
    fn at_index(&self, index: usize) -> Option<T>;
    /// Gather elements at arbitrary indices.
    fn gather_at_indices(&self, indices: &[usize]) -> Vec<T>
    where
        T: Copy;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Hash rows of width `num_cols` and build a Merkle tree.
    fn merklize(
        &self,
        num_cols: usize,
        leaf_hash: EngineId,
        merkle: &merkle_tree::Config,
    ) -> (Self::Nodes, Hash)
    where
        T: Encodable + Send + Sync;
}

/// Read-only operations over a contiguous run of field elements.
///
/// Implemented by owned buffers and borrowed views. A view produced by
/// [`Self::slice`] aliases the original storage.
pub trait BufferRead<F: Field> {
    /// Same-backend owned buffer over another field.
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

    /// Copy this view into a new owned buffer.
    fn copy_to_owned(&self) -> Self::TargetBuffer<F>;
}

/// In-place operations over a contiguous run of field elements.
///
/// Implemented by owned buffers and mutable views. A view produced by
/// [`Self::slice_mut`] aliases the original storage.
///
/// Construction and length-changing operations live on [`Buffer`], because
/// borrowed views cannot construct or resize their backing storage.
pub trait BufferWrite<F: Field>: BufferRead<F> {
    /// Mutable view type produced by slicing this buffer/view.
    type SliceMut<'a>: BufferWrite<F>
    where
        Self: 'a,
        F: 'a;

    /// In-place scalar multiplication: `self[i] *= weight`.
    fn scalar_mul(&mut self, weight: F);

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

/// Owned field-buffer operations.
///
/// This trait contains operations that construct storage or change its length,
/// so it is implemented only by owned buffers.
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
