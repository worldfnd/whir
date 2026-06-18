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

pub trait BufferOps<T> {
    type Nodes: BufferOps<Hash>;

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
    fn merklize(
        &self,
        num_cols: usize,
        leaf_hash: EngineId,
        merkle: &merkle_tree::Config,
    ) -> (Self::Nodes, Hash)
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
