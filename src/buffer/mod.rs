//! Backend-agnostic buffers for protocol data.
//!
//! Protocol code uses the [`ActiveBuffer`] alias to select the backend at
//! compile time. Buffers are owned, backend-managed storage: on the CPU
//! backend they wrap a `Vec`, on an accelerator backend they would own
//! device memory and only [`BufferOps::as_slice`] (and the other readback
//! methods) force a host copy.
//!
//! The trait split follows the element type. [`BufferOps`] is generic over
//! any element and also covers [`struct@Hash`] buffers for Merkle tree
//! nodes. [`Buffer`] adds field arithmetic used by the protocols.
//!
//! [`DefaultRs`] selects the Reed-Solomon encoder for the active backend.

pub mod cpu;

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

pub use cpu::CpuBuffer;

pub type ActiveBuffer<T> = CpuBuffer<T>;
pub type DefaultRs<T> = crate::algebra::ntt::NttEngine<T>;

/// Owned buffer operations over any element type.
///
/// This trait is not field-specific, so it also covers hash buffers and
/// Merkle node layers. Field arithmetic lives on [`Buffer`].
pub trait BufferOps<T: Copy> {
    /// Same-backend buffer type used for Merkle tree nodes.
    type Nodes: BufferOps<Hash>;

    fn from_vec(source: Vec<T>) -> Self;
    fn from_slice(source: &[T]) -> Self;
    /// Read back the buffer contents as a host slice.
    fn as_slice(&self) -> &[T];
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Gather full rows `indices[i] * num_cols .. (indices[i] + 1) * num_cols`.
    fn read_rows(&self, num_cols: usize, indices: &[usize]) -> Vec<T>;
    /// Gather elements at arbitrary indices.
    fn gather_at_indices(&self, indices: &[usize]) -> Vec<T>;
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

/// Field operations on owned buffers.
pub trait Buffer<F: Field>: BufferOps<F> + Clone {
    /// Same-backend owned buffer over another field.
    ///
    /// Used by the `mixed_*` operations that lift base-field data into an
    /// extension field through an [`Embedding`].
    type TargetBuffer<T: Field>: Buffer<T>;

    fn zeros(length: usize) -> Self;

    fn random<R>(rng: &mut R, length: usize) -> Self
    where
        R: RngCore + CryptoRng,
        Standard: Distribution<F>;

    /// Inner product with another buffer of the same length.
    fn dot(&self, other: &Self) -> F;

    /// Sumcheck round coefficients `(c0, c2)` for `dot(self, other)`.
    fn sumcheck_polynomial(&self, other: &Self) -> (F, F);

    fn fold(&mut self, weight: F);

    fn fold_pair(&mut self, other: &mut Self, weight: F) {
        self.fold(weight);
        other.fold(weight);
    }

    fn fold_pair_sumcheck_polynomial(&mut self, other: &mut Self, weight: F) -> (F, F) {
        self.fold_pair(other, weight);
        self.sumcheck_polynomial(other)
    }

    /// In-place scalar multiplication: `self[i] *= weight`.
    fn scalar_mul(&mut self, weight: F);

    /// Accumulate `Σ_j scalars[j] · evaluators[j].point^i` into entry `i`.
    ///
    /// The evaluators must share a common size `s ≤ self.len()`; only the
    /// first `s` entries are updated. This allows accumulating constraints
    /// that cover a prefix of the buffer (e.g. the unmasked message part of
    /// a covector).
    fn accumulate_univariate_evaluations(
        &mut self,
        evaluators: &[UnivariateEvaluation<F>],
        scalars: &[F],
    );

    /// Univariate evaluation at a target-field point.
    fn mixed_univariate_evaluate<M: Embedding<Source = F>>(
        &self,
        embedding: &M,
        point: M::Target,
    ) -> M::Target;

    /// Inner product with a target-field buffer.
    fn mixed_dot<M: Embedding<Source = F>>(
        &self,
        embedding: &M,
        other: &Self::TargetBuffer<M::Target>,
    ) -> M::Target;

    /// `accumulator += weight * self`, lifted into the target field.
    fn mixed_scalar_mul_add_to<M: Embedding<Source = F>>(
        &self,
        embedding: &M,
        accumulator: &mut Self::TargetBuffer<M::Target>,
        weight: M::Target,
    );

    /// Random linear combination of linear forms into a covector buffer.
    fn linear_forms_rlc(
        size: usize,
        linear_forms: &mut [Box<dyn LinearForm<F>>],
        rlc_coeffs: &[F],
    ) -> Self;

    /// Random linear combination of vectors, lifted into the target field.
    fn mixed_linear_combination<M: Embedding<Source = F>>(
        embedding: &M,
        vectors: &[&Self],
        coeffs: &[M::Target],
    ) -> Self::TargetBuffer<M::Target>;
}
