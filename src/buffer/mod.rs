//! Backend-agnostic buffers for protocol data.
//!
//! Protocol code uses the [`Buffer`] alias to select the backend at
//! compile time. Buffers are owned, backend-managed storage: on the CPU
//! backend they wrap a `Vec`, on an accelerator backend they would own
//! device memory and only [`BufferOps::to_slice`] (and the other readback
//! methods) force a host copy.
//!
//! [`BufferOps`] covers host-to-backend and backend-to-host communication,
//! while [`Buffer`] provides backend-specific field arithmetic. The traits
//! are independent so callers can require only the capabilities they use.
//!
//! [`DefaultRs`] selects the Reed-Solomon encoder for the active backend.

pub mod cpu;

use ark_ff::Field;
use ark_std::rand::{
    distributions::{Distribution, Standard},
    CryptoRng, RngCore,
};
pub use cpu::CpuBuffer;

use crate::algebra::{
    embedding::Embedding,
    linear_form::{LinearForm, UnivariateEvaluation},
};

pub type Buffer<T> = CpuBuffer<T>;
pub type DefaultRs<T> = crate::algebra::ntt::NttEngine<T>;

/// Host communication for owned buffers over any copyable element type.
///
/// Construction uses the standard [`From`] implementations of each backend.
/// Field arithmetic lives independently on [`Buffer`].
pub trait BufferOps<T: Copy> {
    /// Read back the buffer contents as a host slice.
    fn to_slice(&self) -> &[T];
    /// Consume the buffer and return its contents as an owned host `Vec`.
    ///
    /// The dual of the `From<Vec<T>>` constructor: on the CPU backend this is a
    /// zero-copy move of the backing storage; an accelerator backend would copy
    /// device memory back to the host once.
    fn into_vec(self) -> Vec<T>
    where
        Self: Sized;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Gather full rows `indices[i] * num_cols .. (indices[i] + 1) * num_cols`.
    fn read_rows(&self, num_cols: usize, indices: &[usize]) -> Vec<T>;
    /// Gather elements at arbitrary indices.
    fn gather_at_indices(&self, indices: &[usize]) -> Vec<T>;
    fn get(&self, index: usize) -> Option<&T>;

    /// Best-effort in-place zeroization of the buffer's contents.
    ///
    /// Used to scrub secret material (blinding masks, witness randomness)
    /// before the buffer is dropped. On the CPU backend this zeroizes the
    /// backing storage; accelerator backends would override with a device wipe.
    fn wipe(&mut self)
    where
        T: zeroize::Zeroize;
}

/// Field operations on owned buffers.
pub trait BufferMath<F: Field>: Clone {
    /// Same-backend owned buffer over another field.
    ///
    /// Used by the `mixed_*` operations that lift base-field data into an
    /// extension field through an [`Embedding`].
    type TargetBuffer<T: Field>: BufferMath<T>;

    fn zeros(length: usize) -> Self;

    /// Buffer of `length` copies of `F::ONE`, filled on the backend.
    ///
    /// Prefer this over `from(vec![F::ONE; length])`, which uploads a host
    /// buffer to the backend.
    fn ones(length: usize) -> Self;

    fn random<R>(rng: &mut R, length: usize) -> Self
    where
        R: RngCore + CryptoRng,
        Standard: Distribution<F>;

    /// Inner product with another buffer of the same length.
    fn dot(&self, other: &Self) -> F;

    /// Bilinear form over `self`, a row-major matrix with `rows.len()` rows
    /// and `cols.len()` columns:
    /// `Σ_i rows[i] · Σ_j cols[j] · self[i · cols.len() + j]`.
    ///
    /// Equivalently `rowsᵀ · self · cols`. Generalizes [`Buffer::dot`] to a
    /// weighted reduction of a matrix; `dot` is the special case where `self`
    /// is the identity.
    fn bilinear_form(&self, rows: &Self, cols: &Self) -> F;

    /// Tensor (outer) product `self ⊗ other`, row-major: length
    /// `self.len() · other.len()` with entry `[i · other.len() + j] = self[i] · other[j]`.
    #[must_use]
    fn tensor_product(&self, other: &Self) -> Self;

    /// Matrix-vector product. `self` is a row-major matrix with `vector.len()`
    /// columns; returns a buffer of length `self.len() / vector.len()` where
    /// `out[i] = dot(row_i, vector)`. `vector` must be non-empty.
    #[must_use]
    fn mat_vec(&self, vector: &Self) -> Self;

    /// Concatenation `[self, other]` into a single buffer of length
    /// `self.len() + other.len()`.
    #[must_use]
    fn concat(&self, other: &Self) -> Self;

    /// Equality-polynomial weights `eq(point, ·)` over the Boolean hypercube
    /// `{0,1}^{point.len()}`, as a buffer of length `1 << point.len()`.
    ///
    /// `point` holds host-side transcript challenges; the weights live on the
    /// backend so downstream reductions never force a readback.
    fn eq_weights(point: &[F]) -> Self;

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
        scalars: &Self,
    );

    /// Random linear combination of linear forms into a covector buffer.
    fn linear_forms_rlc(
        size: usize,
        linear_forms: &mut [Box<dyn LinearForm<F>>],
        rlc_coeffs: &Self,
    ) -> Self;

    fn geometric_challenge<G: Field>(current: G, base: G, length: usize) -> Self::TargetBuffer<G>;

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

    /// Random linear combination of vectors, lifted into the target field.
    fn mixed_linear_combination<M: Embedding<Source = F>>(
        embedding: &M,
        vectors: &[&Self],
        coeffs: &Self::TargetBuffer<M::Target>,
    ) -> Self::TargetBuffer<M::Target>;
}
