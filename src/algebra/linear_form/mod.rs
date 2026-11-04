//! Linear Forms that committed vectors can be opened against.

mod covector;
mod multilinear_extension;
mod univariate_evaluation;

use std::any::Any;

use ark_ff::Field;
use static_assertions::assert_obj_safe;

pub use self::{
    covector::Covector, multilinear_extension::MultilinearExtension,
    univariate_evaluation::UnivariateEvaluation,
};
use crate::algebra::{
    embedding::{self, Embedding},
    fields,
};

/// Represents a linear function $𝔽^n → 𝔽$ used in WHIR openings.
///
/// Note that the trait does not contain a method to actually evaluate the linear form, for that
/// see the [`Evaluate`] trait.
///
/// The `Any` supertrait enables downcasting concrete types (e.g. [`Covector`]) from
/// `dyn LinearForm<F>`, which the prover uses to recycle covector buffers.
pub trait LinearForm<F: Field>: Any + Send + Sync {
    /// The dimension of the domain of this linear form.
    fn size(&self) -> usize;

    /// Evaluate the linear form as a multi-linear extension in a random point.
    ///
    /// Specifically the computed value should be the linear functional applied to vector given by
    /// the tensor product $a = a_0 ⊗ a_1 ⊗ ⋯ ⊗ a_(k-1)$ where $k ≥ ⌈log_2 size⌉$ and
    /// $a_i = (1-point_i, point_i)$ truncated to `self.size()`.
    ///
    /// This function can be used by the verifier to check the [`FinalClaim`].
    fn mle_evaluate(&self, point: &[F]) -> F;

    /// Accumulate the covector representation of the linear form.
    ///
    /// Take $w ∈ 𝔽^n$ such that evaluating the linear form on $v ∈ 𝔽^n$ equals the inner
    /// product $⟨w,v⟩$. Then this function computes $accumulator_i += scalar · w_i$.
    ///
    /// Used by the prover and by batched verifier evaluations of custom forms.
    fn accumulate(&self, accumulator: &mut [F], scalar: F);
}

/// A linear form that can be evaluated on a subfield vector.
///
/// In particular [`Evaluate<Identity<F>>`] allows evaluating the linear form
/// on its native field.
pub trait Evaluate<M: Embedding>: LinearForm<M::Target> {
    /// Evaluate the linear form on a subfield vector.
    ///
    /// - `self` is a linear form $𝔽^n → 𝔽$.
    /// - `embedding` is an embedding $𝔾 → 𝔽$.
    /// - `vector` is a vector in $𝔾^n$.
    ///
    fn evaluate(&self, embedding: &M, vector: &[M::Source]) -> M::Target;
}

assert_obj_safe!(LinearForm<fields::Field64>);

assert_obj_safe!(Evaluate<embedding::Basefield<fields::Field64_2>>);
