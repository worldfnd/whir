use ark_ff::Field;

use super::LinearForm;
use crate::algebra::{embedding::Embedding, linear_form::Evaluate, mixed_univariate_evaluate};

/// Linear form to represent univariate polynomial evaluation.
///
/// Given a vector $v ∈ 𝔽^n$ it computes $sum_i v_i · x^i$ for some fixed $x$.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct UnivariateEvaluation<F: Field> {
    /// Univariate evaluation doesn't have an inherent size, so we need to store one.
    pub size: usize,

    /// The point $x ∈ 𝔽$ to evaluate on.
    pub point: F,
}

impl<F: Field> UnivariateEvaluation<F> {
    pub const fn new(point: F, size: usize) -> Self {
        Self { size, point }
    }
}

impl<F: Field> LinearForm<F> for UnivariateEvaluation<F> {
    fn size(&self) -> usize {
        self.size
    }

    fn mle_evaluate(&self, point: &[F]) -> F {
        // Multilinear extension of (1, x, x^2, ..) = ⨂_i (1, x^2^i).
        let mut x2i = self.point;
        let mut result = F::ONE;
        for &r in point.iter().rev() {
            // TODO: Why rev?
            result *= (F::ONE - r) + r * x2i;
            x2i.square_in_place();
        }
        result
    }

    fn accumulate(&self, accumulator: &mut [F], scalar: F) {
        assert_eq!(accumulator.len(), self.size);
        let mut power = scalar;
        for entry in accumulator {
            *entry += power;
            power *= self.point;
        }
    }
}

impl<M: Embedding> Evaluate<M> for UnivariateEvaluation<M::Target> {
    fn evaluate(&self, embedding: &M, vector: &[M::Source]) -> M::Target {
        mixed_univariate_evaluate(embedding, vector, self.point)
    }
}
