use ark_ff::Field;

use super::LinearForm;
use crate::algebra::{
    embedding::{Embedding, PublicScalar},
    linear_form::Evaluate,
    mixed_univariate_evaluate,
};

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
        if self.size == 0 {
            return F::ZERO;
        }
        let dimensions = (usize::BITS - (self.size - 1).leading_zeros()) as usize;
        assert!(point.len() >= dimensions);
        let (padding, point) = point.split_at(point.len() - dimensions);
        let truncated = !self.size.is_power_of_two();
        let mut bound = self.size;
        let mut x2i = PublicScalar::new(self.point);
        let mut full = F::ONE;
        let mut prefix = F::ZERO;
        // Coordinates are most significant first; build from the least significant bit.
        for &r in point.iter().rev() {
            let zero = F::ONE - r;
            let one = x2i.mul(r);
            if truncated {
                // A set bound bit admits the full zero branch and a truncated one branch.
                prefix = if bound & 1 == 0 {
                    zero * prefix
                } else {
                    zero * full + one * prefix
                };
                bound >>= 1;
            }
            full *= zero + one;
            x2i.square();
        }
        let result = if truncated { prefix } else { full };
        padding
            .iter()
            .fold(result, |value, &r| value * (F::ONE - r))
    }

    fn accumulate(&self, accumulator: &mut [F], scalar: F) {
        assert_eq!(accumulator.len(), self.size);
        let point = PublicScalar::new(self.point);
        let mut power = scalar;
        for entry in accumulator {
            *entry += power;
            power = point.mul(power);
        }
    }
}

impl<M: Embedding> Evaluate<M> for UnivariateEvaluation<M::Target> {
    fn evaluate(&self, embedding: &M, vector: &[M::Source]) -> M::Target {
        mixed_univariate_evaluate(embedding, vector, self.point)
    }
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{
        distributions::Standard, prelude::Distribution, rngs::StdRng, Rng, SeedableRng,
    };

    use super::*;
    use crate::algebra::{
        fields::{Field64, Field64_2, Field64_3},
        linear_form::Covector,
        random_vector,
    };

    fn compare_with_dense<F: Field>()
    where
        Standard: Distribution<F>,
    {
        let mut rng = StdRng::seed_from_u64(47);
        for dimensions in 0..=6 {
            let points = [
                vec![F::ZERO; dimensions],
                vec![F::ONE; dimensions],
                random_vector(&mut rng, dimensions),
            ];
            for size in 0..=1 << dimensions {
                for x in [F::ZERO, F::ONE, F::from(7), rng.gen()] {
                    let form = UnivariateEvaluation::new(x, size);
                    let dense = Covector::from(&form);
                    for point in &points {
                        assert_eq!(
                            form.mle_evaluate(point),
                            dense.mle_evaluate(point),
                            "size={size}, dimensions={dimensions}, x={x}, point={point:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn truncated_geometric_covectors_match_dense_evaluation() {
        compare_with_dense::<Field64>();
        compare_with_dense::<Field64_2>();
        compare_with_dense::<Field64_3>();
    }
}
