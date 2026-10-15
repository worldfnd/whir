use ark_ff::Field;

use super::{Evaluate, LinearForm};
use crate::{
    algebra::{eval_eq, mixed_multilinear_extend, Embedding},
    utils::zip_strict,
};

/// Multilinear extension evaluation as a linear form $𝔽^n → 𝔽$.
///
/// Given a multilinear function $f ∈ 𝔽^(≤ 1)[X_0,…,X_(k-1)]$ represented by a vector $v ∈ 𝔽^n$ with
/// $n = 2^k$ using the boolean hypercube evaluation basis such that $v_i = f( bits(i) )$ where
/// $bits: ℕ → {0,1}^k$ is the little-endian binary decomposition, then this linear form will
/// evaluate to $f(x)$ for some fixed point $x ∈ 𝔽^k$.
///
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct MultilinearExtension<F: Field> {
    pub point: Vec<F>,
}

impl<F: Field> MultilinearExtension<F> {
    pub const fn new(point: Vec<F>) -> Self {
        Self { point }
    }
}

impl<F: Field> LinearForm<F> for MultilinearExtension<F> {
    fn size(&self) -> usize {
        1 << self.point.len()
    }

    fn mle_evaluate(&self, point: &[F]) -> F {
        zip_strict(&self.point, point).fold(F::ONE, |acc, (&l, &r)| {
            let factor = if r.is_zero() {
                F::ONE - l
            } else if r.is_one() {
                l
            } else {
                // l*r + (1-l)*(1-r) = (2*l-1)*r + (1-l).
                (l.double() - F::ONE) * r + (F::ONE - l)
            };
            acc * factor
        })
    }

    fn accumulate(&self, accumulator: &mut [F], scalar: F) {
        eval_eq(accumulator, &self.point, scalar);
    }
}

impl<M: Embedding> Evaluate<M> for MultilinearExtension<M::Target> {
    fn evaluate(&self, embedding: &M, vector: &[M::Source]) -> M::Target {
        mixed_multilinear_extend(embedding, vector, &self.point)
    }
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{
        distributions::Standard, prelude::Distribution, rngs::StdRng, SeedableRng,
    };

    use super::*;
    use crate::algebra::{
        fields::{Field256, Field64, Field64_3},
        linear_form::Covector,
        random_vector,
    };

    fn check_evaluation<F: Field>()
    where
        Standard: Distribution<F>,
    {
        let mut rng = StdRng::seed_from_u64(81);
        for dimensions in 0..=8 {
            for point in [
                vec![F::ZERO; dimensions],
                vec![F::ONE; dimensions],
                random_vector::<F>(&mut rng, dimensions),
            ] {
                let form = MultilinearExtension::new(point);
                let dense = Covector::from(&form);
                let mut mixed = random_vector::<F>(&mut rng, dimensions);
                for (i, coordinate) in mixed.iter_mut().enumerate() {
                    match i % 3 {
                        0 => *coordinate = F::ZERO,
                        1 => *coordinate = F::ONE,
                        _ => {}
                    }
                }
                for evaluation in [
                    vec![F::ZERO; dimensions],
                    vec![F::ONE; dimensions],
                    random_vector::<F>(&mut rng, dimensions),
                    mixed,
                ] {
                    assert_eq!(
                        form.mle_evaluate(&evaluation),
                        dense.mle_evaluate(&evaluation)
                    );
                }
            }
        }
    }

    #[test]
    fn boolean_and_random_coordinates_match_dense_evaluation() {
        check_evaluation::<Field64>();
        check_evaluation::<Field64_3>();
        check_evaluation::<Field256>();
    }
}
