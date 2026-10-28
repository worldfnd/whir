pub mod embedding;
pub mod fields;
pub mod linear_form;
mod multilinear;
pub mod ntt;
pub mod sumcheck;

use ark_ff::{AdditiveGroup, Field};
use ark_std::rand::{distributions::Standard, prelude::Distribution, Rng};
pub use multilinear::{eq_weights, eval_eq, mixed_multilinear_extend, multilinear_extend};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use self::embedding::Embedding;
#[cfg(feature = "parallel")]
use crate::utils::workload_size;
use crate::utils::zip_strict;

pub fn random_vector<F: Field>(mut rng: impl Rng, length: usize) -> Vec<F>
where
    Standard: Distribution<F>,
{
    (0..length).map(|_| rng.gen()).collect::<Vec<F>>()
}

pub fn geometric_sequence<F: Field>(current: F, base: F, length: usize) -> Vec<F> {
    let mut result = Vec::with_capacity(length);
    let mut current = current;
    for _ in 0..length {
        result.push(current);
        current *= base;
    }
    result
}

pub fn dot<F: Field>(a: &[F], b: &[F]) -> F {
    mixed_dot(&embedding::Identity::new(), a, b)
}

pub fn tensor_product<F: Field>(a: &[F], b: &[F]) -> Vec<F> {
    let mut result = Vec::with_capacity(a.len() * b.len());
    for &x in a {
        for &y in b {
            result.push(x * y);
        }
    }
    result
}

/// Lift a vector to an embedding.
pub fn lift<M: Embedding>(embedding: &M, source: &[M::Source]) -> Vec<M::Target> {
    #[cfg(not(feature = "parallel"))]
    let result = source.iter().map(|c| embedding.map(*c)).collect();

    #[cfg(feature = "parallel")]
    let result = source.par_iter().map(|c| embedding.map(*c)).collect();

    result
}

pub fn scalar_mul<F: Field>(vector: &mut [F], weight: F) {
    for value in vector.iter_mut() {
        *value *= weight;
    }
}

/// Scalar mul add into new vector.
///
/// Returns r such that r[i] = a[i] + c · b[i].
pub fn scalar_mul_add_new<F: Field>(a: &[F], c: F, b: &[F]) -> Vec<F> {
    zip_strict(a.iter(), b.iter())
        .map(|(a, b)| *a + c * *b)
        .collect::<Vec<F>>()
}

pub fn scalar_mul_add<F: Field>(accumulator: &mut [F], weight: F, vector: &[F]) {
    mixed_scalar_mul_add(
        &embedding::Identity::<F>::new(),
        accumulator,
        weight,
        vector,
    );
}

/// Mixed scalar-mul add
///
/// `accumulator[i] += weight * vector[i]`
pub fn mixed_scalar_mul_add<M: Embedding>(
    embedding: &M,
    accumulator: &mut [M::Target],
    weight: M::Target,
    vector: &[M::Source],
) {
    assert_eq!(accumulator.len(), vector.len());
    for (accumulator, value) in accumulator.iter_mut().zip(vector) {
        *accumulator += embedding.mixed_mul(weight, *value);
    }
}

pub fn univariate_evaluate<F: Field>(coefficients: &[F], point: F) -> F {
    mixed_univariate_evaluate(&embedding::Identity::new(), coefficients, point)
}

/// Mixed field univariate Horner evaluation.
pub fn mixed_univariate_evaluate<M: Embedding>(
    embedding: &M,
    coefficients: &[M::Source],
    point: M::Target,
) -> M::Target {
    #[cfg(feature = "parallel")]
    if coefficients.len() > workload_size::<M::Source>() {
        let half = coefficients.len() / 2;
        let (low, high) = coefficients.split_at(half);
        let (low, high) = rayon::join(
            || mixed_univariate_evaluate(embedding, low, point),
            || mixed_univariate_evaluate(embedding, high, point),
        );
        return low + high * point.pow([half as u64]);
    }

    let Some(mut acc) = coefficients.last().map(|c| embedding.map(*c)) else {
        return M::Target::ZERO;
    };
    for &c in coefficients.iter().rev().skip(1) {
        acc *= point;
        acc = embedding.mixed_add(acc, c);
    }
    acc
}

pub fn mixed_dot<F: Field, G: Field>(
    embedding: &impl Embedding<Source = F, Target = G>,
    a: &[G],
    b: &[F],
) -> G {
    assert_eq!(a.len(), b.len());

    #[cfg(feature = "parallel")]
    if a.len() > workload_size::<G>() {
        return a
            .par_iter()
            .zip(b)
            .map(|(a, b)| embedding.mixed_mul(*a, *b))
            .sum();
    }

    a.iter()
        .zip(b)
        .map(|(a, b)| embedding.mixed_mul(*a, *b))
        .sum()
}

/// Compute `accumulator[i] += sum_j scalars[j] * points[j]^i`
pub fn geometric_accumulate<F: Field>(accumulator: &mut [F], mut scalars: Vec<F>, points: &[F]) {
    assert_eq!(scalars.len(), points.len());
    geometric_accumulate_inner(accumulator, &mut scalars, points);
}

/// Accumulate geometric sequences over packed polynomial prefixes.
///
/// Prefix `p`, coefficient `i` receives `sum_j scalars[j] * points[j]^(p * message_length + i)`.
/// Advances each scalar by `points[j]^message_length` after every prefix.
pub(crate) fn geometric_accumulate_prefixes<F: Field>(
    accumulator: &mut [F],
    scalars: &mut [F],
    points: &[F],
    prefix_length: usize,
    message_length: usize,
) {
    assert!(prefix_length > 0 && prefix_length <= message_length);
    assert_eq!(accumulator.len() % prefix_length, 0);
    assert_eq!(scalars.len(), points.len());
    let strides: Vec<_> = points
        .iter()
        .map(|&point| point.pow([message_length as u64]))
        .collect();
    let mut accumulation = GeometricAccumulator::new(points);
    // Multiplication advances between prefixes without division, including at zero points.
    for prefix in accumulator.chunks_exact_mut(prefix_length) {
        accumulation.accumulate(prefix, scalars);
        for (scalar, stride) in scalars.iter_mut().zip(&strides) {
            *scalar *= stride;
        }
    }
}

/// Reuse scalar scratch across geometric accumulations.
pub(crate) struct GeometricAccumulator<F: Field> {
    points: Vec<F>,
    scratch: Vec<F>,
}

impl<F: Field> GeometricAccumulator<F> {
    pub(crate) fn new(points: &[F]) -> Self {
        Self {
            points: points.to_vec(),
            scratch: vec![F::ZERO; points.len()],
        }
    }

    /// Add `sum_j scalars[j] * points[j]^i` at each index `i`, starting at zero.
    pub(crate) fn accumulate(&mut self, accumulator: &mut [F], scalars: &[F]) {
        self.scratch.copy_from_slice(scalars);
        geometric_accumulate_inner(accumulator, &mut self.scratch, &self.points);
    }
}

fn geometric_accumulate_inner<F: Field>(accumulator: &mut [F], scalars: &mut [F], points: &[F]) {
    if points.is_empty() {
        return;
    }
    #[cfg(feature = "parallel")]
    if accumulator.len() > workload_size::<F>() {
        let half = accumulator.len() / 2;
        let (low, high) = accumulator.split_at_mut(half);
        let mut scalars_high: Vec<_> = scalars
            .iter()
            .zip(points)
            .map(|(s, x)| *s * x.pow([half as u64]))
            .collect();
        rayon::join(
            || geometric_accumulate_inner(low, scalars, points),
            || geometric_accumulate_inner(high, &mut scalars_high, points),
        );
        return;
    }

    for entry in accumulator {
        for (scalar, point) in scalars.iter_mut().zip(points) {
            *entry += *scalar;
            *scalar *= *point; // TODO: Skip on last
        }
    }
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::algebra::{
        embedding::Basefield,
        fields::{Field64, Field64_3},
    };

    #[test]
    fn mixed_public_points_match_generic_arithmetic() {
        type F = Field64_3;
        let mut rng = StdRng::seed_from_u64(132);
        let mut points = vec![F::ZERO, F::ONE, F::from(7)];
        points.extend(random_vector::<F>(&mut rng, 2));
        points.push(points[2]);
        for size in [0, 1, 3, 8193] {
            let coefficients = random_vector::<F>(&mut rng, size);
            let base_coefficients = random_vector::<Field64>(&mut rng, size);
            for &point in &points {
                let expected = coefficients
                    .iter()
                    .rev()
                    .fold(F::ZERO, |a, &c| a * point + c);
                assert_eq!(univariate_evaluate(&coefficients, point), expected);
                let expected = base_coefficients
                    .iter()
                    .rev()
                    .fold(F::ZERO, |a, &c| a * point + F::from_base_prime_field(c));
                assert_eq!(
                    mixed_univariate_evaluate(&Basefield::<F>::new(), &base_coefficients, point),
                    expected
                );
                let scalar = rng.gen::<F>();
                let expected: Vec<_> = (0..size).map(|i| scalar * point.pow([i as u64])).collect();
                assert_eq!(geometric_sequence(scalar, point, size), expected);
            }
            for queries in [0, 3, points.len()] {
                let points = &points[..queries];
                let mut accumulation = GeometricAccumulator::new(points);
                for _ in 0..3 {
                    let scalars = random_vector::<F>(&mut rng, points.len());
                    let mut actual = coefficients.clone();
                    let expected: Vec<_> = coefficients
                        .iter()
                        .enumerate()
                        .map(|(i, &c)| {
                            c + scalars
                                .iter()
                                .zip(points)
                                .map(|(&s, &p)| s * p.pow([i as u64]))
                                .sum::<F>()
                        })
                        .collect();
                    geometric_accumulate(&mut actual, scalars.clone(), points);
                    assert_eq!(actual, expected);
                    let mut reused = coefficients.clone();
                    accumulation.accumulate(&mut reused, &scalars);
                    assert_eq!(reused, expected);
                }
            }
        }
    }
}
