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

pub fn geometric_sequence<F: Field>(base: F, length: usize) -> Vec<F> {
    let mut result = Vec::with_capacity(length);
    let mut current = F::ONE;
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
    #[cfg(feature = "parallel")]
    if accumulator.len() > workload_size::<F>() {
        let half = accumulator.len() / 2;
        let (low, high) = accumulator.split_at_mut(half);
        let scalars_high = scalars
            .iter()
            .zip(points)
            .map(|(s, x)| *s * x.pow([half as u64]))
            .collect();
        rayon::join(
            || geometric_accumulate(low, scalars, points),
            || geometric_accumulate(high, scalars_high, points),
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
