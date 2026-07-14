//! In-memory CPU backend for the buffer abstraction.

use std::{any::Any, mem};

use ark_ff::Field;
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, Rng, RngCore};

use crate::{
    algebra::{
        embedding::Embedding,
        linear_form::{Covector, LinearForm, UnivariateEvaluation},
    },
    buffer::{ActiveBuffer, Buffer, BufferOps},
    engines::EngineId,
    hash::{self, Hash},
    protocols::{
        matrix_commit::{hash_rows, Encodable, Merklize},
        merkle_tree,
    },
};

#[derive(
    Clone,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Debug,
    Default,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct CpuBuffer<T> {
    data: Vec<T>,
}

impl<T> From<Vec<T>> for CpuBuffer<T> {
    fn from(source: Vec<T>) -> Self {
        Self { data: source }
    }
}

impl<T: Clone> From<&[T]> for CpuBuffer<T> {
    fn from(source: &[T]) -> Self {
        Self {
            data: source.to_vec(),
        }
    }
}

impl<T: Copy> BufferOps<T> for CpuBuffer<T> {
    fn to_slice(&self) -> &[T] {
        &self.data
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn read_rows(&self, num_cols: usize, indices: &[usize]) -> Vec<T> {
        let mut result = Vec::with_capacity(indices.len() * num_cols);
        for i in indices {
            result.extend_from_slice(&self.data[i * num_cols..(i + 1) * num_cols]);
        }
        result
    }

    fn gather_at_indices(&self, indices: &[usize]) -> Vec<T> {
        indices.iter().map(|&i| self.data[i]).collect()
    }

    fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }
}

impl<T: Encodable + Copy + Send + Sync> Merklize<T> for CpuBuffer<T> {
    type Nodes = CpuBuffer<Hash>;

    fn merklize(
        &self,
        num_cols: usize,
        leaf_hash: EngineId,
        merkle: &merkle_tree::Config,
    ) -> (Self::Nodes, Hash) {
        assert_eq!(self.len(), num_cols * merkle.num_leaves);
        let engine = hash::ENGINES
            .retrieve(leaf_hash)
            .expect("Failed to retrieve hash engine");
        #[cfg(feature = "tracing")]
        tracing::Span::current().record("engine", engine.name().as_ref());

        let mut leaves = vec![Hash::default(); merkle.num_leaves];
        hash_rows(&*engine, &self.data, &mut leaves);
        let nodes = merkle.build_nodes(leaves);
        let root = nodes[merkle.num_nodes() - 1];
        (CpuBuffer { data: nodes }, root)
    }
}

impl<F: Field> Buffer<F> for CpuBuffer<F> {
    type TargetBuffer<T: Field> = CpuBuffer<T>;

    fn zeros(length: usize) -> Self {
        Self {
            data: vec![F::ZERO; length],
        }
    }

    fn ones(length: usize) -> Self {
        Self {
            data: vec![F::ONE; length],
        }
    }

    fn random<R>(rng: &mut R, length: usize) -> Self
    where
        R: RngCore + CryptoRng,
        Standard: Distribution<F>,
    {
        Self {
            data: (0..length).map(|_| rng.gen()).collect(),
        }
    }

    fn dot(&self, other: &Self) -> F {
        crate::algebra::dot(&self.data, &other.data)
    }

    fn bilinear_form(&self, rows: &Self, cols: &Self) -> F {
        crate::utils::zip_strict(&rows.data, self.data.chunks_exact(cols.len()))
            .map(|(r, row)| *r * crate::algebra::dot(&cols.data, row))
            .sum()
    }

    fn tensor_product(&self, other: &Self) -> Self {
        Self {
            data: crate::algebra::tensor_product(&self.data, &other.data),
        }
    }

    fn mat_vec(&self, vector: &Self) -> Self {
        Self {
            data: self
                .data
                .chunks_exact(vector.data.len())
                .map(|row| crate::algebra::dot(&vector.data, row))
                .collect(),
        }
    }

    fn concat(&self, other: &Self) -> Self {
        let mut data = Vec::with_capacity(self.data.len() + other.data.len());
        data.extend_from_slice(&self.data);
        data.extend_from_slice(&other.data);
        Self { data }
    }

    fn eq_weights(point: &[F]) -> Self {
        Self {
            data: crate::algebra::eq_weights(point),
        }
    }

    fn sumcheck_polynomial(&self, other: &Self) -> (F, F) {
        crate::algebra::sumcheck::compute_sumcheck_polynomial(&self.data, &other.data)
    }

    fn fold(&mut self, weight: F) {
        crate::algebra::sumcheck::fold(&mut self.data, weight);
    }

    fn fold_pair_sumcheck_polynomial(&mut self, other: &mut Self, weight: F) -> (F, F) {
        crate::algebra::sumcheck::fold_and_compute_polynomial(
            &mut self.data,
            &mut other.data,
            weight,
        )
    }

    fn scalar_mul(&mut self, weight: F) {
        crate::algebra::scalar_mul(&mut self.data, weight);
    }

    fn accumulate_univariate_evaluations(
        &mut self,
        evaluators: &[UnivariateEvaluation<F>],
        scalars: &Self,
    ) {
        let Some(size) = evaluators.first().map(|e| e.size) else {
            return;
        };
        UnivariateEvaluation::accumulate_many(
            evaluators,
            &mut self.data[..size],
            scalars.to_slice(),
        );
    }

    fn mixed_univariate_evaluate<M: Embedding<Source = F>>(
        &self,
        embedding: &M,
        point: M::Target,
    ) -> M::Target {
        crate::algebra::mixed_univariate_evaluate(embedding, &self.data, point)
    }

    fn mixed_dot<M: Embedding<Source = F>>(
        &self,
        embedding: &M,
        other: &CpuBuffer<M::Target>,
    ) -> M::Target {
        crate::algebra::mixed_dot(embedding, &other.data, &self.data)
    }

    fn mixed_scalar_mul_add_to<M: Embedding<Source = F>>(
        &self,
        embedding: &M,
        accumulator: &mut CpuBuffer<M::Target>,
        weight: M::Target,
    ) {
        crate::algebra::mixed_scalar_mul_add(embedding, &mut accumulator.data, weight, &self.data);
    }

    fn geometric_challenge<G: Field>(current: G, base: G, length: usize) -> Self::TargetBuffer<G> {
        CpuBuffer {
            data: crate::algebra::geometric_sequence(current, base, length),
        }
    }

    fn linear_forms_rlc(
        size: usize,
        linear_forms: &mut [Box<dyn LinearForm<F>>],
        rlc_coeffs: &ActiveBuffer<F>,
    ) -> Self {
        assert_eq!(linear_forms.len(), rlc_coeffs.len());
        let rlc_coeffs = rlc_coeffs.to_slice();
        let mut covector = vec![F::ZERO; size];
        if let Some((first, linear_forms)) = linear_forms.split_first_mut() {
            debug_assert_eq!(rlc_coeffs[0], F::ONE);
            if let Some(covector_form) =
                (first.as_mut() as &mut dyn Any).downcast_mut::<Covector<F>>()
            {
                mem::swap(&mut covector, &mut covector_form.vector);
            } else {
                first.accumulate(&mut covector, F::ONE);
            }
            for (rlc_coeff, linear_form) in rlc_coeffs[1..].iter().zip(linear_forms) {
                linear_form.accumulate(&mut covector, *rlc_coeff);
            }
        }
        Self { data: covector }
    }

    fn mixed_linear_combination<M: Embedding<Source = F>>(
        embedding: &M,
        vectors: &[&Self],
        coeffs: &Self::TargetBuffer<M::Target>,
    ) -> CpuBuffer<M::Target> {
        let coeffs = coeffs.to_slice();
        assert_eq!(vectors.len(), coeffs.len());
        let Some((first, vectors)) = vectors.split_first() else {
            return CpuBuffer { data: Vec::new() };
        };
        debug_assert_eq!(coeffs[0], M::Target::ONE);
        let mut accumulator = crate::algebra::lift(embedding, &first.data);
        for (coeff, vector) in coeffs[1..].iter().zip(vectors) {
            crate::algebra::mixed_scalar_mul_add(embedding, &mut accumulator, *coeff, &vector.data);
        }
        CpuBuffer { data: accumulator }
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::AdditiveGroup;

    use super::*;
    use crate::algebra::{fields::Field64, geometric_accumulate};

    type F = Field64;

    #[test]
    fn scalar_mul_multiplies_in_place() {
        let values = vec![F::from(1u64), F::from(2u64), F::from(3u64), F::from(4u64)];
        let weight = F::from(5u64);
        let mut buffer = CpuBuffer::from(values.clone());
        buffer.scalar_mul(weight);
        let expected: Vec<F> = values.iter().map(|&v| v * weight).collect();
        assert_eq!(buffer.to_slice(), expected.as_slice());
    }

    #[test]
    fn accumulate_matches_geometric_accumulate_over_prefix() {
        let len = 8usize;
        let points = vec![F::from(3u64), F::from(5u64)];
        let scalars = vec![F::from(2u64), F::from(9u64)];

        // Full-length and prefix accumulation.
        for size in [len, 5] {
            let mut buffer = CpuBuffer::from(vec![F::ZERO; len]);
            let evaluators: Vec<_> = points
                .iter()
                .map(|&point| UnivariateEvaluation::new(point, size))
                .collect();
            buffer
                .accumulate_univariate_evaluations(&evaluators, &CpuBuffer::from(scalars.clone()));

            // Reference: accumulate Σ_j scalars[j]·points[j]^i into the prefix
            // of a plain vector.
            let mut reference = vec![F::ZERO; len];
            geometric_accumulate(&mut reference[..size], scalars.clone(), &points);

            assert_eq!(
                buffer.to_slice(),
                reference.as_slice(),
                "accumulate mismatch for evaluator size {size}"
            );
        }
    }
}
