use std::{any::Any, mem};

use ark_ff::Field;
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, Rng, RngCore};

use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        linear_form::{Covector, LinearForm, UnivariateEvaluation},
        mixed_dot, mixed_multilinear_extend, mixed_scalar_mul_add, mixed_univariate_evaluate, ntt,
        sumcheck::{compute_sumcheck_polynomial, fold},
    },
    utils::chunks_exact_or_empty,
};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub use super::metal_buffer::MetalBuffer;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub type ActiveBuffer<T> = MetalBuffer<T>;

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub type ActiveBuffer<T> = CpuBuffer<T>;

pub trait BufferOps<T> {
    fn from_vec(source: Vec<T>) -> Self;
    fn from_slice(source: &[T]) -> Self;
    fn as_slice(&self) -> &[T];
    fn num_rows(&self, num_cols: usize) -> usize {
        self.len() / num_cols
    }
    fn read_rows(&self, num_cols: usize, indices: &[usize]) -> Vec<T>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait FieldOps<F: Field>: Clone {
    type TargetBuffer<T: Field>: FieldOps<T>;

    fn zeros(length: usize) -> Self;

    // consider removing this to avoid dependency on ark for Buffer design
    fn random<R>(rng: &mut R, length: usize) -> Self
    where
        R: RngCore + CryptoRng,
        Standard: Distribution<F>;
    fn zero_pad(&mut self);
    fn dot(&self, other: &Self) -> F;
    fn fold(&mut self, weight: F);
    fn sumcheck_polynomial(&self, other: &Self) -> (F, F);
    fn accumulate_univariate_evaluations(
        &mut self,
        evaluators: &[UnivariateEvaluation<F>],
        scalars: &[F],
    );
    fn linear_forms_rlc(
        size: usize,
        linear_forms: &mut [Box<dyn LinearForm<F>>],
        rlc_coeffs: &[F],
    ) -> Self;

    fn mixed_extend<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        embedding: &M,
        point: &[M::Target],
    ) -> M::Target;
    fn mixed_dot<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        embedding: &M,
        other: &Self::TargetBuffer<T>,
    ) -> M::Target;
    fn mixed_univariate_evaluate<M: Embedding<Source = F>>(
        &self,
        embedding: &M,
        point: M::Target,
    ) -> M::Target;
    fn mixed_linear_combination<M: Embedding<Source = F>>(
        embedding: &M,
        vectors: &[&Self],
        coeffs: &[M::Target],
    ) -> Self::TargetBuffer<M::Target>;
    fn mixed_scalar_mul_add_to<M: Embedding<Source = F>>(
        &self,
        embedding: &M,
        accumulator: &mut Self::TargetBuffer<M::Target>,
        weight: M::Target,
    );
    fn interleaved_rs_encode(
        vectors: &[&Self],
        masks: &Self,
        message_length: usize,
        interleaving_depth: usize,
        codeword_length: usize,
    ) -> Self;
}

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
pub struct CpuBuffer<T: Clone> {
    data: Vec<T>,
}

impl<T: Clone> CpuBuffer<T> {
    pub fn from_vec(source: Vec<T>) -> Self {
        Self { data: source }
    }

    pub fn from_slice(source: &[T]) -> Self {
        Self {
            data: Vec::from(source),
        }
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }
}

impl<T: Clone> BufferOps<T> for CpuBuffer<T> {
    fn as_slice(&self) -> &[T] {
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

    fn from_vec(source: Vec<T>) -> Self {
        Self::from_vec(source)
    }

    fn from_slice(source: &[T]) -> Self {
        Self::from_slice(source)
    }
}

impl<F: Field> FieldOps<F> for CpuBuffer<F> {
    type TargetBuffer<T: Field> = CpuBuffer<T>;

    fn zeros(length: usize) -> Self {
        Self {
            data: vec![F::ZERO; length],
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

    fn linear_forms_rlc(
        size: usize,
        linear_forms: &mut [Box<dyn LinearForm<F>>],
        rlc_coeffs: &[F],
    ) -> Self {
        assert_eq!(linear_forms.len(), rlc_coeffs.len());
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

    fn zero_pad(&mut self) {
        if !self.is_empty() {
            self.data.resize(self.len().next_power_of_two(), F::ZERO);
        }
    }

    fn fold(&mut self, weight: F) {
        fold(&mut self.data, weight);
    }

    fn sumcheck_polynomial(&self, other: &Self) -> (F, F) {
        compute_sumcheck_polynomial(&self.data, other.as_slice())
    }

    fn accumulate_univariate_evaluations(
        &mut self,
        evaluators: &[UnivariateEvaluation<F>],
        scalars: &[F],
    ) {
        UnivariateEvaluation::accumulate_many(evaluators, &mut self.data, scalars);
    }

    fn mixed_extend<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        embedding: &M,
        point: &[M::Target],
    ) -> M::Target {
        mixed_multilinear_extend(embedding, &self.data, point)
    }

    fn mixed_dot<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        embedding: &M,
        other: &Self::TargetBuffer<T>,
    ) -> M::Target {
        mixed_dot(embedding, other.as_slice(), self.as_slice())
    }

    fn dot(&self, other: &Self) -> F {
        self.mixed_dot(&Identity::new(), other)
    }

    fn mixed_univariate_evaluate<M: Embedding<Source = F>>(
        &self,
        embedding: &M,
        point: M::Target,
    ) -> M::Target {
        mixed_univariate_evaluate(embedding, &self.data, point)
    }

    fn mixed_linear_combination<M: Embedding<Source = F>>(
        embedding: &M,
        vectors: &[&Self],
        coeffs: &[M::Target],
    ) -> Self::TargetBuffer<M::Target> {
        assert_eq!(vectors.len(), coeffs.len());
        let Some((first, vectors)) = vectors.split_first() else {
            return CpuBuffer::from_vec(Vec::new());
        };
        debug_assert_eq!(coeffs[0], M::Target::ONE);
        let mut accumulator = crate::algebra::lift(embedding, first.as_slice());
        for (coeff, vector) in coeffs[1..].iter().zip(vectors) {
            mixed_scalar_mul_add(embedding, &mut accumulator, *coeff, vector.as_slice());
        }
        CpuBuffer::from_vec(accumulator)
    }

    fn mixed_scalar_mul_add_to<M: Embedding<Source = F>>(
        &self,
        embedding: &M,
        accumulator: &mut Self::TargetBuffer<M::Target>,
        weight: M::Target,
    ) {
        mixed_scalar_mul_add(embedding, &mut accumulator.data, weight, self.as_slice());
    }

    fn interleaved_rs_encode(
        vectors: &[&Self],
        masks: &Self,
        message_length: usize,
        interleaving_depth: usize,
        codeword_length: usize,
    ) -> Self {
        let vectors = vectors.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
        let messages = vectors
            .iter()
            .flat_map(|v| chunks_exact_or_empty(v, message_length, interleaving_depth))
            .collect::<Vec<_>>();
        Self::from_vec(ntt::interleaved_rs_encode(
            &messages,
            masks.as_slice(),
            codeword_length,
        ))
    }
}
