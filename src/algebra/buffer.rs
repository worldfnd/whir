use std::{any::Any, mem};

use ark_ff::Field;
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, Rng, RngCore};
use spongefish::DuplexSpongeInterface;

use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        linear_form::{Covector, LinearForm, UnivariateEvaluation},
        mixed_dot, mixed_multilinear_extend, mixed_scalar_mul_add, mixed_univariate_evaluate, ntt,
        sumcheck::{compute_sumcheck_polynomial, fold, fold_and_compute_polynomial},
    },
    hash::Hash,
    protocols::matrix_commit,
    transcript::{ProverMessage, ProverState},
    type_info::TypeInfo,
    utils::chunks_exact_or_empty,
};

pub trait BufferOps<F: Field>: Clone {
    type Buffer<T: Field>: BufferOps<T>;
    type Matrix: MatrixBufferOps<F>;

    fn from_vec(source: Vec<F>) -> Self;
    fn zeros(length: usize) -> Self;
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn random<R>(rng: &mut R, length: usize) -> Self
    where
        R: RngCore + CryptoRng,
        Standard: Distribution<F>;

    fn linear_forms_rlc(
        size: usize,
        linear_forms: &mut [Box<dyn LinearForm<F>>],
        rlc_coeffs: &[F],
    ) -> Self;

    fn zero_pad(&mut self);
    fn fold(&mut self, weight: F);
    fn sumcheck_polynomial(&self, other: &Self) -> (F, F);
    fn fold_and_sumcheck_polynomial(&mut self, other: &mut Self, weight: F) -> (F, F);
    fn accumulate_univariate_evaluations(
        &mut self,
        evaluators: &[UnivariateEvaluation<F>],
        scalars: &[F],
    );
    fn write_to_prover<H, R>(&self, prover_state: &mut ProverState<H, R>)
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: ProverMessage<[H::U]>;
    fn mixed_extend<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        embedding: &M,
        point: &[M::Target],
    ) -> M::Target;
    fn mixed_dot<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        embedding: &M,
        other: &Self::Buffer<T>,
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
    ) -> Self::Buffer<M::Target>;
    fn mixed_scalar_mul_add_to<M: Embedding<Source = F>>(
        &self,
        embedding: &M,
        accumulator: &mut Self::Buffer<M::Target>,
        weight: M::Target,
    );
    fn mixed_dot_slice<M: Embedding<Source = F>>(
        &self,
        embedding: &M,
        other: &[M::Target],
    ) -> M::Target;
    fn interleaved_rs_encode(
        vectors: &[&Self],
        masks: &Self,
        message_length: usize,
        interleaving_depth: usize,
        codeword_length: usize,
    ) -> Self::Matrix
    where
        F: 'static;
    fn dot(&self, other: &Self) -> F;
}

pub trait MatrixBufferOps<F: Field> {
    type Witness;

    fn len(&self) -> usize;
    fn num_rows(&self) -> usize;
    fn num_cols(&self) -> usize;

    fn commit_rows<H, R>(
        &self,
        config: &matrix_commit::Config<F>,
        prover_state: &mut ProverState<H, R>,
    ) -> Self::Witness
    where
        F: TypeInfo + matrix_commit::Encodable + Send + Sync,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        Hash: ProverMessage<[H::U]>;

    fn open_rows<H, R>(
        &self,
        config: &matrix_commit::Config<F>,
        prover_state: &mut ProverState<H, R>,
        witness: &Self::Witness,
        indices: &[usize],
    ) where
        F: TypeInfo + matrix_commit::Encodable + Send + Sync,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        Hash: ProverMessage<[H::U]>;

    fn read_rows(&self, indices: &[usize]) -> Vec<F>;
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
pub struct CpuMatrix<F: Field> {
    data: Vec<F>,
    num_rows: usize,
    num_cols: usize,
}

impl<F: Field> CpuMatrix<F> {
    pub fn from_vec(data: Vec<F>, num_rows: usize, num_cols: usize) -> Self {
        assert_eq!(data.len(), num_rows * num_cols);
        Self {
            data,
            num_rows,
            num_cols,
        }
    }

    fn row(&self, row: usize) -> &[F] {
        let start = row * self.num_cols;
        let end = start + self.num_cols;
        &self.data[start..end]
    }
}

impl<F> MatrixBufferOps<F> for CpuMatrix<F>
where
    F: Field + TypeInfo + matrix_commit::Encodable + Send + Sync,
{
    type Witness = matrix_commit::Witness;

    fn len(&self) -> usize {
        self.data.len()
    }

    fn num_rows(&self) -> usize {
        self.num_rows
    }

    fn num_cols(&self) -> usize {
        self.num_cols
    }

    fn commit_rows<H, R>(
        &self,
        config: &matrix_commit::Config<F>,
        prover_state: &mut ProverState<H, R>,
    ) -> Self::Witness
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(config.num_rows(), self.num_rows);
        assert_eq!(config.num_cols, self.num_cols);
        config.commit(prover_state, &self.data)
    }

    fn open_rows<H, R>(
        &self,
        config: &matrix_commit::Config<F>,
        prover_state: &mut ProverState<H, R>,
        witness: &Self::Witness,
        indices: &[usize],
    ) where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        Hash: ProverMessage<[H::U]>,
    {
        config.open(prover_state, witness, indices);
    }

    fn read_rows(&self, indices: &[usize]) -> Vec<F> {
        let mut rows = Vec::with_capacity(indices.len() * self.num_cols);
        for &index in indices {
            assert!(index < self.num_rows);
            rows.extend_from_slice(self.row(index));
        }
        rows
    }
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
pub struct CpuBuffer<F: Field> {
    data: Vec<F>,
}

impl<F: Field> CpuBuffer<F> {
    pub fn from_vec(source: Vec<F>) -> Self {
        Self { data: source }
    }

    pub fn from_slice(source: &[F]) -> Self {
        Self {
            data: Vec::from(source),
        }
    }

    pub(crate) fn as_slice(&self) -> &[F] {
        self.data.as_slice()
    }
}

impl<F: Field> BufferOps<F> for CpuBuffer<F> {
    type Buffer<T: Field> = CpuBuffer<T>;
    type Matrix = CpuMatrix<F>;

    fn from_vec(source: Vec<F>) -> Self {
        Self::from_vec(source)
    }

    fn zeros(length: usize) -> Self {
        Self {
            data: vec![F::ZERO; length],
        }
    }

    fn len(&self) -> usize {
        self.data.len()
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

    fn fold_and_sumcheck_polynomial(&mut self, other: &mut Self, weight: F) -> (F, F) {
        fold_and_compute_polynomial(&mut self.data, &mut other.data, weight)
    }

    fn accumulate_univariate_evaluations(
        &mut self,
        evaluators: &[UnivariateEvaluation<F>],
        scalars: &[F],
    ) {
        UnivariateEvaluation::accumulate_many(evaluators, &mut self.data, scalars);
    }

    fn write_to_prover<H, R>(&self, prover_state: &mut ProverState<H, R>)
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: ProverMessage<[H::U]>,
    {
        prover_state.prover_messages(&self.data);
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
        other: &Self::Buffer<T>,
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
    ) -> Self::Buffer<M::Target> {
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
        accumulator: &mut Self::Buffer<M::Target>,
        weight: M::Target,
    ) {
        mixed_scalar_mul_add(embedding, &mut accumulator.data, weight, self.as_slice());
    }

    fn mixed_dot_slice<M: Embedding<Source = F>>(
        &self,
        embedding: &M,
        other: &[M::Target],
    ) -> M::Target {
        mixed_dot(embedding, other, self.as_slice())
    }

    fn interleaved_rs_encode(
        vectors: &[&Self],
        masks: &Self,
        message_length: usize,
        interleaving_depth: usize,
        codeword_length: usize,
    ) -> Self::Matrix
    where
        F: 'static,
    {
        let vectors = vectors.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
        interleaved_rs_encode_slices(
            &vectors,
            masks.as_slice(),
            message_length,
            interleaving_depth,
            codeword_length,
        )
    }
}

fn interleaved_rs_encode_slices<F: Field + 'static>(
    vectors: &[&[F]],
    masks: &[F],
    message_length: usize,
    interleaving_depth: usize,
    codeword_length: usize,
) -> CpuMatrix<F> {
    let messages = vectors
        .iter()
        .flat_map(|v| chunks_exact_or_empty(v, message_length, interleaving_depth))
        .collect::<Vec<_>>();
    CpuMatrix::from_vec(
        ntt::interleaved_rs_encode(&messages, masks, codeword_length),
        codeword_length,
        vectors.len() * interleaving_depth,
    )
}
