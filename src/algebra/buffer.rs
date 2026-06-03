use ark_ff::Field;
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, Rng, RngCore};
use spongefish::DuplexSpongeInterface;

use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        mixed_dot, mixed_multilinear_extend, mixed_univariate_evaluate, ntt,
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

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn random<R>(rng: &mut R, length: usize) -> Self
    where
        R: RngCore + CryptoRng,
        Standard: Distribution<F>;

    fn zero_pad(&mut self);
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

pub trait MatrixBufferOps<F: Field> {
    fn len(&self) -> usize;
    fn num_rows(&self) -> usize;
    fn num_cols(&self) -> usize;

    fn commit_rows<H, R>(
        &self,
        config: &matrix_commit::Config<F>,
        prover_state: &mut ProverState<H, R>,
    ) -> matrix_commit::Witness
    where
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
    ) -> matrix_commit::Witness
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(config.num_rows(), self.num_rows);
        assert_eq!(config.num_cols, self.num_cols);
        config.commit(prover_state, &self.data)
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
#[derive(Clone)]
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

    fn as_slice(&self) -> &[F] {
        self.data.as_slice()
    }
}

impl<F: Field> BufferOps<F> for CpuBuffer<F> {
    type Buffer<T: Field> = CpuBuffer<T>;
    type Matrix = CpuMatrix<F>;

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

    fn zero_pad(&mut self) {
        if !self.is_empty() {
            self.data.resize(self.len().next_power_of_two(), F::ZERO);
        }
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

impl<F: Field> BufferOps<F> for Vec<F> {
    type Buffer<T: Field> = Vec<T>;
    type Matrix = CpuMatrix<F>;

    fn len(&self) -> usize {
        self.len()
    }

    fn random<R>(rng: &mut R, length: usize) -> Self
    where
        R: RngCore + CryptoRng,
        Standard: Distribution<F>,
    {
        (0..length).map(|_| rng.gen()).collect()
    }

    fn zero_pad(&mut self) {
        if !self.is_empty() {
            self.resize(self.len().next_power_of_two(), F::ZERO);
        }
    }

    fn mixed_extend<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        embedding: &M,
        point: &[M::Target],
    ) -> M::Target {
        mixed_multilinear_extend(embedding, self, point)
    }

    fn mixed_dot<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        embedding: &M,
        other: &Self::Buffer<T>,
    ) -> M::Target {
        mixed_dot(embedding, other, self)
    }

    fn mixed_univariate_evaluate<M: Embedding<Source = F>>(
        &self,
        embedding: &M,
        point: M::Target,
    ) -> M::Target {
        mixed_univariate_evaluate(embedding, self, point)
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
            masks,
            message_length,
            interleaving_depth,
            codeword_length,
        )
    }

    fn dot(&self, other: &Self) -> F {
        self.mixed_dot(&Identity::new(), other)
    }
}
