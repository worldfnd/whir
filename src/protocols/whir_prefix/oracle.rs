use ark_ff::FftField;
use ark_std::rand::{CryptoRng, RngCore};
use spongefish::StdHash;

use super::{Config, Dimensions};
use crate::{
    algebra::{
        ntt::{generator, interleaved_rs_encode},
        univariate_evaluate,
    },
    protocols::{
        challenge_indices::challenge_indices,
        matrix_commit::{Commitment, Config as MatrixConfig, Witness as MatrixWitness},
    },
    transcript::{Codec, ProverState, VerificationResult, VerifierMessage, VerifierState},
    verify,
};

pub(super) struct Witness<F: FftField> {
    pub table: Vec<F>,
    matrix: Vec<F>,
    tree: MatrixWitness,
}

pub(super) struct Openings<F> {
    pub indices: Vec<usize>,
    pub points: Vec<F>,
    pub values: Vec<Vec<F>>,
}

impl<F: FftField + Codec> Config<F> {
    pub(super) fn encode<R: RngCore + CryptoRng>(
        dimensions: Dimensions,
        payload: &[F],
        rng: &mut R,
    ) -> Vec<F> {
        let mut table = dimensions.embed_payload(payload);
        for column in table.chunks_exact_mut(dimensions.column_size) {
            for value in &mut column[..dimensions.prefix] {
                *value = F::rand(rng);
            }
        }
        table
    }

    fn matrix_config(&self, dimensions: Dimensions) -> MatrixConfig<F> {
        MatrixConfig::with_hash(self.hash_id, dimensions.rows, dimensions.row_width::<F>())
    }

    pub(super) fn commit_oracle<R: RngCore + CryptoRng>(
        &self,
        state: &mut ProverState<StdHash, R>,
        dimensions: Dimensions,
        table: Vec<F>,
        oracle_id: usize,
    ) -> Witness<F> {
        assert_eq!(table.len(), dimensions.table_size());
        let width = dimensions.row_width::<F>();
        let mut matrix = interleaved_rs_encode(&[&table], dimensions.rows, dimensions.columns);
        let final_size = dimensions.rows * width;
        matrix.reserve_exact(final_size - matrix.len());
        matrix.resize(final_size, F::ZERO);
        for index in (0..dimensions.rows).rev() {
            matrix.copy_within(
                index * dimensions.columns..(index + 1) * dimensions.columns,
                index * width,
            );
            let row = &mut matrix[index * width..(index + 1) * width];
            row[dimensions.columns] = F::from(oracle_id as u64);
            row[dimensions.columns + 1..].fill(F::ZERO);
        }
        let tree = self.matrix_config(dimensions).commit(state, &matrix);
        Witness {
            table,
            matrix,
            tree,
        }
    }

    pub(super) fn receive_oracle(
        &self,
        state: &mut VerifierState<'_>,
        dimensions: Dimensions,
    ) -> VerificationResult<Commitment> {
        self.matrix_config(dimensions).receive_commitment(state)
    }

    pub(super) fn open_oracle<R: RngCore + CryptoRng>(
        &self,
        state: &mut ProverState<StdHash, R>,
        dimensions: Dimensions,
        witness: &Witness<F>,
    ) -> Openings<F> {
        let indices = challenge_indices(state, dimensions.rows, self.queries, true);
        self.open_at_indices(state, dimensions, witness, &indices)
    }

    pub(super) fn open_at_indices<R: RngCore + CryptoRng>(
        &self,
        state: &mut ProverState<StdHash, R>,
        dimensions: Dimensions,
        witness: &Witness<F>,
        indices: &[usize],
    ) -> Openings<F> {
        let width = dimensions.row_width::<F>();
        let values: Vec<F> = indices
            .iter()
            .flat_map(|&i| witness.matrix[i * width..(i + 1) * width].iter().copied())
            .collect();
        for value in &values {
            state.prover_hint(value);
        }
        self.matrix_config(dimensions)
            .open(state, &witness.tree, indices);
        let root = generator::<F>(dimensions.rows).expect("validated RS domain");
        Openings {
            indices: indices.to_vec(),
            points: indices.iter().map(|&i| root.pow([i as u64])).collect(),
            values: values
                .chunks_exact(width)
                .map(|row| row[..dimensions.columns].to_vec())
                .collect(),
        }
    }

    pub(super) fn verify_oracle(
        &self,
        state: &mut VerifierState<'_>,
        dimensions: Dimensions,
        commitment: &Commitment,
        oracle_id: usize,
    ) -> VerificationResult<Openings<F>> {
        let indices = challenge_indices(state, dimensions.rows, self.queries, true);
        self.verify_at_indices(state, dimensions, commitment, oracle_id, &indices)
    }

    pub(super) fn verify_at_indices(
        &self,
        state: &mut VerifierState<'_>,
        dimensions: Dimensions,
        commitment: &Commitment,
        oracle_id: usize,
        indices: &[usize],
    ) -> VerificationResult<Openings<F>> {
        let values = (0..indices.len() * dimensions.row_width::<F>())
            .map(|_| state.prover_hint())
            .collect::<VerificationResult<Vec<F>>>()?;
        self.matrix_config(dimensions)
            .verify(state, commitment, indices, &values)?;
        let width = dimensions.row_width::<F>();
        verify!(values
            .chunks_exact(width)
            .all(|row| row[dimensions.columns] == F::from(oracle_id as u64)));
        let root = generator::<F>(dimensions.rows).expect("validated RS domain");
        Ok(Openings {
            indices: indices.to_vec(),
            points: indices.iter().map(|&i| root.pow([i as u64])).collect(),
            values: values
                .chunks_exact(width)
                .map(|row| row[..dimensions.columns].to_vec())
                .collect(),
        })
    }

    pub(super) fn send_ood<R: RngCore + CryptoRng>(
        &self,
        state: &mut ProverState<StdHash, R>,
        dimensions: Dimensions,
        table: &[F],
    ) -> Vec<(F, F)> {
        let mut samples = Vec::new();
        while samples.len() < self.ood_samples {
            let point: F = state.verifier_message();
            if point.pow([dimensions.rows as u64]) == F::ONE
                || samples.iter().any(|(z, _)| *z == point)
            {
                continue;
            }
            let value = univariate_evaluate(table, point);
            state.prover_message(&value);
            samples.push((point, value));
        }
        samples
    }

    pub(super) fn receive_ood(
        &self,
        state: &mut VerifierState<'_>,
        dimensions: Dimensions,
    ) -> VerificationResult<Vec<(F, F)>> {
        let mut samples = Vec::new();
        while samples.len() < self.ood_samples {
            let point: F = state.verifier_message();
            if point.pow([dimensions.rows as u64]) == F::ONE
                || samples.iter().any(|(z, _)| *z == point)
            {
                continue;
            }
            samples.push((point, state.prover_message()?));
        }
        Ok(samples)
    }
}
