//! Experimental prefix-mask WHIR construction; not a security-certified PCS.
//!
//! Each RS column has an independent random coefficient prefix. A code-switch
//! packs the folded table, sumcheck-mask coefficients, and their encoding prefix
//! into the next payload. Fresh prefixes hide the new columns; the final opening
//! reveals a full-length random mask plus a challenge times the small table.

mod config;
mod linear_form;
mod oracle;
mod sumcheck;

use std::mem::take;

use ark_ff::FftField;
use ark_std::rand::{CryptoRng, RngCore};
pub use config::{Config, Dimensions};
pub use linear_form::TensorLinearForm;
use spongefish::StdHash;

use self::{
    oracle::{Openings, Witness as OracleWitness},
    sumcheck::Challenges,
};
use crate::{
    algebra::{
        dot, geometric_accumulate,
        linear_form::LinearForm,
        multilinear_extend,
        ntt::{interleaved_rs_encode, ntt},
        univariate_evaluate, MultilinearPoint,
    },
    hash::Hash,
    protocols::matrix_commit::Commitment as OracleCommitment,
    transcript::{Codec, ProverState, VerificationResult, VerifierMessage, VerifierState},
    verify,
};

#[must_use]
pub struct Witness<F: FftField> {
    oracle: OracleWitness<F>,
    ood: Vec<(F, F)>,
}

#[must_use]
pub struct Commitment<F: FftField> {
    oracle: OracleCommitment,
    ood: Vec<(F, F)>,
}

struct Statement<F: FftField> {
    form: TensorLinearForm<F>,
    sum: F,
}

struct SwitchOpenings<F> {
    ood: Vec<(F, F)>,
    main: Openings<F>,
    mask: Openings<F>,
}

struct SwitchClaims<F: FftField> {
    payload: TensorLinearForm<F>,
    table: TensorLinearForm<F>,
    sum: F,
}

impl<F: FftField> Statement<F> {
    fn carry(&mut self, challenges: &Challenges<F>, folded_size: usize, next: Dimensions) {
        let mut carry = self.form.fold(&challenges.folding);
        carry.scale(challenges.rho);
        carry.resize(next.payload_size);
        carry.add_coefficients(folded_size, &challenges.mask_weights);
        self.form = carry.insert_prefixes(
            next.payload_size,
            next.columns,
            next.prefix,
            next.column_size,
        );
    }
}

impl<F: FftField + Codec> Config<F> {
    pub fn commit<R: RngCore + CryptoRng>(
        &self,
        state: &mut ProverState<StdHash, R>,
        vector: &[F],
    ) -> VerificationResult<Witness<F>> {
        verify!(vector.len() == self.witness_size);
        let dimensions = self.rounds[0];
        let table = Self::encode(dimensions, vector, state.rng());
        let oracle = self.commit_oracle(state, dimensions, table, 0);
        let ood = self.send_ood(state, dimensions, &oracle.table);
        Ok(Witness { oracle, ood })
    }

    pub fn receive_commitment(
        &self,
        state: &mut VerifierState<'_>,
    ) -> VerificationResult<Commitment<F>> {
        let dimensions = self.rounds[0];
        let oracle = self.receive_oracle(state, dimensions)?;
        let ood = self.receive_ood(state, dimensions)?;
        Ok(Commitment { oracle, ood })
    }

    pub fn prove<R: RngCore + CryptoRng>(
        &self,
        state: &mut ProverState<StdHash, R>,
        mut witness: Witness<F>,
        form: &TensorLinearForm<F>,
        value: F,
    ) -> VerificationResult<()> {
        verify!(form.size() == self.witness_size.next_power_of_two());
        state.prover_message(&form.digest());
        state.prover_message(&value);
        let Statement { form, mut sum } = self.initial_statement(state, form, value, &witness.ood);
        let mut vector = take(&mut witness.oracle.table);
        let mut covector = form.coefficients();
        verify!(dot(&vector, &covector) == sum);
        for round in 0..self.rounds.len() - 1 {
            let (mask, challenges) = self.prove_masked_sumcheck(
                state,
                &mut vector,
                &mut covector,
                &mut sum,
                2 * round + 1,
            );
            let next = self.rounds[round + 1];
            let folded_size = vector.len();
            let mask_len = 2 * self.folds + 1;
            let mask_prefix = self.mask_dimensions().prefix;
            vector.extend_from_slice(&mask.table[mask_prefix..mask_prefix + mask_len]);
            vector.resize(folded_size + self.mask_segment_size(), F::ZERO);
            vector.extend_from_slice(&mask.table[..mask_prefix]);
            vector.resize(next.payload_size, F::ZERO);
            let next_table = Self::encode(next, &vector, state.rng());
            let mut next_witness = self.commit_oracle(state, next, next_table, 2 * round + 2);
            let ood = self.send_ood(state, next, &next_witness.table);
            let main = self.open_oracle(state, self.rounds[round], &witness.oracle);
            let mask = self.open_oracle(state, self.mask_dimensions(), &mask);

            covector.extend_from_slice(&challenges.mask_weights);
            covector.resize(next.payload_size, F::ZERO);
            let batch = self.batch_switch(
                state,
                round,
                &challenges.folding,
                &SwitchOpenings { ood, main, mask },
            );
            vector = take(&mut next_witness.table);
            witness.oracle = next_witness;
            batch.payload.accumulate(&mut covector, F::ONE);
            covector = next.embed_payload(&covector);
            batch.table.accumulate(&mut covector, F::ONE);
            sum += batch.sum;
            debug_assert_eq!(dot(&vector, &covector), sum);
        }
        self.prove_final(state, &witness.oracle, &vector, covector, sum);
        Ok(())
    }

    pub fn verify(
        &self,
        state: &mut VerifierState<'_>,
        mut commitment: Commitment<F>,
        form: &TensorLinearForm<F>,
        value: F,
    ) -> VerificationResult<()> {
        verify!(form.size() == self.witness_size.next_power_of_two());
        let digest: Hash = state.prover_message()?;
        verify!(digest == form.digest());
        verify!(state.prover_message::<F>()? == value);
        let mut statement = self.initial_statement(state, form, value, &commitment.ood);
        for round in 0..self.rounds.len() - 1 {
            let current = self.rounds[round];
            let (mask_commitment, challenges) = self.verify_masked_sumcheck(
                state,
                current.table_size().ilog2() as usize,
                &mut statement.sum,
            )?;
            let next = self.rounds[round + 1];
            let next_commitment = self.receive_oracle(state, next)?;
            let ood = self.receive_ood(state, next)?;
            let main = self.verify_oracle(state, current, &commitment.oracle, 2 * round)?;
            let mask = self.verify_oracle(
                state,
                self.mask_dimensions(),
                &mask_commitment,
                2 * round + 1,
            )?;
            statement.carry(&challenges, current.column_size, next);
            let batch = self.batch_switch(
                state,
                round,
                &challenges.folding,
                &SwitchOpenings { ood, main, mask },
            );
            let mut form = batch.table;
            form.add(batch.payload.insert_prefixes(
                next.payload_size,
                next.columns,
                next.prefix,
                next.column_size,
            ));
            statement.form.add(form);
            statement.sum += batch.sum;
            commitment.oracle = next_commitment;
        }
        self.verify_final(state, &commitment.oracle, &statement)
    }

    fn initial_statement<T: VerifierMessage<U = u8>>(
        &self,
        state: &mut T,
        form: &TensorLinearForm<F>,
        value: F,
        ood: &[(F, F)],
    ) -> Statement<F> {
        let dimensions = self.rounds[0];
        let mut form = form.insert_prefixes(
            dimensions.payload_size,
            dimensions.columns,
            dimensions.prefix,
            dimensions.column_size,
        );
        let mut sum = value;
        for &(point, value) in ood {
            let scalar: F = state.verifier_message();
            form.add_univariate(0, dimensions.table_size(), point, scalar);
            sum += scalar * value;
        }
        Statement { form, sum }
    }

    fn batch_switch<T: VerifierMessage<U = u8>>(
        &self,
        state: &mut T,
        round: usize,
        folding: &[F],
        openings: &SwitchOpenings<F>,
    ) -> SwitchClaims<F> {
        let next = self.rounds[round + 1];
        let mut claims = SwitchClaims {
            payload: TensorLinearForm::zero(next.payload_size),
            table: TensorLinearForm::zero(next.table_size()),
            sum: F::ZERO,
        };
        for &(point, value) in &openings.ood {
            let scalar: F = state.verifier_message();
            claims
                .table
                .add_univariate(0, next.table_size(), point, scalar);
            claims.sum += scalar * value;
        }
        let folded_size = self.rounds[round].column_size;
        let weights = MultilinearPoint(folding.to_vec()).eq_weights();
        for (&index, row) in openings.main.indices.iter().zip(&openings.main.values) {
            let scalar: F = state.verifier_message();
            claims
                .payload
                .add_in_domain(0, folded_size, self.rounds[round].rows, index, scalar);
            claims.sum += scalar * dot(row, &weights);
        }
        let mask_len = 2 * self.folds + 1;
        let mask_dimensions = self.mask_dimensions();
        let mask_prefix = mask_dimensions.prefix;
        let moments = mask_prefix + mask_len;
        let use_fft = moments.saturating_mul(openings.mask.indices.len())
            > mask_dimensions
                .rows
                .saturating_mul(2 * mask_dimensions.rows.ilog2() as usize);
        let mask_size = if use_fft {
            mask_dimensions.rows
        } else {
            moments
        };
        let mut mask_covector = vec![F::ZERO; mask_size];
        let mut scalars = Vec::new();
        for (&index, row) in openings.mask.indices.iter().zip(&openings.mask.values) {
            let scalar: F = state.verifier_message();
            if use_fft {
                mask_covector[index] += scalar;
            } else {
                scalars.push(scalar);
            }
            claims.sum += scalar * row[0];
        }
        if use_fft {
            // FFT(weights)[i] = sum_j alpha_j * z_j^i for the sampled domain points z_j.
            ntt(&mut mask_covector);
        } else {
            geometric_accumulate(&mut mask_covector, scalars, &openings.mask.points);
        }
        claims.payload.add_coefficients(
            folded_size,
            &mask_covector[mask_prefix..mask_prefix + mask_len],
        );
        claims.payload.add_coefficients(
            folded_size + self.mask_segment_size(),
            &mask_covector[..mask_prefix],
        );
        claims
    }

    fn prove_final<R: RngCore + CryptoRng>(
        &self,
        state: &mut ProverState<StdHash, R>,
        witness: &OracleWitness<F>,
        vector: &[F],
        mut form: Vec<F>,
        claim: F,
    ) {
        let dimensions = *self.rounds.last().expect("nonempty schedule");
        let id = 2 * (self.rounds.len() - 1);
        let mask: Vec<_> = (0..vector.len()).map(|_| F::rand(state.rng())).collect();
        let mut mask_witness = self.commit_oracle(state, dimensions, mask, id + 1);
        let mut mask = take(&mut mask_witness.table);
        let sum = dot(&form, &mask);
        state.prover_message(&sum);
        let epsilon: F = state.verifier_message();
        for (value, &witness_value) in mask.iter_mut().zip(vector) {
            *value += epsilon * witness_value;
            state.prover_message(value);
        }
        debug_assert_eq!(dot(&form, &mask), sum + epsilon * claim);
        let old = self.open_oracle(state, dimensions, witness);
        // Replay the same locations for both roots without drawing new challenges.
        let mask_openings = self.open_at_indices(state, dimensions, &mask_witness, &old.indices);
        debug_assert!(old
            .points
            .iter()
            .zip(&old.values)
            .zip(&mask_openings.values)
            .all(|((&point, row), masked)| univariate_evaluate(&mask, point)
                == masked[0] + epsilon * row[0]));
        let mut target = sum + epsilon * claim;
        self.final_sumcheck
            .prove(state, &mut mask, &mut form, &mut target);
    }

    fn verify_final(
        &self,
        state: &mut VerifierState<'_>,
        commitment: &OracleCommitment,
        statement: &Statement<F>,
    ) -> VerificationResult<()> {
        let dimensions = *self.rounds.last().expect("nonempty schedule");
        let id = 2 * (self.rounds.len() - 1);
        let mask_commitment = self.receive_oracle(state, dimensions)?;
        let sum: F = state.prover_message()?;
        let epsilon: F = state.verifier_message();
        let masked: Vec<F> = state.prover_messages_vec(dimensions.table_size())?;
        let old = self.verify_oracle(state, dimensions, commitment, id)?;
        let mask =
            self.verify_at_indices(state, dimensions, &mask_commitment, id + 1, &old.indices)?;
        let evaluations = (masked.len().saturating_mul(old.indices.len())
            > dimensions
                .rows
                .saturating_mul(2 * dimensions.rows.ilog2() as usize))
        .then(|| interleaved_rs_encode(&[&masked], dimensions.rows, 1));
        for (((&index, &point), row), mask_value) in old
            .indices
            .iter()
            .zip(&old.points)
            .zip(&old.values)
            .zip(mask.values)
        {
            let value = evaluations.as_ref().map_or_else(
                || univariate_evaluate(&masked, point),
                |evaluations| evaluations[index],
            );
            verify!(value == mask_value[0] + epsilon * row[0]);
        }
        let mut target = sum + epsilon * statement.sum;
        let challenges = self.final_sumcheck.verify(state, &mut target)?;
        verify!(
            target
                == multilinear_extend(&masked, &challenges.0)
                    * statement.form.mle_evaluate(&challenges.0)
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests;
