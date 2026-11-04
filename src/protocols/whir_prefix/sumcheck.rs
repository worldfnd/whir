use ark_ff::FftField;
use ark_std::rand::{CryptoRng, RngCore};
use spongefish::StdHash;

use super::{oracle::Witness, Config};
use crate::{
    algebra::sumcheck::{compute_sumcheck_polynomial, fold},
    protocols::matrix_commit::Commitment,
    transcript::{Codec, ProverState, VerificationResult, VerifierMessage, VerifierState},
};

pub(super) struct Challenges<F> {
    pub folding: Vec<F>,
    pub mask_weights: Vec<F>,
    pub rho: F,
}

impl<F: FftField + Codec> Config<F> {
    pub(super) fn prove_masked_sumcheck<R: RngCore + CryptoRng>(
        &self,
        state: &mut ProverState<StdHash, R>,
        vector: &mut Vec<F>,
        covector: &mut Vec<F>,
        target: &mut F,
        oracle_id: usize,
    ) -> (Witness<F>, Challenges<F>) {
        let variables = vector.len().ilog2() as usize;
        let coefficients: Vec<_> = (0..=2 * self.folds).map(|_| F::rand(state.rng())).collect();
        let mask_dimensions = self.mask_dimensions();
        let mask_table = Self::encode(mask_dimensions, &coefficients, state.rng());
        let witness = self.commit_oracle(state, mask_dimensions, mask_table, oracle_id);

        let inverse_two = F::from(2).inverse().expect("odd-characteristic field");
        let mut suffix: F = coefficients[1..].iter().sum();
        let mut scale = F::from(2).pow([(variables - 1) as u64]);
        let mut constant = coefficients[0];
        let mask_sum = scale * (constant.double() + suffix);
        state.prover_message(&mask_sum);
        let rho: F = state.verifier_message();
        *target = rho * *target + mask_sum;
        for value in covector.iter_mut() {
            *value *= rho;
        }

        let mut folding = Vec::with_capacity(self.folds);
        for pair in coefficients[1..].chunks_exact(2) {
            let (a, b) = (pair[0], pair[1]);
            suffix -= a + b;
            let (ordinary_constant, ordinary_quadratic) =
                compute_sumcheck_polynomial(vector, covector);
            let c0 = ordinary_constant + scale * (constant + suffix * inverse_two);
            let c2 = ordinary_quadratic + scale * b;
            let c1 = *target - c0.double() - c2;
            state.prover_message(&c0);
            state.prover_message(&c2);
            let challenge: F = state.verifier_message();
            folding.push(challenge);
            fold(vector, challenge);
            fold(covector, challenge);
            *target = (c2 * challenge + c1) * challenge + c0;
            constant += a * challenge + b * challenge.square();
            scale *= inverse_two;
        }
        let mask_weights = mask_weights(variables, &folding);
        (
            witness,
            Challenges {
                folding,
                mask_weights,
                rho,
            },
        )
    }

    pub(super) fn verify_masked_sumcheck(
        &self,
        state: &mut VerifierState<'_>,
        variables: usize,
        target: &mut F,
    ) -> VerificationResult<(Commitment, Challenges<F>)> {
        let commitment = self.receive_oracle(state, self.mask_dimensions())?;
        let mask_sum: F = state.prover_message()?;
        let rho: F = state.verifier_message();
        *target = rho * *target + mask_sum;
        let mut sumcheck = self.final_sumcheck.clone();
        sumcheck.initial_size = 1 << variables;
        sumcheck.num_rounds = self.folds;
        let folding = sumcheck.verify(state, target)?.0;
        let mask_weights = mask_weights(variables, &folding);
        Ok((
            commitment,
            Challenges {
                folding,
                mask_weights,
                rho,
            },
        ))
    }
}

fn mask_weights<F: FftField>(variables: usize, folding: &[F]) -> Vec<F> {
    let scale = F::from(2).pow([(variables - folding.len()) as u64]);
    let mut weights = vec![scale];
    for &challenge in folding {
        weights.extend([scale * challenge, scale * challenge.square()]);
    }
    weights
}
