//! Non-succinct linear opening (Construction 7.2, p.43). HVZK in ZK mode.

use ark_ff::Field;
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use spongefish::{Decoding, VerificationResult};

use crate::{
    algebra::{embedding::Identity, multilinear_extend, univariate_evaluate},
    buffer::{ActiveBuffer, Buffer, BufferOps},
    hash::Hash,
    protocols::{irs_commit, proof_of_work, sumcheck},
    transcript::{
        codecs::U64, Codec, DuplexSpongeInterface, ProverMessage, ProverState, VerifierMessage,
        VerifierState,
    },
    utils::zip_strict,
    verify,
};

#[must_use]
pub struct Opening<F: Field> {
    pub evaluation_points: Vec<F>,
    pub linear_form_evaluation: F,
}

/// Standard / ZeroKnowledge selector for basecase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BasecaseMode {
    Standard,
    ZeroKnowledge,
}

#[must_use]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Config<F: Field> {
    commit: irs_commit::Config<Identity<F>>,
    sumcheck: sumcheck::Config<F>,
    mode: BasecaseMode,
    pow: proof_of_work::Config,
}

impl<F: Field> Config<F> {
    /// Standard basecase has no γ challenge — PoW must be `none()`. ZK
    /// basecase has a γ-combination slot (Lemma 7.4) and may or may not need
    /// PoW depending on whether the analytic floor already clears the target
    /// (under unique decoding it often does).
    pub fn new(
        commit: irs_commit::Config<Identity<F>>,
        sumcheck: sumcheck::Config<F>,
        mode: BasecaseMode,
        pow: proof_of_work::Config,
    ) -> Self {
        let has_pow = pow != proof_of_work::Config::none();
        debug_assert!(
            !matches!(mode, BasecaseMode::Standard) || !has_pow,
            "Standard basecase has no γ challenge — pow must be none()",
        );
        Self {
            commit,
            sumcheck,
            mode,
            pow,
        }
    }

    pub const fn commit(&self) -> &irs_commit::Config<Identity<F>> {
        &self.commit
    }

    pub const fn sumcheck(&self) -> &sumcheck::Config<F> {
        &self.sumcheck
    }

    pub const fn mode(&self) -> BasecaseMode {
        self.mode
    }

    pub const fn pow(&self) -> proof_of_work::Config {
        self.pow
    }

    #[cfg(test)]
    pub(crate) const fn set_pow_for_test(&mut self, pow: proof_of_work::Config) {
        self.pow = pow;
    }

    pub const fn size(&self) -> usize {
        self.sumcheck.initial_size()
    }

    pub const fn is_zk(&self) -> bool {
        matches!(self.mode, BasecaseMode::ZeroKnowledge)
    }

    pub fn prove<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        mut vector: ActiveBuffer<F>,
        witness: &irs_commit::Witness<F>,
        mut covector: ActiveBuffer<F>,
        mut sum: F,
    ) -> Opening<F>
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
        Standard: Distribution<F>,
    {
        assert_eq!(self.commit.interleaving_depth(), 1);
        assert_eq!(self.commit.num_vectors(), 1);
        assert_eq!(self.commit.vector_size(), self.sumcheck.initial_size());
        assert_eq!(self.sumcheck.final_size(), 1.min(self.commit.vector_size()));
        debug_assert_eq!(vector.dot(&covector), sum);
        if self.size() == 0 {
            return Opening {
                evaluation_points: Vec::new(),
                linear_form_evaluation: F::ZERO,
            };
        }

        let blinding_witness =
            self.maybe_blind_prove(prover_state, &mut vector, witness, &covector, &mut sum);

        let witnesses: Vec<&irs_commit::Witness<F>> = blinding_witness
            .as_ref()
            .map_or_else(|| vec![witness], |b| vec![b, witness]);
        let _ = self.commit.open(prover_state, &witnesses);

        let point = self
            .sumcheck
            .prove(prover_state, &mut vector, &mut covector, &mut sum, &[])
            .round_challenges;

        // Negligible event over a challenge-sized field; without it the verifier
        // cannot derive `l(r) = sum / vector_mle(r)`.
        assert!(
            !vector.to_slice().first().expect("Proof failed").is_zero(),
            "Proof failed"
        );

        Opening {
            evaluation_points: point,
            linear_form_evaluation: *covector.to_slice().first().expect("Proof failed"),
        }
    }

    /// ZK: commits a blinding codeword, runs the RLC, mutates `vector`/`sum` to
    /// the combined values, sends them cleartext. Standard: sends `vector` and
    /// `witness.masks` cleartext (no ZK).
    fn maybe_blind_prove<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        vector: &mut ActiveBuffer<F>,
        witness: &irs_commit::Witness<F>,
        covector: &ActiveBuffer<F>,
        sum: &mut F,
    ) -> Option<irs_commit::Witness<F>>
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
        Standard: Distribution<F>,
    {
        match self.mode {
            BasecaseMode::Standard => {
                prover_state.prover_messages(vector.to_slice());
                prover_state.prover_messages(witness.masks.to_slice());
                None
            }
            BasecaseMode::ZeroKnowledge => {
                let mut blinding_vector = ActiveBuffer::random(prover_state.rng(), vector.len());
                let blinding_witness = self.commit.commit(prover_state, &[&blinding_vector]);
                let blinding_inner_product = blinding_vector.dot(covector);
                prover_state.prover_message(&blinding_inner_product);

                // Grind the Theorem 7.1 γ-combination gap before γ is sampled.
                self.pow.prove(prover_state);

                let combination_randomness = prover_state.verifier_message::<F>();
                assert!(!combination_randomness.is_zero(), "Proof failed");

                vector.mixed_scalar_mul_add_to(
                    &Identity::<F>::new(),
                    &mut blinding_vector,
                    combination_randomness,
                );
                *vector = blinding_vector;
                prover_state.prover_messages(vector.to_slice());

                let mut combined_irs_randomness = blinding_witness.masks.clone();
                witness.masks.mixed_scalar_mul_add_to(
                    &Identity::<F>::new(),
                    &mut combined_irs_randomness,
                    combination_randomness,
                );
                prover_state.prover_messages(combined_irs_randomness.to_slice());

                *sum = blinding_inner_product + combination_randomness * *sum;
                Some(blinding_witness)
            }
        }
    }

    pub fn verify<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
        commitment: &irs_commit::Commitment,
        mut sum: F,
    ) -> VerificationResult<Opening<F>>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(self.commit.interleaving_depth(), 1);
        assert_eq!(self.commit.num_vectors(), 1);
        assert_eq!(self.commit.vector_size(), self.sumcheck.initial_size());
        assert_eq!(self.sumcheck.final_size(), 1.min(self.commit.vector_size()));
        if self.size() == 0 {
            return Ok(Opening {
                evaluation_points: Vec::new(),
                linear_form_evaluation: F::ZERO,
            });
        }

        let blind = self.maybe_receive_blind(verifier_state, &mut sum)?;

        let vector = verifier_state.prover_messages_vec(self.commit.vector_size())?;
        let irs_randomness = verifier_state
            .prover_messages_vec(self.commit.mask_length() * self.commit.num_messages())?;

        let (commitments, weights): (Vec<&irs_commit::Commitment>, Vec<F>) = match &blind {
            Some((b, gamma)) => (vec![b, commitment], vec![F::ONE, *gamma]),
            None => (vec![commitment], vec![F::ONE]),
        };
        let evals = self.commit.verify(verifier_state, &commitments)?;

        // Spot-check: Enc_C(vector, irs_randomness)(x) = Σ weights · opened_row(x).
        for (&point, value) in zip_strict(&evals.points, evals.values(&weights)) {
            let expected = univariate_evaluate(&vector, point)
                + point.pow([self.commit.message_length() as u64])
                    * univariate_evaluate(&irs_randomness, point);
            verify!(value == expected);
        }

        let point = self
            .sumcheck
            .verify(verifier_state, &mut sum)?
            .round_challenges;

        // l(r) = sum / vector_mle(r), where l is the implicit linear form.
        let mle = multilinear_extend(&vector, &point);
        verify!(!mle.is_zero());
        let linear_mle = sum / mle;

        Ok(Opening {
            evaluation_points: point,
            linear_form_evaluation: linear_mle,
        })
    }

    /// ZK: reads the blinding commitment + μ' + γ, mutates `sum` to the
    /// combined value, returns `(commitment, γ)`. Standard: no-op.
    fn maybe_receive_blind<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
        sum: &mut F,
    ) -> VerificationResult<Option<(irs_commit::Commitment, F)>>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        match self.mode {
            BasecaseMode::Standard => Ok(None),
            BasecaseMode::ZeroKnowledge => {
                let blinding_commitment = self.commit.receive_commitment(verifier_state)?;
                let blinding_inner_product: F = verifier_state.prover_message()?;
                // Grind the Theorem 7.1 γ-combination gap before γ is sampled.
                self.pow.verify(verifier_state)?;
                let combination_randomness: F = verifier_state.verifier_message();
                verify!(!combination_randomness.is_zero());
                *sum = blinding_inner_product + combination_randomness * *sum;
                Ok(Some((blinding_commitment, combination_randomness)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use proptest::{bool, prelude::Strategy, proptest};
    #[cfg(feature = "tracing")]
    use tracing::instrument;

    use super::*;
    use crate::{algebra::fields, protocols::proof_of_work, transcript::DomainSeparator};

    impl<F: Field> Config<F> {
        pub fn arbitrary(size: usize, mask_length: usize) -> impl Strategy<Value = Self> {
            let commit =
                irs_commit::Config::arbitrary(Identity::<F>::new(), 1, size, mask_length, 1);
            (commit, bool::weighted(0.8)).prop_map(move |(commit, is_zk)| Self {
                commit,
                sumcheck: sumcheck::Config::new(
                    size,
                    proof_of_work::Config::none(),
                    size.next_power_of_two().trailing_zeros() as usize,
                    sumcheck::SumcheckMode::Standard,
                ),
                mode: if is_zk {
                    BasecaseMode::ZeroKnowledge
                } else {
                    BasecaseMode::Standard
                },
                pow: proof_of_work::Config::none(),
            })
        }
    }

    #[cfg_attr(feature = "tracing", instrument)]
    fn test_config<F>(seed: u64, config: &Config<F>)
    where
        F: Field + Codec,
        Standard: Distribution<F>,
    {
        let instance = U64(seed);
        let ds = DomainSeparator::protocol(config)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&instance);
        let mut rng = StdRng::seed_from_u64(seed);
        let vector = ActiveBuffer::random(&mut rng, config.size());
        let covector = ActiveBuffer::random(&mut rng, config.size());
        let sum = vector.dot(&covector);

        let mut prover_state = ProverState::new_std(&ds);
        let witness = config.commit.commit(&mut prover_state, &[&vector]);
        let prover_result = config.prove(
            &mut prover_state,
            vector.clone(),
            &witness,
            covector.clone(),
            sum,
        );
        assert_eq!(
            multilinear_extend(covector.to_slice(), &prover_result.evaluation_points),
            prover_result.linear_form_evaluation
        );
        let proof = prover_state.proof();

        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let commitment = config
            .commit
            .receive_commitment(&mut verifier_state)
            .unwrap();
        let verifier_result = config
            .verify(&mut verifier_state, &commitment, sum)
            .unwrap();
        assert_eq!(
            verifier_result.evaluation_points,
            prover_result.evaluation_points
        );
        assert_eq!(
            verifier_result.linear_form_evaluation,
            prover_result.linear_form_evaluation
        );
        verifier_state.check_eof().unwrap();
    }

    fn test<F: Field + Codec>()
    where
        Standard: Distribution<F>,
    {
        crate::tests::init();
        let configs = (0_usize..1 << 10, 0_usize..1 << 10)
            .prop_flat_map(|(size, mask_length)| Config::arbitrary(size, mask_length));
        proptest!(|(seed: u64, config in configs)| {
            test_config(seed, &config);
        });
    }

    #[test]
    fn test_field64_1() {
        test::<fields::Field64>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field64_2() {
        test::<fields::Field64_2>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field64_3() {
        test::<fields::Field64_3>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field128() {
        test::<fields::Field128>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field192() {
        test::<fields::Field192>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field256() {
        test::<fields::Field256>();
    }
}
