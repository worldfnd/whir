// NOTE ON BUFFER ABSTRACTION: since verifier is not succinct, full vector readbacks are forced
// consider alternative approaches here

//! Base Case Linear Opening Protocol
//!
//! It support honest verifier zero-knowledge (HVZK), but is not succinct.
//!
//! <https://eprint.iacr.org/2026/391.pdf> § 7.

use ark_ff::Field;
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use spongefish::{Decoding, VerificationResult};

use crate::{
    algebra::{embedding::Identity, multilinear_extend, univariate_evaluate},
    buffer::{ActiveBuffer, Buffer, BufferOps},
    hash::Hash,
    protocols::{irs_commit, sumcheck},
    transcript::{
        codecs::U64, Codec, DuplexSpongeInterface, ProverMessage, ProverState, VerifierMessage,
        VerifierState,
    },
    utils::zip_strict,
    verify,
};

/// Output from the base case protocol (shared by prover and verifier).
#[must_use]
pub struct Opening<F: Field> {
    pub evaluation_points: Vec<F>,
    pub linear_form_evaluation: F,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Config<F: Field> {
    pub commit: irs_commit::Config<Identity<F>>,
    pub sumcheck: sumcheck::Config<F>,

    /// Whether to mask the vectors, which adds HVZK.
    pub masked: bool,
}

impl<F: Field> Config<F> {
    pub const fn size(&self) -> usize {
        self.sumcheck.initial_size
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
        assert_eq!(self.commit.interleaving_depth, 1);
        assert_eq!(self.commit.num_vectors, 1);
        assert_eq!(self.commit.vector_size, self.sumcheck.initial_size);
        assert_eq!(self.sumcheck.final_size(), 1.min(self.commit.vector_size));
        debug_assert_eq!(vector.dot(&covector), sum);
        if self.size() == 0 {
            return Opening {
                evaluation_points: Vec::new(),
                linear_form_evaluation: F::ZERO,
            };
        }

        // Even more trivial non-zk protocol: send f and r directly.
        if !self.masked {
            // TODO: avoid these big readbacks even though they are transcript-sized
            prover_state.prover_messages(vector.as_slice());
            prover_state.prover_messages(witness.masks.as_slice());
            let _ = self.commit.open(prover_state, &[witness]);
            let point = self
                .sumcheck
                .prove(prover_state, &mut vector, &mut covector, &mut sum, &[])
                .round_challenges;
            assert!(
                !vector.as_slice().first().expect("Proof failed").is_zero(),
                "Proof failed"
            );
            return Opening {
                evaluation_points: point,
                linear_form_evaluation: *covector.as_slice().first().expect("Proof failed"),
            };
        }

        // Create masking vector.
        let mask = ActiveBuffer::random(prover_state.rng(), vector.len());

        // Commit to the masking vector.
        let mask_witness = self.commit.commit(prover_state, &[&mask]);

        // Compute and send linear form of mask (μ' in paper).
        let mask_sum = mask.dot(&covector);
        prover_state.prover_message(&mask_sum);

        // RLC the mask with the vector
        let mask_rlc = prover_state.verifier_message::<F>();
        assert!(!mask_rlc.is_zero(), "Proof failed");
        let mut masked_vector = mask;
        vector.mixed_scalar_mul_add_to(&Identity::<F>::new(), &mut masked_vector, mask_rlc);
        prover_state.prover_messages(masked_vector.as_slice());

        // Send combined IRS randomness. (r^* in paper)
        let mut masked_masks = mask_witness.masks.clone();
        witness
            .masks
            .mixed_scalar_mul_add_to(&Identity::<F>::new(), &mut masked_masks, mask_rlc);
        prover_state.prover_messages(masked_masks.as_slice());

        // Open the commitment and mask simultaneously.
        let _ = self.commit.open(prover_state, &[&mask_witness, witness]);

        // Run sumcheck to reduce linear form claim
        let mut masked_sum = mask_sum + mask_rlc * sum;
        let point = self
            .sumcheck
            .prove(
                prover_state,
                &mut masked_vector,
                &mut covector,
                &mut masked_sum,
                &[],
            )
            .round_challenges;

        // If the MLE of `masked_vector` evaluates to zero, the verifier can not proceed.
        // Basically the sumcheck equation has degenerated to 0 * l(r) = 0, which provides
        // no constraints on l(r) that the verifier can return.
        // This event is cryptographically unlikely as `F` is challenge sized.
        assert!(
            !masked_vector
                .as_slice()
                .first()
                .expect("Proof failed")
                .is_zero(),
            "Proof failed"
        );

        Opening {
            evaluation_points: point,
            linear_form_evaluation: *covector.as_slice().first().expect("Proof failed"),
        }
    }

    pub fn verify<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
        commitment: &irs_commit::Commitment<F>,
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
        assert_eq!(self.commit.interleaving_depth, 1);
        assert_eq!(self.commit.num_vectors, 1);
        assert_eq!(self.commit.vector_size, self.sumcheck.initial_size);
        assert_eq!(self.sumcheck.final_size(), 1.min(self.commit.vector_size));
        if self.size() == 0 {
            return Ok(Opening {
                evaluation_points: Vec::new(),
                linear_form_evaluation: F::ZERO,
            });
        }

        // Unmasked protocol
        if !self.masked {
            let vector = verifier_state.prover_messages_vec(self.commit.vector_size)?;
            let masks = verifier_state
                .prover_messages_vec(self.commit.mask_length * self.commit.num_messages())?;
            let evals = self.commit.verify(verifier_state, &[commitment])?;
            let point = self
                .sumcheck
                .verify(verifier_state, &mut sum)?
                .round_challenges;

            for (&point, value) in zip_strict(&evals.points, evals.values(&[F::ONE])) {
                // We expected `f(x) + x^l · g(x)` where l = deg(f) + 1, f is the message and g the mask.
                let expected = univariate_evaluate(&vector, point)
                    + point.pow([self.commit.message_length() as u64])
                        * univariate_evaluate(&masks, point);
                verify!(value == expected);
            }
            let mle = multilinear_extend(&vector, &point);
            verify!(!mle.is_zero());
            let linear_mle = sum / mle;
            return Ok(Opening {
                evaluation_points: point,
                linear_form_evaluation: linear_mle,
            });
        }

        let mask_commitment = self.commit.receive_commitment(verifier_state)?;
        let mask_sum: F = verifier_state.prover_message()?;
        let mask_rlc: F = verifier_state.verifier_message();
        verify!(!mask_rlc.is_zero());
        let masked_vector: Vec<F> = verifier_state.prover_messages_vec(self.commit.vector_size)?;
        let masked_masks: Vec<F> = verifier_state.prover_messages_vec(self.commit.mask_length)?;

        // Open the commitment and mask simultaneously.
        let evals = self
            .commit
            .verify(verifier_state, &[&mask_commitment, commitment])?;

        // Spot check evaluations.
        for (&point, value) in zip_strict(&evals.points, evals.values(&[F::ONE, mask_rlc])) {
            // We expected `f(x) + x^l · g(x)` where l = deg(f) + 1, f is the message and g the mask.
            let expected = univariate_evaluate(&masked_vector, point)
                + point.pow([self.commit.message_length() as u64])
                    * univariate_evaluate(&masked_masks, point);
            verify!(value == expected);
        }

        // Sumcheck on masked inner product
        let mut masked_sum = mask_sum + mask_rlc * sum;
        let point = self
            .sumcheck
            .verify(verifier_state, &mut masked_sum)?
            .round_challenges;

        // Compute implied MLE of the linear form
        // f*(r) · l(r) = sum  =>  l(r) = sum / f*(r)
        let masked_mle = multilinear_extend(&masked_vector, &point);
        verify!(!masked_mle.is_zero());
        let linear_mle = masked_sum / masked_mle;

        Ok(Opening {
            evaluation_points: point,
            linear_form_evaluation: linear_mle,
        })
    }
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use proptest::{bool, prelude::Strategy, proptest};
    #[cfg(feature = "tracing")]
    use tracing::instrument;

    use super::*;
    use crate::{
        algebra::fields, protocols::proof_of_work, transcript::DomainSeparator, type_info::Type,
    };

    impl<F: Field> Config<F> {
        pub fn arbitrary(size: usize, mask_length: usize) -> impl Strategy<Value = Self> {
            let commit =
                irs_commit::Config::arbitrary(Identity::<F>::new(), 1, size, mask_length, 1);
            (commit, bool::weighted(0.8)).prop_map(move |(commit, masked)| Self {
                commit: irs_commit::Config {
                    out_domain_samples: 0,
                    ..commit
                },
                sumcheck: sumcheck::Config {
                    field: Type::new(),
                    initial_size: size,
                    round_pow: proof_of_work::Config::none(),
                    num_rounds: size.next_power_of_two().trailing_zeros() as usize,
                    mask_length: 0,
                },
                masked,
            })
        }
    }

    #[cfg_attr(feature = "tracing", instrument)]
    fn test_config<F>(seed: u64, config: &Config<F>)
    where
        F: Field + Codec,
        Standard: Distribution<F>,
    {
        // Pseudo-random Instance
        let instance = U64(seed);
        let ds = DomainSeparator::protocol(config)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&instance);
        let mut rng = StdRng::seed_from_u64(seed);
        let vector = ActiveBuffer::random(&mut rng, config.size());
        let covector = ActiveBuffer::random(&mut rng, config.size());
        let sum = vector.dot(&covector);

        // Prover
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
            multilinear_extend(covector.as_slice(), &prover_result.evaluation_points),
            prover_result.linear_form_evaluation
        );
        let proof = prover_state.proof();

        // Verifier
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
