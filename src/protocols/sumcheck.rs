//! Quadratic sumcheck protocol.

use std::fmt;

use ark_ff::Field;
use ark_std::rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::{
    algebra::univariate_evaluate,
    buffer::{ActiveBuffer, Buffer, BufferOps},
    protocols::proof_of_work,
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverState, VerificationResult,
        VerifierMessage, VerifierState,
    },
    type_info::Type,
    utils::chunks_exact_or_empty,
};

/// Output from the sumcheck protocol (shared by prover and verifier).
#[must_use]
pub struct SumcheckOpening<F: Field> {
    pub round_challenges: Vec<F>,
    pub mask_rlc: F,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Config<F>
where
    F: Field,
{
    pub field: Type<F>,
    pub initial_size: usize,
    pub round_pow: proof_of_work::Config,
    pub num_rounds: usize,
    pub mask_length: usize,
}

impl<F: Field> Config<F> {
    pub fn final_size(&self) -> usize {
        assert!(
            self.num_rounds == 0 || self.initial_size.next_power_of_two() >= 1 << self.num_rounds
        );
        if self.initial_size == 0 || self.num_rounds == 0 {
            self.initial_size
        } else {
            self.initial_size.next_power_of_two() >> self.num_rounds
        }
    }

    /// Runs the quadratic sumcheck protocol as configured.
    ///
    /// It reduces a claim of the form `dot(a, b) == sum` to an exponentially
    /// smaller claim `dot(a', b') == sum'` where `a'` is `a` folded in place
    /// and similarly for `b`.
    ///
    /// This function:
    /// - Samples random values to progressively reduce the polynomial.
    /// - Applies proof-of-work grinding if required.
    /// - Returns the sampled folding randomness values used in each reduction step.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn prove<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        a: &mut ActiveBuffer<F>,
        b: &mut ActiveBuffer<F>,
        sum: &mut F,
        masks: &[F],
    ) -> SumcheckOpening<F>
    where
        H: DuplexSpongeInterface,
        R: CryptoRng + RngCore,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
    {
        assert!(
            self.num_rounds == 0 || self.initial_size.next_power_of_two() >= 1 << self.num_rounds
        );
        // Mask must cover all 3 sumcheck polynomial coefficients (c0, c1, c2)
        assert!(self.mask_length == 0 || self.mask_length >= 3);
        assert_eq!(a.len(), self.initial_size);
        assert_eq!(b.len(), self.initial_size);
        debug_assert_eq!(a.dot(b), *sum);
        assert_eq!(masks.len(), self.num_rounds * self.mask_length);
        let half = F::from(2).inverse().unwrap();

        let mut mask_sum = F::ZERO;
        let mut mask_rlc = F::ONE;
        if self.mask_length > 0 && self.num_rounds > 0 {
            let sum_multiple = F::from(1 << self.num_rounds.saturating_sub(1));
            mask_sum = masks.chunks_exact(self.mask_length).map(eval_01).sum::<F>() * sum_multiple;
            prover_state.prover_message(&mask_sum);
            mask_rlc = prover_state.verifier_message();
        }

        // We do a staggered Sumcheck loop so we can merge the inner fold+compute loops.
        let mut univariate = Vec::new();
        let mut round_challenges = Vec::with_capacity(self.num_rounds);
        let mut prev_round_challenge = None;
        for (round, mask) in
            chunks_exact_or_empty(masks, self.mask_length, self.num_rounds).enumerate()
        {
            // Fold and compute sumcheck polynomial in one pass.
            let (c0, c2) = if let Some(w) = prev_round_challenge {
                a.fold_pair_sumcheck_polynomial(b, w)
            } else {
                a.sumcheck_polynomial(b)
            };
            let c1 = *sum - c0.double() - c2;

            // Optionally mask with univariate
            if mask.is_empty() {
                prover_state.prover_messages(&[c0, c2]);
            } else {
                // Initialize to round masking univariate polynomial.
                univariate.clear();
                let sum_multiple = F::from(1 << self.num_rounds.saturating_sub(round + 1));
                univariate.extend(mask.iter().map(|m| sum_multiple * *m));

                // Add constant term from previous and future masks.
                univariate[0] += (mask_sum - sum_multiple * eval_01(mask)) * half;

                // Add plain sumcheck polynomial
                univariate[0] += mask_rlc * c0;
                univariate[1] += mask_rlc * c1;
                univariate[2] += mask_rlc * c2;

                prover_state.prover_message(&univariate[0]);
                prover_state.prover_messages(&univariate[2..]);
            }

            // Receive the random evaluation point and update the sum
            self.round_pow.prove(prover_state);
            let r = prover_state.verifier_message::<F>();
            round_challenges.push(r);
            *sum = (c2 * r + c1) * r + c0;
            if self.mask_length > 0 && self.num_rounds > 0 {
                let masked_sum = univariate_evaluate(&univariate, r);
                mask_sum = masked_sum - mask_rlc * *sum;
            }
            prev_round_challenge = Some(r);
        }
        if let Some(w) = prev_round_challenge {
            // Final fold of the inputs (no polynomial computation)
            a.fold_pair(b, w);
        }

        *sum = mask_sum + mask_rlc * *sum;
        SumcheckOpening {
            round_challenges,
            mask_rlc,
        }
    }

    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn verify<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
        sum: &mut F,
    ) -> VerificationResult<SumcheckOpening<F>>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
    {
        assert!(
            self.num_rounds == 0 || self.initial_size.next_power_of_two() >= 1 << self.num_rounds
        );
        // Mask must cover all 3 sumcheck polynomial coefficients (c0, c1, c2)
        assert!(self.mask_length == 0 || self.mask_length >= 3);

        let mut mask_rlc = F::ONE;
        if self.mask_length > 0 && self.num_rounds > 0 {
            let mask_sum: F = verifier_state.prover_message()?;
            mask_rlc = verifier_state.verifier_message();
            *sum = mask_sum + mask_rlc * *sum;
        }

        let mut univariate = vec![F::ZERO; self.mask_length.max(3)];
        let mut round_challenges = Vec::with_capacity(self.num_rounds);
        for _ in 0..self.num_rounds {
            // Receive all but linear coefficient.
            univariate[0] = verifier_state.prover_message()?;
            for c in &mut univariate[2..] {
                *c = verifier_state.prover_message()?;
            }

            // Derive linear coefficient from relation `univariate(0) + univariate(1) = sum`
            univariate[1] = *sum - univariate[0].double() - univariate[2..].iter().sum::<F>();

            // Check proof of work (if any)
            self.round_pow.verify(verifier_state)?;

            // Receive the random evaluation point
            let round_challenge = verifier_state.verifier_message::<F>();
            round_challenges.push(round_challenge);

            // Update the sum
            *sum = univariate_evaluate(&univariate, round_challenge);
        }
        Ok(SumcheckOpening {
            round_challenges,
            mask_rlc,
        })
    }
}

impl<F: Field> fmt::Display for Config<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "size {} rounds {} pow {:.2} ℓ_zk {}",
            self.initial_size,
            self.num_rounds,
            self.round_pow.difficulty(),
            self.mask_length
        )
    }
}

// Evaluated a univariate as p(0) + p(1)
fn eval_01<F: Field>(coefficients: &[F]) -> F {
    if coefficients.is_empty() {
        return F::ZERO;
    }
    coefficients[0] + coefficients.iter().sum::<F>()
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{
        distributions::{Distribution, Standard},
        rngs::StdRng,
        SeedableRng,
    };
    use proptest::{prelude::Just, prop_oneof, proptest, strategy::Strategy};
    #[cfg(feature = "tracing")]
    use tracing::instrument;

    use super::*;
    use crate::{
        algebra::{
            dot,
            fields::{self, Field64},
            multilinear_extend, random_vector,
        },
        buffer::ActiveBuffer,
        transcript::DomainSeparator,
    };

    impl<F: Field + 'static> Config<F>
    where
        Standard: Distribution<F>,
    {
        pub fn arbitrary() -> impl Strategy<Value = Self> {
            let mask_length = prop_oneof![
                3 => Just(0_usize),
                7 => 3_usize..20,
            ];
            (0_usize..(1 << 12), 0_usize..12, mask_length).prop_map(
                |(initial_size, num_rounds, mask_length)| {
                    let num_rounds =
                        num_rounds.min(initial_size.next_power_of_two().trailing_zeros() as usize);
                    Self {
                        field: Type::new(),
                        initial_size,
                        num_rounds,
                        round_pow: proof_of_work::Config::none(),
                        mask_length,
                    }
                },
            )
        }
    }

    #[cfg_attr(feature = "tracing", instrument)]
    fn test_config<F>(seed: u64, config: &Config<F>)
    where
        F: Field + Codec<[u8]> + 'static,
        Standard: Distribution<F>,
    {
        // Pseudo-random Instance
        let instance = U64(seed);
        let ds = DomainSeparator::protocol(config)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&instance);
        let mut rng = StdRng::seed_from_u64(seed);
        let initial_vector = random_vector(&mut rng, config.initial_size);
        let initial_covector = random_vector(&mut rng, config.initial_size);
        let initial_sum = dot(&initial_vector, &initial_covector);
        let masks = random_vector(&mut rng, config.mask_length * config.num_rounds);

        // Prover
        let mut vector = ActiveBuffer::from_slice(&initial_vector);
        let mut covector = ActiveBuffer::from_slice(&initial_covector);
        let mut sum = initial_sum;
        let mut prover_state = ProverState::new_std(&ds);
        let SumcheckOpening {
            round_challenges: point,
            mask_rlc,
        } = config.prove(
            &mut prover_state,
            &mut vector,
            &mut covector,
            &mut sum,
            &masks,
        );
        assert_eq!(vector.len(), config.final_size());
        assert_eq!(covector.len(), config.final_size());
        if config.final_size() == 1 {
            assert_eq!(
                multilinear_extend(&initial_vector, &point),
                vector.as_slice()[0]
            );
            assert_eq!(
                multilinear_extend(&initial_covector, &point),
                covector.as_slice()[0]
            );
        } else {
            // TODO: Check correct folding.
        }

        let expected_mask_sum: F =
            chunks_exact_or_empty(&masks, config.mask_length, config.num_rounds)
                .zip(&point)
                .map(|(m, x)| univariate_evaluate(m, *x))
                .sum();
        assert_eq!(
            sum,
            expected_mask_sum + mask_rlc * dot(vector.as_slice(), covector.as_slice())
        );

        let proof = prover_state.proof();

        // Verifier
        let mut verifier_sum = initial_sum;
        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let SumcheckOpening {
            round_challenges: verifier_point,
            mask_rlc: verifier_mask_rlc,
        } = config
            .verify(&mut verifier_state, &mut verifier_sum)
            .unwrap();
        assert_eq!(verifier_point, point);
        assert_eq!(verifier_mask_rlc, mask_rlc);
        assert_eq!(verifier_sum, sum);
        verifier_state.check_eof().unwrap();

        // Non-ZK path: mask_rlc defaults to ONE (no combination randomness sampled)
        if config.mask_length == 0 || config.num_rounds == 0 {
            assert_eq!(mask_rlc, F::ONE);
        }
    }

    fn test<F: Field + Codec<[u8]> + 'static>()
    where
        Standard: Distribution<F>,
    {
        crate::tests::init();
        proptest!(|(seed: u64, config in Config::arbitrary())| {
            test_config(seed, &config);
        });
    }

    #[test]
    fn test_single_round() {
        test_config(
            0,
            &Config::<Field64> {
                field: Type::new(),
                initial_size: 2,
                round_pow: proof_of_work::Config::none(),
                num_rounds: 1,
                mask_length: 3,
            },
        );
    }

    #[test]
    fn test_two_rounds() {
        test_config(
            0,
            &Config::<Field64> {
                field: Type::new(),
                initial_size: 3,
                round_pow: proof_of_work::Config::none(),
                num_rounds: 2,
                mask_length: 3,
            },
        );
    }

    #[test]
    fn test_three_rounds() {
        test_config(
            0,
            &Config::<Field64> {
                field: Type::new(),
                initial_size: 5,
                round_pow: proof_of_work::Config::none(),
                num_rounds: 3,
                mask_length: 3,
            },
        );
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
