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

/// ZK sumcheck mask polynomial dimension.
///
/// Validated at construction to be at least `MIN = 3` — the round polynomial
/// has 3 coefficients (degree-2), so the mask must have at least as many to
/// hide it. Lemma 6.4 itself only requires `ℓ_zk ≥ 2`; the `3` floor is a
/// WHIR design choice tied to the degree-2 round polynomial (see
/// `params::sumcheck::zk_mask_length`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SumcheckMaskLen(usize);

impl SumcheckMaskLen {
    pub const MIN: usize = 3;

    pub const fn new(n: usize) -> Self {
        assert!(n >= Self::MIN);
        Self(n)
    }

    pub const fn get(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SumcheckMode {
    Standard,
    ZeroKnowledge { mask_length: SumcheckMaskLen },
}

#[must_use]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Config<F>
where
    F: Field,
{
    field: Type<F>,
    initial_size: usize,
    round_pow: proof_of_work::Config,
    num_rounds: usize,
    mode: SumcheckMode,
}

impl<F: Field> Config<F> {
    pub fn new(
        initial_size: usize,
        round_pow: proof_of_work::Config,
        num_rounds: usize,
        mode: SumcheckMode,
    ) -> Self {
        assert!(num_rounds == 0 || initial_size.next_power_of_two() >= 1 << num_rounds);
        // `SumcheckMaskLen::new` already enforces the ≥ 3 floor at construction;
        // here we only need the field-characteristic precondition from Lemma 6.4.
        if matches!(mode, SumcheckMode::ZeroKnowledge { .. }) {
            assert!(
                !F::ONE.double().is_zero(),
                "ZK sumcheck requires char(F) ≠ 2"
            );
        }
        Self {
            field: Type::new(),
            initial_size,
            round_pow,
            num_rounds,
            mode,
        }
    }

    pub const fn initial_size(&self) -> usize {
        self.initial_size
    }

    pub const fn round_pow(&self) -> proof_of_work::Config {
        self.round_pow
    }

    pub const fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    pub const fn mode(&self) -> &SumcheckMode {
        &self.mode
    }

    const fn mask_length(&self) -> usize {
        match &self.mode {
            SumcheckMode::Standard => 0,
            SumcheckMode::ZeroKnowledge { mask_length } => mask_length.get(),
        }
    }

    #[cfg(test)]
    pub(crate) const fn override_round_pow_for_test(&mut self, round_pow: proof_of_work::Config) {
        self.round_pow = round_pow;
    }

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
        assert_eq!(a.len(), self.initial_size);
        assert_eq!(b.len(), self.initial_size);
        debug_assert_eq!(a.dot(b), *sum);
        assert_eq!(masks.len(), self.num_rounds * self.mask_length());
        let half = F::from(2).inverse().unwrap();
        let polynomial_len = self.mask_length().max(3);

        let (mut mask_sum, mask_rlc) = self.maybe_send_initial_mask_sum(prover_state, masks);

        let mut univariate = Vec::with_capacity(polynomial_len);
        let mut round_challenges = Vec::with_capacity(self.num_rounds);
        let mut prev_round_challenge = None;
        for (round, mask) in
            chunks_exact_or_empty(masks, self.mask_length(), self.num_rounds).enumerate()
        {
            // Fold and compute sumcheck polynomial in one pass.
            let (c0, c2) = if let Some(w) = prev_round_challenge {
                a.fold_pair_sumcheck_polynomial(b, w)
            } else {
                a.sumcheck_polynomial(b)
            };
            let c1 = *sum - c0.double() - c2;

            // Build round polynomial. In Standard (`mask = []`, `mask_rlc = 1`,
            // `mask_sum = 0`) this collapses to `[c0, c1, c2]`.
            univariate.clear();
            univariate.resize(polynomial_len, F::ZERO);
            let sum_multiple = F::from(1 << self.num_rounds.saturating_sub(round + 1));
            for (u, m) in univariate.iter_mut().zip(mask.iter()) {
                *u = sum_multiple * *m;
            }
            univariate[0] += (mask_sum - sum_multiple * eval_01(mask)) * half;
            univariate[0] += mask_rlc * c0;
            univariate[1] += mask_rlc * c1;
            univariate[2] += mask_rlc * c2;

            prover_state.prover_message(&univariate[0]);
            prover_state.prover_messages(&univariate[2..]);

            // Receive the random evaluation point and update the sum.
            self.round_pow.prove(prover_state);
            let r = prover_state.verifier_message::<F>();
            round_challenges.push(r);
            *sum = (c2 * r + c1) * r + c0;

            mask_sum = univariate_evaluate(&univariate, r) - mask_rlc * *sum;
            prev_round_challenge = Some(r);
        }
        if let Some(w) = prev_round_challenge {
            // Final fold of the inputs (no polynomial computation).
            a.fold_pair(b, w);
        }

        *sum = mask_sum + mask_rlc * *sum;
        SumcheckOpening {
            round_challenges,
            mask_rlc,
        }
    }

    fn maybe_send_initial_mask_sum<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        masks: &[F],
    ) -> (F, F)
    where
        H: DuplexSpongeInterface,
        R: CryptoRng + RngCore,
        F: Codec<[H::U]>,
    {
        match &self.mode {
            SumcheckMode::Standard => (F::ZERO, F::ONE),
            SumcheckMode::ZeroKnowledge { mask_length } => {
                if self.num_rounds == 0 {
                    return (F::ZERO, F::ONE);
                }
                let sum_multiple = F::from(1 << self.num_rounds.saturating_sub(1));
                let mask_sum = masks
                    .chunks_exact(mask_length.get())
                    .map(eval_01)
                    .sum::<F>()
                    * sum_multiple;
                prover_state.prover_message(&mask_sum);
                let mask_rlc = prover_state.verifier_message();
                (mask_sum, mask_rlc)
            }
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

        let mask_rlc = self.maybe_receive_initial_mask_sum(verifier_state, sum)?;

        let mut univariate = vec![F::ZERO; self.mask_length().max(3)];
        let mut round_challenges = Vec::with_capacity(self.num_rounds);
        for _ in 0..self.num_rounds {
            // Receive all but linear coefficient.
            univariate[0] = verifier_state.prover_message()?;
            for c in &mut univariate[2..] {
                *c = verifier_state.prover_message()?;
            }

            // Derive linear coefficient from relation `univariate(0) + univariate(1) = sum`.
            univariate[1] = *sum - univariate[0].double() - univariate[2..].iter().sum::<F>();

            // Check proof of work (if any).
            self.round_pow.verify(verifier_state)?;

            // Receive the random evaluation point.
            let round_challenge = verifier_state.verifier_message::<F>();
            round_challenges.push(round_challenge);

            // Update the sum.
            *sum = univariate_evaluate(&univariate, round_challenge);
        }
        Ok(SumcheckOpening {
            round_challenges,
            mask_rlc,
        })
    }

    fn maybe_receive_initial_mask_sum<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
        sum: &mut F,
    ) -> VerificationResult<F>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
    {
        match &self.mode {
            SumcheckMode::Standard => Ok(F::ONE),
            SumcheckMode::ZeroKnowledge { .. } => {
                if self.num_rounds == 0 {
                    return Ok(F::ONE);
                }
                let mask_sum: F = verifier_state.prover_message()?;
                let mask_rlc = verifier_state.verifier_message();
                *sum = mask_sum + mask_rlc * *sum;
                Ok(mask_rlc)
            }
        }
    }
}

impl<F: Field> fmt::Display for Config<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mode_str = match &self.mode {
            SumcheckMode::Standard => "standard".to_string(),
            SumcheckMode::ZeroKnowledge { mask_length } => {
                format!("zk ℓ_zk={}", mask_length.get())
            }
        };
        write!(
            f,
            "size {} rounds {} pow {:.2} {}",
            self.initial_size,
            self.num_rounds,
            self.round_pow.difficulty(),
            mode_str,
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
            let mode_strategy = prop_oneof![
                3 => Just(SumcheckMode::Standard),
                7 => (3_usize..20).prop_map(|n| SumcheckMode::ZeroKnowledge {
                    mask_length: SumcheckMaskLen::new(n),
                }),
            ];
            (0_usize..(1 << 12), 0_usize..12, mode_strategy).prop_map(
                |(initial_size, num_rounds, mode)| {
                    let num_rounds =
                        num_rounds.min(initial_size.next_power_of_two().trailing_zeros() as usize);
                    Self::new(
                        initial_size,
                        proof_of_work::Config::none(),
                        num_rounds,
                        mode,
                    )
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
        let masks = random_vector(&mut rng, config.mask_length() * config.num_rounds);

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
                vector.to_slice()[0]
            );
            assert_eq!(
                multilinear_extend(&initial_covector, &point),
                covector.to_slice()[0]
            );
        } else {
            // TODO: Check correct folding.
        }

        let expected_mask_sum: F =
            chunks_exact_or_empty(&masks, config.mask_length(), config.num_rounds)
                .zip(&point)
                .map(|(m, x)| univariate_evaluate(m, *x))
                .sum();
        assert_eq!(
            sum,
            expected_mask_sum + mask_rlc * dot(vector.to_slice(), covector.to_slice())
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

        // Standard path: mask_rlc defaults to ONE (no combination randomness sampled).
        if matches!(config.mode, SumcheckMode::Standard) || config.num_rounds == 0 {
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
            &Config::<Field64>::new(
                2,
                proof_of_work::Config::none(),
                1,
                SumcheckMode::ZeroKnowledge {
                    mask_length: SumcheckMaskLen::new(3),
                },
            ),
        );
    }

    #[test]
    fn test_two_rounds() {
        test_config(
            0,
            &Config::<Field64>::new(
                3,
                proof_of_work::Config::none(),
                2,
                SumcheckMode::ZeroKnowledge {
                    mask_length: SumcheckMaskLen::new(3),
                },
            ),
        );
    }

    #[test]
    fn test_three_rounds() {
        test_config(
            0,
            &Config::<Field64>::new(
                5,
                proof_of_work::Config::none(),
                3,
                SumcheckMode::ZeroKnowledge {
                    mask_length: SumcheckMaskLen::new(3),
                },
            ),
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
