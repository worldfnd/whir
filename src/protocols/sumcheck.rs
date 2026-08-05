//! Generic sumcheck protocol.
//!
//! The transcript / mask / challenge machinery is degree-agnostic. The
//! round polynomial computation is delegated to a [`RoundPolyOracle`]:
//!   - [`Config::prove`] is a thin wrapper that builds a dot-product oracle
//!     (degree 2) — the legacy `⟨a, b⟩ = sum` reduction.
//!   - [`Config::prove_with_oracle`] takes any oracle and is used by the
//!     selector sumcheck (degree 3) and any future higher-degree variants.
//!
//! The verifier ([`Config::verify`]) is fully degree-generic: it derives `c_1`
//! from the sumcheck invariant `p(0) + p(1) = sum` and accepts any
//! degree-`d` round polynomial whose `d` non-`c_1` coefficients the prover
//! sends.

use std::{any::Any, fmt, num::NonZeroUsize};

use ark_ff::Field;
use ark_std::rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::{
    algebra::{embedding::Embedding, lift, univariate_evaluate},
    buffer::{Buffer, BufferMath, BufferOps},
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

/// Per-round polynomial provider for [`Config::prove_with_oracle`].
///
/// The prover loop owns the transcript, masking, challenge sampling, and
/// per-round PoW; the oracle owns the round polynomial computation and the
/// internal-state fold. Implementations should pick the in-place fold
/// strategy that best fits their state representation — the dot-product
/// oracle fuses fold and round-poly into one pass via
/// [`crate::algebra::sumcheck::fold_and_compute_polynomial`], for example.
pub trait RoundPolyOracle<F: Field> {
    /// Degree `d` of every round polynomial.
    fn degree(&self) -> usize;

    /// Compute the round polynomial's `d` non-`c_1` coefficients
    /// `[c_0, c_2, c_3, …, c_d]` for the current round. If `prev_challenge`
    /// is `Some(r)`, the oracle must first fold its internal state by `r`
    /// (the challenge sampled in the previous round) and then compute the
    /// new round polynomial.
    ///
    /// `c_1` is the verifier-derivable coefficient
    /// `c_1 = sum − 2·c_0 − Σ_{i≥2} c_i` and is recovered by the prover
    /// loop, so the oracle never returns it.
    fn fold_and_compute(&mut self, prev_challenge: Option<F>) -> Vec<F>;

    /// Apply the final round's challenge so the oracle's internal state
    /// reflects the fully folded representation when the loop exits.
    fn finalize(&mut self, final_challenge: F);
}

impl<F: Field, O: RoundPolyOracle<F>> RoundPolyOracle<F> for &mut O {
    fn degree(&self) -> usize {
        O::degree(self)
    }

    fn fold_and_compute(&mut self, prev_challenge: Option<F>) -> Vec<F> {
        O::fold_and_compute(self, prev_challenge)
    }

    fn finalize(&mut self, final_challenge: F) {
        O::finalize(self, final_challenge);
    }
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
    /// Maximum degree of each round polynomial in the folded variable.
    /// `2` for the legacy dot-product sumcheck; `3` for the selector
    /// sumcheck (where `P_r(X) = eq(X, β) · ⟨M(X), V(X)⟩` is cubic).
    degree: NonZeroUsize,
}

impl<F: Field> Config<F> {
    pub fn new(
        initial_size: usize,
        round_pow: proof_of_work::Config,
        num_rounds: usize,
        mode: SumcheckMode,
        degree: NonZeroUsize,
    ) -> Self {
        assert!(num_rounds == 0 || initial_size.next_power_of_two() >= 1 << num_rounds);
        assert!(degree.get() >= 2, "sumcheck degree must be ≥ 2");
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
            degree,
        }
    }

    pub const fn degree(&self) -> NonZeroUsize {
        self.degree
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

    /// Reduce a claim `dot(a, b) == sum` via the degree-2 dot-product sumcheck.
    ///
    /// Thin wrapper around [`Self::prove_with_oracle`] that builds a
    /// dot-product oracle, returning the folded `a'` (with `b` folded in
    /// place).
    ///
    /// `a` lives in the embedding's source field, `b` (and the transcript) in
    /// its target field `F`; pass `&Identity::new()` when both coincide. `a`
    /// stays in the source field until the first fold lifts it into the target
    /// field, so the first round's polynomial and fold run as source × target
    /// products. When the fields coincide the first fold happens in place, so
    /// no second buffer is held alongside `a`. The transcript is bit-identical
    /// to lifting `a` up front (the embedding is a ring homomorphism).
    ///
    /// # Panics
    ///
    /// Panics if `self.degree() != 2`. Use [`Self::prove_with_oracle`]
    /// directly for higher-degree sumchecks.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn prove<M, H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        embedding: &M,
        a: Buffer<M::Source>,
        b: &mut Buffer<F>,
        sum: &mut F,
        masks: &[F],
    ) -> (Buffer<F>, SumcheckOpening<F>)
    where
        M: Embedding<Target = F>,
        H: DuplexSpongeInterface,
        R: CryptoRng + RngCore,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
    {
        assert_eq!(
            self.degree.get(),
            2,
            "Config::prove only supports degree-2 dot-product sumchecks; use prove_with_oracle"
        );
        assert_eq!(a.len(), self.initial_size);
        assert_eq!(b.len(), self.initial_size);
        debug_assert_eq!(a.mixed_dot(embedding, b), *sum);

        let mut oracle = DotProductOracle {
            embedding,
            a_source: Some(a),
            a: None,
            b,
        };
        let opening = self.prove_with_oracle(prover_state, sum, masks, &mut oracle);
        let a_folded = match (oracle.a, oracle.a_source) {
            (Some(folded), _) => folded,
            // No rounds: nothing folds, but the caller still expects a
            // target-field buffer. Cold path; a plain lift is fine.
            (None, Some(a)) => Buffer::from(lift(embedding, a.to_slice())),
            (None, None) => unreachable!("oracle consumed the source buffer without folding"),
        };
        (a_folded, opening)
    }

    /// Generic sumcheck prover. Drives the transcript / mask / challenge /
    /// PoW loop; delegates per-round polynomial computation to `oracle`.
    ///
    /// The `degree()` reported by `oracle` must match this config's degree.
    /// On return, `oracle`'s internal state reflects the fully folded
    /// representation (via [`RoundPolyOracle::finalize`]).
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn prove_with_oracle<O, H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        sum: &mut F,
        masks: &[F],
        mut oracle: O,
    ) -> SumcheckOpening<F>
    where
        O: RoundPolyOracle<F>,
        H: DuplexSpongeInterface,
        R: CryptoRng + RngCore,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
    {
        assert!(
            self.num_rounds == 0 || self.initial_size.next_power_of_two() >= 1 << self.num_rounds
        );
        assert_eq!(
            oracle.degree(),
            self.degree.get(),
            "RoundPolyOracle::degree must match Config::degree"
        );
        assert_eq!(masks.len(), self.num_rounds * self.mask_length());

        let degree = self.degree.get();
        let half = F::from(2).inverse().unwrap();
        let polynomial_len = self.mask_length().max(degree + 1);

        let (mut mask_sum, mask_rlc) = self.maybe_send_initial_mask_sum(prover_state, masks);

        let mut univariate = Vec::with_capacity(polynomial_len);
        let mut round_challenges = Vec::with_capacity(self.num_rounds);
        let mut prev_round_challenge = None;
        for (round, mask) in
            chunks_exact_or_empty(masks, self.mask_length(), self.num_rounds).enumerate()
        {
            // Oracle computes the round polynomial's non-c1 coefficients
            // `[c_0, c_2, c_3, …, c_d]` (length d). The fold of the oracle's
            // internal state by `prev_round_challenge` is fused into this call
            // when the oracle supports it.
            let coeffs_no_c1 = oracle.fold_and_compute(prev_round_challenge);
            debug_assert_eq!(coeffs_no_c1.len(), degree);
            let c0 = coeffs_no_c1[0];
            let high = &coeffs_no_c1[1..]; // c_2, c_3, …, c_d
            let c1 = *sum - c0.double() - high.iter().copied().sum::<F>();

            // Build round polynomial. In Standard (`mask = []`, `mask_rlc = 1`,
            // `mask_sum = 0`) this collapses to `[c_0, c_1, c_2, …, c_d]`.
            univariate.clear();
            univariate.resize(polynomial_len, F::ZERO);
            let sum_multiple = F::from(1 << self.num_rounds.saturating_sub(round + 1));
            for (u, m) in univariate.iter_mut().zip(mask.iter()) {
                *u = sum_multiple * *m;
            }
            univariate[0] += (mask_sum - sum_multiple * eval_01(mask)) * half;
            univariate[0] += mask_rlc * c0;
            univariate[1] += mask_rlc * c1;
            for (slot, c) in univariate.iter_mut().skip(2).zip(high.iter()) {
                *slot += mask_rlc * *c;
            }

            prover_state.prover_message(&univariate[0]);
            prover_state.prover_messages(&univariate[2..]);

            // Receive the random evaluation point and update the sum.
            self.round_pow.prove(prover_state);
            let r = prover_state.verifier_message::<F>();
            round_challenges.push(r);
            // Update sum to p(r). Horner over [c_0, c_1, c_2, …, c_d].
            let mut s = *high.last().unwrap_or(&F::ZERO);
            for &c in high.iter().rev().skip(1) {
                s = s * r + c;
            }
            s = s * r + c1;
            s = s * r + c0;
            *sum = s;

            mask_sum = univariate_evaluate(&univariate, r) - mask_rlc * *sum;
            prev_round_challenge = Some(r);
        }
        if let Some(r) = prev_round_challenge {
            oracle.finalize(r);
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

        let mut univariate = vec![F::ZERO; self.mask_length().max(self.degree.get() + 1)];
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
/// If the embedding's source and target fields coincide (e.g. `Identity`),
/// recover the source buffer as a target-field buffer without copying;
/// otherwise hand the buffer back.
fn same_field_buffer<M: Embedding>(
    a: Buffer<M::Source>,
) -> Result<Buffer<M::Target>, Buffer<M::Source>> {
    match (Box::new(a) as Box<dyn Any>).downcast::<Buffer<M::Target>>() {
        Ok(same_field) => Ok(*same_field),
        Err(a) => Err(*a.downcast::<Buffer<M::Source>>().expect("roundtrip")),
    }
}

fn eval_01<F: Field>(coefficients: &[F]) -> F {
    if coefficients.is_empty() {
        return F::ZERO;
    }
    coefficients[0] + coefficients.iter().sum::<F>()
}

/// Degree-2 dot-product oracle for [`Config::prove_with_oracle`]: reduces
/// `⟨a, b⟩ = sum` via the legacy quadratic sumcheck. Folds `a` and `b` in
/// place using [`BufferMath::fold_pair_sumcheck_polynomial`] to fuse the
/// previous-round fold with the current-round polynomial computation in a
/// single pass.
struct DotProductOracle<'a, M: Embedding> {
    embedding: &'a M,
    /// `a` before the first fold, in the source field.
    a_source: Option<Buffer<M::Source>>,
    /// `a` from the first fold on, in the target field.
    a: Option<Buffer<M::Target>>,
    b: &'a mut Buffer<M::Target>,
}

impl<M: Embedding> DotProductOracle<'_, M> {
    /// First fold: `a` crosses from the source to the target field. When the
    /// fields coincide (`Identity`) fold in place; only a genuine base → ext
    /// crossing allocates the target buffer while the source one is alive.
    /// Folds `b` alongside.
    fn first_fold(&mut self, w: M::Target) -> Buffer<M::Target> {
        let a = self.a_source.take().expect("source buffer consumed once");
        let folded = match same_field_buffer::<M>(a) {
            Ok(mut same_field) => {
                same_field.fold(w);
                same_field
            }
            Err(a) => a.mixed_fold(self.embedding, w),
        };
        self.b.fold(w);
        folded
    }
}

impl<M: Embedding> RoundPolyOracle<M::Target> for DotProductOracle<'_, M> {
    fn degree(&self) -> usize {
        2
    }

    fn fold_and_compute(&mut self, prev_challenge: Option<M::Target>) -> Vec<M::Target> {
        let (c0, c2) = if let Some(folded) = self.a.as_mut() {
            let w = prev_challenge.expect("folded buffer implies a prior challenge");
            folded.fold_pair_sumcheck_polynomial(self.b, w)
        } else if let Some(w) = prev_challenge {
            let folded = self.first_fold(w);
            let coefficients = folded.sumcheck_polynomial(self.b);
            self.a = Some(folded);
            coefficients
        } else {
            let a = self
                .a_source
                .as_ref()
                .expect("source buffer available before the first fold");
            a.mixed_sumcheck_polynomial(self.embedding, self.b)
        };
        vec![c0, c2]
    }

    fn finalize(&mut self, final_challenge: M::Target) {
        if let Some(folded) = self.a.as_mut() {
            folded.fold_pair(self.b, final_challenge);
        } else {
            let folded = self.first_fold(final_challenge);
            self.a = Some(folded);
        }
    }
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
            embedding::Identity,
            fields::{self, Field64},
            multilinear_extend, random_vector,
        },
        buffer::Buffer,
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
                        NonZeroUsize::new(2).expect("2 is non-zero"),
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
        let vector = Buffer::from(initial_vector.as_slice());
        let mut covector = Buffer::from(initial_covector.as_slice());
        let mut sum = initial_sum;
        let mut prover_state = ProverState::new_std(&ds);
        let (
            vector,
            SumcheckOpening {
                round_challenges: point,
                mask_rlc,
            },
        ) = config.prove(
            &mut prover_state,
            &Identity::new(),
            vector,
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

    /// The delayed lift's core claim: proving with a source-field `a` through
    /// a real embedding is byte-identical (proof and outputs) to lifting `a`
    /// up front and proving through `Identity`.
    #[test]
    fn mixed_prove_matches_lifted_transcript() {
        use crate::algebra::{embedding::Basefield, fields::Field64_3, lift};
        crate::tests::init();
        let embedding = Basefield::<Field64_3>::new();
        proptest!(|(seed: u64, config in Config::<Field64_3>::arbitrary())| {
            let instance = U64(seed);
            let ds = DomainSeparator::protocol(&config)
                .session(&format!("Mixed vs lifted at {}:{}", file!(), line!()))
                .instance(&instance);
            let mut rng = StdRng::seed_from_u64(seed);
            let a_source: Vec<Field64> = random_vector(&mut rng, config.initial_size);
            let covector: Vec<Field64_3> = random_vector(&mut rng, config.initial_size);
            let masks: Vec<Field64_3> =
                random_vector(&mut rng, config.mask_length() * config.num_rounds);
            let a_lifted = lift(&embedding, &a_source);
            let initial_sum = dot(&a_lifted, &covector);

            let run = |mixed: bool| {
                let mut b = Buffer::from(covector.as_slice());
                let mut sum = initial_sum;
                let mut prover_state = ProverState::new_std(&ds);
                let (folded, opening) = if mixed {
                    config.prove(
                        &mut prover_state,
                        &embedding,
                        Buffer::from(a_source.as_slice()),
                        &mut b,
                        &mut sum,
                        &masks,
                    )
                } else {
                    config.prove(
                        &mut prover_state,
                        &Identity::new(),
                        Buffer::from(a_lifted.as_slice()),
                        &mut b,
                        &mut sum,
                        &masks,
                    )
                };
                (
                    prover_state.proof(),
                    folded.into_vec(),
                    b.into_vec(),
                    sum,
                    opening.round_challenges,
                    opening.mask_rlc,
                )
            };
            assert_eq!(run(true), run(false));
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
                NonZeroUsize::new(2).expect("2 is non-zero"),
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
                NonZeroUsize::new(2).expect("2 is non-zero"),
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
                NonZeroUsize::new(2).expect("2 is non-zero"),
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
