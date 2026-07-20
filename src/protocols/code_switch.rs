//! Code-switching IOR: R_{C, C_zk, sl} → R_{C', C_zk, sl'}
//!
//! Reduces a proximity claim about oracle f (source code C) to a proximity
//! claim about oracle g (target code C'). Supports optional ZK via mask oracle.

use std::{fmt, num::NonZeroUsize};

use ark_ff::Field;
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::{
    algebra::{
        dot,
        embedding::{Embedding, Identity},
        eq_weights, lift,
        linear_form::UnivariateEvaluation,
        mixed_dot,
    },
    buffer::{Buffer, BufferMath, BufferOps},
    hash::Hash,
    protocols::{
        geometric_challenge::{geometric_challenge, geometric_challenge_groups},
        irs_commit::{Commitment as IrsCommitment, Config as IrsConfig, Witness as IrsWitness},
        proof_of_work,
    },
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState,
        VerificationResult, VerifierMessage, VerifierState,
    },
    verify,
};

/// Standard / ZeroKnowledge selector for code-switch.
#[derive(Clone, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
pub enum CodeSwitchMode {
    Standard,
    ZeroKnowledge { message_mask_length: NonZeroUsize },
}

/// Code-switching IOR config with optional ZK.
#[must_use]
#[derive(Clone, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Config<M: Embedding> {
    source: IrsConfig<M>,
    target: IrsConfig<Identity<M::Target>>,
    mode: CodeSwitchMode,
    out_domain_samples: usize,
    pow: proof_of_work::Config,
}

/// Prover output from the code-switch.
#[must_use]
#[derive(Clone, Debug)]
pub struct Witness<F: Field> {
    pub message: Buffer<F>,
    pub target_witness: IrsWitness<F>,
}

/// Verifier output from the code-switch.
pub type Commitment = IrsCommitment;

#[inline]
fn univariate_evaluators<F: Field>(points: &[F], size: usize) -> Vec<UnivariateEvaluation<F>> {
    points
        .iter()
        .map(|&point| UnivariateEvaluation::new(point, size))
        .collect()
}

impl<M: Embedding> Config<M> {
    /// Create a code-switch config.
    pub fn new(
        source_config: IrsConfig<M>,
        target_config: IrsConfig<Identity<M::Target>>,
        out_domain_samples: usize,
        mode: CodeSwitchMode,
        pow: proof_of_work::Config,
    ) -> Self {
        assert_eq!(
            source_config.num_vectors(),
            1,
            "code-switch requires a single source vector"
        );
        assert_eq!(
            target_config.num_vectors(),
            1,
            "code-switch requires a single target vector"
        );
        // Construction 9.7 needs at least one OOD challenge; unique-decoding
        // Standard mode (`t_ood = 0`) is incompatible with code-switch.
        assert!(
            out_domain_samples > 0,
            "code-switch requires t_ood ≥ 1 (Construction 9.7)",
        );
        // Target encodes one polynomial of length ℓ = source.message_length()
        // under C' = D^{ι_t}. The IRS splits the input of length ℓ into ι_t
        // parallel slices of length ℓ/ι_t, each encoded under D.
        assert_eq!(
            target_config.vector_size(),
            source_config.message_length(),
            "target vector_size must equal source message_length (target encodes one polynomial of length ℓ)"
        );
        assert!(
            target_config.interleaving_depth().is_power_of_two(),
            "target.interleaving_depth must be a power of 2"
        );
        assert!(
            source_config.interleaving_depth().is_power_of_two(),
            "source.interleaving_depth must be a power of 2"
        );
        if let CodeSwitchMode::ZeroKnowledge {
            message_mask_length,
        } = &mode
        {
            let l_zk = message_mask_length.get();
            // Theorem 9.6: ℓ_zk ≥ r (mask oracle must cover source randomness).
            assert!(
                l_zk >= source_config.mask_length(),
                "message_mask_length ({l_zk}) must be >= source randomness length ({})",
                source_config.mask_length(),
            );
            assert!(
                l_zk - source_config.mask_length() >= out_domain_samples,
                "sampled randomness (s) length must cover all out-of-domain sample requests"
            );
            // t' = target in-domain queries + OOD queries (Construction 9.7 step 4).
            // Definition 3.16: a t'-query ZK encoding requires r' ≥ t'; here
            // r' = target.mask_length.
            assert!(
                target_config.mask_length()
                    >= target_config.in_domain_samples() + out_domain_samples,
                "target encoder violates t' ≤ r': queries must be covered by target mask"
            );
        } else {
            assert_eq!(
                source_config.mask_length(),
                0,
                "source with IRS randomness requires ZK mode",
            );
        }

        Self {
            source: source_config,
            target: target_config,
            mode,
            out_domain_samples,
            pow,
        }
    }

    pub const fn source(&self) -> &IrsConfig<M> {
        &self.source
    }

    pub const fn target(&self) -> &IrsConfig<Identity<M::Target>> {
        &self.target
    }

    pub const fn mode(&self) -> &CodeSwitchMode {
        &self.mode
    }

    pub const fn out_domain_samples(&self) -> usize {
        self.out_domain_samples
    }

    pub const fn pow(&self) -> proof_of_work::Config {
        self.pow
    }

    #[cfg(test)]
    pub(crate) const fn target_mut_for_test(&mut self) -> &mut IrsConfig<Identity<M::Target>> {
        &mut self.target
    }

    /// Mask oracle length `ℓ_zk`. Returns 0 in Standard mode.
    pub const fn message_mask_length(&self) -> usize {
        match &self.mode {
            CodeSwitchMode::Standard => 0,
            CodeSwitchMode::ZeroKnowledge {
                message_mask_length,
            } => message_mask_length.get(),
        }
    }

    /// `true` iff the protocol is configured for ZK.
    pub const fn is_zk(&self) -> bool {
        matches!(&self.mode, CodeSwitchMode::ZeroKnowledge { .. })
    }

    /// Length of the covector for this code-switch.
    pub fn covector_length(&self) -> usize {
        self.source.message_length() + self.message_mask_length()
    }

    /// Prove the code-switch.
    ///
    /// # Soundness-critical inputs
    ///
    /// `folding_randomness` is the **sumcheck folding randomness `γ`** that
    /// was sampled from the verifier in the preceding sumcheck protocol
    /// (Construction 6.3, p.37-38). It must be the same `γ` the verifier
    /// derived from the transcript — it is NOT caller-supplied randomness.
    ///
    /// Used by the verifier to collapse ι_s parallel codeword columns into a
    /// single value of `Fold(f, γ)` via `eq_weights(γ)`. Passing different
    /// randomness here breaks IOR completeness; passing locally-sampled
    /// randomness breaks Fiat-Shamir soundness in the composed protocol.
    ///
    /// `message` is `Fold(f, γ)`, the post-sumcheck polynomial of length
    /// `source.message_length()`.
    ///
    /// `mask` is `(r || s)` from the orchestrator's shared mask tree
    /// (see Construction 9.7 Step 1, p.55). Length must equal
    /// `self.message_mask_length()` — pass an empty slice in Standard mode.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn prove<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        message: Buffer<M::Target>,
        witness: &IrsWitness<M::Source>,
        covector: &mut Buffer<M::Target>,
        folding_randomness: &[M::Target],
        mask: &Buffer<M::Target>,
    ) -> Witness<M::Target>
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        Standard: Distribution<M::Target>,
        M::Target: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(message.len(), self.source.message_length());
        assert_eq!(covector.len(), self.covector_length());
        assert_eq!(mask.len(), self.message_mask_length());
        assert_eq!(
            1 << folding_randomness.len(),
            self.source.interleaving_depth(),
            "folding_randomness must have length log2(source.interleaving_depth) ({} != log2({}))",
            folding_randomness.len(),
            self.source.interleaving_depth(),
        );
        // Step 1: g := Enc_{C'}(f, r') — Construction 9.7 Step 1, p.55
        let target_witness = self.target.commit(prover_state, &[&message]);

        // Grind Lemma 9.9 OOD gap before α is sampled.
        self.pow.prove(prover_state);

        // Step 2-3: OOD challenge + answers — Construction 9.7 Steps 2-3, p.55
        let ood_points: Vec<M::Target> = prover_state.verifier_message_vec(self.out_domain_samples);
        self.maybe_send_ood_answers(prover_state, &message, mask, &ood_points);

        // Step 4: in-domain queries — Construction 9.7 Step 4, p.55
        let source_evaluations = self.source.open(prover_state, &[witness]);

        // Step 4.1: batching — Construction 9.7 Step 4, p.55
        let num_ood = self.out_domain_samples;
        let num_in_domain = source_evaluations.points.len();
        // The batching challenge [1, x, x², …] splits into the `sl'` scalar,
        // the OODS run, and the in-domain run (the verifier splits the same
        // sequence identically).
        let mut groups =
            geometric_challenge_groups::<_, M::Target>(prover_state, &[1, num_ood, num_in_domain])
                .into_iter();
        let original_sl_coeff = *groups.next().unwrap().get(0).unwrap();
        let ood_rlc_coeffs = groups.next().unwrap();
        let in_domain_rlc_coeffs = groups.next().unwrap();

        // Covector update — sl' from Completeness proof (p.55-56).
        covector.scalar_mul(original_sl_coeff);
        let eval_points = lift(self.source.embedding(), &source_evaluations.points);
        self.update_covector(
            covector,
            &ood_rlc_coeffs,
            &ood_points,
            &in_domain_rlc_coeffs,
            &eval_points,
        );

        Witness {
            message,
            target_witness,
        }
    }

    /// Send OOD answers `y_i = f(α_i) [+ α_i^ℓ · (r ‖ s)(α_i)]`.
    /// In Standard mode the bracketed term is omitted.
    fn maybe_send_ood_answers<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        message: &Buffer<M::Target>,
        mask: &Buffer<M::Target>,
        ood_points: &[M::Target],
    ) where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        M::Target: Codec<[H::U]>,
    {
        let msg_len = message.len();
        for &point in ood_points {
            let f_eval = message.mixed_univariate_evaluate(&Identity::<M::Target>::new(), point);
            let answer = match &self.mode {
                CodeSwitchMode::Standard => f_eval,
                CodeSwitchMode::ZeroKnowledge { .. } => {
                    let mask_eval =
                        mask.mixed_univariate_evaluate(&Identity::<M::Target>::new(), point);
                    let shift = point.pow([msg_len as u64]);
                    f_eval + shift * mask_eval
                }
            };
            prover_state.prover_message(&answer);
        }
    }

    /// Accumulate OOD and in-domain weights into the covector.
    /// Standard mode treats all points uniformly; ZK mode applies OOD over
    /// the full `[f; r; s]` and in-domain over the `[f; r]` prefix only.
    fn update_covector(
        &self,
        covector: &mut Buffer<M::Target>,
        ood_rlc_coeffs: &Buffer<M::Target>,
        ood_points: &[M::Target],
        in_domain_rlc_coeffs: &Buffer<M::Target>,
        in_domain_points: &[M::Target],
    ) {
        match &self.mode {
            CodeSwitchMode::Standard => {
                let mut evaluators = univariate_evaluators(ood_points, covector.len());
                evaluators.extend(univariate_evaluators(in_domain_points, covector.len()));
                let scalars = ood_rlc_coeffs.concat(in_domain_rlc_coeffs);
                covector.accumulate_univariate_evaluations(&evaluators, &scalars);
            }
            CodeSwitchMode::ZeroKnowledge { .. } => {
                let ood_evaluators = univariate_evaluators(ood_points, covector.len());
                covector.accumulate_univariate_evaluations(&ood_evaluators, ood_rlc_coeffs);
                let in_domain_evaluators =
                    univariate_evaluators(in_domain_points, self.source.masked_message_length());
                covector
                    .accumulate_univariate_evaluations(&in_domain_evaluators, in_domain_rlc_coeffs);
            }
        }
    }

    /// Verify the code-switch.
    ///
    /// `folding_randomness` is the **sumcheck folding randomness `γ`** the
    /// verifier derived from the transcript during the preceding sumcheck.
    /// It must match what the prover received from the same transcript —
    /// not caller-supplied randomness. See `prove` doc for details.
    ///
    /// Returns the target commitment. In ZK mode, the caller **must**
    /// additionally run `mask_proximity::verify` on the mask commitment
    /// to ensure the mask oracle `(r, s)` is close to a `C_zk` codeword.
    /// Without this check, soundness is not guaranteed.
    ///
    /// # Soundness composition note
    ///
    /// This verifier checks the OOD/in-domain consistency of the target
    /// codeword `g` against transcript-supplied mask values `s(α_i)`. It
    /// does **not** check that `s` is close to a `C_zk` codeword — that
    /// is the job of mask-proximity (Construction 7.2). Without a
    /// downstream mask-proximity invocation against the same `s`, a
    /// prover can submit non-codeword mask values that satisfy the OOD
    /// equation, breaking the soundness reduction in Theorem 9.10.
    ///
    /// In the orchestrated WHIR protocol, the orchestrator owns the
    /// per-round mask tree containing `s` and is responsible for
    /// running `mask_proximity::verify` on that same tree before
    /// accepting the round.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn verify<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
        sum: &mut M::Target,
        folding_randomness: &[M::Target],
        commitment: &IrsCommitment,
    ) -> VerificationResult<Commitment>
    where
        H: DuplexSpongeInterface,
        Standard: Distribution<M::Target>,
        M::Target: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        verify!(1 << folding_randomness.len() == self.source.interleaving_depth());

        let collapse_weights = eq_weights(folding_randomness);

        // Step 1: target commitment — Construction 9.7 Step 1, p.55
        // Mask oracle is committed in the shared mask tree by the orchestrator.
        let target_commitment = self.target.receive_commitment(verifier_state)?;

        // Grind Lemma 9.9 OOD gap before α is sampled.
        self.pow.verify(verifier_state)?;

        // Step 2-3: OOD — Construction 9.7 Steps 2-3, p.55
        // In ZK mode, ood_answers = f(α) + α^ℓ · (r,s)(α) where (r,s) is
        // the mask oracle message committed in the shared tree.
        let _ood_points: Vec<M::Target> =
            verifier_state.verifier_message_vec(self.out_domain_samples);
        let ood_answers: Vec<M::Target> =
            verifier_state.prover_messages_vec(self.out_domain_samples)?;

        // Step 4: source opening — Construction 9.7 Step 4, p.55
        let source_evaluations = self.source.verify(verifier_state, &[commitment])?;
        let collapsed_values: Vec<M::Target> = source_evaluations
            .matrix
            .to_slice()
            .chunks_exact(self.source.interleaving_depth())
            .map(|row| mixed_dot(self.source.embedding(), &collapse_weights, row))
            .collect();

        // Step 4.1: batching + μ' — Construction 9.7 Decision phase, p.55
        let num_ood = self.out_domain_samples;
        let num_in_domain = source_evaluations.points.len();
        let coeffs = geometric_challenge(verifier_state, 1 + num_ood + num_in_domain);
        let (&original_sl_coeff, constraint_rlc_coeffs) = coeffs.split_first().unwrap();
        let (ood_rlc_coeffs, in_domain_rlc_coeffs) = constraint_rlc_coeffs.split_at(num_ood);

        *sum = original_sl_coeff * *sum
            + dot(ood_rlc_coeffs, &ood_answers)
            + dot(in_domain_rlc_coeffs, &collapsed_values);

        Ok(target_commitment)
    }
}

impl<M: Embedding> fmt::Display for Config<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CodeSwitch(source={}, target={}, ood={}, zk={})",
            self.source,
            self.target,
            self.out_domain_samples,
            self.is_zk(),
        )
    }
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{
        distributions::Standard, prelude::Distribution, rngs::StdRng, Rng, SeedableRng,
    };
    use proptest::{bool, prelude::Strategy, prop_assume, proptest, sample::select};

    use super::*;
    use crate::{
        algebra::{embedding::Identity, fields, ntt, random_vector},
        transcript::{codecs::U64, DomainSeparator},
    };

    impl<M: Embedding> Config<M> {
        pub fn arbitrary(embedding: M) -> impl Strategy<Value = Self>
        where
            M: Default + 'static,
        {
            // Sizes ≥ 4 to allow ι ∈ {1, 2, 4} with non-trivial message_length.
            let valid_sizes = (4..=256)
                .filter(|&n| ntt::next_order::<M::Source>(n) == Some(n))
                .filter(|&n| n.is_power_of_two())
                .collect::<Vec<_>>();

            let scalars = (
                select(valid_sizes),
                0_usize..=3, // src_mask_len (source IRS randomness, post-fold)
                bool::ANY,   // zk
                1_usize..=5, // ood (= code-switch t_ood; ≥ 1 per Construction 9.7)
                0_usize..=5, // fresh_s_len (≥ ood for assumption (c))
                select(vec![1_usize, 2, 4]), // ι_s (source interleaving)
                0_usize..=10, // target.in_domain_samples (t'_in)
            );

            scalars.prop_flat_map(
                move |(size, src_mask_len, zk, ood, fresh_s_len, iota_s, t_in)| {
                    // Bound 3 assumption (c): ℓ_zk - r ≥ t_ood ⇒ fresh_s_len ≥ ood.
                    // Also enforce `ℓ_zk = r + fresh_s_len > 0` so NonZeroUsize
                    // construction below is total in ZK mode.
                    let fresh_s_len = if zk {
                        let min_fresh = usize::from(src_mask_len == 0);
                        fresh_s_len.max(ood).max(min_fresh)
                    } else {
                        fresh_s_len
                    };
                    // Bound 4 assumption (a): target.mask_length ≥ t' = t_in + ood.
                    let target_mask = if zk { t_in + ood } else { 0 };
                    let source_mask = if zk { src_mask_len } else { 0 };

                    IrsConfig::arbitrary(embedding.clone(), 1, size, source_mask, iota_s)
                        .prop_flat_map(move |source| {
                            // ι_t must divide msg_len and be a power of 2.
                            let msg_len = source.message_length();
                            let iota_t_choices: Vec<usize> = [1, 2, 4]
                                .into_iter()
                                .filter(|&i| msg_len.is_multiple_of(i))
                                .collect();

                            select(iota_t_choices).prop_flat_map(move |iota_t| {
                                // target.vector_size = ℓ; C' = D^{ι_t} where D's
                                // message length = ℓ / ι_t.
                                let target = IrsConfig::<Identity<M::Target>>::arbitrary(
                                    Identity::new(),
                                    1,
                                    msg_len,
                                    target_mask,
                                    iota_t,
                                );
                                let source = source.clone();
                                target.prop_map(move |mut target| {
                                    // IrsConfig::arbitrary samples in_domain_samples
                                    // in [0,10] independently of mask_length; pin it
                                    // to the value target_mask was sized for so
                                    // assumption (a) holds.
                                    if zk {
                                        target.set_in_domain_samples_for_test(t_in);
                                    }
                                    // r = post-fold randomness length (ι_s parallel
                                    // masks fold to a single length-mask_length chunk).
                                    let r = source.mask_length();
                                    let mode = if zk {
                                        CodeSwitchMode::ZeroKnowledge {
                                            message_mask_length: NonZeroUsize::new(r + fresh_s_len)
                                                .expect("ZK ⇒ r + fresh_s_len > 0"),
                                        }
                                    } else {
                                        CodeSwitchMode::Standard
                                    };
                                    Self::new(
                                        source.clone(),
                                        target,
                                        ood,
                                        mode,
                                        proof_of_work::Config::none(),
                                    )
                                })
                            })
                        })
                },
            )
        }
    }

    /// Fold ι parallel chunks of length `chunk_len` into a single chunk via
    /// eq_weights(γ). Layout: values = [chunk_0; chunk_1; ...; chunk_{ι-1}],
    /// each of length `chunk_len`. Returns Σ_l eq_weights(γ)[l] · chunk_l.
    fn fold_chunks<F: Field>(values: &[F], chunk_len: usize, folding_randomness: &[F]) -> Vec<F> {
        let iota = 1 << folding_randomness.len();
        assert_eq!(values.len(), chunk_len * iota);
        if iota == 1 {
            return values.to_vec();
        }
        let weights = eq_weights(folding_randomness);
        (0..chunk_len)
            .map(|j| {
                (0..iota)
                    .map(|l| weights[l] * values[l * chunk_len + j])
                    .sum()
            })
            .collect()
    }

    /// Sample folding randomness of length log2(source.interleaving_depth).
    fn sample_folding_randomness<F: Field>(
        config: &Config<Identity<F>>,
        rng: &mut impl RngCore,
    ) -> Vec<F>
    where
        Standard: Distribution<F>,
    {
        let log_iota = config.source.interleaving_depth().trailing_zeros() as usize;
        random_vector(rng, log_iota)
    }

    /// Simulate what the orchestrator does: build (r || fresh_s) where r is
    /// the *folded* source IRS randomness. Returns empty vec in non-ZK mode.
    fn build_mask_msg<F: Field>(
        config: &Config<Identity<F>>,
        source_witness: &IrsWitness<F>,
        folding_randomness: &[F],
        rng: &mut impl RngCore,
    ) -> Vec<F>
    where
        Standard: Distribution<F>,
    {
        if !config.is_zk() {
            return Vec::new();
        }
        // Lift ι parallel masks (total length source.mask_length × ι) and fold
        // chunks of length source.mask_length down to a single chunk.
        let raw = lift(config.source.embedding(), source_witness.masks.to_slice());
        let mut mask = fold_chunks(&raw, config.source.mask_length(), folding_randomness);
        // Append fresh padding s of length message_mask_length - source.mask_length.
        mask.extend(random_vector::<F>(
            rng,
            config.message_mask_length() - mask.len(),
        ));
        mask
    }

    fn test_config<F: Field + Codec<[u8]>>(seed: u64, config: &Config<Identity<F>>)
    where
        Standard: Distribution<F>,
        Hash: ProverMessage<[u8]>,
    {
        let mut rng = StdRng::seed_from_u64(seed);
        // Commit the full pre-fold vector of length source.vector_size
        // (= ι · message_length), which IRS encodes as ι parallel codewords.
        let f_full: Vec<F> = random_vector(&mut rng, config.source.vector_size());
        let initial_sum: F = rng.gen();

        let mut covector: Vec<F> = random_vector(&mut rng, config.source.message_length());
        covector.resize(config.covector_length(), F::ZERO);
        let mut covector = Buffer::from(covector);

        let instance = U64(seed);
        let ds = DomainSeparator::protocol(config)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&instance);
        let mut prover_state = ProverState::new_std(&ds);
        let f_full_buffer = Buffer::from(f_full.as_slice());
        let source_witness = config.source.commit(&mut prover_state, &[&f_full_buffer]);

        // Sample γ for sumcheck folding (length log2(ι)).
        let folding_randomness = sample_folding_randomness(config, &mut rng);
        // Post-fold message Fold(f_full, γ) of length message_length.
        let folded_message = Buffer::from(fold_chunks(
            &f_full,
            config.source.message_length(),
            &folding_randomness,
        ));
        let mask_msg = build_mask_msg(config, &source_witness, &folding_randomness, &mut rng);

        let witness = config.prove(
            &mut prover_state,
            folded_message.clone(),
            &source_witness,
            &mut covector,
            &folding_randomness,
            &Buffer::from(mask_msg.as_slice()),
        );
        let proof = prover_state.proof();

        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let source_commitment = config
            .source
            .receive_commitment(&mut verifier_state)
            .unwrap();
        let mut verifier_sum = initial_sum;
        let _ = config
            .verify(
                &mut verifier_state,
                &mut verifier_sum,
                &folding_randomness,
                &source_commitment,
            )
            .unwrap();
        verifier_state.check_eof().unwrap();
        assert_eq!(witness.message, folded_message);
    }

    fn test_ior_identity_config<F: Field + Codec<[u8]>>(seed: u64, config: &Config<Identity<F>>)
    where
        Standard: Distribution<F>,
        Hash: ProverMessage<[u8]>,
    {
        let mut rng = StdRng::seed_from_u64(seed);
        let f_full: Vec<F> = random_vector(&mut rng, config.source.vector_size());

        let mut covector: Vec<F> = random_vector(&mut rng, config.source.message_length());
        covector.resize(config.covector_length(), F::ZERO);
        let mut covector = Buffer::from(covector);

        let instance = U64(seed);
        let ds = DomainSeparator::protocol(config)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&instance);
        let mut prover_state = ProverState::new_std(&ds);
        let f_full_buffer = Buffer::from(f_full.as_slice());
        let source_witness = config.source.commit(&mut prover_state, &[&f_full_buffer]);

        let folding_randomness = sample_folding_randomness(config, &mut rng);
        let folded_message = Buffer::from(fold_chunks(
            &f_full,
            config.source.message_length(),
            &folding_randomness,
        ));
        let mask_msg = build_mask_msg(config, &source_witness, &folding_randomness, &mut rng);

        // h is the post-fold polynomial whose inner product with covector
        // should equal the verifier sum:
        // - non-ZK: h = folded_message (length message_length)
        // - ZK:     h = [folded_message; mask_msg] (length message_length + l_zk)
        let h: Buffer<F> = if mask_msg.is_empty() {
            folded_message.clone()
        } else {
            Buffer::from(
                folded_message
                    .to_slice()
                    .iter()
                    .chain(mask_msg.iter())
                    .copied()
                    .collect::<Vec<_>>(),
            )
        };
        let initial_mu = h.dot(&covector);

        let _witness = config.prove(
            &mut prover_state,
            folded_message,
            &source_witness,
            &mut covector,
            &folding_randomness,
            &Buffer::from(mask_msg.as_slice()),
        );
        let proof = prover_state.proof();

        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let source_commitment = config
            .source
            .receive_commitment(&mut verifier_state)
            .unwrap();
        let mut verifier_sum = initial_mu;
        let _ = config
            .verify(
                &mut verifier_state,
                &mut verifier_sum,
                &folding_randomness,
                &source_commitment,
            )
            .unwrap();
        verifier_state.check_eof().unwrap();

        assert_eq!(h.dot(&covector), verifier_sum);
    }

    fn test_tampered_ood_config<F: Field + Codec<[u8]>>(seed: u64, config: &Config<Identity<F>>)
    where
        Standard: Distribution<F>,
        Hash: ProverMessage<[u8]>,
    {
        let instance = U64(seed);
        let ds = DomainSeparator::protocol(config)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&instance);
        let mut rng = StdRng::seed_from_u64(seed);
        let f_full: Vec<F> = random_vector(&mut rng, config.source.vector_size());

        let mut covector: Vec<F> = random_vector(&mut rng, config.source.message_length());
        covector.resize(config.covector_length(), F::ZERO);
        let mut covector = Buffer::from(covector);

        // Commit honest f_full, fold to get the honest post-fold message.
        let mut prover_state = ProverState::new_std(&ds);
        let f_full_buffer = Buffer::from(f_full.as_slice());
        let source_witness = config.source.commit(&mut prover_state, &[&f_full_buffer]);
        let folding_randomness = sample_folding_randomness(config, &mut rng);
        let folded_message =
            fold_chunks(&f_full, config.source.message_length(), &folding_randomness);

        // For non-ZK and source.mask_length == 0, h = folded_message and identity holds.
        let folded_message_buffer = Buffer::from(folded_message.as_slice());
        let initial_mu = folded_message_buffer.dot(&covector);

        // Tamper the post-fold message before proving.
        let mut tampered = folded_message;
        tampered[0] += F::ONE;
        let tampered = Buffer::from(tampered);
        let _witness = config.prove(
            &mut prover_state,
            tampered,
            &source_witness,
            &mut covector,
            &folding_randomness,
            &Buffer::from([].as_slice()),
        );
        let proof = prover_state.proof();

        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let source_commitment = config
            .source
            .receive_commitment(&mut verifier_state)
            .unwrap();
        let mut verifier_sum = initial_mu;
        let _ = config
            .verify(
                &mut verifier_state,
                &mut verifier_sum,
                &folding_randomness,
                &source_commitment,
            )
            .unwrap();
        verifier_state.check_eof().unwrap();

        // Sum diverges — downstream sumcheck would reject
        assert_ne!(folded_message_buffer.dot(&covector), verifier_sum);
    }

    fn test<F: Field + Codec<[u8]> + 'static>()
    where
        Standard: Distribution<F>,
        Hash: ProverMessage<[u8]>,
    {
        crate::tests::init();
        let configs = Config::arbitrary(Identity::<F>::new());
        proptest!(|(seed: u64, config in configs)| {
            test_config(seed, &config);
        });
    }

    #[test]
    fn test_field64() {
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

    #[test]
    fn test_ior_identity() {
        crate::tests::init();
        let configs = Config::arbitrary(Identity::<fields::Field64>::new());
        proptest!(|(seed: u64, config in configs)| {
            prop_assume!(config.source.in_domain_samples() > 0);
            test_ior_identity_config(seed, &config);
        });
    }

    #[test]
    fn test_tampered_ood() {
        crate::tests::init();
        let configs = Config::arbitrary(Identity::<fields::Field64>::new())
            .prop_filter("non-ZK", |config| {
                !config.is_zk() && config.source.mask_length() == 0
            });
        proptest!(|(seed: u64, config in configs)| {
            test_tampered_ood_config(seed, &config);
        });
    }
}
