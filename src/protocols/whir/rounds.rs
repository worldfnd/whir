//! Shared round execution logic for the WHIR protocol.
//!
//! These free functions implement the round body for rounds 1+ and the final
//! round. Both the base WHIR prover/verifier and the zkWHIR 2.0 prover/verifier
//! call these functions to avoid duplicating the round loop.

use ark_ff::FftField;
use ark_std::rand::{CryptoRng, RngCore};

use super::RoundConfig;
use crate::{
    algebra::{
        dot,
        embedding::Identity,
        linear_form::{Evaluate, UnivariateEvaluation},
        MultilinearPoint,
    },
    hash::Hash,
    protocols::{geometric_challenge::geometric_challenge, irs_commit, proof_of_work, sumcheck},
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState,
        VerificationResult, VerifierState,
    },
    utils::zip_strict,
    verify,
};

/// Mutable sumcheck accumulation state threaded through WHIR prover rounds.
#[derive(Debug)]
pub struct SumcheckState<'a, F> {
    pub(crate) vector: &'a mut Vec<F>,
    pub(crate) covector: &'a mut Vec<F>,
    pub(crate) the_sum: &'a mut F,
}

/// Configuration for the final WHIR round (sumcheck + proof-of-work).
#[derive(Debug)]
pub struct FinalRoundConfig<'a, F: FftField> {
    pub(crate) sumcheck: &'a sumcheck::Config<F>,
    pub(crate) pow: &'a proof_of_work::Config,
}

/// Result of a single verifier round (rounds 1+).
#[must_use]
#[derive(Debug)]
pub struct VerifyRoundResult<F: FftField> {
    pub commitment: irs_commit::Commitment<F>,
    pub in_domain: irs_commit::Evaluations<F>,
    pub stir_rlc_coeffs: Vec<F>,
    pub stir_challenges: Vec<UnivariateEvaluation<F>>,
    pub folding_randomness: MultilinearPoint<F>,
}

/// Result of a single prover round (rounds 1+).
#[must_use]
#[derive(Debug)]
pub struct ProveRoundResult<F: FftField> {
    pub witness: irs_commit::Witness<F, F>,
    pub in_domain: irs_commit::Evaluations<F>,
    pub folding_randomness: MultilinearPoint<F>,
}

/// Single prover round body for rounds 1+ of the WHIR protocol.
///
/// Commits the current vector, opens the previous round's witness, accumulates
/// STIR constraints (OOD + in-domain), and runs the round's sumcheck.
fn prove_round<F, H, R>(
    round_config: &RoundConfig<F>,
    prev_round_config: &RoundConfig<F>,
    prover_state: &mut ProverState<H, R>,
    state: &mut SumcheckState<'_, F>,
    prev_witness: &irs_commit::Witness<F, F>,
    folding_randomness: &MultilinearPoint<F>,
) -> ProveRoundResult<F>
where
    F: FftField + Codec<[H::U]>,
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
    [u8; 32]: Decoding<[H::U]>,
    U64: Codec<[H::U]>,
    u8: Decoding<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    let new_witness = round_config
        .irs_committer
        .commit(prover_state, &[state.vector.as_slice()]);
    round_config.pow.prove(prover_state);

    let in_domain = prev_round_config
        .irs_committer
        .open(prover_state, &[prev_witness]);

    let stir_challenges: Vec<_> = new_witness
        .out_of_domain()
        .evaluators(round_config.initial_size())
        .chain(in_domain.evaluators(round_config.initial_size()))
        .collect();
    let stir_evaluations: Vec<F> = new_witness
        .out_of_domain()
        .values(&[F::ONE])
        .chain(in_domain.values(&folding_randomness.eq_weights()))
        .collect();

    let stir_rlc_coeffs: Vec<F> = geometric_challenge(prover_state, 1, stir_challenges.len());
    UnivariateEvaluation::accumulate_many(&stir_challenges, state.covector, &stir_rlc_coeffs);
    *state.the_sum += dot(&stir_rlc_coeffs, &stir_evaluations);
    debug_assert_eq!(dot(state.vector, state.covector), *state.the_sum);

    let new_folding =
        round_config
            .sumcheck
            .prove(prover_state, state.vector, state.covector, state.the_sum);
    debug_assert_eq!(dot(state.vector, state.covector), *state.the_sum);

    ProveRoundResult {
        witness: new_witness,
        in_domain,
        folding_randomness: new_folding,
    }
}

/// Final prover round.
///
/// Sends the (small) final folded vector directly, runs PoW, opens the last
/// commitment, and runs the final sumcheck.
fn prove_final_round<F, H, R>(
    final_config: &FinalRoundConfig<'_, F>,
    last_round_config: &RoundConfig<F>,
    prover_state: &mut ProverState<H, R>,
    state: &mut SumcheckState<'_, F>,
    prev_witness: &irs_commit::Witness<F, F>,
) -> (irs_commit::Evaluations<F>, MultilinearPoint<F>)
where
    F: FftField + Codec<[H::U]>,
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
    [u8; 32]: Decoding<[H::U]>,
    U64: Codec<[H::U]>,
    u8: Decoding<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    assert_eq!(state.vector.len(), final_config.sumcheck.initial_size);
    for coeff in state.vector.iter() {
        prover_state.prover_message(coeff);
    }

    final_config.pow.prove(prover_state);

    let in_domain = last_round_config
        .irs_committer
        .open(prover_state, &[prev_witness]);

    let final_folding =
        final_config
            .sumcheck
            .prove(prover_state, state.vector, state.covector, state.the_sum);

    (in_domain, final_folding)
}

/// Single verifier round body for rounds 1+.
///
/// Receives commitment, verifies PoW, opens the previous round's commitment,
/// accumulates STIR constraints, and runs the round's sumcheck.
fn verify_round<F, H>(
    round_config: &RoundConfig<F>,
    prev_round_config: &RoundConfig<F>,
    verifier_state: &mut VerifierState<'_, H>,
    the_sum: &mut F,
    prev_commitment: &irs_commit::Commitment<F>,
    folding_randomness: &MultilinearPoint<F>,
) -> VerificationResult<VerifyRoundResult<F>>
where
    F: FftField + Codec<[H::U]>,
    H: DuplexSpongeInterface,
    u8: Decoding<[H::U]>,
    [u8; 32]: Decoding<[H::U]>,
    U64: Codec<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    let commitment = round_config
        .irs_committer
        .receive_commitment(verifier_state)?;
    round_config.pow.verify(verifier_state)?;

    let in_domain = prev_round_config
        .irs_committer
        .verify(verifier_state, &[prev_commitment])?;

    let stir_challenges: Vec<UnivariateEvaluation<F>> = commitment
        .out_of_domain()
        .evaluators(round_config.initial_size())
        .chain(in_domain.evaluators(round_config.initial_size()))
        .collect();
    let stir_evaluations: Vec<F> = commitment
        .out_of_domain()
        .values(&[F::ONE])
        .chain(in_domain.values(&folding_randomness.eq_weights()))
        .collect();

    let stir_rlc_coeffs: Vec<F> = geometric_challenge(verifier_state, 1, stir_challenges.len());
    *the_sum += dot(&stir_rlc_coeffs, &stir_evaluations);

    let folding_randomness = round_config.sumcheck.verify(verifier_state, the_sum)?;

    Ok(VerifyRoundResult {
        commitment,
        in_domain,
        stir_rlc_coeffs,
        stir_challenges,
        folding_randomness,
    })
}

/// Final verifier round.
///
/// Receives the final vector, verifies PoW, opens the last commitment, checks
/// in-domain evaluations directly, and runs the final sumcheck.
///
/// Returns `(final_vector, in_domain, final_folding_randomness)`.
fn verify_final_round<F, H>(
    final_config: &FinalRoundConfig<'_, F>,
    last_round_config: &RoundConfig<F>,
    verifier_state: &mut VerifierState<'_, H>,
    the_sum: &mut F,
    prev_commitment: &irs_commit::Commitment<F>,
    folding_randomness: &MultilinearPoint<F>,
) -> VerificationResult<(Vec<F>, irs_commit::Evaluations<F>, MultilinearPoint<F>)>
where
    F: FftField + Codec<[H::U]>,
    H: DuplexSpongeInterface,
    u8: Decoding<[H::U]>,
    [u8; 32]: Decoding<[H::U]>,
    U64: Codec<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    let final_vector: Vec<F> =
        verifier_state.prover_messages_vec(final_config.sumcheck.initial_size)?;

    final_config.pow.verify(verifier_state)?;

    let in_domain = last_round_config
        .irs_committer
        .verify(verifier_state, &[prev_commitment])?;

    for (weight, eval) in zip_strict(
        in_domain.evaluators(final_vector.len()),
        in_domain.values(&folding_randomness.eq_weights()),
    ) {
        verify!(weight.evaluate(&Identity::<F>::new(), &final_vector) == eval);
    }

    let final_folding = final_config.sumcheck.verify(verifier_state, the_sum)?;

    Ok((final_vector, in_domain, final_folding))
}

/// Result of running remaining prover rounds (rounds 1..N + final round).
#[must_use]
#[derive(Debug)]
pub struct ProveRemainingResult<F> {
    /// In-domain points from the first opening after round 0.
    pub first_in_domain_points: Vec<F>,
    /// Folding randomness from each remaining round + final round.
    pub round_folding_randomness: Vec<MultilinearPoint<F>>,
}

/// Run the remaining prover rounds (1..N) plus the final round.
///
/// After the caller handles round 0 (which may differ between base WHIR and
/// zkWHIR 2.0), this function executes all subsequent fold-commit-sumcheck
/// rounds and the final direct-send round.
pub fn prove_remaining_rounds<F, H, R>(
    round_configs: &[RoundConfig<F>],
    final_config: &FinalRoundConfig<'_, F>,
    prover_state: &mut ProverState<H, R>,
    state: &mut SumcheckState<'_, F>,
    round0_witness: irs_commit::Witness<F, F>,
    round0_folding: &MultilinearPoint<F>,
) -> ProveRemainingResult<F>
where
    F: FftField + Codec<[H::U]>,
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
    [u8; 32]: Decoding<[H::U]>,
    U64: Codec<[H::U]>,
    u8: Decoding<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    assert!(!round_configs.is_empty());
    let mut prev_witness = round0_witness;
    let mut folding_randomness = round0_folding.clone();
    let mut first_in_domain_points = Vec::new();
    let mut round_folding_randomness = Vec::new();

    for (i, window) in round_configs.windows(2).enumerate() {
        let (prev_rc, rc) = (&window[0], &window[1]);
        let ProveRoundResult {
            witness,
            in_domain,
            folding_randomness: new_folding,
        } = prove_round(
            rc,
            prev_rc,
            prover_state,
            state,
            &prev_witness,
            &folding_randomness,
        );
        if i == 0 {
            first_in_domain_points = in_domain.points;
        }
        folding_randomness = new_folding.clone();
        round_folding_randomness.push(new_folding);
        prev_witness = witness;
    }

    let last_rc = round_configs.last().unwrap();
    let (final_in_domain, final_folding) =
        prove_final_round(final_config, last_rc, prover_state, state, &prev_witness);
    if round_configs.len() == 1 {
        first_in_domain_points = final_in_domain.points;
    }
    round_folding_randomness.push(final_folding);

    ProveRemainingResult {
        first_in_domain_points,
        round_folding_randomness,
    }
}

/// Result of running remaining verifier rounds (rounds 1..N + final round).
#[must_use]
#[derive(Debug)]
pub struct VerifyRemainingResult<F: FftField> {
    /// The final folded vector sent by the prover.
    pub final_vector: Vec<F>,
    /// In-domain evaluations from the first opening after round 0.
    pub first_in_domain: irs_commit::Evaluations<F>,
    /// STIR constraints accumulated during rounds 1..N.
    pub round_constraints: Vec<(Vec<F>, Vec<UnivariateEvaluation<F>>)>,
    /// Folding randomness from rounds 1..N (excludes the final sumcheck).
    pub round_folding_randomness: Vec<MultilinearPoint<F>>,
    /// Folding randomness from the final sumcheck. Stored separately because
    /// callers need it both in the evaluation point AND for the MLE check.
    pub final_sumcheck_randomness: MultilinearPoint<F>,
}

/// Run the remaining verifier rounds (1..N) plus the final round.
///
/// After the caller handles round 0, this function verifies all subsequent
/// fold-commit-sumcheck rounds and the final direct-send round.
pub fn verify_remaining_rounds<F, H>(
    round_configs: &[RoundConfig<F>],
    final_config: &FinalRoundConfig<'_, F>,
    verifier_state: &mut VerifierState<'_, H>,
    the_sum: &mut F,
    round0_commitment: &irs_commit::Commitment<F>,
    round0_folding: &MultilinearPoint<F>,
) -> VerificationResult<VerifyRemainingResult<F>>
where
    F: FftField + Codec<[H::U]>,
    H: DuplexSpongeInterface,
    u8: Decoding<[H::U]>,
    [u8; 32]: Decoding<[H::U]>,
    U64: Codec<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    assert!(!round_configs.is_empty());
    let mut prev_commitment = round0_commitment.clone();
    let mut folding_randomness = round0_folding.clone();
    let mut first_in_domain = None;
    let mut round_constraints = Vec::new();
    let mut round_folding_randomness = Vec::new();

    for (i, window) in round_configs.windows(2).enumerate() {
        let (prev_rc, rc) = (&window[0], &window[1]);
        let result = verify_round(
            rc,
            prev_rc,
            verifier_state,
            the_sum,
            &prev_commitment,
            &folding_randomness,
        )?;
        if i == 0 {
            first_in_domain = Some(result.in_domain);
        }
        round_constraints.push((result.stir_rlc_coeffs, result.stir_challenges));
        let new_folding = result.folding_randomness;
        folding_randomness = new_folding.clone();
        round_folding_randomness.push(new_folding);
        prev_commitment = result.commitment;
    }

    let last_rc = round_configs.last().unwrap();
    let (final_vector, final_in_domain, final_sumcheck_randomness) = verify_final_round(
        final_config,
        last_rc,
        verifier_state,
        the_sum,
        &prev_commitment,
        &folding_randomness,
    )?;

    let first_in_domain = first_in_domain.unwrap_or(final_in_domain);

    Ok(VerifyRemainingResult {
        final_vector,
        first_in_domain,
        round_constraints,
        round_folding_randomness,
        final_sumcheck_randomness,
    })
}
