//! The atomic per-round operation shared by every Zook flow.
//!
//! Both `prove_whir_round` and `verify_whir_round` are thin orchestrators
//! around the multi-source code-switch and mask paths. Calling them with a
//! single source (`t = 1`, `theta = [F::ONE]`) produces transcript-byte-
//! identical output to the legacy single-source path because the original
//! `code_switch::prove` / `verify_for_implicit` and `bind_code_switch_mask`
//! were themselves wrappers around the multi-source primitives.

use ark_ff::{AdditiveGroup, Field};
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, RngCore};
use zeroize::Zeroize;

use crate::{
    algebra::{
        dot,
        embedding::{Embedding, Identity},
        mixed_dot,
    },
    buffer::{Buffer, BufferOps},
    hash::Hash,
    protocols::{
        code_switch::{self, CovectorUpdateParams},
        irs_commit::{Commitment as IrsCommitment, Witness as IrsWitness},
        params::config::RoundConfig,
        zook::{
            block::{ProverBlock, VerifierBlock},
            prover::RoundMaskOracle,
            slot_weights,
            verifier::RoundMaskOracleCheck,
        },
    },
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState,
        VerificationResult, VerifierState,
    },
};

/// Prove one WHIR round. Round 0 carries the base→ext embedding `M` (its
/// `witnesses` are `IrsWitness<M::Source>`); the code-switch yields an
/// `M::Target` witness, so the output block is `ProverBlock<Identity<M::Target>>`.
#[cfg_attr(feature = "tracing", tracing::instrument(skip_all, name = "zook::prove_whir_round", fields(msg_len = round.code_switch().source().message_length(), t = block.witnesses.len())))]
pub fn prove_whir_round<M, H, R>(
    round: &RoundConfig<M>,
    block: ProverBlock<M>,
    ps: &mut ProverState<H, R>,
) -> ProverBlock<Identity<M::Target>>
where
    M: Embedding,
    M::Target: Field + Default + Zeroize + Codec<[H::U]>,
    Standard: Distribution<M::Target>,
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
    u8: Decoding<[H::U]>,
    [u8; 32]: Decoding<[H::U]>,
    U64: Codec<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    let ProverBlock {
        message,
        covector,
        mut sum,
        witnesses,
        theta,
    } = block;

    let embedding = round.code_switch().source().embedding();
    debug_assert_eq!(witnesses.len(), theta.len());
    debug_assert!(!witnesses.is_empty());
    debug_assert_eq!(
        mixed_dot(embedding, &covector, &message),
        sum,
        "prove_whir_round entry: dot(message, covector) must equal sum"
    );

    let msg_len = round.code_switch().source().message_length();

    let mut masker = RoundMaskOracle::begin(round, ps);

    // Sumcheck lifts the source-field message into `M::Target` at its first
    // fold and returns the folded buffer (the covector folds in place). Move
    // the host-side `Vec` round state into buffers, fold, and move the folded
    // result back into `Vec` (downstream steps resize/truncate/index directly,
    // and code-switch takes `Vec` message). The hops are zero-copy on the CPU
    // backend.
    let message_buf = Buffer::from(message);
    let mut covector_buf = Buffer::from(covector);
    let (message_buf, opening) = round.sumcheck().prove(
        ps,
        embedding,
        message_buf,
        &mut covector_buf,
        &mut sum,
        masker.sumcheck_blinding(),
    );
    let message = message_buf.into_vec();
    let mut covector = covector_buf.into_vec();

    // Round 0's witnesses are base-field; the θ-combine inside the masker lifts
    // them into `M::Target` via the embedding before folding.
    let witness_refs: Vec<&IrsWitness<M::Source>> = witnesses.iter().collect();
    masker.bind_code_switch_mask_multi_source(
        embedding,
        &witness_refs,
        &theta,
        &opening,
        &mut sum,
        ps,
    );

    debug_assert_eq!(
        dot(&message, &covector),
        sum,
        "prove_whir_round post-reconcile: dot(message, covector) must equal sum"
    );

    covector.resize(msg_len + masker.covector_extension(), M::Target::ZERO);

    let slot_weights = slot_weights::build(&theta, &opening.round_challenges);
    let cs_witness = round.code_switch().prove_virtual(
        ps,
        message,
        &witness_refs,
        &slot_weights,
        code_switch::Claim {
            covector: &mut covector,
            sum: &mut sum,
        },
        masker.code_switch_blinding(),
    );

    masker.finish(
        &opening.round_challenges,
        &covector[msg_len..],
        &mut sum,
        ps,
    );

    covector.truncate(cs_witness.message.len());

    debug_assert_eq!(
        dot(&cs_witness.message, &covector),
        sum,
        "prove_whir_round exit: dot(message, covector) must equal sum"
    );

    ProverBlock::single_source(cs_witness.message, covector, sum, cs_witness.target_witness)
}

/// Verifier-side output of one round: the data needed by the caller to
/// accumulate implicit constraints and track per-round scale factors.
pub struct VerifyRoundOutput<F: Field> {
    pub(crate) round_challenges: Vec<F>,
    pub(crate) update_params: CovectorUpdateParams<F>,
}

/// The next-block-plus-round-output pair returned by [`verify_whir_round`].
type VerifyRoundResult<F> = VerificationResult<(VerifierBlock<F>, VerifyRoundOutput<F>)>;

/// Verify one WHIR round. The verifier holds no base-field state (commitments
/// are field-agnostic and all arithmetic is in `M::Target`), so only the round
/// config's embedding `M` varies — round 0 opens a base source IRS.
#[cfg_attr(feature = "tracing", tracing::instrument(skip_all, name = "zook::verify_whir_round", fields(msg_len = round.code_switch().source().message_length(), t = block.commitments.len())))]
pub fn verify_whir_round<M, H>(
    round: &RoundConfig<M>,
    block: VerifierBlock<M::Target>,
    vs: &mut VerifierState<H>,
) -> VerifyRoundResult<M::Target>
where
    M: Embedding,
    M::Target: Field + Default + Codec<[H::U]>,
    Standard: Distribution<M::Target>,
    H: DuplexSpongeInterface,
    u8: Decoding<[H::U]>,
    [u8; 32]: Decoding<[H::U]>,
    U64: Codec<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    let VerifierBlock {
        mut sum,
        commitments,
        theta,
    } = block;

    debug_assert_eq!(commitments.len(), theta.len());
    debug_assert!(!commitments.is_empty());

    let msg_len = round.code_switch().source().message_length();

    let mut masker = RoundMaskOracleCheck::begin(round, vs)?;
    let opening = round.sumcheck().verify(vs, &mut sum)?;
    masker.receive_cs_mask_and_reconcile(&opening, vs, &mut sum)?;

    let slot_weights = slot_weights::build(&theta, &opening.round_challenges);
    let commitment_refs: Vec<&IrsCommitment> = commitments.iter().collect();
    let (target_commitment, update_params) = round.code_switch().verify_virtual_for_implicit(
        vs,
        &mut sum,
        &commitment_refs,
        &slot_weights,
    )?;

    masker.verify_and_discharge(
        &opening.round_challenges,
        msg_len,
        &update_params,
        vs,
        &mut sum,
    )?;

    let next = VerifierBlock::single_source(sum, target_commitment);
    let out = VerifyRoundOutput {
        round_challenges: opening.round_challenges,
        update_params,
    };
    Ok((next, out))
}
