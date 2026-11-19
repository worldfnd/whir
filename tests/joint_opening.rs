//! Staged commitments share one mask collection across application challenges.

use std::panic::{catch_unwind, AssertUnwindSafe};

use ark_ff::{AdditiveGroup, Field};
use ark_std::rand::{distributions::Standard, prelude::Distribution};
use whir::{
    algebra::{
        embedding::{Basefield, Embedding, Identity},
        fields::{Field256, Field64_3},
        linear_form::{Covector, LinearForm},
    },
    buffer::Buffer,
    hash::Hash,
    protocols::{
        params::{DecodingRegime, Mode, Params, RateSchedule},
        zook::{JointOpeningConfig, ProverClaim},
    },
    transcript::{
        codecs::U64, Codec, DomainSeparator, Proof, ProverState, VerifierMessage, VerifierState,
    },
};

fn config<M: Embedding + Default>(mode: Mode, n: usize, first: usize) -> JointOpeningConfig<M> {
    JointOpeningConfig::derive(Params {
        mode,
        regime: DecodingRegime::Johnson,
        bits: 128,
        pow: 0,
        hash: whir::hash::BLAKE3,
        size: n,
        rate: 1,
        folds: [first, 2],
        schedule: RateSchedule::Capped(1),
    })
    .unwrap()
}

struct Example<M: Embedding> {
    config: JointOpeningConfig<M>,
    proof: Proof,
    forms: Vec<Covector<M::Target>>,
    values: Vec<M::Target>,
    claim: ProverClaim<M::Target>,
}

fn example<M>(config: JointOpeningConfig<M>) -> Example<M>
where
    M: Embedding + Default,
    M::Target: Codec,
    Standard: Distribution<M::Source> + Distribution<M::Target>,
{
    let n = config.witness_commitment().vector_size();
    // This fixed instance specifies the two public forms used below.
    let ds = DomainSeparator::protocol(&config).instance(b"joint-two-sparse-forms");
    let mut ps = ProverState::new_std(&ds);
    let mut left = vec![M::Source::ZERO; n];
    left[0] = M::Source::ONE;
    let first = config.commit_first(&mut ps, Buffer::from(left));
    ps.prover_message(&Hash([7; 32]));
    let challenge: [u8; 32] = ps.verifier_message();
    let mut right = vec![M::Source::ZERO; n];
    right[0] = M::Source::from(u64::from(challenge[0]));
    let joint = first.commit_second(&mut ps, Buffer::from(right));
    let mut forms = Vec::new();
    for (l, r) in [(2, 3), (5, 7)] {
        let mut weights = vec![M::Target::ZERO; 2 * n];
        weights[0] = M::Target::from(l);
        weights[n] = M::Target::from(r);
        forms.push(Covector::new(weights));
    }
    let values = vec![
        M::Target::from(2 + 3 * u64::from(challenge[0])),
        M::Target::from(5 + 7 * u64::from(challenge[0])),
    ];
    let refs: Vec<&dyn LinearForm<M::Target>> = forms.iter().map(|f| f as _).collect();
    let claim = joint.prove(&mut ps, &refs, &values);
    ps.prover_message(&M::Target::from(987));
    Example {
        proof: ps.proof(),
        config,
        forms,
        values,
        claim,
    }
}

fn verify<M>(example: &Example<M>, observe_header: bool) -> Result<(), ()>
where
    M: Embedding + Default,
    M::Target: Codec,
    Standard: Distribution<M::Target>,
{
    let ds = DomainSeparator::protocol(&example.config).instance(b"joint-two-sparse-forms");
    let mut vs = VerifierState::new_std(&ds, &example.proof);
    let first = example.config.receive_first(&mut vs).map_err(|_| ())?;
    assert_eq!(vs.prover_message::<Hash>().map_err(|_| ())?, Hash([7; 32]));
    let _ = vs.verifier_message::<[u8; 32]>();
    let joint = first.receive_second(&mut vs).map_err(|_| ())?;
    if observe_header {
        assert_eq!(vs.prover_message::<U64>().map_err(|_| ())?.0, 2);
        assert_eq!(
            vs.prover_messages_vec::<M::Target>(2).map_err(|_| ())?,
            example.values
        );
        assert_eq!(
            vs.verifier_message::<M::Target>(),
            example.claim.rlc_coefficients[1]
        );
        return Ok(());
    }
    let refs: Vec<&dyn LinearForm<M::Target>> = example.forms.iter().map(|f| f as _).collect();
    let claim = joint
        .verify(&mut vs, &refs, &example.values)
        .map_err(|_| ())?;
    claim.verify(&refs).map_err(|_| ())?;
    assert_eq!(claim.evaluation_point, example.claim.evaluation_point);
    assert_eq!(claim.rlc_coefficients, example.claim.rlc_coefficients);
    assert_eq!(
        claim.evaluation_point.len(),
        example.config.witness_commitment().vector_size().ilog2() as usize + 1
    );
    assert_eq!(
        vs.prover_message::<M::Target>().map_err(|_| ())?,
        M::Target::from(987)
    );
    vs.check_eof().map_err(|_| ())
}

fn rejects<T, E>(run: impl FnOnce() -> Result<T, E>) {
    let result = catch_unwind(AssertUnwindSafe(run));
    #[cfg(feature = "verifier_panics")]
    assert!(!matches!(result, Ok(Ok(_))));
    #[cfg(not(feature = "verifier_panics"))]
    assert!(
        matches!(result, Ok(Err(_))),
        "verification must reject without panicking"
    );
}

#[test]
fn challenge_separated_witnesses_open_in_both_modes_and_embeddings() {
    for mode in [Mode::Standard, Mode::ZeroKnowledge] {
        verify(&example(config::<Identity<Field256>>(mode, 4096, 2)), false).unwrap();
        verify(
            &example(config::<Basefield<Field64_3>>(mode, 4096, 2)),
            false,
        )
        .unwrap();
        verify(
            &example(config::<Identity<Field64_3>>(mode, 4096, 2)),
            false,
        )
        .unwrap();
    }
}

#[test]
fn selector_fold_supports_different_first_blocks_and_immediate_basecases() {
    for (n, first) in [(4096, 3), (2048, 2)] {
        verify(
            &example(config::<Identity<Field256>>(Mode::ZeroKnowledge, n, first)),
            false,
        )
        .unwrap();
    }
}

#[test]
fn all_claim_values_precede_the_batching_challenge() {
    for mode in [Mode::Standard, Mode::ZeroKnowledge] {
        verify(&example(config::<Identity<Field256>>(mode, 4096, 2)), true).unwrap();
    }
}

#[test]
fn altered_values_and_truncated_proofs_are_rejected() {
    for mode in [Mode::Standard, Mode::ZeroKnowledge] {
        let mut example = example(config::<Identity<Field256>>(mode, 4096, 2));
        example.values[0] += Field256::ONE;
        rejects(|| verify(&example, false));
        example.values[0] -= Field256::ONE;
        example
            .proof
            .narg_string
            .truncate(example.proof.narg_string.len() / 2);
        rejects(|| verify(&example, false));
    }
}

#[test]
fn altered_mask_root_is_rejected() {
    let mut example = example(config::<Identity<Field256>>(Mode::ZeroKnowledge, 4096, 2));
    example.proof.narg_string[0] ^= 1;
    rejects(|| verify(&example, false));
}
