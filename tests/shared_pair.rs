//! One mask collection across two challenge-separated witness commitments.

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
        params::{FoldingFactor, Mode, PowBudget, RateSchedule, SecuritySpec, TuningSpec},
        zook::{ProtocolConfig, ProverClaim},
    },
    transcript::{
        codecs::Empty, Codec, DomainSeparator, NargSerialize, Proof, ProverState,
        VerificationError, VerificationResult, VerifierMessage, VerifierState,
    },
};

fn config<M: Embedding + Default>(mode: Mode, n: usize, first: usize) -> ProtocolConfig<M> {
    ProtocolConfig::derive(
        SecuritySpec::new(128)
            .with_mode(mode)
            .with_pow_budget(PowBudget::Forbidden),
        TuningSpec {
            vector_size: 2 * n,
            starting_log_inv_rate: 1,
            folding_factor: FoldingFactor::ConstantFromSecondRound {
                initial: first + 1,
                rest: 2,
            },
            rate_schedule: RateSchedule::Capped {
                max_log_inv_rate: 1,
            },
        },
    )
    .unwrap()
}

struct Example<M: Embedding> {
    config: ProtocolConfig<M>,
    proof: Proof,
    forms: Vec<Covector<M::Target>>,
    values: Vec<M::Target>,
    result: ProverClaim<M::Target>,
    stage: [u8; 32],
    instance: M::Target,
}

fn example<M>(config: ProtocolConfig<M>, instance: M::Target) -> Example<M>
where
    M: Embedding + Default,
    M::Target: Codec,
    Standard: Distribution<M::Source> + Distribution<M::Target>,
{
    let n = config.tuning().vector_size / 2;
    let mut left = vec![M::Source::ZERO; n];
    left[..3].copy_from_slice(&[1u64, 2, 3].map(M::Source::from));
    let ds = DomainSeparator::protocol(&config).instance(&instance);
    let mut ps = ProverState::new_std(&ds);
    let mut stage = [0; 32];
    let pair = config
        .commit_pair(&mut ps, Buffer::from(left), |ps| -> VerificationResult<_> {
            // An application commitment can precede its challenge inside the callback.
            ps.prover_message(&Hash([7; 32]));
            stage = ps.verifier_message();
            let mut right = vec![M::Source::ZERO; n];
            right[..2].copy_from_slice(&[u64::from(stage[0]), 7].map(M::Source::from));
            Ok(Buffer::from(right))
        })
        .unwrap();
    let form = |entries: &[(usize, M::Target)]| {
        let mut weights = vec![M::Target::ZERO; 2 * n];
        for &(index, weight) in entries {
            weights[index] = weight;
        }
        Covector::new(weights)
    };
    // The instance and fixed protocol determine both public forms.
    let forms = vec![
        form(&[
            (0, 2u64.into()),
            (1, 3u64.into()),
            (2, 4u64.into()),
            (n, 11u64.into()),
            (n + 1, 13u64.into()),
        ]),
        form(&[(1, instance + M::Target::ONE), (n, instance)]),
    ];
    let c = u64::from(stage[0]);
    let values = vec![
        M::Target::from(111 + 11 * c),
        instance * M::Target::from(c + 2) + M::Target::from(2),
    ];
    let refs: Vec<&dyn LinearForm<M::Target>> = forms.iter().map(|f| f as _).collect();
    ps.prover_messages(&values);
    let result = config.prove(&mut ps, pair, &refs, &values);
    ps.prover_message(&M::Target::from(987));
    Example {
        config,
        proof: ps.proof(),
        forms,
        values,
        result,
        stage,
        instance,
    }
}

fn verify<M>(example: &Example<M>, bad_claim: bool) -> Result<(), ()>
where
    M: Embedding + Default,
    M::Target: Default + Codec,
    Standard: Distribution<M::Target>,
{
    let ds = DomainSeparator::protocol(&example.config).instance(&example.instance);
    let mut vs = VerifierState::new_std(&ds, &example.proof);
    let mut stage = [0; 32];
    let pair = example
        .config
        .receive_pair(&mut vs, |vs| -> VerificationResult<()> {
            if vs.prover_message::<Hash>()? != Hash([7; 32]) {
                return Err(VerificationError);
            }
            stage = vs.verifier_message();
            Ok(())
        })
        .map_err(|_| ())?;
    let refs: Vec<&dyn LinearForm<M::Target>> = example.forms.iter().map(|f| f as _).collect();
    let mut values = example.values.clone();
    if bad_claim {
        values[0] += M::Target::ONE;
    }
    if vs
        .prover_messages_vec::<M::Target>(values.len())
        .map_err(|_| ())?
        != values
    {
        return Err(());
    }
    let result = example
        .config
        .verify(&mut vs, pair, &refs, &values)
        .map_err(|_| ())?;
    result.verify(&refs).map_err(|_| ())?;
    assert_eq!(stage, example.stage);
    assert_eq!(result.evaluation_point, example.result.evaluation_point);
    assert_eq!(result.rlc_coefficients, example.result.rlc_coefficients);
    assert_eq!(
        result.evaluation_point.len(),
        example.config.tuning().vector_size.ilog2() as usize
    );
    assert_eq!(
        vs.prover_message::<M::Target>().map_err(|_| ())?,
        M::Target::from(987)
    );
    vs.check_eof().map_err(|_| ())
}

#[test]
fn staged_pairs_support_both_modes_and_field_embeddings() {
    let e = Field64_3::new(2u64.into(), 3u64.into(), 5u64.into());
    verify(
        &example(
            config::<Basefield<Field64_3>>(Mode::ZeroKnowledge, 4096, 2),
            e,
        ),
        false,
    )
    .unwrap();
    verify(
        &example(
            config::<Identity<Field64_3>>(Mode::ZeroKnowledge, 4096, 2),
            e,
        ),
        false,
    )
    .unwrap();
    for mode in [Mode::Standard, Mode::ZeroKnowledge] {
        verify(
            &example(
                config::<Identity<Field256>>(mode, 4096, 2),
                Field256::from(19),
            ),
            false,
        )
        .unwrap();
    }
}

#[test]
fn shared_pairs_cover_short_tails_and_immediate_basecases() {
    for (n, first) in [(4096, 3), (2048, 2)] {
        let mut scheme = config::<Identity<Field256>>(Mode::ZeroKnowledge, n, first);
        if n == 4096 {
            let mut tuning = scheme.tuning().clone();
            tuning.rate_schedule = RateSchedule::Stepping;
            scheme = ProtocolConfig::derive(scheme.security().clone(), tuning).unwrap();
            assert!(scheme.num_rounds() > 1);
        } else {
            assert_eq!(scheme.num_rounds(), 1);
        }
        verify(&example(scheme, Field256::from(19)), false).unwrap();
    }
}

#[test]
fn changed_roots_claims_and_truncation_are_rejected() {
    let mut example = example(
        config::<Identity<Field256>>(Mode::ZeroKnowledge, 4096, 2),
        Field256::from(19),
    );
    let original = example.proof.clone();
    let rejected = |example: &Example<Identity<Field256>>, bad_claim| {
        #[cfg(feature = "verifier_panics")]
        let rejected = std::panic::catch_unwind(|| verify(example, bad_claim))
            .map_or(true, |result| result.is_err());
        #[cfg(not(feature = "verifier_panics"))]
        let rejected = verify(example, bad_claim).is_err();
        assert!(rejected);
    };
    rejected(&example, true);
    let mut encoded = Vec::new();
    Field256::ZERO.serialize_into_narg(&mut encoded);
    // Change the public claim and its transcript encoding together.
    example.values[0] += Field256::ONE;
    let mut changed = Vec::new();
    example.values[0].serialize_into_narg(&mut changed);
    let claim_offset = 128 + 2 * encoded.len();
    example.proof.narg_string[claim_offset..claim_offset + changed.len()].copy_from_slice(&changed);
    rejected(&example, false);
    example.values[0] -= Field256::ONE;
    // Mask root, left root, two binding values, callback hash, right root.
    for offset in [0, 32, 96 + 2 * encoded.len()] {
        example.proof = original.clone();
        example.proof.narg_string[offset] ^= 1;
        rejected(&example, false);
    }
    example.proof = original;
    example
        .proof
        .narg_string
        .truncate(example.proof.narg_string.len() / 2);
    rejected(&example, false);
}

#[test]
fn invalid_inputs_and_callback_errors_do_not_commit_the_second_witness() {
    type F = Field256;
    let config = config::<Identity<F>>(Mode::ZeroKnowledge, 4096, 2);
    let ds = DomainSeparator::protocol(&config).instance(&Empty);
    for case in 0..3 {
        let mut ps = ProverState::new_std(&ds);
        let mut called = false;
        let first_len = if case == 0 { 4095 } else { 4096 };
        assert!(config
            .commit_pair(
                &mut ps,
                Buffer::from(vec![F::ONE; first_len]),
                |_| -> VerificationResult<_> {
                    called = true;
                    if case == 1 {
                        return Err(VerificationError);
                    }
                    Ok(Buffer::from(vec![F::ONE; 4095]))
                }
            )
            .is_err());
        assert_eq!(called, case != 0);
        ps.prover_message(&Hash([9; 32]));
        let proof = ps.proof();
        let mut vs = VerifierState::new_std(&ds, &proof);
        if case != 0 {
            assert!(config
                .receive_pair(&mut vs, |_| -> VerificationResult<()> {
                    Err(VerificationError)
                })
                .is_err());
        }
        assert_eq!(vs.prover_message::<Hash>().unwrap(), Hash([9; 32]));
        vs.check_eof().unwrap();
    }
}
