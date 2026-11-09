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
        codecs::Empty, Codec, DomainSeparator, Proof, ProverState, VerifierMessage, VerifierState,
    },
};

fn config<M: Embedding + Default>(mode: Mode, n: usize, first: usize) -> ProtocolConfig<M> {
    ProtocolConfig::derive(
        SecuritySpec::new(128)
            .with_mode(mode)
            .with_pow_budget(PowBudget::Forbidden),
        TuningSpec {
            vector_size: n,
            starting_log_inv_rate: 1,
            folding_factor: FoldingFactor::ConstantFromSecondRound {
                initial: first,
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

fn example<M>(config: ProtocolConfig<M>, instance: M::Target, first: u64) -> Example<M>
where
    M: Embedding + Default,
    M::Target: Codec,
    Standard: Distribution<M::Source> + Distribution<M::Target>,
{
    let n = config.tuning().vector_size;
    let mut left = vec![M::Source::ZERO; n];
    left[..3].copy_from_slice(&[first, 2, 3].map(M::Source::from));
    let ds = DomainSeparator::protocol(&config).instance(&instance);
    let mut ps = ProverState::new_std(&ds);
    let mut left = config
        .commit_pair(&mut ps, Buffer::from(left), None)
        .unwrap();
    // An application commitment and challenge separate the two witness commitments.
    ps.prover_message(&Hash([7; 32]));
    let stage: [u8; 32] = ps.verifier_message();
    let mut right = vec![M::Source::ZERO; n];
    right[..2].copy_from_slice(&[u64::from(stage[0]), 7].map(M::Source::from));
    let right = config
        .commit_pair(&mut ps, Buffer::from(right), Some(&mut left))
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
        M::Target::from(109 + 2 * first + 11 * c),
        instance * M::Target::from(c + 2) + M::Target::from(2),
    ];
    let refs: Vec<&dyn LinearForm<M::Target>> = forms.iter().map(|f| f as _).collect();
    let result = config
        .prove_pair(&mut ps, [left, right], &refs, &values)
        .unwrap();
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
    let left = example.config.receive_pair(&mut vs, None).map_err(|_| ())?;
    if vs.prover_message::<Hash>().map_err(|_| ())? != Hash([7; 32]) {
        return Err(());
    }
    let stage: [u8; 32] = vs.verifier_message();
    let right = example
        .config
        .receive_pair(&mut vs, Some(&left))
        .map_err(|_| ())?;
    let refs: Vec<&dyn LinearForm<M::Target>> = example.forms.iter().map(|f| f as _).collect();
    let mut values = example.values.clone();
    if bad_claim {
        values[0] += M::Target::ONE;
    }
    let result = example
        .config
        .verify_pair(&mut vs, [left, right], &refs, &values)
        .map_err(|_| ())?;
    result.verify(&refs).map_err(|_| ())?;
    assert_eq!(stage, example.stage);
    assert_eq!(result.evaluation_point, example.result.evaluation_point);
    assert_eq!(result.rlc_coefficients, example.result.rlc_coefficients);
    assert_eq!(
        result.evaluation_point.len(),
        example.config.tuning().vector_size.ilog2() as usize + 1
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
    for mode in [Mode::Standard, Mode::ZeroKnowledge] {
        verify(
            &example(config::<Basefield<Field64_3>>(mode, 4096, 2), e, 1),
            false,
        )
        .unwrap();
        verify(
            &example(config::<Identity<Field64_3>>(mode, 4096, 2), e, 1),
            false,
        )
        .unwrap();
        verify(
            &example(
                config::<Identity<Field256>>(mode, 4096, 2),
                Field256::from(19),
                1,
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
        verify(&example(scheme, Field256::from(19), 1), false).unwrap();
    }
}

fn rejects<T, E>(run: impl FnOnce() -> Result<T, E>) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(run));
    #[cfg(feature = "verifier_panics")]
    assert!(!matches!(result, Ok(Ok(_))));
    #[cfg(not(feature = "verifier_panics"))]
    assert!(
        matches!(result, Ok(Err(_))),
        "verification must reject without panicking"
    );
}

#[test]
fn changed_roots_claims_and_truncation_are_rejected() {
    for mode in [Mode::Standard, Mode::ZeroKnowledge] {
        let mut example = example(
            config::<Identity<Field256>>(mode, 4096, 2),
            Field256::from(19),
            1,
        );
        rejects(|| verify(&example, true));
        example.proof.narg_string[0] ^= 1;
        rejects(|| verify(&example, false));
        example.proof.narg_string[0] ^= 1;
        example
            .proof
            .narg_string
            .truncate(example.proof.narg_string.len() / 2);
        rejects(|| verify(&example, false));
    }
}

#[test]
fn swapped_and_mixed_handles_are_rejected() {
    type F = Field256;
    for mode in [Mode::Standard, Mode::ZeroKnowledge] {
        let config = config::<Identity<F>>(mode, 4096, 2);
        let first = example(config.clone(), F::from(19), 1);
        let other = example(config.clone(), F::from(19), 2);
        let ds = DomainSeparator::protocol(&config).instance(&first.instance);
        let receive = |vs: &mut VerifierState<'_>| {
            let left = config.receive_pair(vs, None).unwrap();
            assert_eq!(vs.prover_message::<Hash>().unwrap(), Hash([7; 32]));
            let _: [u8; 32] = vs.verifier_message();
            let right = config.receive_pair(vs, Some(&left)).unwrap();
            [left, right]
        };
        let refs: Vec<&dyn LinearForm<F>> = first.forms.iter().map(|form| form as _).collect();
        for replace in 0..3 {
            let mut vs = VerifierState::new_std(&ds, &first.proof);
            let mut handles = receive(&mut vs);
            if replace == 2 {
                handles.swap(0, 1);
            } else {
                let mut other_vs = VerifierState::new_std(&ds, &other.proof);
                handles[replace] = receive(&mut other_vs)[replace].clone();
            }
            rejects(|| config.verify_pair(&mut vs, handles, &refs, &first.values));
        }
    }
}

#[test]
fn rejected_inputs_and_repeated_completion_leave_the_transcript_unchanged() {
    type F = Field256;
    for mode in [Mode::Standard, Mode::ZeroKnowledge] {
        let config = config::<Identity<F>>(mode, 4096, 2);
        let other = self::config::<Identity<F>>(mode, 8192, 2);
        let ds = DomainSeparator::protocol(&config).instance(&Empty);
        let mut ps = ProverState::new_std(&ds);
        assert!(config
            .commit_pair(&mut ps, Buffer::from(vec![F::ONE; 4095]), None)
            .is_err());
        let mut first = config
            .commit_pair(&mut ps, Buffer::from(vec![F::ONE; 4096]), None)
            .unwrap();
        assert!(config
            .commit_pair(&mut ps, Buffer::from(vec![F::ONE; 4095]), Some(&mut first))
            .is_err());
        assert!(other
            .commit_pair(&mut ps, Buffer::from(vec![F::ONE; 8192]), Some(&mut first))
            .is_err());
        let second = config
            .commit_pair(&mut ps, Buffer::from(vec![F::ONE; 4096]), Some(&mut first))
            .unwrap();
        assert!(config
            .commit_pair(&mut ps, Buffer::from(vec![F::ONE; 4096]), Some(&mut first))
            .is_err());
        let short = Covector::new(vec![F::ZERO; 4096]);
        assert!(config
            .prove_pair(&mut ps, [first, second], &[&short], &[F::ZERO])
            .is_err());
        let challenge: F = ps.verifier_message();
        ps.prover_message(&Hash([9; 32]));
        let proof = ps.proof();
        // This control skips every rejected call, detecting even symmetric transcript mutations.
        for attempt_invalid_opening in [false, true] {
            let mut vs = VerifierState::new_std(&ds, &proof);
            let first = config.receive_pair(&mut vs, None).unwrap();
            let second = config.receive_pair(&mut vs, Some(&first)).unwrap();
            if attempt_invalid_opening {
                assert!(config
                    .verify_pair(&mut vs, [first, second], &[&short], &[F::ZERO])
                    .is_err());
            }
            assert_eq!(vs.verifier_message::<F>(), challenge);
            assert_eq!(vs.prover_message::<Hash>().unwrap(), Hash([9; 32]));
            vs.check_eof().unwrap();
        }
    }
}

#[test]
fn unfinished_pair_handles_cannot_use_single_openings() {
    type F = Field256;
    for mode in [Mode::Standard, Mode::ZeroKnowledge] {
        let config = config::<Identity<F>>(mode, 4096, 2);
        let ds = DomainSeparator::protocol(&config).instance(&Empty);
        let mut ps = ProverState::new_std(&ds);
        let first = config
            .commit_pair(&mut ps, Buffer::from(vec![F::ONE; 4096]), None)
            .unwrap();
        let proof = ps.proof();
        let form = Covector::new(vec![F::ZERO; 4096]);
        let mut vs = VerifierState::new_std(&ds, &proof);
        let commitment = config.receive_pair(&mut vs, None).unwrap();
        rejects(|| config.verify(&mut vs, commitment, &[&form], &[F::ZERO]));
        let mut ps = ProverState::new_std(&ds);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            config.prove(&mut ps, first, &[&form], &[F::ZERO])
        }))
        .is_err());
    }
}
