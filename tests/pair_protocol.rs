//! Small examples for the complete two-witness opening.
use ark_ff::{AdditiveGroup, Field};
use ark_std::rand::{distributions::Standard, prelude::Distribution};
use whir::{
    algebra::{
        embedding::{Basefield, Embedding, Identity},
        fields::{Field256, Field64_3},
        linear_form::{Covector, LinearForm},
    },
    buffer::Buffer,
    hash,
    protocols::{
        params::{
            DecodingRegime, FoldingFactor, Mode, PowBudget, RateSchedule, SecuritySpec, TuningSpec,
        },
        zook::{ProtocolConfig, ProverClaim},
    },
    transcript::{
        codecs::Empty, Codec, DomainSeparator, Proof, ProverState, VerifierMessage, VerifierState,
    },
};
use zeroize::Zeroize;

fn source<M: Embedding + Default>(mode: Mode) -> ProtocolConfig<M> {
    ProtocolConfig::derive(
        SecuritySpec {
            mode,
            decoding_regime: DecodingRegime::Johnson,
            target_security_bits: 128,
            pow_budget: PowBudget::per_slot(10),
            hash_id: hash::BLAKE3,
        },
        TuningSpec {
            vector_size: 256,
            starting_log_inv_rate: 1,
            folding_factor: FoldingFactor::Constant(2),
            rate_schedule: RateSchedule::Stepping,
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
    stage_challenge: M::Target,
}

fn prove_example<M>(config: ProtocolConfig<M>, extension_coefficient: M::Target) -> Example<M>
where
    M: Embedding + Default,
    M::Target: Default + Zeroize + Codec,
    Standard: Distribution<M::Source> + Distribution<M::Target>,
{
    let config: ProtocolConfig<M> =
        serde_json::from_value(serde_json::to_value(config).unwrap()).unwrap();
    let n = config.tuning().vector_size;
    let mut left = vec![M::Source::ZERO; n];
    let mut right = left.clone();
    // Three entries and two entries are independently padded to N.
    left[..3].copy_from_slice(&[1u64, 2, 3].map(M::Source::from));
    right[..2].copy_from_slice(&[5u64, 7].map(M::Source::from));
    let form = |entries: &[(usize, M::Target)]| {
        let mut weights = vec![M::Target::ZERO; 2 * n];
        for &(i, weight) in entries {
            weights[i] = weight;
        }
        Covector::new(weights)
    };
    let e = extension_coefficient;
    let forms = vec![
        form(&[
            (0, 2u64.into()),
            (1, 3u64.into()),
            (2, 4u64.into()),
            (n, 11u64.into()),
            (n + 1, 13u64.into()),
        ]),
        form(&[(1, e + M::Target::ONE), (n, e)]),
        form(&[(2, M::Target::ONE)]),
    ];
    // 2*1 + 3*2 + 4*3 = 20; 11*5 + 13*7 = 146; total = 166.
    // The extension-valued claim is (e+1)*2 + e*5 = 7e+2.
    let values = vec![
        166u64.into(),
        e * M::Target::from(7) + M::Target::from(2),
        3u64.into(),
    ];
    let refs: Vec<&dyn LinearForm<M::Target>> = forms.iter().map(|f| f as _).collect();
    let ds = DomainSeparator::protocol(&"two-witness-examples").instance(&Empty);
    let mut ps = ProverState::new_std(&ds);
    let left = config.commit(&mut ps, Buffer::from(left));
    let stage_challenge = ps.verifier_message();
    let right = config.commit(&mut ps, Buffer::from(right));
    let result = config
        .prove_pair(&mut ps, [left, right], &refs, &values)
        .unwrap();
    // A separate protocol can continue after the pair opening.
    ps.prover_message(&M::Target::from(987));
    Example {
        config,
        proof: ps.proof(),
        forms,
        values,
        result,
        stage_challenge,
    }
}

#[derive(Clone, Copy, Debug)]
enum Change {
    None,
    SwapRoots,
    LeftRoot,
    RightRoot,
    Claim,
    Form,
}

fn verify_example<M>(example: &Example<M>, change: Change) -> Result<(), ()>
where
    M: Embedding + Default,
    M::Target: Default + Codec,
    Standard: Distribution<M::Target>,
{
    let ds = DomainSeparator::protocol(&"two-witness-examples").instance(&Empty);
    let mut vs = VerifierState::new_std(&ds, &example.proof);
    let left = example.config.receive_commitment(&mut vs).map_err(|_| ())?;
    let challenge: M::Target = vs.verifier_message();
    assert_eq!(challenge, example.stage_challenge);
    let right = example.config.receive_commitment(&mut vs).map_err(|_| ())?;
    let commitments = match change {
        Change::SwapRoots => [right, left],
        Change::LeftRoot => [right.clone(), right],
        Change::RightRoot => [left.clone(), left],
        _ => [left, right],
    };
    let refs: Vec<&dyn LinearForm<M::Target>> = example.forms.iter().map(|f| f as _).collect();
    let mut values = example.values.clone();
    if matches!(change, Change::Claim) {
        values[0] += M::Target::ONE;
    }
    let result = example
        .config
        .verify_pair(&mut vs, commitments, &refs, &values)
        .map_err(|_| ())?;
    assert_eq!(
        result.evaluation_point.len(),
        example.config.tuning().vector_size.ilog2() as usize + 1,
        "selector plus the original variables"
    );
    assert_eq!(result.evaluation_point, example.result.evaluation_point);
    assert_eq!(result.rlc_coefficients, example.result.rlc_coefficients);
    let mut final_forms = refs;
    if matches!(change, Change::Form) {
        final_forms.swap(0, 1);
    }
    result.verify(&final_forms).map_err(|_| ())?;
    let sentinel: M::Target = vs.prover_message().map_err(|_| ())?;
    assert_eq!(sentinel, M::Target::from(987));
    vs.check_eof().map_err(|_| ())
}

#[test]
fn hand_calculated_claims_use_both_existing_commitments_at_128_bits() {
    for mode in [Mode::Standard, Mode::ZeroKnowledge] {
        let example = prove_example(source::<Identity<Field256>>(mode), Field256::from(19));
        verify_example(&example, Change::None).unwrap();
    }
}

#[test]
fn genuine_extension_coefficients_open_base_field_witnesses_at_128_bits() {
    let e = Field64_3::new(2u64.into(), 3u64.into(), 5u64.into());
    assert_ne!(e, Field64_3::from(2));
    for mode in [Mode::Standard, Mode::ZeroKnowledge] {
        let example = prove_example(source::<Basefield<Field64_3>>(mode), e);
        assert_ne!(example.values[1], Field64_3::from(16));
        verify_example(&example, Change::None).unwrap();
    }
}

#[test]
fn selector_fold_preserves_a_short_tail_or_an_immediate_basecase() {
    type F = Field256;
    for mode in [Mode::Standard, Mode::ZeroKnowledge] {
        let source = source::<Identity<F>>(mode);
        for (vector_size, initial, rounds) in [
            // Source folds 256 -> 32 -> 8, with factors 3 and 2.
            (256, 3, 2),
            // Source folds 16 -> 4, then enters the basecase directly.
            (16, 2, 1),
        ] {
            let config = ProtocolConfig::<Identity<F>>::derive(
                source.security().clone(),
                TuningSpec {
                    vector_size,
                    folding_factor: FoldingFactor::ConstantFromSecondRound { initial, rest: 2 },
                    ..source.tuning().clone()
                },
            )
            .unwrap();
            assert_eq!(config.num_rounds(), rounds);
            let example = prove_example(config, F::from(19));
            verify_example(&example, Change::None).unwrap();
        }
    }
}

#[test]
fn changed_roots_claim_values_and_final_forms_are_rejected() {
    for mode in [Mode::Standard, Mode::ZeroKnowledge] {
        let example = prove_example(source::<Identity<Field256>>(mode), Field256::from(19));
        for change in [
            Change::SwapRoots,
            Change::LeftRoot,
            Change::RightRoot,
            Change::Claim,
            Change::Form,
        ] {
            // The debug feature deliberately panics when verification rejects a proof.
            #[cfg(feature = "verifier_panics")]
            let rejected = std::panic::catch_unwind(|| verify_example(&example, change))
                .map_or(true, |result| result.is_err());
            #[cfg(not(feature = "verifier_panics"))]
            let rejected = verify_example(&example, change).is_err();
            assert!(rejected, "{mode:?}: {change:?} must fail");
        }
    }
}

#[test]
fn truncated_joint_proofs_are_rejected() {
    for mode in [Mode::Standard, Mode::ZeroKnowledge] {
        let mut example = prove_example(source::<Identity<Field256>>(mode), Field256::from(19));
        example
            .proof
            .narg_string
            .truncate(example.proof.narg_string.len() / 2);
        assert!(verify_example(&example, Change::None).is_err());
    }
}

#[test]
fn all_claim_values_precede_the_shared_batching_challenge() {
    type F = Field256;
    for mode in [Mode::Standard, Mode::ZeroKnowledge] {
        let example = prove_example(source::<Identity<F>>(mode), F::from(19));
        let ds = DomainSeparator::protocol(&"two-witness-examples").instance(&Empty);
        let mut observer = VerifierState::new_std(&ds, &example.proof);
        let _ = example.config.receive_commitment(&mut observer).unwrap();
        let _: F = observer.verifier_message();
        let _ = example.config.receive_commitment(&mut observer).unwrap();
        let _: hash::Hash = observer.prover_message().unwrap();
        let count: whir::transcript::codecs::U64 = observer.prover_message().unwrap();
        assert_eq!(count.0 as usize, example.values.len());
        for expected in &example.values {
            let bound_value: F = observer.prover_message().unwrap();
            assert_eq!(&bound_value, expected);
        }
        let gamma: F = observer.verifier_message();
        assert_eq!(example.result.rlc_coefficients[0], F::ONE);
        assert_eq!(example.result.rlc_coefficients[1], gamma);
        assert_eq!(example.result.rlc_coefficients[2], gamma * gamma);
    }
}

#[test]
fn malformed_claim_dimensions_fail_before_transcript_mutation() {
    type F = Field256;
    type ClaimInput<'a> = (&'a [&'a dyn LinearForm<F>], &'a [F]);
    for mode in [Mode::Standard, Mode::ZeroKnowledge] {
        let config = source::<Identity<F>>(mode);
        let ds = DomainSeparator::protocol(&"pair-shape-checks").instance(&Empty);
        let mut ps = ProverState::new_std(&ds);
        let committed = [
            config.commit(&mut ps, Buffer::from(vec![F::ONE; 256])),
            config.commit(&mut ps, Buffer::from(vec![F::ONE; 256])),
        ];
        let form = Covector::new(vec![F::ZERO; 512]);
        let short = Covector::new(vec![F::ZERO; 256]);
        let malformed: [ClaimInput<'_>; 3] = [(&[], &[]), (&[&form], &[]), (&[&short], &[F::ZERO])];
        for (forms, values) in malformed {
            assert!(config
                .prove_pair(&mut ps, committed.clone(), forms, values)
                .is_err());
        }
        config
            .prove_pair(&mut ps, committed, &[&form], &[F::ZERO])
            .unwrap();
        let proof = ps.proof();
        let mut vs = VerifierState::new_std(&ds, &proof);
        let commitments = [
            config.receive_commitment(&mut vs).unwrap(),
            config.receive_commitment(&mut vs).unwrap(),
        ];
        for (forms, values) in malformed {
            assert!(config
                .verify_pair(&mut vs, commitments.clone(), forms, values)
                .is_err());
        }
        config
            .verify_pair(&mut vs, commitments, &[&form], &[F::ZERO])
            .unwrap()
            .verify(&[&form])
            .unwrap();
        vs.check_eof().unwrap();
    }
}

#[test]
fn incompatible_source_handles_fail_before_transcript_mutation() {
    type M = Identity<Field256>;
    type F = Field256;
    for mode in [Mode::Standard, Mode::ZeroKnowledge] {
        let source = source::<M>(mode);
        let config = &source;
        let mut different_rate = source.tuning().clone();
        different_rate.starting_log_inv_rate += 1;
        let mut different_folding = source.tuning().clone();
        different_folding.folding_factor = FoldingFactor::Constant(3);
        let alternatives = [
            ProtocolConfig::<M>::derive(source.security().clone(), different_rate).unwrap(),
            ProtocolConfig::<M>::derive(source.security().clone(), different_folding).unwrap(),
            ProtocolConfig::<M>::derive(
                source.security().clone().with_mode(match mode {
                    Mode::Standard => Mode::ZeroKnowledge,
                    Mode::ZeroKnowledge => Mode::Standard,
                }),
                source.tuning().clone(),
            )
            .unwrap(),
        ];
        for other in alternatives {
            for sources in [[&source, &other], [&other, &source]] {
                let ds = DomainSeparator::protocol(&"incompatible-source-handles").instance(&Empty);
                let mut ps = ProverState::new_std(&ds);
                let handles =
                    sources.map(|source| source.commit(&mut ps, Buffer::from(vec![F::ONE; 256])));
                let form = Covector::new(vec![F::ONE; 512]);
                assert!(config
                    .prove_pair(&mut ps, handles, &[&form], &[F::from(512)])
                    .is_err());
                ps.prover_message(&F::from(987));
                let proof = ps.proof();
                let mut vs = VerifierState::new_std(&ds, &proof);
                for source in sources {
                    let _ = source.receive_commitment(&mut vs).unwrap();
                }
                let sentinel: F = vs.prover_message().unwrap();
                assert_eq!(sentinel, F::from(987));
                vs.check_eof().unwrap();
            }
        }
    }
}
