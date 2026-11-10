//! PR #283 parameter baselines and seeded proofs through deserialized configurations.
use ark_ff::Field;
use ark_std::rand::{distributions::Standard, prelude::Distribution, rngs::StdRng, SeedableRng};
use serde_json::{json, Value};
use zeroize::Zeroize;

use super::*;
use crate::{
    algebra::{
        embedding::{Basefield, Embedding, Identity},
        fields::{Field256, Field64_3},
        linear_form::{Covector, Evaluate, LinearForm},
        random_vector,
    },
    buffer::Buffer,
    protocols::{irs_commit, proof_of_work, sumcheck},
    transcript::{codecs::U64, Codec, DomainSeparator, ProverState, VerifierState},
};

fn code_parameters<M: Embedding>(code: &irs_commit::Config<M>) -> Value {
    let matrix = code.matrix_commit();
    json!({
        "vectors": code.num_vectors(),
        "vector_size": code.vector_size(),
        "codeword_length": code.codeword_length(),
        "interleaving": code.interleaving_depth(),
        "queries": code.in_domain_samples(),
        "deduplicate": code.deduplicate_in_domain(),
        "prefix_mask_length": code.mask_length(),
        "unique_decoding": code.unique_decoding(),
        "list_size_bits": code.list_size().to_bits(),
        "leaf_hash": hex::encode(matrix.leaf_hash_id.as_slice()),
        "tree_hashes": matrix.merkle_tree.layers.iter()
            .map(|layer| hex::encode(layer.hash_id.as_slice())).collect::<Vec<_>>(),
    })
}

fn pow_parameters(pow: proof_of_work::Config) -> Value {
    json!({"threshold": pow.threshold, "hash": hex::encode(pow.hash_id.as_slice())})
}

fn sumcheck_parameters<F: Field>(config: &sumcheck::Config<F>) -> Value {
    let mask = match config.mode() {
        sumcheck::SumcheckMode::Standard => 0,
        sumcheck::SumcheckMode::ZeroKnowledge { mask_length } => mask_length.get(),
    };
    json!({
        "initial_size": config.initial_size(),
        "rounds": config.num_rounds(),
        "mask_length": mask,
        "pow": pow_parameters(config.round_pow()),
    })
}

fn round_parameters<M: Embedding>(round: &RoundConfig<'_, M>) -> Value {
    let switch = &round.code_switch;
    json!({
        "index": round.index,
        "zk": round.masks.is_some(),
        "source": code_parameters(switch.source()),
        "target": code_parameters(switch.target()),
        "out_domain_samples": switch.out_domain_samples(),
        "message_mask_length": switch.message_mask_length(),
        "switch_zk": switch.is_zk(),
        "pow": pow_parameters(switch.pow()),
        "sumcheck": sumcheck_parameters(&round.sumcheck),
        "sumcheck_mask_coefficients": round.masks.map(|_| 2 * round.sumcheck.num_rounds() + 1),
    })
}

fn parameters<M: Embedding>(config: &ProtocolConfig<M>) -> Value {
    let mut rounds = Vec::new();
    if let Some(first) = config.first_round() {
        rounds.push(round_parameters(&first));
    }
    rounds.extend(config.tail_rounds().map(|round| round_parameters(&round)));
    let basecase = config.basecase();
    let masks: Vec<_> = (0..=config.num_rounds())
        .map(|index| {
            config
                .witness_mask(index)
                .map(|mask| [mask.num_polynomials, mask.coefficients_per_polynomial])
        })
        .collect();
    let joint = config.joint_masks().map(|joint| json!({
        "code": code_parameters(&joint.commit),
        "sumcheck_ranges": joint.sumcheck_masks.iter().map(|range| [range.start, range.end]).collect::<Vec<_>>(),
        "witness_ranges": joint.witness_masks.iter().map(|range| [range.start, range.end]).collect::<Vec<_>>(),
        "blinder": [joint.blinder.start, joint.blinder.end],
        "basefield_blinder": [joint.basefield_blinder.start, joint.basefield_blinder.end],
    }));
    json!({
        "padded_vector_size": config.padded_vector_size(),
        "rounds": rounds,
        "basecase": {
            "zk": config.params().zk,
            "code": code_parameters(config.basecase().commit()),
            "sumcheck": sumcheck_parameters(basecase.sumcheck()),
            "pow": pow_parameters(basecase.pow()),
        },
        "witness_masks": masks,
        "joint_masks": joint,
    })
}

fn check<M: Embedding + Default>(
    expected: &Value,
    config: &ProtocolConfig<M>,
) -> ProtocolConfig<M> {
    if let Some(first) = config.first_round() {
        let mut previous = first.code_switch.target();
        for round in config.tail_rounds() {
            assert!(std::ptr::eq(previous, round.code_switch.source()));
            previous = round.code_switch.target();
        }
        assert!(std::ptr::eq(previous, config.final_commit()));
    }
    assert!(std::ptr::eq(
        config.final_commit(),
        config.basecase().commit()
    ));
    if expected["name"].as_str().unwrap().ends_with("pair") {
        config.validate_pair().unwrap();
        let first = config.first_round().unwrap();
        let node = config.pair_source().unwrap();
        let paired = config.pair_round(&node).unwrap();
        assert_eq!(
            paired.sumcheck.num_rounds(),
            first.sumcheck.num_rounds() + 1
        );
        assert_eq!(
            paired.sumcheck.initial_size(),
            2 * first.sumcheck.initial_size()
        );
        assert_eq!(paired.sumcheck.final_size(), first.sumcheck.final_size());
        assert!(std::ptr::eq(
            paired.code_switch.target(),
            first.code_switch.target()
        ));
    }
    let name = expected["name"].as_str().unwrap();
    let projection = serde_json::to_vec(&parameters(config)).unwrap();
    assert_eq!(
        *expected,
        json!({"name": name, "rounds": config.num_rounds(),
        "parameters_blake3": blake3::hash(&projection).to_hex().to_string()})
    );
    let restored = serde_json::from_value(serde_json::to_value(config).unwrap()).unwrap();
    assert_eq!(config, &restored);
    let mut bytes = Vec::new();
    ciborium::into_writer(config, &mut bytes).unwrap();
    assert_eq!(
        config,
        &ciborium::from_reader::<ProtocolConfig<M>, _>(bytes.as_slice()).unwrap()
    );
    restored
}

fn prove_case<M>(expected: &Value)
where
    M: Embedding + Default,
    M::Target: Default + Zeroize + Codec,
    Standard: Distribution<M::Source> + Distribution<M::Target>,
{
    let name = expected["name"].as_str().unwrap();
    let (prefix, kind) = name.split_once('_').unwrap();
    let mode = match prefix {
        "standard" => false,
        "zk" => true,
        _ => panic!("unknown mode"),
    };
    let regime = match kind {
        "unique" => DecodingRegime::Unique,
        "capacity" => DecodingRegime::Capacity,
        _ => DecodingRegime::Johnson,
    };
    let schedule = match kind {
        "adaptive" => RateSchedule::Adaptive(0.5.into()),
        "capped" => RateSchedule::Capped(2),
        _ => RateSchedule::Stepping,
    };
    let basecase = kind == "basecase";
    let pair = kind.ends_with("pair");
    let proof_case = matches!(
        kind,
        "basecase" | "rounds" | "mixed" | "pair" | "mixed_pair"
    );
    let config = ProtocolConfig::<M>::derive(Params {
        bits: 128,
        zk: mode,
        regime,
        pow: if proof_case { 0 } else { 16 },
        hash: crate::hash::BLAKE3,
        size: if basecase { 8 } else { 4096 },
        rate: 1,
        folds: [2, 2],
        schedule,
    })
    .unwrap();
    let from_json = check(expected, &config);
    if !proof_case {
        return;
    }
    let n = config.params().size;
    let count = if pair { 2 } else { 1 };
    let mut rng = StdRng::seed_from_u64(0x7768_6972_2830_0001);
    let witness: Vec<M::Source> = random_vector(&mut rng, n * count);
    // Full-field coefficients exercise the genuine extension-field path.
    let forms: Vec<Covector<M::Target>> = (0..2)
        .map(|_| Covector::new(random_vector(&mut rng, n * count)))
        .collect();
    let values: Vec<_> = forms
        .iter()
        .map(|form| form.evaluate(&M::default(), &witness))
        .collect();
    let refs: Vec<&dyn LinearForm<M::Target>> = forms.iter().map(|form| form as _).collect();
    let instance = U64(0x7768_6972_2830_0001);
    let ds = DomainSeparator::protocol(&config)
        .session(&name)
        .instance(&instance);
    let mut prover = ProverState::new_std_seeded(&ds, 0x7061_7261_6d73_2830);
    // Bind every public claim before any commitment or opening challenge.
    for form in &forms {
        prover.prover_messages(&form.vector);
    }
    prover.prover_messages(&values);
    let left = config.commit(&mut prover, Buffer::from(&witness[..n]));
    let claim = if pair {
        let right = config.commit(&mut prover, Buffer::from(&witness[n..]));
        config
            .prove_pair(&mut prover, [left, right], &refs, &values)
            .unwrap()
    } else {
        config.prove(&mut prover, left, &refs, &values)
    };
    prover.prover_message(&M::Target::from(987));
    let proof = prover.proof();

    // Verify with the deserialized configuration to cover persisted configurations too.
    let mut verifier = VerifierState::new_std(&ds, &proof);
    for form in &forms {
        let received: Vec<M::Target> = verifier.prover_messages_vec(form.vector.len()).unwrap();
        assert_eq!(received, form.vector);
    }
    let received: Vec<M::Target> = verifier.prover_messages_vec(values.len()).unwrap();
    assert_eq!(received, values);
    let left = from_json.receive_commitment(&mut verifier).unwrap();
    let final_claim = if pair {
        let right = from_json.receive_commitment(&mut verifier).unwrap();
        from_json
            .verify_pair(&mut verifier, [left, right], &refs, &values)
            .unwrap()
    } else {
        from_json
            .verify(&mut verifier, left, &refs, &values)
            .unwrap()
    };
    final_claim.verify(&refs).unwrap();
    assert_eq!(final_claim.evaluation_point, claim.evaluation_point);
    assert_eq!(final_claim.rlc_coefficients, claim.rlc_coefficients);
    let sentinel: M::Target = verifier.prover_message().unwrap();
    assert_eq!(sentinel, M::Target::from(987));
    verifier.check_eof().unwrap();
}

#[test]
fn all_parameter_baselines_and_seeded_proofs() {
    let fixtures: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/parameter_regressions.json"
    ))
    .unwrap();
    assert_eq!(
        fixtures["baseline_revision"],
        "b9a4e7823cadce9d2d5727380a1cab3d5304c54f"
    );
    let cases = fixtures["cases"].as_array().unwrap();
    let choices = fixtures["configurations"].as_array().unwrap();
    assert_eq!((cases.len(), choices.len()), (10, 6));
    for expected in cases.iter().chain(choices) {
        if expected["name"].as_str().unwrap().contains("mixed") {
            prove_case::<Basefield<Field64_3>>(expected);
        } else {
            prove_case::<Identity<Field256>>(expected);
        }
    }
}
