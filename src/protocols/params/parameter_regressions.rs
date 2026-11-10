//! Parameter-selection baselines captured from PR #283 before the refactor.
//! The projection excludes config serialization, type names, and proof bytes.
//! Seeded protocol roundtrips also exercise deserialized configurations.

use ark_ff::Field;
use ark_std::rand::{distributions::Standard, prelude::Distribution, rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use zeroize::Zeroize;

use super::{
    BasecasePlan, DecodingRegime, FoldingFactor, KneeWeight, Mode, PowBudget, ProtocolConfig,
    RateSchedule, RoundConfig, SecuritySpec, TuningSpec,
};
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

const BASELINE_REVISION: &str = "b9a4e7823cadce9d2d5727380a1cab3d5304c54f";
const PRIVATE_SEED: u64 = 0x7061_7261_6d73_2830;
const WITNESS_SEED: u64 = 0x7768_6972_2830_0001;

#[derive(Debug, Deserialize)]
struct Fixtures {
    baseline_revision: String,
    cases: Vec<Fingerprint>,
    configurations: Vec<Fingerprint>,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Fingerprint {
    name: String,
    rounds: usize,
    parameters_blake3: String,
}

// This schema records protocol decisions through primitive values and accessors.
// It deliberately excludes computed security reports and private storage layout.
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
    let switch = round.code_switch();
    json!({
        "index": round.round_index(),
        "zk": round.masks().is_some(),
        "source": code_parameters(switch.source()),
        "target": code_parameters(switch.target()),
        "out_domain_samples": switch.out_domain_samples(),
        "message_mask_length": switch.message_mask_length(),
        "switch_zk": switch.is_zk(),
        "pow": pow_parameters(switch.pow()),
        "sumcheck": sumcheck_parameters(round.sumcheck()),
        "sumcheck_mask_coefficients": round.masks().map(|masks| masks.sumcheck_coefficients),
    })
}

fn parameters<M: Embedding>(config: &ProtocolConfig<M>) -> Value {
    let mut rounds = Vec::new();
    if let Some(first) = config.first_round() {
        rounds.push(round_parameters(&first));
    }
    rounds.extend(config.tail_rounds().map(|round| round_parameters(&round)));
    let (sumcheck, pow) = match config.basecase() {
        BasecasePlan::Standard { config, .. } => (
            sumcheck_parameters(config.sumcheck()),
            pow_parameters(config.pow()),
        ),
        BasecasePlan::ZeroKnowledge { sumcheck, .. } => (
            sumcheck_parameters(&sumcheck),
            pow_parameters(proof_of_work::Config::none()),
        ),
    };
    let masks: Vec<_> = (0..=config.num_rounds())
        .map(|index| {
            config
                .witness_mask(index)
                .map(|mask| [mask.num_polynomials, mask.coefficients_per_polynomial])
        })
        .collect();
    let joint = config.joint_masks().map(|joint| json!({
        "code": code_parameters(joint.commit()),
        "sumcheck_ranges": joint.sumcheck_masks().iter().map(|range| [range.start, range.end]).collect::<Vec<_>>(),
        "witness_ranges": joint.witness_masks().iter().map(|range| [range.start, range.end]).collect::<Vec<_>>(),
        "blinder": [joint.blinder().start, joint.blinder().end],
        "basefield_blinder": [joint.basefield_blinder().start, joint.basefield_blinder().end],
    }));
    json!({
        "padded_vector_size": config.padded_vector_size(),
        "rounds": rounds,
        "basecase": {
            "zk": config.basecase().is_zk(),
            "code": code_parameters(config.basecase().commit()),
            "sumcheck": sumcheck,
            "pow": pow,
        },
        "witness_masks": masks,
        "joint_masks": joint,
    })
}

fn fingerprint<M: Embedding>(name: &str, config: &ProtocolConfig<M>) -> Fingerprint {
    let projection = serde_json::to_vec(&parameters(config)).unwrap();
    Fingerprint {
        name: name.to_owned(),
        rounds: config.num_rounds(),
        parameters_blake3: blake3::hash(&projection).to_hex().to_string(),
    }
}

fn serialization_roundtrip<M: Embedding + Default>(
    config: &ProtocolConfig<M>,
) -> ProtocolConfig<M> {
    let json = serde_json::to_vec(config).unwrap();
    let mut cbor = Vec::new();
    ciborium::into_writer(config, &mut cbor).unwrap();
    let from_json: ProtocolConfig<M> = serde_json::from_slice(&json).unwrap();
    let from_cbor: ProtocolConfig<M> = ciborium::from_reader(cbor.as_slice()).unwrap();
    assert_eq!(config, &from_json);
    assert_eq!(config, &from_cbor);
    from_json.validate().unwrap();
    from_cbor.validate().unwrap();
    from_json
}

fn capture<M>(name: &str, mode: Mode, basecase: bool, pair: bool) -> Fingerprint
where
    M: Embedding + Default,
    M::Target: Default + Zeroize + Codec,
    Standard: Distribution<M::Source> + Distribution<M::Target>,
{
    let config = ProtocolConfig::<M>::derive(
        SecuritySpec::new(128)
            .with_mode(mode)
            .with_pow_budget(PowBudget::Forbidden),
        TuningSpec {
            vector_size: if basecase { 8 } else { 4096 },
            starting_log_inv_rate: 1,
            folding_factor: FoldingFactor::Constant(2),
            rate_schedule: RateSchedule::Stepping,
        },
    )
    .unwrap();
    assert_eq!(config.has_rounds(), !basecase, "{name}: expected layout");
    if pair {
        config.validate_pair().unwrap();
    }

    let from_json = serialization_roundtrip(&config);

    let n = config.tuning().vector_size;
    let count = if pair { 2 } else { 1 };
    let mut rng = StdRng::seed_from_u64(WITNESS_SEED);
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
    let instance = U64(WITNESS_SEED);
    let ds = DomainSeparator::protocol(&config)
        .session(&name)
        .instance(&instance);
    let mut prover = ProverState::new_std_seeded(&ds, PRIVATE_SEED);
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

    fingerprint(name, &config)
}

fn check(actual: &Fingerprint) {
    let fixtures: Fixtures = serde_json::from_str(include_str!(
        "../../../tests/fixtures/parameter_regressions.json"
    ))
    .unwrap();
    assert_eq!(fixtures.baseline_revision, BASELINE_REVISION);
    let expected = fixtures
        .cases
        .iter()
        .find(|fixture| fixture.name == actual.name)
        .expect("baseline fixture exists");
    assert_eq!(actual, expected, "PR #283 parameter selection changed");
}

macro_rules! fixture {
    ($name:ident, $embedding:ty, $mode:ident, $basecase:expr, $pair:expr) => {
        #[test]
        fn $name() {
            check(&capture::<$embedding>(
                stringify!($name),
                Mode::$mode,
                $basecase,
                $pair,
            ));
        }
    };
}

fixture!(standard_basecase, Identity<Field256>, Standard, true, false);
fixture!(zk_basecase, Identity<Field256>, ZeroKnowledge, true, false);
fixture!(standard_rounds, Identity<Field256>, Standard, false, false);
fixture!(zk_rounds, Identity<Field256>, ZeroKnowledge, false, false);
fixture!(standard_mixed, Basefield<Field64_3>, Standard, false, false);
fixture!(zk_mixed, Basefield<Field64_3>, ZeroKnowledge, false, false);
fixture!(standard_pair, Identity<Field256>, Standard, false, true);
fixture!(zk_pair, Identity<Field256>, ZeroKnowledge, false, true);
fixture!(
    standard_mixed_pair,
    Basefield<Field64_3>,
    Standard,
    false,
    true
);
fixture!(
    zk_mixed_pair,
    Basefield<Field64_3>,
    ZeroKnowledge,
    false,
    true
);

fn parameter_choices() -> Vec<Fingerprint> {
    let choices = [
        (
            "standard_adaptive",
            Mode::Standard,
            DecodingRegime::Johnson,
            RateSchedule::Adaptive {
                knee_weight: KneeWeight::DEFAULT,
            },
        ),
        (
            "zk_adaptive",
            Mode::ZeroKnowledge,
            DecodingRegime::Johnson,
            RateSchedule::Adaptive {
                knee_weight: KneeWeight::DEFAULT,
            },
        ),
        (
            "standard_capped",
            Mode::Standard,
            DecodingRegime::Johnson,
            RateSchedule::Capped {
                max_log_inv_rate: 2,
            },
        ),
        (
            "zk_capped",
            Mode::ZeroKnowledge,
            DecodingRegime::Johnson,
            RateSchedule::Capped {
                max_log_inv_rate: 2,
            },
        ),
        (
            "standard_unique",
            Mode::Standard,
            DecodingRegime::Unique,
            RateSchedule::Stepping,
        ),
        (
            "standard_capacity",
            Mode::Standard,
            DecodingRegime::Capacity,
            RateSchedule::Stepping,
        ),
    ];
    choices
        .into_iter()
        .map(|(name, mode, regime, rate_schedule)| {
            let config = ProtocolConfig::<Identity<Field256>>::derive(
                SecuritySpec::new(128)
                    .with_mode(mode)
                    .with_decoding_regime(regime)
                    .with_pow_budget(PowBudget::per_slot(16)),
                TuningSpec {
                    vector_size: 4096,
                    starting_log_inv_rate: 1,
                    folding_factor: FoldingFactor::Constant(2),
                    rate_schedule,
                },
            )
            .unwrap();
            config.validate().unwrap();
            serialization_roundtrip(&config);
            fingerprint(name, &config)
        })
        .collect()
}

#[test]
fn alternate_schedules_and_decoding_regimes() {
    let fixtures: Fixtures = serde_json::from_str(include_str!(
        "../../../tests/fixtures/parameter_regressions.json"
    ))
    .unwrap();
    assert_eq!(fixtures.baseline_revision, BASELINE_REVISION);
    assert_eq!(parameter_choices(), fixtures.configurations);
}
