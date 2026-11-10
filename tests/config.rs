//! Configuration loading checks execution assumptions before returning a config.

use serde_json::{json, Value};
use whir::{
    algebra::{embedding::Identity, fields::Field64_2},
    protocols::params::{
        DeriveError, FoldingFactor, Mode, PowBudget, ProtocolConfig, RateSchedule, SecuritySpec,
        TuningSpec,
    },
};

type Config = ProtocolConfig<Identity<Field64_2>>;

fn config(mode: Mode, vector_size: usize) -> Config {
    Config::derive(
        SecuritySpec::new(40)
            .with_mode(mode)
            .with_pow_budget(PowBudget::Forbidden),
        TuningSpec {
            vector_size,
            starting_log_inv_rate: 2,
            folding_factor: FoldingFactor::Constant(2),
            rate_schedule: RateSchedule::Stepping,
        },
    )
    .unwrap()
}

fn rejects(config: &Config, pointer: &str, value: Value) {
    let mut input = serde_json::to_value(config).unwrap();
    *input.pointer_mut(pointer).expect("existing config field") = value;
    let result = std::panic::catch_unwind(|| serde_json::from_value::<Config>(input));
    assert!(result.is_ok(), "deserialization panicked for {pointer}");
    assert!(result.unwrap().is_err(), "accepted invalid {pointer}");
}

#[test]
fn valid_json_and_cbor_configs_keep_their_parameters() {
    for mode in [Mode::Standard, Mode::ZeroKnowledge] {
        for size in [8, 4096] {
            let original = config(mode, size);
            let decoded: Config =
                serde_json::from_slice(&serde_json::to_vec(&original).unwrap()).unwrap();
            assert_eq!(decoded, original);
            let mut bytes = Vec::new();
            ciborium::into_writer(&original, &mut bytes).unwrap();
            let decoded: Config = ciborium::from_reader(bytes.as_slice()).unwrap();
            assert_eq!(decoded, original);
            assert!(decoded.validate().is_ok());
        }
    }
}

#[test]
fn rejects_standard_basecase_sumcheck_dimension_mismatches() {
    let config = config(Mode::Standard, 8);
    rejects(
        &config,
        "/basecase/Standard/config/sumcheck/initial_size",
        json!(4),
    );
    rejects(
        &config,
        "/basecase/Standard/config/sumcheck/num_rounds",
        json!(1),
    );
    rejects(
        &config,
        "/basecase/Standard/config/sumcheck/mode",
        json!({"ZeroKnowledge": {"mask_length": 3}}),
    );
}

#[test]
fn rejects_round_sumcheck_mode_mismatches() {
    let zk = config(Mode::ZeroKnowledge, 4096);
    rejects(&zk, "/first_round/sumcheck/mode", json!("Standard"));
    rejects(
        &zk,
        "/first_round/sumcheck/mode/ZeroKnowledge/mask_length",
        json!(0),
    );
    let standard = config(Mode::Standard, 256);
    rejects(
        &standard,
        "/first_round/sumcheck/mode",
        json!({"ZeroKnowledge": {"mask_length": 3}}),
    );
}

#[test]
fn rejects_nested_irs_and_matrix_dimension_mismatches() {
    let config = config(Mode::Standard, 8);
    for (suffix, value) in [
        ("matrix_commit/num_cols", json!(2)),
        ("matrix_commit/merkle_tree/num_leaves", json!(3)),
        ("matrix_commit/merkle_tree/layers", json!([])),
        ("interleaving_depth", json!(0)),
        ("codeword_length", json!(0)),
        ("in_domain_samples", json!(0)),
        ("vector_size", json!(usize::MAX)),
    ] {
        rejects(
            &config,
            &format!("/basecase/Standard/config/commit/{suffix}"),
            value,
        );
    }
}

#[test]
fn rejects_invalid_decoding_and_grinding_inputs_without_panics() {
    let config = config(Mode::Standard, 256);
    rejects(
        &config,
        "/first_round/code_switch/source/regime/Johnson/slack",
        json!(0.0),
    );
    rejects(
        &config,
        "/first_round/sumcheck/round_pow/threshold",
        json!(0),
    );
    rejects(
        &config,
        "/first_round/code_switch/out_domain_samples",
        json!(usize::MAX),
    );
    rejects(
        &config,
        "/first_round/code_switch/source/matrix_commit/leaf_hash_id",
        json!(vec![0u8; 32]),
    );
}

#[test]
fn rejects_request_dimension_overflow_without_panicking() {
    let tuning = TuningSpec {
        vector_size: 1usize << (usize::BITS - 1),
        starting_log_inv_rate: 1,
        folding_factor: FoldingFactor::ConstantFromSecondRound {
            initial: (usize::BITS - 2) as usize,
            rest: 1,
        },
        rate_schedule: RateSchedule::Capped {
            max_log_inv_rate: 1,
        },
    };
    let config = config(Mode::Standard, 8);
    assert!(matches!(
        Config::derive(config.security().clone(), tuning.clone()),
        Err(DeriveError::SizeOverflow)
    ));
    rejects(&config, "/tuning", serde_json::to_value(tuning).unwrap());
}
