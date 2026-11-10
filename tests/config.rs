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

fn rejects(input: &Value) {
    let json = std::panic::catch_unwind(|| serde_json::from_value::<Config>(input.clone()));
    assert!(json.is_ok(), "JSON deserialization panicked");
    assert!(json.unwrap().is_err(), "JSON accepted invalid request");
    let mut bytes = Vec::new();
    ciborium::into_writer(input, &mut bytes).unwrap();
    let cbor = std::panic::catch_unwind(|| ciborium::from_reader::<Config, _>(bytes.as_slice()));
    assert!(cbor.is_ok(), "CBOR deserialization panicked");
    assert!(cbor.unwrap().is_err(), "CBOR accepted invalid request");
}

#[test]
fn valid_json_and_cbor_configs_keep_their_parameters() {
    for mode in [Mode::Standard, Mode::ZeroKnowledge] {
        for size in [8, 4096] {
            let original = config(mode, size);
            let encoded = serde_json::to_value(&original).unwrap();
            assert_eq!(
                encoded,
                json!({"embedding": whir::type_info::Typed::new(Identity::<Field64_2>::new()),
                    "security": original.security(), "tuning": original.tuning()})
            );
            let decoded: Config = serde_json::from_value(encoded).unwrap();
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
fn round_views_share_adjacent_commitments() {
    for mode in [Mode::Standard, Mode::ZeroKnowledge] {
        let config = config(mode, 4096);
        let first = config.first_round().unwrap();
        let mut previous = first.code_switch().target();
        for round in config.tail_rounds() {
            assert!(std::ptr::eq(previous, round.code_switch().source()));
            previous = round.code_switch().target();
        }
        assert!(std::ptr::eq(previous, config.final_commit()));
        assert!(std::ptr::eq(
            config.final_commit(),
            config.basecase().commit()
        ));
    }
}

#[test]
fn rejects_derived_fields_in_serialized_requests() {
    let config = config(Mode::Standard, 8);
    for field in [
        "padded_vector_size",
        "first_round",
        "tail_rounds",
        "basecase",
        "joint_masks",
        "first",
        "targets",
    ] {
        let mut input = serde_json::to_value(&config).unwrap();
        input[field] = Value::Null;
        rejects(&input);
    }
}

#[test]
fn rejects_invalid_request_fields_without_panicking() {
    let config = config(Mode::Standard, 8);
    for (pointer, value) in [
        ("/tuning/vector_size", json!(0)),
        ("/tuning/vector_size", json!(3)),
        ("/tuning/vector_size", json!(usize::MAX)),
        ("/tuning/starting_log_inv_rate", json!(0)),
        ("/tuning/starting_log_inv_rate", json!(u32::MAX)),
        ("/tuning/folding_factor", json!({"Constant": 0})),
        (
            "/tuning/folding_factor",
            json!({"ConstantFromSecondRound": {"initial": 0, "rest": 2}}),
        ),
        (
            "/tuning/folding_factor",
            json!({"ConstantFromSecondRound": {"initial": 2, "rest": 0}}),
        ),
        (
            "/tuning/rate_schedule",
            json!({"Capped": {"max_log_inv_rate": 0}}),
        ),
        ("/security/target_security_bits", json!(u32::MAX)),
        ("/security/pow_budget", json!({"PerSlot": {"bits": 0}})),
        ("/security/hash_id", json!([])),
    ] {
        let mut input = serde_json::to_value(&config).unwrap();
        *input.pointer_mut(pointer).unwrap() = value;
        rejects(&input);
    }
    for regime in ["Unique", "Capacity"] {
        let mut input = serde_json::to_value(&config).unwrap();
        input["security"]["mode"] = json!("ZeroKnowledge");
        input["security"]["decoding_regime"] = json!(regime);
        rejects(&input);
    }
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
    let mut input = serde_json::to_value(&config).unwrap();
    input["tuning"] = serde_json::to_value(tuning).unwrap();
    rejects(&input);
}

#[test]
fn serialized_requests_bind_the_field_and_embedding() {
    use whir::algebra::{
        embedding::{Basefield, Frobenius},
        fields::Field64_3,
    };
    let input = serde_json::to_value(config(Mode::Standard, 256)).unwrap();
    assert!(serde_json::from_value::<ProtocolConfig<Identity<Field64_3>>>(input.clone()).is_err());
    assert!(serde_json::from_value::<ProtocolConfig<Basefield<Field64_2>>>(input).is_err());
    let config = config(Mode::Standard, 256);
    let frobenius = ProtocolConfig::<Frobenius<Field64_2>>::derive(
        config.security().clone(),
        config.tuning().clone(),
    )
    .unwrap();
    let mut input = serde_json::to_value(frobenius).unwrap();
    input["embedding"]["value"]["power"] = json!(1);
    assert!(serde_json::from_value::<ProtocolConfig<Frobenius<Field64_2>>>(input).is_err());
}
