//! Serialized requests cannot inject derived state or change the field and embedding.
use serde_json::{json, Value};
use whir::{
    algebra::{
        embedding::{Basefield, Identity},
        fields::{Field64_2, Field64_3},
    },
    protocols::params::{Mode, ProtocolConfig, RateSchedule, SecuritySpec, TuningSpec},
};
type Config = ProtocolConfig<Identity<Field64_2>>;

fn rejects(input: &Value) {
    let result = std::panic::catch_unwind(|| serde_json::from_value::<Config>(input.clone()));
    assert!(result.unwrap().is_err(), "accepted JSON request: {input}");
    let mut bytes = Vec::new();
    ciborium::into_writer(input, &mut bytes).unwrap();
    let result = std::panic::catch_unwind(|| ciborium::from_reader::<Config, _>(bytes.as_slice()));
    assert!(result.unwrap().is_err(), "accepted CBOR request: {input}");
}

#[test]
fn rejects_malformed_requests_and_embedding_changes() {
    let config = Config::derive(
        SecuritySpec::new(40).with_pow_budget(0),
        TuningSpec {
            vector_size: 8,
            starting_log_inv_rate: 2,
            folding_factor: [2, 2],
            rate_schedule: RateSchedule::Stepping,
        },
    )
    .unwrap();
    let original = serde_json::to_value(&config).unwrap();
    assert_eq!(original.as_object().unwrap().len(), 3);
    assert!(
        serde_json::from_value::<ProtocolConfig<Identity<Field64_3>>>(original.clone()).is_err()
    );
    assert!(
        serde_json::from_value::<ProtocolConfig<Basefield<Field64_2>>>(original.clone()).is_err()
    );
    for field in [
        "padded_vector_size",
        "first_round",
        "tail_rounds",
        "basecase",
        "joint_masks",
        "first",
        "targets",
    ] {
        let mut input = original.clone();
        input[field] = Value::Null;
        rejects(&input);
    }
    for (pointer, invalid) in [
        ("/embedding", Value::Null),
        ("/tuning/vector_size", json!(0)),
        ("/tuning/vector_size", json!(3)),
        ("/tuning/vector_size", json!(usize::MAX)),
        ("/tuning/starting_log_inv_rate", json!(0)),
        ("/tuning/starting_log_inv_rate", json!(u32::MAX)),
        ("/tuning/folding_factor", json!([0, 2])),
        ("/tuning/folding_factor", json!([2, 0])),
        (
            "/tuning/rate_schedule",
            json!({"Capped": {"max_log_inv_rate": 0}}),
        ),
        ("/security/target_security_bits", json!(u32::MAX)),
        ("/security/hash_id", json!([])),
    ] {
        let mut input = original.clone();
        *input.pointer_mut(pointer).unwrap() = invalid;
        rejects(&input);
    }
    for regime in ["Unique", "Capacity"] {
        let mut input = original.clone();
        input["security"]["mode"] = json!(Mode::ZeroKnowledge);
        input["security"]["decoding_regime"] = json!(regime);
        rejects(&input);
    }
    let mut input = original;
    input["tuning"] = json!({"vector_size": 1usize << (usize::BITS - 1),
        "starting_log_inv_rate": 1, "folding_factor": [usize::BITS - 2, 1],
        "rate_schedule": {"Capped": {"max_log_inv_rate": 1}}});
    rejects(&input);
}
