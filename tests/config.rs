//! Serialized requests cannot inject derived state or change the field and embedding.
use serde_json::{json, Value};
use whir::{
    algebra::{
        embedding::{Basefield, Identity},
        fields::{Field64_2, Field64_3},
    },
    protocols::params::{Params, ProtocolConfig, RateSchedule},
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
    let config = Config::derive(Params {
        bits: 40,
        zk: false,
        regime: whir::protocols::params::DecodingRegime::Johnson,
        pow: 0,
        hash: whir::hash::BLAKE3,
        size: 8,
        rate: 2,
        folds: [2, 2],
        schedule: RateSchedule::Stepping,
    })
    .unwrap();
    let original = serde_json::to_value(&config).unwrap();
    assert_eq!(original.as_object().unwrap().len(), 2);
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
        ("/params/size", json!(0)),
        ("/params/size", json!(3)),
        ("/params/size", json!(usize::MAX)),
        ("/params/rate", json!(0)),
        ("/params/rate", json!(u32::MAX)),
        ("/params/folds", json!([0, 2])),
        ("/params/folds", json!([2, 0])),
        ("/params/schedule", json!({"Capped": 0})),
        ("/params/bits", json!(u32::MAX)),
        ("/params/hash", json!([])),
    ] {
        let mut input = original.clone();
        *input.pointer_mut(pointer).unwrap() = invalid;
        rejects(&input);
    }
    for regime in ["Unique", "Capacity"] {
        let mut input = original.clone();
        input["params"]["zk"] = json!(true);
        input["params"]["regime"] = json!(regime);
        rejects(&input);
    }
    let mut input = original;
    input["params"]["size"] = json!(1usize << (usize::BITS - 1));
    input["params"]["rate"] = json!(1);
    input["params"]["folds"] = json!([usize::BITS - 2, 1]);
    input["params"]["schedule"] = json!({"Capped": 1});
    rejects(&input);
}
