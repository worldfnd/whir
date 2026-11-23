//! Helpers shared by unit tests across modules. Type-specific fixtures stay with their type.

use std::{fmt::Debug, sync::Once};

use ark_ff::Field;
use serde::{Deserialize, Serialize};

/// Initialize the test logger once; with `tracing`, honors `RUST_LOG`.
pub fn init() {
    static INIT: Once = Once::new();

    #[cfg(not(feature = "tracing"))]
    INIT.call_once(|| {});

    #[cfg(feature = "tracing")]
    INIT.call_once(|| {
        use tracing_subscriber::{fmt, fmt::format::FmtSpan, EnvFilter};

        // Respect RUST_LOG if set, otherwise default for tests.
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug"));

        // Create a writer compatible with the testing harnesses
        fmt()
            .with_env_filter(filter)
            .with_span_events(FmtSpan::ENTER)
            .with_test_writer()
            .init();

        tracing::debug!("Initialized test logger");
    });
}

/// Assert that `value` roundtrips through JSON and CBOR unchanged.
#[track_caller]
pub fn test_serde<T: Debug + PartialEq + Serialize + for<'a> Deserialize<'a>>(value: &T) {
    // Test in human-readable format
    let json = serde_json::to_string_pretty(value).expect("json serialization failed");
    let deserialized = serde_json::from_str(&json).expect("json deserialization failed");
    assert_eq!(value, &deserialized, "json serde roundtrip failed");

    // Test in schemaless binary format

    let bytes = {
        let mut writer = Vec::new();
        ciborium::into_writer(value, &mut writer).expect("ciborium serialization failed");
        writer
    };
    let deserialized =
        ciborium::from_reader(bytes.as_slice()).expect("ciborium deserialization failed");
    assert_eq!(value, &deserialized, "ciborium serde roundtrip failed");
}

/// Zero-pad to the next power of two.
pub fn zero_pad<F: Field>(values: &[F]) -> Vec<F> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut vec = values.to_vec();
    vec.resize(vec.len().next_power_of_two(), F::ZERO);
    vec
}

/// Assert two floats agree to 1e-9.
pub fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "actual {actual}, expected {expected}"
    );
}
