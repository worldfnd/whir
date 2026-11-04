//! Shared parameter-test fixtures.

use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        fields::Field64,
    },
    bits::Bits,
    hash::BLAKE3,
    protocols::{
        irs_commit::{Config as IrsConfig, IrsMode, IrsParams},
        params::spec::DecodingRegime,
    },
};

pub type TestField = Field64;
pub type TestEmbedding = Identity<TestField>;

/// Compare analytic values within floating-point rounding error.
pub fn assert_close(got: f64, expected: f64) {
    assert!(
        (got - expected).abs() < 1e-9,
        "actual {got}, expected {expected}"
    );
}

pub fn johnson_code<M: Embedding + Default>(
    vector_size: usize,
    depth: usize,
    rate: f64,
) -> IrsConfig<M> {
    johnson_code_with_vectors(vector_size, depth, rate, 1)
}

pub fn johnson_code_with_vectors<M: Embedding + Default>(
    vector_size: usize,
    depth: usize,
    rate: f64,
    num_vectors: usize,
) -> IrsConfig<M> {
    IrsConfig::new(IrsParams {
        security_target: 80.0,
        decoding_regime: DecodingRegime::Johnson,
        hash_id: BLAKE3,
        num_vectors,
        vector_size,
        interleaving_depth: depth,
        rate,
        mode: IrsMode::Standard,
    })
}

pub fn assert_probability_bits(actual: Bits, probability: f64) {
    let expected = -probability.log2();
    assert!(
        (f64::from(actual) - expected).abs() < 1e-10,
        "actual {actual}, expected {expected}"
    );
}
