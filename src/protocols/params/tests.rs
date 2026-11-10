//! Independent equation anchors and construction properties complement the frozen baselines.
use DecodingRegime::{Capacity, Johnson, Unique};

use super::{security::*, *};
use crate::{
    algebra::{
        embedding::{Basefield, Embedding, Identity},
        fields::{Field64, Field64_2},
    },
    bits::Bits,
    hash::BLAKE3,
    protocols::irs_commit::{Config as IrsConfig, IrsMode, IrsParams},
};

pub(super) fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "actual {actual}, expected {expected}"
    );
}
#[test]
fn adaptive_weight_preserves_normalization() {
    for (input, expected) in [
        (-1.0, 0.0),
        (0.0, 0.0),
        (0.3, 0.3),
        (2.0, 1.0),
        (f64::NAN, 0.5),
    ] {
        assert_close(KneeWeight::from(input).0.into_inner(), expected);
    }
}

#[derive(ark_ff::MontConfig)]
#[modulus = "17"]
#[generator = "3"]
struct UnregisteredFieldConfig;

#[test]
fn unsupported_zook_regime_fails_before_ntt_lookup() {
    type F = ark_ff::Fp64<ark_ff::MontBackend<UnregisteredFieldConfig, 1>>;
    for regime in [Unique, Capacity] {
        let result = ProtocolConfig::<Identity<F>>::derive(Params {
            zk: true,
            regime,
            bits: 40,
            pow: 0,
            hash: BLAKE3,
            size: 8,
            rate: 2,
            folds: [2, 2],
            schedule: RateSchedule::Stepping,
        });
        assert_eq!(
            result.unwrap_err(),
            DeriveError("Zook requires Johnson decoding")
        );
    }
}
pub(super) fn code<M: Embedding + Default>(
    size: usize,
    depth: usize,
    rate: f64,
    vectors: usize,
) -> IrsConfig<M> {
    IrsConfig::new(IrsParams {
        security_target: 80.0,
        decoding_regime: Johnson,
        hash_id: BLAKE3,
        num_vectors: vectors,
        vector_size: size,
        interleaving_depth: depth,
        rate,
        mode: IrsMode::Standard,
    })
}
#[test]
fn independent_decoding_and_probability_anchors() {
    // Unique, BCSS25 Theorem 1.5, and STIR Conjecture 5.6 at k=16, rho=1/4, |F|=2^64.
    for (policy, list, distance, mca) in [
        (Unique, 1.0, 0.625, -58.0),
        (Johnson, 40.0, 0.525, -38.623_375_386_827_36),
        (Capacity, 5120.0, 0.2625, -49.678_071_905_112_64),
    ] {
        let actual = decoding(policy, 0.25, 16, 64.0);
        assert_close(actual[0], list);
        assert_close(actual[1], f64::log2(distance));
        assert_close(actual[2], mca);
        assert_eq!(
            policy.to_string().parse::<DecodingRegime>().unwrap(),
            policy
        );
    }
    for (a, b, sum) in [
        (f64::NEG_INFINITY, -80.0, -80.0),
        (-2000.0, -2000.0, -1999.0),
    ] {
        assert_close(log2_sum(a, b), sum);
    }
    assert_eq!(
        log2_sum(f64::NEG_INFINITY, f64::NEG_INFINITY).to_bits(),
        f64::NEG_INFINITY.to_bits()
    );
    for n in [1, 128] {
        assert_close(
            out_of_domain_bits(8.0, n).unwrap(),
            ((256 - n) as f64).log2(),
        );
    }
    assert!(out_of_domain_bits(8.0, 256).is_err());
}

#[test]
fn zook_equations_and_hypotheses() {
    let source = code::<Basefield<Field64_2>>(256, 4, 0.25, 1);
    let target = code::<Identity<Field64_2>>(64, 2, 0.125, 1);
    let witness = code::<Identity<Field64_2>>(64, 1, 0.25, 1);
    let masks = code::<Identity<Field64_2>>(256, 1, 0.25, 2);
    // Independent §6.4 probabilities with p=18446744069414584321, Q=p², and 87 queries.
    // Agreement: L*((2/3)*L^4*rho+1)*n/Q; DEEP: L²*m/(2*(Q-n)).
    // Binding: L*(L+1)*m/(2*(Q-1)); final queries sum two equal errors.
    let (queries, aggregation) = zook_query_error_bits(&source, &target).unwrap();
    let (sumcheck, _) = round_bits(&source, Some(&target), 1, true).unwrap();
    let (binding, combination, final_queries) = final_bits(&witness, &masks).unwrap();
    for (actual, expected) in [
        (sumcheck, 95.975_318_644_3),
        (queries, 80.876_128_471_521_71),
        (aggregation, 115.235_128_408_592_1),
        (binding, 110.320_519_899_822_74),
        (combination, 93.975_318_644_3),
        (final_queries, 79.876_128_473_448_38),
    ] {
        assert_close(f64::from(actual), expected);
    }
    for vectors in 1..=3 {
        for depth in [1, 2] {
            let code = code::<Identity<Field64_2>>(128, depth, 0.25, vectors);
            assert_eq!(
                validate_zook_code(&code, Some(2)).is_ok(),
                vectors == 2 && depth == 1
            );
        }
    }
    assert!(validate_zook_code(&code::<Identity<Field64>>(2, 1, 0.5, 1), None).is_err());
    assert!(zook_query_error_bits(&source, &code::<Identity<Field64_2>>(32, 1, 0.25, 1)).is_err());
}

#[test]
fn standard_round_equations_and_layout() {
    // At rho=1/4 and 20 analytic bits, query counts are 30,22,11 respectively.
    for (regime, sumcheck, queries) in [
        (Unique, 56.0, 20.342_157_153_379_13),
        (Johnson, 36.623_375_386_827_36, 20.451_434_786_389_246),
        (Capacity, 47.678_071_905_112_64, 21.225_717_393_194_62),
    ] {
        for folds in 1..=3 {
            let config = ProtocolConfig::<Identity<Field64>>::derive(Params {
                zk: false,
                regime,
                bits: 80,
                pow: 60,
                hash: BLAKE3,
                size: 256,
                rate: 2,
                folds: [folds, 2],
                schedule: RateSchedule::Stepping,
            })
            .unwrap();
            let first = config.first_round().unwrap();
            let switch = &first.code_switch;
            let source = switch.source();
            let (bound, actual) = round_bits(
                source,
                Some(switch.target()),
                switch.out_domain_samples(),
                false,
            )
            .unwrap();
            assert_close(f64::from(bound), sumcheck + folds as f64 - 2.0);
            assert_close(f64::from(actual.unwrap()), queries);
            assert!(f64::from(bound + first.sumcheck.round_pow().difficulty()) >= 80.0 - 1e-9);
            assert!(queries + f64::from(first.code_switch.pow().difficulty()) >= 80.0 - 1e-9);
            let mut rate = 2;
            for round in config.first_round().into_iter().chain(config.tail_rounds()) {
                let source = round.code_switch.source();
                assert_eq!(
                    round.sumcheck.num_rounds(),
                    if round.index == 0 { folds } else { 2 }
                );
                assert_eq!(
                    round.code_switch.target().vector_size(),
                    source.message_length()
                );
                assert_close(source.rate(), 2_f64.powf(-f64::from(rate)));
                rate += round.sumcheck.num_rounds() as u32 - 1;
            }
        }
    }
}

#[test]
fn grinding_respects_fractional_targets_and_mode() {
    let spec = Params {
        zk: false,
        regime: Johnson,
        bits: 80,
        pow: 60,
        hash: BLAKE3,
        size: 256,
        rate: 2,
        folds: [2, 2],
        schedule: RateSchedule::Stepping,
    };
    for analytic in [20.3422, 35.623_375_386_491_45] {
        let pow = grind_to_at(&spec, Bits::new(analytic), "test").unwrap();
        assert!(analytic + f64::from(pow.difficulty()) >= 80.0 - 1e-9);
        assert!(f64::from(pow.difficulty()) <= 60.0);
    }
    for (spec, bound) in [
        (Params { pow: 59, ..spec }, 20.3422),
        (Params { zk: true, ..spec }, 79.9999),
    ] {
        assert!(grind_to_at(&spec, Bits::new(bound), "test").is_err());
    }
}
