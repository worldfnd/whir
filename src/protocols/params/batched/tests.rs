use super::*;
use crate::{
    hash,
    protocols::params::{
        error::{DeriveError, Pow},
        spec::{DecodingRegime, FoldingFactor, Mode, PowBudget, RateSchedule, SecuritySpec},
        test_utils::TestEmbedding,
    },
};

fn test_spec(mode: Mode) -> SecuritySpec {
    SecuritySpec {
        mode,
        decoding_regime: DecodingRegime::Johnson,
        target_security_bits: 40,
        pow_budget: PowBudget::per_slot(20),
        hash_id: hash::BLAKE3,
    }
}

fn test_tuning(bundles: Vec<BundleSpec>) -> BatchedTuningSpec {
    BatchedTuningSpec {
        bundles,
        starting_log_inv_rate: 1,
        folding_factor: FoldingFactor::Constant(2),
        rate_schedule: RateSchedule::Stepping,
    }
}

fn derive(tuning: BatchedTuningSpec) -> Result<BatchedProtocolConfig<TestEmbedding>, DeriveError> {
    BatchedProtocolConfig::<TestEmbedding>::derive(test_spec(Mode::ZeroKnowledge), tuning)
}

// ── Case 1: one poly / committed-together (single bundle) ───────────────────

#[test]
fn derive_single_bundle_single_poly() {
    let cfg = derive(test_tuning(vec![BundleSpec {
        length: 1 << 8,
        num_polys: 1,
    }]))
    .unwrap();
    assert_eq!(cfg.bundle_configs().len(), 1);
    assert_eq!(cfg.schedule().joins.len(), 1);
    let join = &cfg.schedule().joins[0];
    assert_eq!(join.round_index, Some(0));
    assert_eq!(join.t, 1);
    assert_eq!(join.selector.selector_dim(), 0);
    assert!(cfg.bundle_configs()[0].pre_merge_rounds.is_empty());
}

#[test]
fn derive_committed_together_multi_poly_flattens_to_round_zero() {
    let cfg = derive(test_tuning(vec![BundleSpec {
        length: 1 << 8,
        num_polys: 4,
    }]))
    .unwrap();
    assert_eq!(cfg.bundle_configs().len(), 1);
    // Flattened: inner witness is num_polys × length.
    assert_eq!(cfg.inner().tuning().vector_size, 4 * (1 << 8));
    assert!(cfg.bundle_configs()[0].pre_merge_rounds.is_empty());
}

// ── Case 2: separate, same size → round-0 merge ─────────────────────────────

#[test]
fn derive_same_size_bundles_merge_at_round_zero() {
    let cfg = derive(test_tuning(vec![
        BundleSpec {
            length: 1 << 8,
            num_polys: 1,
        };
        3
    ]))
    .unwrap();
    assert_eq!(cfg.bundle_configs().len(), 3);
    assert_eq!(cfg.schedule().joins.len(), 1);
    let join = &cfg.schedule().joins[0];
    assert_eq!(join.round_index, Some(0));
    assert_eq!(join.t, 3);
    assert_eq!(join.bundle_indices, vec![0, 1, 2]);
    assert_eq!(join.selector.selector_dim(), 2);
    // No pre-merge; identical IRS configs (one shared commit shape).
    let first = &cfg.bundle_configs()[0].irs_config;
    for b in cfg.bundle_configs() {
        assert!(b.pre_merge_rounds.is_empty());
        assert_eq!(&b.irs_config, first);
    }
}

// ── Case 3: separate, different size → one-round converge to round 0 ─────────

#[test]
fn derive_smaller_bundle_at_merge_point_only_bigger_folds() {
    // Δ = 2 = folding factor → merge point sits at the smaller bundle's size:
    // the bigger (2^8) folds once by 2 to 2^6, the smaller (2^6) is already there.
    let cfg = derive(test_tuning(vec![
        BundleSpec {
            length: 1 << 8,
            num_polys: 1,
        },
        BundleSpec {
            length: 1 << 6,
            num_polys: 1,
        },
    ]))
    .unwrap();
    assert_eq!(cfg.bundle_configs()[0].pre_merge_rounds.len(), 1);
    assert!(cfg.bundle_configs()[1].pre_merge_rounds.is_empty());
    assert_eq!(cfg.bundle_configs()[0].join_round, Some(0));
    assert_eq!(cfg.bundle_configs()[1].join_round, Some(0));
    // Single round-0 merge.
    assert_eq!(cfg.schedule().joins.len(), 1);
    assert_eq!(cfg.schedule().joins[0].round_index, Some(0));
    assert_eq!(cfg.schedule().joins[0].t, 2);
    // Merge point = 2^6.
    assert_eq!(
        cfg.inner().rounds()[0]
            .code_switch()
            .config()
            .source()
            .vector_size(),
        1 << 6
    );
}

#[test]
fn derive_two_diff_sizes_both_run_one_pre_merge_round() {
    // Δ = 1 < folding factor 2 → merge below both: 2^10 folds by 2, 2^9 by 1,
    // both reach 2^8. This is the "run both for one round, then merge" case.
    let cfg = derive(test_tuning(vec![
        BundleSpec {
            length: 1 << 10,
            num_polys: 1,
        },
        BundleSpec {
            length: 1 << 9,
            num_polys: 1,
        },
    ]))
    .unwrap();
    assert_eq!(cfg.bundle_configs()[0].pre_merge_rounds.len(), 1);
    assert_eq!(cfg.bundle_configs()[1].pre_merge_rounds.len(), 1);
    assert_eq!(cfg.bundle_configs()[0].join_round, Some(0));
    assert_eq!(cfg.bundle_configs()[1].join_round, Some(0));
    assert_eq!(cfg.schedule().joins.len(), 1);
    assert_eq!(cfg.schedule().joins[0].t, 2);
    // Both committed at their natural sizes.
    assert_eq!(cfg.bundle_configs()[0].irs_config.vector_size(), 1 << 10);
    assert_eq!(cfg.bundle_configs()[1].irs_config.vector_size(), 1 << 9);
    // Merge point = 2^8 (one fold below the largest).
    assert_eq!(
        cfg.inner().rounds()[0]
            .code_switch()
            .config()
            .source()
            .vector_size(),
        1 << 8
    );
}

#[test]
fn derive_2_pow_19_and_2_pow_20_fold_2_converges_at_2_pow_18() {
    // Δ = 1 < folding factor 2 → merge one fold below the largest: the bigger
    // (2^20) folds by 2, the smaller (2^19) folds by 1, both reach 2^18. Each
    // commits at its natural size and runs a single pre-merge round before the
    // shared round-0 selector merge.
    let cfg = derive(test_tuning(vec![
        BundleSpec {
            length: 1 << 19,
            num_polys: 1,
        },
        BundleSpec {
            length: 1 << 20,
            num_polys: 1,
        },
    ]))
    .unwrap();

    // Both bundles run exactly one pre-merge round, then join at round 0.
    assert_eq!(cfg.bundle_configs()[0].pre_merge_rounds.len(), 1);
    assert_eq!(cfg.bundle_configs()[1].pre_merge_rounds.len(), 1);
    assert_eq!(cfg.bundle_configs()[0].join_round, Some(0));
    assert_eq!(cfg.bundle_configs()[1].join_round, Some(0));

    // Each is committed at its own natural size.
    assert_eq!(cfg.bundle_configs()[0].irs_config.vector_size(), 1 << 19);
    assert_eq!(cfg.bundle_configs()[1].irs_config.vector_size(), 1 << 20);

    // Single round-0 merge gathering both bundles.
    assert_eq!(cfg.schedule().joins.len(), 1);
    assert_eq!(cfg.schedule().joins[0].round_index, Some(0));
    assert_eq!(cfg.schedule().joins[0].t, 2);
    assert_eq!(cfg.schedule().joins[0].bundle_indices, vec![0, 1]);

    // Merge point = 2^18 (one fold below the largest bundle).
    assert_eq!(
        cfg.inner().rounds()[0]
            .code_switch()
            .config()
            .source()
            .vector_size(),
        1 << 18
    );
}

// ── Negative / guard tests ──────────────────────────────────────────────────

#[test]
fn derive_rejects_non_constant_folding() {
    let mut tuning = test_tuning(vec![BundleSpec {
        length: 1 << 8,
        num_polys: 1,
    }]);
    tuning.folding_factor = FoldingFactor::PerRound(vec![1, 2, 2]);
    let err = derive(tuning).unwrap_err();
    match err {
        DeriveError::BatchedUnsupported { reason } => {
            assert!(
                reason.contains("FoldingFactor::Constant"),
                "expected Constant-required message, got: {reason}"
            );
        }
        other => panic!("expected BatchedUnsupported, got: {other:?}"),
    }
}

#[test]
fn derive_rejects_gap_larger_than_folding_factor() {
    // Δ = 4 > folding factor 2 → the smaller bundle can't reach the merge point
    // in one fold.
    let tuning = test_tuning(vec![
        BundleSpec {
            length: 1 << 10,
            num_polys: 1,
        },
        BundleSpec {
            length: 1 << 6,
            num_polys: 1,
        },
    ]);
    let err = derive(tuning).unwrap_err();
    assert!(matches!(err, DeriveError::BatchedUnsupported { .. }));
}

#[test]
fn derive_rejects_mixed_num_polys() {
    let tuning = test_tuning(vec![
        BundleSpec {
            length: 1 << 8,
            num_polys: 2,
        },
        BundleSpec {
            length: 1 << 8,
            num_polys: 4,
        },
    ]);
    let err = derive(tuning).unwrap_err();
    assert!(matches!(err, DeriveError::BatchedUnsupported { .. }));
}

#[test]
fn derive_rejects_non_power_of_two_length() {
    let err = derive(test_tuning(vec![BundleSpec {
        length: 100,
        num_polys: 1,
    }]))
    .unwrap_err();
    assert!(matches!(err, DeriveError::BatchedUnsupported { .. }));
}

#[test]
fn derive_rejects_non_power_of_two_num_polys() {
    let err = derive(test_tuning(vec![BundleSpec {
        length: 1 << 8,
        num_polys: 3,
    }]))
    .unwrap_err();
    match err {
        DeriveError::BatchedUnsupported { reason } => {
            assert!(
                reason.contains("num_polys") && reason.contains("not a power of 2"),
                "expected num_polys/power-of-2 message, got: {reason}",
            );
        }
        other => panic!("expected BatchedUnsupported, got: {other:?}"),
    }
}

#[test]
fn derive_rejects_empty_bundles() {
    let err = derive(test_tuning(vec![])).unwrap_err();
    assert!(matches!(err, DeriveError::BatchedUnsupported { .. }));
}

// ── Security-slot + invariant checks ────────────────────────────────────────

#[test]
fn selector_merge_gets_pow_when_analytic_slot_is_below_target() {
    let mut spec = test_spec(Mode::Standard);
    spec.target_security_bits = 59;
    spec.pow_budget = PowBudget::per_slot(60);
    let cfg = BatchedProtocolConfig::<TestEmbedding>::derive(
        spec,
        test_tuning(vec![
            BundleSpec {
                length: 1 << 8,
                num_polys: 1,
            };
            512
        ]),
    )
    .expect("selector merge should be grindable");

    let pow = cfg.schedule().joins[0]
        .selector
        .selector()
        .sumcheck()
        .round_pow();
    assert!(
        f64::from(pow.difficulty()) > 0.0,
        "selector merge should carry PoW when analytic bits are below target"
    );
    cfg.validate_security_target_met_for_claims(&vec![1; 512])
        .expect("selector PoW should close the security gap");
}

#[test]
fn runtime_claim_count_validation_rejects_oversized_intra_bundle_rlc() {
    let cfg = derive(test_tuning(vec![BundleSpec {
        length: 1 << 8,
        num_polys: 1,
    }]))
    .unwrap();

    let err = cfg
        .validate_security_target_met_for_claims(&[1 << 30])
        .expect_err("huge claim counts should violate Field64's 40-bit target");
    match err {
        DeriveError::SecurityTargetNotMet {
            pow: Pow::BatchedIntraBundleRlc { index: 0 },
            ..
        } => {}
        other => panic!("expected intra-bundle RLC security failure, got {other:?}"),
    }
}

#[test]
fn try_new_rejects_inconsistent_join_t() {
    let cfg = derive(test_tuning(vec![
        BundleSpec {
            length: 1 << 8,
            num_polys: 1,
        };
        2
    ]))
    .unwrap();

    let BatchedProtocolConfig {
        inner,
        mut schedule,
        bundle_configs,
        tuning,
    } = cfg;
    schedule.joins[0].t += 1;

    let err =
        BatchedProtocolConfig::<TestEmbedding>::try_new(inner, schedule, bundle_configs, tuning)
            .expect_err("try_new should reject inconsistent join t");
    assert!(matches!(err, DeriveError::BatchedUnsupported { .. }));
}

#[test]
fn schedule_lookup_round_vs_basecase() {
    use crate::{
        algebra::fields::Field64,
        protocols::{proof_of_work, zook::batched::selector::merge as selector_merge},
    };
    type F = Field64;

    let joins = vec![
        RoundJoin {
            round_index: Some(0),
            bundle_indices: vec![0, 1],
            t: 2,
            selector: selector_merge::Config::new(2, 16, proof_of_work::Config::none()),
        },
        RoundJoin {
            round_index: None,
            bundle_indices: vec![2],
            t: 2,
            selector: selector_merge::Config::new(2, 1, proof_of_work::Config::none()),
        },
    ];
    let schedule = MergeSchedule::<F> { joins };

    assert_eq!(schedule.bundle_count(), 3);
    assert_eq!(schedule.join_at(0).unwrap().bundle_indices, vec![0, 1]);
    assert!(schedule.join_at(1).is_none());
    assert_eq!(schedule.basecase_join().unwrap().bundle_indices, vec![2]);
}
