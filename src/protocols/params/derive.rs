//! Derives a [`ProtocolConfig`] from a spec + tuning.

use crate::{
    algebra::embedding::Embedding,
    protocols::params::{
        basecase as basecase_params,
        branch::{Branch, RoundBuildMode, RoundBuildPayload},
        build_round::build_round_config,
        error::DeriveError,
        layout::{round_layout, RoundLayout},
        protocol_config::{ProtocolConfig, RoundConfig},
        spec::{LogInvRate, SecuritySpec, TuningSpec},
    },
};

impl<M: Embedding + Default> ProtocolConfig<M> {
    /// Fails with [`DeriveError`] when the spec/tuning combination is
    /// infeasible.
    pub fn derive(spec: SecuritySpec, tuning: TuningSpec) -> Result<Self, DeriveError> {
        let mode: RoundBuildMode<'_> = spec.as_zk().map_or(Branch::Standard, |zk_spec| {
            Branch::ZeroKnowledge(RoundBuildPayload {
                zk_spec,
                // Hardcoded to 2^-4 to decouple the C_zk code rate from the
                // message rate. The mask-proximity γ-combination discharge
                // (Lemma 7.4 slot) needs headroom in |Λ(C_zk, δ_zk)|, and
                // tightening to `tuning.starting_log_inv_rate` (which is set
                // by the witness rate, not the C_zk rate) shrinks that
                // headroom unpredictably. Adaptive schedules can still tune
                // the witness side without touching C_zk.
                c_zk_log_inv_rate: LogInvRate::new(4),
            })
        });

        let RoundLayout {
            shapes,
            basecase_vector_size,
            basecase_log_inv_rate,
        } = round_layout::<M>(&spec, &tuning, mode)?;

        let rounds: Vec<RoundConfig<M>> = shapes
            .iter()
            .map(|shape| build_round_config::<M>(&spec, shape, mode))
            .collect::<Result<_, _>>()?;

        // When at least one round exists, the last round's `code_switch.target`
        // is exactly the IRS that holds the folded basecase message — it has
        // `interleaving_depth = 1` thanks to the override in `round_layout`.
        // Build basecase around that same IRS so `zook::prove` can pass the
        // existing witness directly to `basecase.prove` without re-encoding.
        // PoW (sumcheck + γ-combination) is re-solved against the swapped
        // IRS so analytic + PoW still meets the security target.
        let basecase = if let Some(last) = rounds.last() {
            basecase_params::solve_with_commit(&spec, last.code_switch().target().clone())?
        } else {
            basecase_params::solve(&spec, basecase_vector_size, basecase_log_inv_rate)?
        };

        let plan = Self::new(spec, tuning, rounds, basecase);
        plan.validate()?;
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use crate::{
        algebra::{
            embedding::Embedding,
            fields::{Field64, FieldWithSize},
        },
        protocols::{
            basecase::BasecaseMode,
            params::{
                error::{ChainSource, ChainTarget, DeriveError, Pow},
                protocol_config::{ProtocolConfig, RoundMode},
                spec::{
                    DecodingRegime, FoldingFactor, KneeWeight, Mode, PowBudget, RateSchedule,
                    SecuritySpec, TuningSpec,
                },
                test_utils::{assert_close, assert_pow_closes_gap, TestEmbedding},
            },
        },
    };

    fn arb_tuning() -> impl Strategy<Value = TuningSpec> {
        let folding = prop_oneof![
            (1usize..=3).prop_map(FoldingFactor::Constant),
            (1usize..=3, 1usize..=3).prop_map(|(initial, rest)| {
                FoldingFactor::ConstantFromSecondRound { initial, rest }
            }),
        ];
        // Mix the three RateSchedule variants so proptest covers Capped and
        // Adaptive code paths, not just Stepping.
        let schedule = prop_oneof![
            Just(RateSchedule::Stepping),
            (4u32..=10).prop_map(|max_log_inv_rate| RateSchedule::Capped { max_log_inv_rate }),
            Just(RateSchedule::Adaptive {
                knee_weight: KneeWeight::DEFAULT,
            }),
        ];
        (4u32..=8, 1u32..=3, folding, schedule).prop_map(
            |(log_size, log_inv_rate, folding_factor, rate_schedule)| TuningSpec {
                vector_size: 1usize << log_size,
                starting_log_inv_rate: log_inv_rate,
                folding_factor,
                rate_schedule,
            },
        )
    }

    const FIXTURE_FOLDING_FACTOR: usize = 2;
    const FIXTURE_LOG_INV_RATE: u32 = 1;

    const LOG_VECTOR_SIZE_NO_ROUNDS: u32 = 3;
    const LOG_VECTOR_SIZE_MULTI_ROUND: u32 = 8;

    fn tuning_with(vector_size: usize) -> TuningSpec {
        TuningSpec {
            vector_size,
            starting_log_inv_rate: FIXTURE_LOG_INV_RATE,
            folding_factor: FoldingFactor::Constant(FIXTURE_FOLDING_FACTOR),
            rate_schedule: RateSchedule::Stepping,
        }
    }

    const PLAN_FIXTURE_TARGET_BITS: u32 = 40;

    fn test_spec(mode: Mode) -> SecuritySpec {
        SecuritySpec::new(PLAN_FIXTURE_TARGET_BITS)
            .with_mode(mode)
            .with_pow_budget(PowBudget::per_slot(LOOSE_POW_BUDGET_BITS))
    }

    #[test]
    fn derive_standard_with_no_rounds_uses_basecase_only() {
        let spec = test_spec(Mode::Standard);
        let vector_size = 1usize << LOG_VECTOR_SIZE_NO_ROUNDS;
        let plan = ProtocolConfig::<TestEmbedding>::derive(spec, tuning_with(vector_size)).unwrap();
        assert!(plan.rounds().is_empty());
        assert_eq!(plan.basecase().commit().vector_size(), vector_size);
    }

    #[test]
    fn derive_zk_with_no_rounds_uses_zk_basecase_only() {
        let spec = test_spec(Mode::ZeroKnowledge);
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_NO_ROUNDS),
        )
        .unwrap();
        assert!(plan.rounds().is_empty());
        assert!(matches!(
            plan.basecase().mode(),
            BasecaseMode::ZeroKnowledge
        ));
    }

    #[test]
    fn t_ood_nonzero_in_johnson_zk() {
        let spec = SecuritySpec {
            decoding_regime: DecodingRegime::Johnson,
            ..test_spec(Mode::ZeroKnowledge)
        };
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        for r in plan.rounds() {
            let RoundMode::ZeroKnowledge { t_ood, .. } = r.mode() else {
                panic!("expected ZK round")
            };
            assert!(t_ood.get() >= 1);
        }
    }

    #[test]
    fn t_ood_pinned_to_one_in_unique_zk() {
        let spec = SecuritySpec {
            decoding_regime: DecodingRegime::Unique,
            ..test_spec(Mode::ZeroKnowledge)
        };
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        for r in plan.rounds() {
            let RoundMode::ZeroKnowledge { t_ood, .. } = r.mode() else {
                panic!("expected ZK round")
            };
            assert_eq!(t_ood.get(), 1);
        }
    }

    #[test]
    fn c_zk_keeps_code_switch_mask_under_unique() {
        let spec = SecuritySpec {
            decoding_regime: DecodingRegime::Unique,
            ..test_spec(Mode::ZeroKnowledge)
        };
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        for r in plan.rounds() {
            let mask_oracle = r.mask_oracle().expect("ZK round has a mask oracle");
            let k = r
                .code_switch()
                .source()
                .interleaving_depth()
                .trailing_zeros() as usize;
            // Split-tree mask oracle: sumcheck tree carries the `k` sumcheck
            // masks (2·k vectors after originals + freshes); cs_mask tree
            // carries the single `(r ‖ s)` code-switch mask (2 vectors).
            assert_eq!(
                mask_oracle.sumcheck_masks().c_zk_commit().num_vectors(),
                2 * k,
                "sumcheck_masks tree must carry 2·k originals+freshes",
            );
            assert_eq!(
                mask_oracle.cs_mask().c_zk_commit().num_vectors(),
                2,
                "cs_mask tree carries one (r ‖ s) mask + its fresh pair",
            );
        }
    }

    #[test]
    fn analytic_bits_finite_and_positive_standard() {
        let spec = test_spec(Mode::Standard);
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        let bits: f64 = plan.analytic_bits().into();
        assert!(bits.is_finite() && bits > 0.0, "bits = {bits}");
        let min_round = plan
            .rounds()
            .iter()
            .map(|r| f64::from(r.analytic_bits()))
            .fold(f64::INFINITY, f64::min);
        let expected = min_round.min(f64::from(plan.basecase().analytic_bits()));
        assert_close(bits, expected);
    }

    #[test]
    fn analytic_bits_includes_mask_oracle_in_zk() {
        let spec = test_spec(Mode::ZeroKnowledge);
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        let plan_bits: f64 = plan.analytic_bits().into();
        let mo_floor = plan
            .rounds()
            .iter()
            .filter_map(|r| r.mask_oracle().map(|mo| f64::from(mo.analytic_bits())))
            .fold(f64::INFINITY, f64::min);
        assert!(
            mo_floor.is_finite(),
            "ZK plan must contribute mask-oracle bits"
        );
        let min_round = plan
            .rounds()
            .iter()
            .map(|r| f64::from(r.analytic_bits()))
            .fold(f64::INFINITY, f64::min);
        let expected = mo_floor
            .min(min_round)
            .min(f64::from(plan.basecase().analytic_bits()));
        assert_close(plan_bits, expected);
    }

    #[test]
    fn derive_plans_basecase() {
        let spec = test_spec(Mode::ZeroKnowledge);
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        assert!(matches!(
            plan.basecase().mode(),
            BasecaseMode::ZeroKnowledge
        ));
        assert_eq!(plan.basecase().commit().interleaving_depth(), 1);
        assert_eq!(plan.basecase().sumcheck().final_size(), 1);
    }

    const LOOSE_POW_BUDGET_BITS: u32 = 60;
    const OVER_BUDGET_INJECTED_BITS: f64 = 50.0;

    /// Bounds doc §5.3 + §5.7: HVZK privacy error in bits matches the closed
    /// form `−log Σ_r (t_ood_r² + t_ood_r) / (2|F|)` over ZK rounds.
    #[test]
    fn privacy_error_bits_matches_bound_3_sum() {
        let spec = test_spec(Mode::ZeroKnowledge);
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        let field_bits = <Field64 as FieldWithSize>::field_size_bits();
        let mut expected_total = 0.0_f64;
        for r in plan.rounds() {
            let RoundMode::ZeroKnowledge { t_ood, .. } = r.mode() else {
                panic!("expected ZK round");
            };
            let t = t_ood.get() as f64;
            expected_total += 2_f64.powf(f64::midpoint(t * t, t).log2() - field_bits);
        }
        let expected_bits = -expected_total.log2();
        let got = f64::from(plan.privacy_error_bits());
        assert_close(got, expected_bits);
    }

    #[test]
    fn privacy_error_bits_standard_returns_target_sentinel() {
        let spec = test_spec(Mode::Standard);
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        assert_close(
            f64::from(plan.privacy_error_bits()),
            f64::from(PLAN_FIXTURE_TARGET_BITS),
        );
    }

    #[test]
    fn check_pow_bits_passes_on_derived_plan() {
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            test_spec(Mode::ZeroKnowledge),
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        assert!(plan.check_pow_bits());
    }

    #[test]
    fn check_pow_bits_detects_over_budget_slot() {
        use crate::{bits::Bits, protocols::proof_of_work::Config as PowConfig};
        const MODERATE_POW_BUDGET_BITS: u32 = 30;
        let spec = SecuritySpec {
            pow_budget: PowBudget::per_slot(MODERATE_POW_BUDGET_BITS),
            ..test_spec(Mode::ZeroKnowledge)
        };
        let mut plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        plan.override_basecase_pow_for_test(PowConfig::from_difficulty(Bits::new(
            OVER_BUDGET_INJECTED_BITS,
        )));
        assert!(!plan.check_pow_bits());
    }

    #[test]
    fn validate_round_chaining_detects_adjacent_round_mismatch() {
        let spec = test_spec(Mode::ZeroKnowledge);
        let mut plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        let n = plan.rounds().len();
        assert!(n >= 2, "need ≥ 2 rounds to break a mid-chain link");
        assert!(plan.check_all_invariants(), "fresh plan must validate");

        let bad_size = plan.rounds()[0].code_switch().target().vector_size() + 1;
        plan.corrupt_round_target_vector_size_for_test(0, bad_size);

        let err = plan
            .validate_round_chaining()
            .expect_err("adjacent-round mismatch must trip the chain check");
        assert!(
            matches!(
                err,
                DeriveError::RoundChainBroken {
                    from: ChainSource::Round(0),
                    to: ChainTarget::NextRound(1),
                    ..
                }
            ),
            "got {err:?}",
        );
        assert!(!plan.check_all_invariants());
    }

    #[test]
    fn validate_round_chaining_detects_basecase_mismatch() {
        let spec = test_spec(Mode::ZeroKnowledge);
        let mut plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        let n = plan.rounds().len();
        assert!(n >= 2, "need ≥ 2 rounds to break the chain by truncation");
        assert!(plan.check_all_invariants(), "fresh plan must validate");

        plan.truncate_rounds_for_test(n - 1);
        let err = plan
            .validate_round_chaining()
            .expect_err("truncated tail breaks basecase chaining");
        assert!(
            matches!(
                err,
                DeriveError::RoundChainBroken {
                    to: ChainTarget::Basecase,
                    ..
                }
            ),
            "got {err:?}",
        );
        assert!(!plan.check_all_invariants());
    }

    #[test]
    fn validate_security_target_met_passes_on_fresh_plan() {
        let spec = test_spec(Mode::ZeroKnowledge);
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        plan.validate_security_target_met()
            .expect("fresh plan must satisfy per-slot target check");
    }

    #[test]
    fn validate_security_target_met_catches_recorded_analytic_drift() {
        use crate::bits::Bits;
        let spec = test_spec(Mode::ZeroKnowledge);
        let mut plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        assert!(!plan.rounds().is_empty(), "need a round to corrupt");
        let recorded = plan
            .rounds()
            .first()
            .map(|r| r.sumcheck().analytic())
            .expect("params solver records sumcheck analytic");
        // Bump the recorded value far from the recompute → triggers drift.
        plan.corrupt_round_sumcheck_analytic_for_test(0, Bits::new(f64::from(recorded) + 10.0));
        let err = plan
            .validate_security_target_met()
            .expect_err("recorded vs recompute mismatch must trip drift check");
        assert!(
            matches!(
                err,
                DeriveError::AnalyticDrift {
                    pow: Pow::RoundSumcheck { index: 0 },
                    ..
                }
            ),
            "got {err:?}",
        );
    }

    #[test]
    fn validate_security_target_met_catches_zeroed_basecase_pow() {
        // ZK basecase has a γ-combination PoW slot whose analytic floor is
        // below the security target under the test fixture. Wiping the PoW
        // must trip `validate_security_target_met`.
        use crate::{bits::Bits, protocols::proof_of_work::Config as PowConfig};
        let spec = test_spec(Mode::ZeroKnowledge);
        let mut plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        assert!(
            plan.check_all_invariants(),
            "fresh plan must validate including soundness"
        );

        // Skip the test if this fixture's basecase γ-combination doesn't
        // actually require PoW (i.e., analytic ≥ target already). Otherwise
        // force a violation by zeroing the basecase PoW.
        let pre_pow = plan.basecase().pow().difficulty();
        if f64::from(pre_pow) == 0.0 {
            return;
        }

        plan.override_basecase_pow_for_test(PowConfig::from_difficulty(Bits::new(0.0)));
        let err = plan
            .validate_security_target_met()
            .expect_err("zeroed γ-combination PoW must trip soundness validation");
        assert!(
            matches!(
                err,
                DeriveError::SecurityTargetNotMet {
                    pow: Pow::BasecaseGammaCombination,
                    ..
                }
            ),
            "got {err:?}",
        );
    }

    /// Adaptive plan must validate end-to-end across modes and produce a
    /// strictly less aggressive rate schedule than the unbounded `Stepping`
    /// baseline at the tail.
    #[test]
    fn adaptive_plan_validates_and_caps_tail_rate() {
        for mode in [Mode::Standard, Mode::ZeroKnowledge] {
            let spec = test_spec(mode);
            let tuning = TuningSpec {
                vector_size: 1 << LOG_VECTOR_SIZE_MULTI_ROUND,
                starting_log_inv_rate: FIXTURE_LOG_INV_RATE,
                folding_factor: FoldingFactor::Constant(FIXTURE_FOLDING_FACTOR),
                rate_schedule: RateSchedule::Adaptive {
                    knee_weight: KneeWeight::DEFAULT,
                },
            };
            let plan =
                ProtocolConfig::<TestEmbedding>::derive(spec.clone(), tuning.clone()).unwrap();
            assert!(
                plan.check_all_invariants(),
                "Adaptive plan must satisfy validate() in mode={mode:?}"
            );

            // Unbounded `Stepping` is the maximally-aggressive baseline (the
            // canonical WHIR step-forever schedule). Adaptive should never
            // propose a schedule whose basecase rate exceeds it — that's the
            // whole point: Adaptive may stop growing rate earlier.
            let unbounded = ProtocolConfig::<TestEmbedding>::derive(
                spec,
                TuningSpec {
                    rate_schedule: RateSchedule::Stepping,
                    ..tuning
                },
            )
            .unwrap();
            assert!(
                plan.basecase().commit().rate() >= unbounded.basecase().commit().rate(),
                "Adaptive basecase rate ({}) must be ≥ unbounded baseline's ({}) — \
                 Adaptive should pick a *smaller* `1/ρ` at the tail",
                plan.basecase().commit().rate(),
                unbounded.basecase().commit().rate(),
            );
        }
    }

    /// When rounds exist, basecase IRS must equal the last round's
    /// `code_switch.target` IRS — `zook::prove` skips re-encoding the folded
    /// message at basecase entry.
    #[test]
    fn basecase_irs_aliases_last_round_target_when_rounds_present() {
        for mode in [Mode::Standard, Mode::ZeroKnowledge] {
            let spec = test_spec(mode);
            for schedule in [
                RateSchedule::Stepping,
                RateSchedule::Capped {
                    max_log_inv_rate: 6,
                },
                RateSchedule::Adaptive {
                    knee_weight: KneeWeight::DEFAULT,
                },
            ] {
                let tuning = TuningSpec {
                    vector_size: 1 << LOG_VECTOR_SIZE_MULTI_ROUND,
                    starting_log_inv_rate: FIXTURE_LOG_INV_RATE,
                    folding_factor: FoldingFactor::Constant(FIXTURE_FOLDING_FACTOR),
                    rate_schedule: schedule,
                };
                let plan = ProtocolConfig::<TestEmbedding>::derive(spec.clone(), tuning).unwrap();
                assert!(!plan.rounds().is_empty(), "fixture must have rounds");
                let last_target = plan.rounds().last().unwrap().code_switch().target();
                let basecase_commit = plan.basecase().commit();
                assert_eq!(
                    last_target, basecase_commit,
                    "basecase.commit must alias the last round's code_switch.target — \
                     mode={mode:?} schedule={schedule:?}",
                );
                assert_eq!(basecase_commit.interleaving_depth(), 1);
            }
        }
    }

    #[test]
    fn derive_with_capped_rate_shrinks_basecase_codeword() {
        // Same shape that historically inflates basecase via rate stepping:
        // ~2^20 witness, folding 3, 5 rounds → stepped log_inv_rate ≈ 12.
        const LOG_VECTOR_SIZE: u32 = 12;
        let spec = test_spec(Mode::ZeroKnowledge);
        let folding = FoldingFactor::Constant(3);

        let stepped_tuning = TuningSpec {
            vector_size: 1usize << LOG_VECTOR_SIZE,
            starting_log_inv_rate: 2,
            folding_factor: folding,
            rate_schedule: RateSchedule::Stepping,
        };
        let stepped_plan =
            ProtocolConfig::<TestEmbedding>::derive(spec.clone(), stepped_tuning).unwrap();

        let capped_tuning = TuningSpec {
            vector_size: 1usize << LOG_VECTOR_SIZE,
            starting_log_inv_rate: 2,
            folding_factor: folding,
            rate_schedule: RateSchedule::Capped {
                max_log_inv_rate: 4,
            },
        };
        let capped_plan = ProtocolConfig::<TestEmbedding>::derive(spec, capped_tuning).unwrap();

        // Same number of rounds — cap only affects per-round rate, not layout shape.
        assert_eq!(stepped_plan.rounds().len(), capped_plan.rounds().len());
        // Cap forces a strictly smaller basecase codeword (the saving we're after).
        let stepped_codeword = stepped_plan.basecase().commit().codeword_length();
        let capped_codeword = capped_plan.basecase().commit().codeword_length();
        assert!(
            capped_codeword < stepped_codeword,
            "capped basecase codeword ({capped_codeword}) should be smaller than stepped \
             ({stepped_codeword})",
        );
        // For this fixture the cap should produce at least a 4× reduction.
        assert!(
            capped_codeword * 4 <= stepped_codeword,
            "expected ≥4× reduction; got stepped={stepped_codeword}, capped={capped_codeword}",
        );
    }

    /// Adaptive must not be worse than the unbounded-`Stepping` baseline on
    /// basecase NTT work for the folding-3 / 2^12 ZK fixture that historically
    /// inflates basecase via unbounded rate stepping. At minimum, Adaptive's
    /// basecase codeword must be ≤ the baseline's.
    #[test]
    fn adaptive_basecase_no_worse_than_unbounded_on_inflating_fixture() {
        const LOG_VECTOR_SIZE: u32 = 12;
        let spec = test_spec(Mode::ZeroKnowledge);
        let base = TuningSpec {
            vector_size: 1usize << LOG_VECTOR_SIZE,
            starting_log_inv_rate: 2,
            folding_factor: FoldingFactor::Constant(3),
            rate_schedule: RateSchedule::Stepping,
        };
        let unbounded =
            ProtocolConfig::<TestEmbedding>::derive(spec.clone(), base.clone()).unwrap();
        let adaptive = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            TuningSpec {
                rate_schedule: RateSchedule::Adaptive {
                    knee_weight: KneeWeight::DEFAULT,
                },
                ..base
            },
        )
        .unwrap();
        let unbounded_basecode = unbounded.basecase().commit().codeword_length();
        let adaptive_basecode = adaptive.basecase().commit().codeword_length();
        assert!(
            adaptive_basecode <= unbounded_basecode,
            "Adaptive basecase codeword ({adaptive_basecode}) must be ≤ unbounded \
             baseline's ({unbounded_basecode})",
        );
    }

    #[test]
    fn derive_reports_pow_ungrindable() {
        const UNREACHABLE_TARGET_BITS: u32 = 200;
        let spec = SecuritySpec {
            target_security_bits: UNREACHABLE_TARGET_BITS,
            ..test_spec(Mode::Standard)
        };
        let err = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .expect_err("target above grind cap must fail");
        assert!(
            matches!(err, DeriveError::PowUngrindable { .. }),
            "got {err:?}",
        );
    }

    #[test]
    fn derive_reports_pow_budget_exceeded() {
        const TIGHT_MAX_POW: u32 = 5;
        let spec = SecuritySpec {
            pow_budget: PowBudget::per_slot(TIGHT_MAX_POW),
            ..test_spec(Mode::ZeroKnowledge)
        };
        let err = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .expect_err("tight pow_budget must trip auto-validation");
        assert!(
            matches!(err, DeriveError::PowBudgetExceeded { .. }),
            "got {err:?}",
        );
    }

    #[test]
    fn derive_threads_unique_decoding_standard() {
        let spec = SecuritySpec {
            decoding_regime: DecodingRegime::Unique,
            ..test_spec(Mode::Standard)
        };
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_NO_ROUNDS),
        )
        .unwrap();
        assert!(plan.rounds().is_empty());
        assert!(plan.basecase().commit().unique_decoding());
    }

    #[test]
    fn derive_threads_unique_decoding_zk() {
        let spec = SecuritySpec {
            decoding_regime: DecodingRegime::Unique,
            ..test_spec(Mode::ZeroKnowledge)
        };
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_NO_ROUNDS),
        )
        .unwrap();
        assert!(plan.rounds().is_empty());
        assert!(plan.basecase().commit().unique_decoding());
    }

    #[test]
    fn derive_multi_round_unique_decoding_succeeds() {
        let spec = SecuritySpec {
            decoding_regime: DecodingRegime::Unique,
            ..test_spec(Mode::Standard)
        };
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        assert!(!plan.rounds().is_empty(), "expected multi-round plan");
        for r in plan.rounds() {
            let cs = r.code_switch();
            assert!(cs.source().unique_decoding());
            assert!(cs.target().unique_decoding());
            assert!(cs.out_domain_samples() >= 1);
        }
        assert!(plan.basecase().commit().unique_decoding());
    }

    #[test]
    fn derive_multi_round_unique_decoding_zk_succeeds() {
        let spec = SecuritySpec {
            decoding_regime: DecodingRegime::Unique,
            ..test_spec(Mode::ZeroKnowledge)
        };
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        assert!(!plan.rounds().is_empty(), "expected multi-round plan");
        for r in plan.rounds() {
            let mo = r.mask_oracle().expect("ZK round must own a mask oracle");
            assert!(mo.cs_mask().c_zk_commit().unique_decoding());
            assert!(r.code_switch().source().unique_decoding());
            assert!(r.code_switch().out_domain_samples() >= 1);
        }
        assert!(plan.basecase().commit().unique_decoding());
    }

    #[test]
    fn derive_multi_round_capacity_decoding_succeeds() {
        let spec = SecuritySpec {
            decoding_regime: DecodingRegime::Capacity,
            ..test_spec(Mode::Standard)
        };
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        assert!(!plan.rounds().is_empty(), "expected multi-round plan");
        for r in plan.rounds() {
            assert!(r.code_switch().out_domain_samples() >= 1);
        }
    }

    #[test]
    fn derive_multi_round_capacity_decoding_zk_succeeds() {
        let spec = SecuritySpec {
            decoding_regime: DecodingRegime::Capacity,
            ..test_spec(Mode::ZeroKnowledge)
        };
        let plan = ProtocolConfig::<TestEmbedding>::derive(
            spec,
            tuning_with(1 << LOG_VECTOR_SIZE_MULTI_ROUND),
        )
        .unwrap();
        assert!(!plan.rounds().is_empty(), "expected multi-round plan");
        for r in plan.rounds() {
            r.mask_oracle().expect("ZK round must own a mask oracle");
            assert!(r.code_switch().out_domain_samples() >= 1);
        }
    }

    /// Every slot from [`ProtocolConfig::pow_slots`] must close the gap to the
    /// target with its configured grind, judged from a fresh recompute.
    fn assert_plan_meets_target_per_slot<M: Embedding>(
        spec: &SecuritySpec,
        plan: &ProtocolConfig<M>,
    ) {
        for slot in plan.pow_slots() {
            assert_pow_closes_gap(spec, slot.recompute, &slot.pow);
        }
    }

    proptest! {
        #[test]
        fn derived_plan_meets_target_per_slot_standard(tuning in arb_tuning()) {
            let spec = test_spec(Mode::Standard);
            let plan = ProtocolConfig::<TestEmbedding>::derive(spec.clone(), tuning).unwrap();
            assert_plan_meets_target_per_slot(&spec, &plan);
        }

        #[test]
        fn derived_plan_meets_target_per_slot_zk(tuning in arb_tuning()) {
            let log_threshold =
                tuning.folding_factor.at_round(0) + tuning.folding_factor.at_round(1);
            prop_assume!(tuning.vector_size.trailing_zeros() as usize >= log_threshold);
            let spec = test_spec(Mode::ZeroKnowledge);
            let plan = ProtocolConfig::<TestEmbedding>::derive(spec.clone(), tuning).unwrap();
            assert_plan_meets_target_per_slot(&spec, &plan);
        }

        #[test]
        fn derive_standard_succeeds_over_tunings(tuning in arb_tuning()) {
            let spec = test_spec(Mode::Standard);
            let plan = ProtocolConfig::<TestEmbedding>::derive(spec, tuning).unwrap();
            for r in plan.rounds() {
                prop_assert!(matches!(r.mode(), RoundMode::Standard));
                prop_assert!(r.mask_oracle().is_none());
            }
            prop_assert!(matches!(
                plan.basecase().mode(),
                BasecaseMode::Standard
            ));
            prop_assert_eq!(plan.basecase().commit().interleaving_depth(), 1);
        }

        #[test]
        fn derive_zk_succeeds_over_tunings(tuning in arb_tuning()) {
            let log_threshold =
                tuning.folding_factor.at_round(0) + tuning.folding_factor.at_round(1);
            prop_assume!(tuning.vector_size.trailing_zeros() as usize >= log_threshold);

            let spec = test_spec(Mode::ZeroKnowledge);
            let plan = ProtocolConfig::<TestEmbedding>::derive(spec, tuning).unwrap();
            for r in plan.rounds() {
                let mask_oracle = r
                    .mask_oracle()
                    .expect("ZK round must have a mask oracle");
                let RoundMode::ZeroKnowledge { t_ood, .. } = r.mode() else {
                    panic!("expected ZK round");
                };
                let cs = r.code_switch();
                let k = cs.source().interleaving_depth().trailing_zeros() as usize;
                // Split-tree mask oracle: sumcheck tree has 2·k vectors, cs tree has 2.
                prop_assert_eq!(mask_oracle.sumcheck_masks().c_zk_commit().num_vectors(), 2 * k);
                prop_assert_eq!(mask_oracle.cs_mask().c_zk_commit().num_vectors(), 2);
                let source_mask = cs.source().mask_length();
                prop_assert!(mask_oracle.l_zk().get() >= source_mask + t_ood.get());
            }
            prop_assert!(matches!(
                plan.basecase().mode(),
                BasecaseMode::ZeroKnowledge
            ));
        }

        #[test]
        fn analytic_bits_finite_and_non_negative_standard(tuning in arb_tuning()) {
            let spec = test_spec(Mode::Standard);
            let plan = ProtocolConfig::<TestEmbedding>::derive(spec, tuning).unwrap();
            let analytic = f64::from(plan.analytic_bits());
            prop_assert!(analytic.is_finite());
            prop_assert!(analytic >= 0.0);
        }
    }
}
