use super::*;

#[cfg(test)]
mod decoding_regime_tests {
    use super::*;

    #[test]
    fn from_str_round_trips_display() {
        for r in [
            DecodingRegime::Unique,
            DecodingRegime::Johnson,
            DecodingRegime::Capacity,
        ] {
            assert_eq!(r.to_string().parse::<DecodingRegime>().unwrap(), r);
        }
    }

    #[test]
    fn from_str_rejects_unknown() {
        assert!("johnson".parse::<DecodingRegime>().is_err()); // case-sensitive
        assert!("".parse::<DecodingRegime>().is_err());
        assert!("capacity".parse::<DecodingRegime>().is_err());
    }
}

#[cfg(test)]
mod policy_tests {
    use super::*;
    use crate::hash;

    const TARGET_BITS: u32 = 100;

    fn spec(pow_budget: PowBudget) -> SecuritySpec {
        SecuritySpec::new(TARGET_BITS)
            .with_mode(Mode::ZeroKnowledge)
            .with_pow_budget(pow_budget)
    }

    #[test]
    fn new_uses_documented_defaults() {
        let spec = SecuritySpec::new(128);
        assert_eq!(spec.mode, Mode::Standard);
        assert_eq!(spec.decoding_regime, DecodingRegime::Johnson);
        assert_eq!(spec.target_security_bits, 128);
        assert_eq!(
            spec.pow_budget,
            PowBudget::per_slot(DEFAULT_POW_BUDGET_BITS)
        );
        assert_eq!(spec.hash_id, hash::BLAKE3);
    }

    #[test]
    fn forbidden_means_no_pow_credit() {
        assert_eq!(
            spec(PowBudget::Forbidden).protocol_security_target_bits(),
            Bits::new(f64::from(TARGET_BITS)),
        );
    }

    #[test]
    fn per_slot_zero_collapses_to_forbidden() {
        assert_eq!(PowBudget::per_slot(0), PowBudget::Forbidden);
    }

    #[test]
    fn per_slot_bits_round_trip() {
        assert_eq!(PowBudget::per_slot(20).bits(), 20);
        assert_eq!(PowBudget::Forbidden.bits(), 0);
    }

    #[test]
    fn pow_credit_shifts_analytic_floor() {
        assert_eq!(
            spec(PowBudget::per_slot(20)).protocol_security_target_bits(),
            Bits::new(80.0),
        );
        assert_eq!(
            spec(PowBudget::per_slot(60)).protocol_security_target_bits(),
            Bits::new(40.0),
        );
    }

    #[test]
    fn pow_exceeding_target_saturates_to_zero() {
        let pow_over_target = TARGET_BITS + 100;
        assert_eq!(
            spec(PowBudget::per_slot(pow_over_target)).protocol_security_target_bits(),
            Bits::new(0.0),
        );
    }

    #[test]
    fn rate_schedule_stepping_unbounded() {
        // Stepping adds `folding − 1` per round regardless of magnitude
        // (legacy unbounded-stepping behavior).
        assert_eq!(RateSchedule::Stepping.step(2, 3), 4);
        assert_eq!(RateSchedule::Stepping.step(100, 5), 104);
    }

    #[test]
    fn rate_schedule_capped_clamps_above_cap() {
        let cap = RateSchedule::Capped {
            max_log_inv_rate: 5,
        };
        assert_eq!(cap.step(2, 3), 4); // below cap → step normally
        assert_eq!(cap.step(4, 3), 5); // would step to 6 → clamp to 5
        assert_eq!(cap.step(5, 3), 5); // already at cap → stays
        assert_eq!(cap.step(10, 3), 5); // above cap → snaps back to cap
    }

    #[test]
    fn rate_schedule_folding_factor_one_never_steps() {
        // folding == 1 means rate step is `1 − 1 = 0`. The IRS layer disallows
        // folding < 1, but the math here should still be consistent.
        assert_eq!(RateSchedule::Stepping.step(2, 1), 2);
        assert_eq!(
            RateSchedule::Capped {
                max_log_inv_rate: 5
            }
            .step(2, 1),
            2,
        );
    }
}
