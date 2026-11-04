//! Joint-mask commitment sizing and initial binding security.

use std::ops::Range;

use ark_ff::Field;

use crate::{
    algebra::{embedding::Identity, fields::FieldWithSize},
    bits::Bits,
    protocols::{
        irs_commit::{num_in_domain_queries, Config as IrsConfig},
        params::{
            basecase::ZookFinalOpeningSoundness,
            bounds::{error_bits, out_of_domain_bits, rate, usize_to_f64},
            error::{grind_to_at, DeriveError, Pow, ZookSecurityError},
            irs_commit::{solve_with_num_vectors, validate_joint_mask_code},
            protocol_config::JointMaskConfig,
            solved::Solved,
            spec::{RoundContext, SecuritySpec},
        },
    },
};

// Fixed tuning choice inherited from the mask solver, not a security assumption.
// The resulting code must still satisfy the binding and final-opening bounds.
const MASK_LOG_INV_RATE: u32 = 4;

/// The joint-mask column contains sparse sumcheck masks, witness masks, and two blinders.
///
/// ProveKit Prover Whitepaper §5.4, “ZO0K”, “The required masks”.
/// Sizes are summed from the components: each folding mask has `2 * folds + 1`
/// coefficients, the commitment blinder has `queries + 1`, and the basefield
/// blinder has one coefficient per extension dimension.
pub(super) fn solve<F: Field>(
    spec: &SecuritySpec,
    sumcheck_lengths: &[usize],
    witness_lengths: &[usize],
    final_commit: &IrsConfig<Identity<F>>,
    extension_degree: usize,
) -> Result<JointMaskConfig<F>, DeriveError> {
    let mut end = 0;
    let sumcheck_masks = sumcheck_lengths
        .iter()
        .map(|&len| next_range(&mut end, len))
        .collect::<Result<Vec<_>, _>>()?;
    let witness_masks = witness_lengths
        .iter()
        .map(|&len| next_range(&mut end, len))
        .collect::<Result<Vec<_>, _>>()?;
    let commit = solve_commit(spec, end, extension_degree)?;
    let blinder_len = commit
        .in_domain_samples()
        .checked_add(1)
        .ok_or(DeriveError::SizeOverflow)?;
    let blinder = next_range(&mut end, blinder_len)?;
    let basefield_blinder = next_range(&mut end, extension_degree)?;
    if end > commit.vector_size() {
        return Err(DeriveError::InvalidMaskLayout);
    }
    let binding = zook_mask_binding_bits(&commit)?;
    let opening = ZookFinalOpeningSoundness::for_codes(final_commit, &commit)?;
    grind_to_at(spec, binding, Pow::JointMaskBinding)?;
    grind_to_at(spec, opening.combination, Pow::FinalOpeningCombination)?;
    grind_to_at(spec, opening.queries, Pow::FinalOpeningQueries)?;
    Ok(JointMaskConfig {
        commit: Solved::new(commit, binding),
        sumcheck_masks,
        witness_masks,
        blinder,
        basefield_blinder,
        opening,
    })
}

/// Size the shared code using an upper bound on its query count.
///
/// NTT rounding can lower the effective rate, so it cannot require more queries
/// than this reservation. Actual query counts determine the stored mask ranges.
/// Only the codeword length needs NTT rounding; the mask message is not folded.
pub(super) fn solve_commit<F: Field>(
    spec: &SecuritySpec,
    payload_length: usize,
    extension_degree: usize,
) -> Result<IrsConfig<Identity<F>>, DeriveError> {
    let queries = num_in_domain_queries(
        spec.decoding_regime,
        f64::from(spec.target_security_bits) + 1.0,
        rate(f64::from(MASK_LOG_INV_RATE)),
    )
    .get();
    let vector_size = payload_length
        .checked_add(queries)
        .and_then(|n| n.checked_add(1))
        .and_then(|n| n.checked_add(extension_degree))
        .ok_or(DeriveError::SizeOverflow)?;
    solve_with_num_vectors(
        spec,
        &RoundContext {
            vector_size,
            log_inv_rate: MASK_LOG_INV_RATE,
            folding_factor: 0,
        },
        2,
    )
}

fn next_range(end: &mut usize, length: usize) -> Result<Range<usize>, DeriveError> {
    let start = *end;
    *end = start.checked_add(length).ok_or(DeriveError::SizeOverflow)?;
    Ok(start..*end)
}

/// Joint-mask DEEP binding plus the initial-mask Frobenius check.
///
/// `epsilon <= L*(L-1)*m/(2*(Q-1)) + L*m/(Q-1) = L*(L+1)*m/(2*(Q-1))`.
/// Here `L = masks.list_size()`, `m = masks.message_length()`, and `Q = |F|`.
/// The first term bounds DEEP collisions; the second bounds the Frobenius check.
/// Both concern only the joint-mask column; the other column is its final-opening blinder.
/// Unlike §6.3, “The zk-IOPP”, Phase 0, sampling excludes zero, where the
/// shifted membership blinder vanishes. Nonzero evaluation-domain points remain
/// allowed, so the root-count bounds use the challenge-domain size `Q-1`.
///
/// ProveKit Prover Whitepaper §6.4, “Soundness”, theorem “IOPP soundness”,
/// Phase 0 bound and the following union-bound remark.
pub fn zook_mask_binding_bits<F: Field>(
    masks: &IrsConfig<Identity<F>>,
) -> Result<Bits, ZookSecurityError> {
    validate_joint_mask_code(masks)?;
    let list = masks.list_size();
    Ok(error_bits(
        list.log2() + (list + 1.0).log2() - 1.0 + usize_to_f64(masks.message_length()).log2()
            - out_of_domain_bits(F::field_size_bits(), 1)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        algebra::fields::{Field64_2, Field64_3},
        protocols::params::{
            spec::{Mode, PowBudget},
            test_utils::{assert_probability_bits, johnson_code_with_vectors},
        },
    };

    #[test]
    fn mask_message_keeps_its_length_when_codeword_is_rounded() {
        let spec = SecuritySpec::new(128)
            .with_mode(Mode::ZeroKnowledge)
            .with_pow_budget(PowBudget::Forbidden);
        let commit = solve_commit::<Field64_3>(&spec, 4495, 3).unwrap();
        assert_eq!(commit.vector_size(), 4566);
        assert!(commit.codeword_length() >= 16 * commit.vector_size());
        assert!(4495 + commit.in_domain_samples() + 1 + 3 <= commit.vector_size());
        validate_joint_mask_code(&commit).unwrap();
        assert!(zook_mask_binding_bits(&commit).unwrap() >= Bits::from(128.0));
    }

    #[test]
    fn mask_binding_includes_frobenius_error() {
        let masks = johnson_code_with_vectors::<Identity<Field64_2>>(256, 1, 0.25, 2);
        let challenges = 2.0_f64.powf(Field64_2::field_size_bits()) - 1.0;
        let list = masks.list_size();
        let degree = masks.message_length() as f64;
        assert_probability_bits(
            zook_mask_binding_bits(&masks).unwrap(),
            list * (list - 1.0) / 2.0 * degree / challenges + list * degree / challenges,
        );
    }
}
