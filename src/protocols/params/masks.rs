//! Joint-mask commitment sizing and mask ranges.

use std::ops::Range;

use ark_ff::Field;
use serde::{Deserialize, Serialize};

use crate::{
    algebra::embedding::{Embedding, Identity},
    bits::Bits,
    protocols::{
        irs_commit::{num_in_domain_queries, Config as IrsConfig},
        params::{
            derive::solve_with_num_vectors,
            error::{DeriveError, SecurityEvent, ZookSecurityError},
            security::{grind_to_at, rate, zook_mask_binding_bits, ZookFinalOpeningSoundness},
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
    grind_to_at(spec, binding, SecurityEvent::JointMaskBinding)?;
    grind_to_at(
        spec,
        opening.combination,
        SecurityEvent::FinalOpeningCombination,
    )?;
    grind_to_at(spec, opening.queries, SecurityEvent::FinalOpeningQueries)?;
    Ok(JointMaskConfig {
        commit,
        sumcheck_masks,
        witness_masks,
        blinder,
        basefield_blinder,
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

/// Sparse sumcheck mask length: one constant and two coefficients per variable.
///
/// ProveKit Prover Whitepaper §5.4, “ZO0K”, “The required masks”: `2 * folds + 1`.
pub(super) fn mask_coefficients(folds: usize) -> Result<usize, DeriveError> {
    folds
        .checked_mul(2)
        .and_then(|n| n.checked_add(1))
        .ok_or(DeriveError::SizeOverflow)
}

/// Prefix mask for each contiguous coefficient block encoded by IRS.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WitnessMaskConfig {
    /// Independent encoded polynomials in this commitment.
    pub num_polynomials: usize,
    /// Random prefix length in each polynomial.
    pub coefficients_per_polynomial: usize,
}

impl WitnessMaskConfig {
    /// Cover every opened row and incoming DEEP evaluation without increasing degree.
    ///
    /// Section 5.4, “ZO0K”, “The required masks”, budgets openings per witness.
    /// IRS rows reveal every column, so each polynomial receives its own prefix.
    /// An incoming DEEP evaluation reserves one extra coefficient per polynomial.
    /// This includes the final witness: it is evaluated before the final opening.
    pub(crate) fn for_code<M: Embedding>(
        code: &IrsConfig<M>,
        incoming_deep: usize,
    ) -> Result<Self, DeriveError> {
        let config = Self {
            num_polynomials: code.interleaving_depth(),
            coefficients_per_polynomial: code
                .in_domain_samples()
                .checked_add(incoming_deep)
                .ok_or(DeriveError::SizeOverflow)?,
        };
        config.validate(code, incoming_deep)?;
        Ok(config)
    }

    /// Number of random coefficients to sample for this witness commitment.
    pub fn num_coefficients(self) -> Result<usize, DeriveError> {
        self.num_polynomials
            .checked_mul(self.coefficients_per_polynomial)
            .ok_or(DeriveError::SizeOverflow)
    }

    pub(super) fn validate<M: Embedding>(
        &self,
        code: &IrsConfig<M>,
        incoming_deep: usize,
    ) -> Result<(), DeriveError> {
        let required = code
            .in_domain_samples()
            .checked_add(incoming_deep)
            .ok_or(DeriveError::SizeOverflow)?;
        if code.num_vectors() != 1
            || self.num_polynomials != code.interleaving_depth()
            || self.coefficients_per_polynomial != required
            || code.mask_length() != 0
        {
            return Err(DeriveError::InvalidMaskLayout);
        }
        if required > code.message_length() {
            return Err(DeriveError::MaskDoesNotFit {
                required,
                available: code.message_length(),
            });
        }
        Ok(())
    }
}

/// Masks consumed by one folding round.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoundMasks {
    /// Independent overlaid prefix in every source polynomial.
    pub witness: WitnessMaskConfig,
    /// One constant and two coefficients per folded variable: `2 * folds + 1`.
    pub sumcheck_coefficients: usize,
}

/// One mask commitment, with disjoint ranges in coefficient order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct JointMaskConfig<F: Field> {
    pub(crate) commit: IrsConfig<Identity<F>>,
    pub(crate) sumcheck_masks: Vec<Range<usize>>,
    pub(crate) witness_masks: Vec<Range<usize>>,
    pub(crate) blinder: Range<usize>,
    pub(crate) basefield_blinder: Range<usize>,
}

impl<F: Field> JointMaskConfig<F> {
    /// Ordinary IRS commitment for the joint masks.
    pub const fn commit(&self) -> &IrsConfig<Identity<F>> {
        &self.commit
    }
    /// Sparse sumcheck coefficient ranges, in round order.
    pub fn sumcheck_masks(&self) -> &[Range<usize>] {
        &self.sumcheck_masks
    }
    /// Witness mask ranges, from the initial commitment through the final witness.
    pub fn witness_masks(&self) -> &[Range<usize>] {
        &self.witness_masks
    }
    /// Random coefficients covering joint-mask queries and its binding evaluation.
    pub const fn blinder(&self) -> &Range<usize> {
        &self.blinder
    }
    /// Membership-check blinder sampled in the initial witness commitment's field.
    pub const fn basefield_blinder(&self) -> &Range<usize> {
        &self.basefield_blinder
    }
    /// Security bounds for opening the final witness and accumulated mask claims.
    pub fn opening(
        &self,
        witness: &IrsConfig<Identity<F>>,
    ) -> Result<ZookFinalOpeningSoundness, ZookSecurityError> {
        ZookFinalOpeningSoundness::for_codes(witness, &self.commit)
    }

    pub fn analytic_bits(&self) -> Result<Bits, ZookSecurityError> {
        zook_mask_binding_bits(&self.commit)
    }
}
