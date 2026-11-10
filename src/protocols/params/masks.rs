//! Degree-preserving witness prefixes and the shared mask commitment (§5.4, §6.4).
use std::ops::Range;

use ark_ff::Field;

use super::{
    derive::solve_with_num_vectors,
    security::{final_bits, grind_to_at},
    DeriveError, Params, Result, RoundContext,
};
use crate::{
    algebra::embedding::{Embedding, Identity},
    protocols::irs_commit::{num_in_domain_queries, Config as IrsConfig},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WitnessMaskConfig {
    pub num_polynomials: usize,
    pub coefficients_per_polynomial: usize,
}
impl WitnessMaskConfig {
    pub(crate) fn for_code<M: Embedding>(code: &IrsConfig<M>, incoming: usize) -> Result<Self> {
        let length = code
            .in_domain_samples()
            .checked_add(incoming)
            .ok_or(DeriveError("dimension overflow"))?;
        if code.num_vectors() != 1 || code.mask_length() != 0 {
            return Err(DeriveError("invalid witness mask"));
        }
        if length > code.message_length() {
            return Err(DeriveError("mask does not fit"));
        }
        Ok(Self {
            num_polynomials: code.interleaving_depth(),
            coefficients_per_polynomial: length,
        })
    }
    pub fn num_coefficients(self) -> Result<usize> {
        self.num_polynomials
            .checked_mul(self.coefficients_per_polynomial)
            .ok_or(DeriveError("dimension overflow"))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JointMaskConfig<F: Field> {
    pub commit: IrsConfig<Identity<F>>,
    pub sumcheck_masks: Vec<Range<usize>>,
    pub witness_masks: Vec<Range<usize>>,
    pub blinder: Range<usize>,
    pub basefield_blinder: Range<usize>,
}

/// Reserve queries at rate 1/16, then use actual NTT-rounded queries for the blinder.
pub(super) fn joint_code<F: Field>(
    spec: &Params,
    sumchecks: &[usize],
    witnesses: &[usize],
    final_commit: &IrsConfig<Identity<F>>,
    degree: usize,
) -> Result<JointMaskConfig<F>> {
    let mut end = 0usize;
    let mut allocate = |length: usize| {
        let start = end;
        end = end
            .checked_add(length)
            .ok_or(DeriveError("dimension overflow"))?;
        Ok(start..end)
    };
    let sumcheck_masks = sumchecks
        .iter()
        .map(|&n| allocate(n))
        .collect::<Result<Vec<_>>>()?;
    let witness_masks = witnesses
        .iter()
        .map(|&n| allocate(n))
        .collect::<Result<Vec<_>>>()?;
    let payload = witness_masks
        .last()
        .or_else(|| sumcheck_masks.last())
        .map_or(0, |range| range.end);
    let commit = checked_code(spec, payload, degree, final_commit)?;
    let blinder = allocate(
        commit
            .in_domain_samples()
            .checked_add(1)
            .ok_or(DeriveError("dimension overflow"))?,
    )?;
    let basefield_blinder = allocate(degree)?;
    if end > commit.vector_size() {
        return Err(DeriveError("mask does not fit"));
    }
    Ok(JointMaskConfig {
        commit,
        sumcheck_masks,
        witness_masks,
        blinder,
        basefield_blinder,
    })
}

/// Construction and adaptive search share commitment sizing and all final-opening bounds.
pub(super) fn checked_code<F: Field>(
    spec: &Params,
    payload: usize,
    degree: usize,
    final_commit: &IrsConfig<Identity<F>>,
) -> Result<IrsConfig<Identity<F>>> {
    let queries = num_in_domain_queries(spec.regime, f64::from(spec.bits) + 1.0, 0.0625).get();
    let vector_size = payload
        .checked_add(queries)
        .and_then(|n| n.checked_add(1))
        .and_then(|n| n.checked_add(degree))
        .ok_or(DeriveError("dimension overflow"))?;
    let commit = solve_with_num_vectors(
        spec,
        &RoundContext {
            size: vector_size,
            rate: 4,
            folds: 0,
        },
        2,
    )?;
    let (binding, combination, queries) = final_bits(final_commit, &commit)?;
    for (bits, event) in [
        (binding, "mask binding"),
        (combination, "final combination"),
        (queries, "final queries"),
    ] {
        grind_to_at(spec, bits, event)?;
    }
    Ok(commit)
}
