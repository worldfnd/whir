//! Combined parameters for two staged witness commitments with one mask collection.

use serde::Serialize;

use crate::{
    algebra::embedding::Embedding,
    protocols::{
        irs_commit::Config,
        params::{DeriveError, Params, ProtocolConfig},
    },
};

/// Parameters for two equally sized witnesses committed separately and opened together.
/// The joint opening plan includes a leading selector and one shared mask collection.
#[derive(Clone, Debug, Serialize)]
#[serde(bound = "")]
pub struct JointOpeningConfig<M: Embedding> {
    /// Protocol parameters for the concatenated witnesses, including the selector and shared masks.
    combined: ProtocolConfig<M>,
    /// IRS parameters for each witness: half the combined source columns at the same polynomial length.
    witness_commitment: Config<M>,
}

impl<M: Embedding + Default> JointOpeningConfig<M> {
    /// Derive a joint plan from the desired per-witness size and security settings.
    /// `params.size` is the size of each witness; the first folding block gains
    /// one selector round. Requires an unpadded joint opening plan with a first folding block.
    pub fn derive(mut params: Params) -> Result<Self, DeriveError> {
        if !params.size.is_power_of_two() {
            return Err(DeriveError("vector length must be a power of two"));
        }
        if params.folds.contains(&0) {
            return Err(DeriveError("folding factors must be positive"));
        }
        params.size = params
            .size
            .checked_mul(2)
            .ok_or(DeriveError("dimension overflow"))?;
        params.folds[0] = params.folds[0]
            .checked_add(1)
            .ok_or(DeriveError("dimension overflow"))?;
        let combined = ProtocolConfig::derive(params)?;
        if combined.params().size != combined.padded_vector_size() {
            return Err(DeriveError("unsupported joint opening layout"));
        }
        let witness_commitment = combined
            .first_round()
            .and_then(|round| round.code_switch.source().pair_half())
            .ok_or(DeriveError("unsupported joint opening layout"))?;
        Ok(Self {
            combined,
            witness_commitment,
        })
    }
}

impl<M: Embedding> JointOpeningConfig<M> {
    /// Opening parameters for `[L | R]`, with twice the per-witness size.
    pub const fn combined(&self) -> &ProtocolConfig<M> {
        &self.combined
    }

    /// Commitment parameters for either `L` or `R`.
    /// Each has half the combined source columns, with the same polynomial length.
    pub const fn witness_commitment(&self) -> &Config<M> {
        &self.witness_commitment
    }
}
