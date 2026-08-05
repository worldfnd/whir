use std::fmt::{Debug, Display};

use serde::{Deserialize, Serialize};

use crate::{engines::EngineId, protocols::params::DecodingRegime};

/// Configuration parameters for WHIR proofs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolParameters {
    /// Reed–Solomon decoding regime: `Unique` or `Johnson` (list-decoding).
    pub decoding_regime: DecodingRegime,
    /// The logarithmic inverse rate for sampling.
    pub starting_log_inv_rate: usize,
    /// Folding factor for the initial round.
    pub initial_folding_factor: usize,
    /// Folding factor for rounds after the initial round.
    pub folding_factor: usize,
    /// The security level in bits.
    pub security_level: usize,
    /// The maximum number of bits required for proof-of-work (PoW).
    pub pow_bits: usize,
    /// Number of vectors committed in the batch.
    pub batch_size: usize,
    /// Hash function identifier.
    pub hash_id: EngineId,
}

impl Display for ProtocolParameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Targeting {}-bits of security with {}-bits of PoW using {} decoding",
            self.security_level, self.pow_bits, self.decoding_regime,
        )?;
        writeln!(
            f,
            "Starting rate: 2^-{}, initial_folding_factor: {}, folding_factor: {}",
            self.starting_log_inv_rate, self.initial_folding_factor, self.folding_factor,
        )
    }
}
