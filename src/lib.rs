pub mod algebra;
pub mod ark_serde;
pub mod bits;
pub mod buffer;
pub mod engines;
pub mod hash;
pub mod protocols;
mod soundness;
pub mod transcript;
pub mod type_info;
pub mod type_map;
pub mod utils; // Utils in general
mod whir;

pub use soundness::DecodingRegime;
pub use whir::{
    params::{
        DeriveError, JointOpeningConfig, KneeWeight, Mode, Params, ProtocolConfig, ProtocolSpec,
        RateSchedule,
    },
    Commitment, CommittedWitness, FinalClaim, FirstCommitment, FirstWitness, JointCommitment,
    JointWitness, ProverClaim,
};

#[cfg(test)]
mod test_utils;
