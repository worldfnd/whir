//! WHIR: a Reed–Solomon proximity-based polynomial commitment scheme with
//! standard and zero-knowledge modes.
//!
//! Start at [`ProtocolConfig`]: derive it from [`Params`], then `commit`,
//! `prove`, `receive_commitment`, and `verify`. [`JointOpeningConfig`] opens two
//! separately committed witnesses together.
//!
//! # Where things live
//!
//! ```text
//! whir/                 the protocol
//!   params/             Params → ProtocolConfig
//!     request.rs        Params, Mode, RateSchedule, ProtocolSpec
//!     derive.rs         ProtocolConfig::derive, per-round construction
//!     security.rs       per-round soundness bits
//!     adaptive_rates.rs RateSchedule::Adaptive planner
//!     masks.rs          ZK mask dimensions
//!     joint_opening.rs  JointOpeningConfig::derive
//!   commit.rs           ProtocolConfig::commit, receive_commitment
//!   prover.rs           ProtocolConfig::prove, ProverClaim
//!   verifier.rs         ProtocolConfig::verify, FinalClaim
//!   round.rs            prover/verifier round loop
//!   kernel.rs           verifier's implicit kernel
//!   zk/                 ZK masks: sampling, overlay, binding, claims
//!   joint_opening.rs    joint commit/prove/verify of two witnesses
//! protocols/            sub-protocols, each with its own Config:
//!                       sumcheck, code_switch, basecase, irs_commit,
//!                       matrix_commit, merkle_tree, mask_proximity,
//!                       proof_of_work; plus challenge samplers
//! decoding.rs           Reed–Solomon decoding regimes and bounds
//! algebra/              fields, embeddings, linear forms, NTT, vector math
//! buffer/               backend-agnostic vector storage
//! hash/, engines.rs     hash engines and their registry
//! transcript/           Fiat–Shamir transcript (spongefish)
//! bits.rs               security levels and error probabilities in bits
//! type_info.rs          type tags checked on deserialization
//! ark_serde.rs          Serde support for ark_ff types
//! utils.rs              iterator and sizing helpers
//! bin/main.rs           command-line demo
//! ```

pub mod algebra;
pub mod ark_serde;
pub mod bits;
pub mod buffer;
mod decoding;
pub mod engines;
pub mod hash;
pub mod protocols;
pub mod transcript;
pub mod type_info;
pub mod utils; // Utils in general
mod whir;

pub use decoding::DecodingRegime;
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
