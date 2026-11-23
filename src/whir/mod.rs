//! WHIR opening in standard and zero-knowledge modes.
//! Zero-knowledge mode uses overlaid witness masks and a joint mask commitment.

mod binding;
mod claims;
mod commit;
mod joint;
pub mod params;
mod prover;
mod round;
mod verifier;

#[cfg(test)]
mod tests;

pub use commit::{Commitment, CommittedWitness};
pub use joint::{FirstCommitment, FirstWitness, JointCommitment, JointWitness};
pub use prover::ProverClaim;
pub use verifier::FinalClaim;
