//! WHIR opening in standard and zero-knowledge modes.
//! Zero-knowledge mode uses overlaid witness masks and a joint mask commitment.

mod commit;
mod joint;
mod kernel;
pub mod params;
mod prover;
mod round;
mod verifier;
mod zk;

#[cfg(test)]
mod tests;

pub use commit::{Commitment, CommittedWitness};
pub use joint::{FirstCommitment, FirstWitness, JointCommitment, JointWitness};
pub use prover::ProverClaim;
pub use verifier::FinalClaim;
