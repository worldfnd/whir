//! The atomic round state shared by single-track, pre-merge, and post-merge batched flows.
//!
//! Each flow reduces to a sequence of `prove_whir_round` / `verify_whir_round`
//! calls on a `Block` that carries the round's `(message, covector, sum)` plus
//! the active source(s) being opened. What differs across flows is how the
//! initial `Block` is built (single witness vs. γ-RLC of a bundle) and how
//! many active sources it carries (1 or `t` after a selector merge).

use ark_ff::Field;

use crate::{
    algebra::embedding::Embedding,
    protocols::irs_commit::{Commitment as IrsCommitment, Witness as IrsWitness},
};

/// Prover-side round state.
///
/// The `(covector, sum, theta)` scalars live in the target field `M::Target`;
/// the `message` and the source IRS `witnesses` live in `M::Source` (the
/// round's sumcheck lifts the message into `M::Target` at its first fold).
/// For round 0 the source is the base field; every later round runs over
/// `Identity<M::Target>` (source = target). A round's code-switch consumes the `M::Source` witnesses
/// and yields an `M::Target` one, so a round maps `ProverBlock<M>` to
/// `ProverBlock<Identity<M::Target>>`.
///
/// `witnesses` holds the active source IRS witnesses; in single-track and
/// pre-merge rounds it has length 1 with `theta = [ONE]`. After a selector
/// merge it has length `t` with `theta` from the merge opening.
pub struct ProverBlock<M: Embedding> {
    pub(crate) message: Vec<M::Source>,
    pub(crate) covector: Vec<M::Target>,
    pub(crate) sum: M::Target,
    pub(crate) witnesses: Vec<IrsWitness<M::Source>>,
    pub(crate) theta: Vec<M::Target>,
}

impl<M: Embedding> ProverBlock<M> {
    pub(crate) fn single_source(
        message: Vec<M::Source>,
        covector: Vec<M::Target>,
        sum: M::Target,
        witness: IrsWitness<M::Source>,
    ) -> Self {
        Self {
            message,
            covector,
            sum,
            witnesses: vec![witness],
            theta: vec![<M::Target as Field>::ONE],
        }
    }
}

/// Verifier-side round state.
///
/// Mirror of [`ProverBlock`] on the receive side: holds the active source IRS
/// commitments and the post-sumcheck running `sum`. The verifier never
/// materialises `(message, covector)` — they're folded into the implicit
/// constraint accumulator owned by the caller.
pub struct VerifierBlock<F: Field> {
    pub(crate) sum: F,
    pub(crate) commitments: Vec<IrsCommitment>,
    pub(crate) theta: Vec<F>,
}

impl<F: Field> VerifierBlock<F> {
    pub(crate) fn single_source(sum: F, commitment: IrsCommitment) -> Self {
        Self {
            sum,
            commitments: vec![commitment],
            theta: vec![F::ONE],
        }
    }
}
