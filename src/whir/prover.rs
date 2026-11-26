//! WHIR prover: joint-mask round reduction followed by final opening.

use std::mem::take;

use ark_ff::Zero;
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, RngCore};
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        linear_form::LinearForm,
    },
    buffer::{Buffer, BufferMath, BufferOps},
    transcript::{Codec, DuplexSpongeInterface, ProverMessage, ProverState, VerifierMessage},
    whir::{
        commit::CommittedWitness,
        params::ProtocolConfig,
        round::{reduce_prover, ProverRoundState},
    },
};

/// Values derived by [`ProtocolConfig::prove`] from the prover transcript.
#[derive(Clone, Debug)]
pub struct ProverClaim<F: ark_ff::Field> {
    /// Sumcheck challenges for the original witness dimensions, excluding padding.
    pub evaluation_point: Vec<F>,
    /// Fiat-Shamir RLC coefficients for the input forms.
    pub rlc_coefficients: Vec<F>,
}

impl<M: Embedding + Default> ProtocolConfig<M> {
    /// Prove the supplied linear evaluations against the committed witness.
    #[cfg_attr(feature = "tracing", instrument(skip_all, name = "whir::prove"))]
    pub fn prove<H, R>(
        &self,
        ps: &mut ProverState<H, R>,
        committed: CommittedWitness<M>,
        linear_forms: &[&dyn LinearForm<M::Target>],
        evaluations: &[M::Target],
    ) -> ProverClaim<M::Target>
    where
        M::Target: Codec,
        M::Source: ProverMessage,
        Standard: Distribution<M::Target>,
        H: DuplexSpongeInterface<U = u8>,
        R: RngCore + CryptoRng,
    {
        let reduced = reduce_prover(self, ps, committed, linear_forms, evaluations, None);
        self.prove_final_opening(ps, reduced)
    }

    pub(super) fn prove_final_opening<H, R>(
        &self,
        ps: &mut ProverState<H, R>,
        mut state: ProverRoundState<Identity<M::Target>>,
    ) -> ProverClaim<M::Target>
    where
        M::Target: Codec,
        Standard: Distribution<M::Target>,
        H: DuplexSpongeInterface<U = u8>,
        R: RngCore + CryptoRng,
    {
        let basecase = self.basecase();
        let suffix = if self.params().mode.is_zk() {
            self.blind_final_claim(ps, &mut state);
            let (value, opening) = basecase.sumcheck().prove(
                ps,
                &Identity::new(),
                state.message,
                &mut state.covector,
                &mut state.sum,
                &[],
            );
            assert!(
                !value.to_slice()[0].is_zero(),
                "final witness evaluation is zero"
            );
            opening.round_challenges
        } else {
            assert!(state.masks.is_none());
            basecase
                .prove(
                    ps,
                    state.message,
                    &state.irs_witness,
                    state.covector,
                    state.sum,
                )
                .evaluation_points
        };
        state.challenges.extend(suffix);
        let dimensions = self.params().size.trailing_zeros() as usize;
        let padding = state.challenges.len() - dimensions;
        ProverClaim {
            evaluation_point: state.challenges.split_off(padding),
            rlc_coefficients: state.claim_weights,
        }
    }

    /// Reveal the final witness and joint masks combined with fresh full-length blinders.
    ///
    /// §5.4, “The final opening protocol”: `H' = H + lambda*p`, `M' = M + lambda*m`,
    /// where `p` is the final masked witness, `m` the joint masks, and `H, M` the blinders.
    /// Send `d = <H, K> + B(M)`, then reduce `<H', K> = d + lambda*c - B(M')`
    /// with Standard sumcheck, where `c` is the accumulated claim and `K` its kernel.
    fn blind_final_claim<H, R>(
        &self,
        ps: &mut ProverState<H, R>,
        state: &mut ProverRoundState<Identity<M::Target>>,
    ) where
        M::Target: Codec,
        Standard: Distribution<M::Target>,
        H: DuplexSpongeInterface<U = u8>,
        R: RngCore + CryptoRng,
    {
        let (Some(layout), Some(masks)) = (self.joint_masks(), state.masks.as_mut()) else {
            panic!("final opening requires the joint mask commitment");
        };
        let commit = self.final_commit();
        let mask_commit = &layout.commit;
        let mut witness_blinder = Buffer::random(ps.rng(), commit.vector_size());
        let mut mask_blinder = take(&mut masks.joint.opening_blinder);
        let mut witness_blind_commit = commit.commit(ps, &[&witness_blinder]);
        let blind_sum =
            witness_blinder.dot(&state.covector) + masks.claims.evaluate(mask_blinder.to_slice());
        ps.prover_message(&blind_sum);
        for value in masks.binding.evaluate(layout, mask_blinder.to_slice()) {
            ps.prover_message(&value);
        }
        let lambda = ps.verifier_message::<M::Target>();
        state
            .message
            .mixed_scalar_mul_add_to(&Identity::new(), &mut witness_blinder, lambda);
        masks.joint.coefficients.mixed_scalar_mul_add_to(
            &Identity::new(),
            &mut mask_blinder,
            lambda,
        );
        ps.prover_messages(witness_blinder.to_slice());
        ps.prover_messages(mask_blinder.to_slice());
        let _ = commit.open(ps, &[&witness_blind_commit, &state.irs_witness]);
        let _ = mask_commit.open(ps, &[&masks.joint.irs_witness]);
        witness_blind_commit.matrix.wipe();
        state.sum = blind_sum + lambda * state.sum - masks.claims.evaluate(mask_blinder.to_slice());
        state.message.wipe();
        state.message = witness_blinder;
        debug_assert_eq!(state.message.dot(&state.covector), state.sum);
    }
}
