//! Zook prover: joint-mask round reduction followed by final opening.

use ark_ff::Zero;
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, RngCore};
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        linear_form::{Evaluate, LinearForm},
    },
    buffer::{Buffer, BufferMath, BufferOps},
    hash::Hash,
    protocols::{
        params::{BasecasePlan, ProtocolConfig},
        zook::{
            commit::CommittedWitness,
            round::{reduce_prover, ProverRoundState},
            ProverClaim,
        },
    },
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState,
        VerifierMessage,
    },
};

impl<M: Embedding + Default> ProtocolConfig<M> {
    /// Prove the supplied linear evaluations against the committed witness.
    #[cfg_attr(feature = "tracing", instrument(skip_all, name = "zook::prove"))]
    pub fn prove<H, R>(
        &self,
        ps: &mut ProverState<H, R>,
        committed: CommittedWitness<M>,
        linear_forms: &[&dyn LinearForm<M::Target>],
        evaluations: &[M::Target],
    ) -> ProverClaim<M::Target>
    where
        M::Target: Codec<[H::U]>,
        Standard: Distribution<M::Target>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let reduced = reduce_prover(self, ps, committed, linear_forms, evaluations);
        self.prove_final_opening(ps, reduced)
    }

    fn prove_final_opening<H, R>(
        &self,
        ps: &mut ProverState<H, R>,
        mut state: ProverRoundState<Identity<M::Target>>,
    ) -> ProverClaim<M::Target>
    where
        M::Target: Codec<[H::U]>,
        Standard: Distribution<M::Target>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let suffix = match self.basecase() {
            BasecasePlan::Standard { config, .. } => {
                assert!(state.masks.is_none());
                config
                    .prove(
                        ps,
                        state.message,
                        &state.irs_witness,
                        state.covector,
                        state.sum,
                    )
                    .evaluation_points
            }
            BasecasePlan::ZeroKnowledge { sumcheck, .. } => {
                self.blind_final_claim(ps, &mut state);
                let (value, opening) = sumcheck.prove(
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
            }
        };
        state.challenges.extend(suffix);
        let dimensions = self.tuning().vector_size.trailing_zeros() as usize;
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
        M::Target: Codec<[H::U]>,
        Standard: Distribution<M::Target>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let (Some(layout), Some(masks)) = (self.joint_masks(), state.masks.as_ref()) else {
            panic!("final opening requires the joint mask commitment");
        };
        let commit = self.basecase().commit();
        let mut witness_blinder = Buffer::random(ps.rng(), commit.vector_size());
        let mut mask_blinder = Buffer::random(ps.rng(), layout.commit().vector_size());
        let mut witness_blind_commit = commit.commit(ps, &[&witness_blinder]);
        let mut mask_blind_commit = layout.commit().commit(ps, &[&mask_blinder]);
        let blind_sum =
            witness_blinder.dot(&state.covector) + masks.claims.evaluate(mask_blinder.to_slice());
        ps.prover_message(&blind_sum);
        for form in &masks.binding.forms {
            ps.prover_message(&form.evaluate(&Identity::new(), mask_blinder.to_slice()));
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
        let _ = layout
            .commit()
            .open(ps, &[&mask_blind_commit, &masks.joint.irs_witness]);
        witness_blind_commit.matrix.wipe();
        mask_blind_commit.matrix.wipe();
        state.sum = blind_sum + lambda * state.sum - masks.claims.evaluate(mask_blinder.to_slice());
        state.message.wipe();
        state.message = witness_blinder;
        debug_assert_eq!(state.message.dot(&state.covector), state.sum);
    }
}
