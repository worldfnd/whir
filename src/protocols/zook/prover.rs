//! Zook prover: joint-mask round reduction followed by final opening.

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
    hash::Hash,
    protocols::{
        params::ProtocolConfig,
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
        let reduced = reduce_prover(self, ps, committed, None, None, linear_forms, evaluations);
        self.prove_final_opening(ps, reduced, self.params().size)
    }

    pub(super) fn prove_final_opening<H, R>(
        &self,
        ps: &mut ProverState<H, R>,
        mut state: ProverRoundState<Identity<M::Target>>,
        original_size: usize,
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
        let basecase = self.basecase();
        let suffix = if self.params().zk {
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
            assert!(state.masks.is_empty());
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
        let dimensions = original_size.trailing_zeros() as usize;
        let padding = state.challenges.len() - dimensions;
        ProverClaim {
            evaluation_point: state.challenges.split_off(padding),
            rlc_coefficients: state.claim_weights,
        }
    }

    /// Reveal the final witness and each mask vector combined with its full-length blinder.
    /// Send `d = <H, K> + sum_i B_i(U_i)` before the combination challenge.
    /// The revealed vectors are `H' = H + lambda*p` and `m_i' = U_i + lambda*m_i`.
    /// Reduce `<H', K> = d + lambda*c - sum_i B_i(m_i')` with Standard sumcheck,
    /// where `c` is the accumulated claim and `K` its kernel.
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
        let layout = self.joint_masks().expect("final opening mask layout");
        assert!(!state.masks.is_empty());
        let commit = self.final_commit();
        let mask_commit = &layout.commit;
        let mut witness_blinder = Buffer::random(ps.rng(), commit.vector_size());
        let mut mask_blinders: Vec<_> = state
            .masks
            .iter_mut()
            .map(|masks| take(&mut masks.joint.opening_blinder))
            .collect();
        let mut witness_blind_commit = commit.commit(ps, &[&witness_blinder]);
        let blind_sum = witness_blinder.dot(&state.covector)
            + state
                .masks
                .iter()
                .zip(&mask_blinders)
                .map(|(masks, blinder)| masks.claims.evaluate(blinder.to_slice()))
                .sum::<M::Target>();
        ps.prover_message(&blind_sum);
        for (masks, blinder) in state.masks.iter().zip(&mask_blinders) {
            for value in masks.binding.evaluate(layout, blinder.to_slice()) {
                ps.prover_message(&value);
            }
        }
        let lambda = ps.verifier_message::<M::Target>();
        state
            .message
            .mixed_scalar_mul_add_to(&Identity::new(), &mut witness_blinder, lambda);
        for (masks, blinder) in state.masks.iter().zip(&mut mask_blinders) {
            masks
                .joint
                .coefficients
                .mixed_scalar_mul_add_to(&Identity::new(), blinder, lambda);
        }
        ps.prover_messages(witness_blinder.to_slice());
        for blinder in &mask_blinders {
            ps.prover_messages(blinder.to_slice());
        }
        let _ = commit.open(ps, &[&witness_blind_commit, &state.irs_witness]);
        // Query all mask columns at common indices: they form one interleaved code.
        let roots: Vec<_> = state
            .masks
            .iter()
            .map(|masks| &masks.joint.irs_witness)
            .collect();
        let _ = mask_commit.open(ps, &roots);
        witness_blind_commit.matrix.wipe();
        state.sum = blind_sum + lambda * state.sum
            - state
                .masks
                .iter()
                .zip(&mask_blinders)
                .map(|(masks, blinder)| masks.claims.evaluate(blinder.to_slice()))
                .sum::<M::Target>();
        for blinder in &mut mask_blinders {
            blinder.wipe();
        }
        state.message.wipe();
        state.message = witness_blinder;
        debug_assert_eq!(state.message.dot(&state.covector), state.sum);
    }
}
