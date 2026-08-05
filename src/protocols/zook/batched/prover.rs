//! Batched Zook prover.
//!
//! Composes the shared atom ([`prove_whir_round`]) with the batched-specific
//! producers and merges:
//!   - [`intro_bundle`] turns each committed bundle into a single-source [`ProverBlock`].
//!   - Each bundle's `pre_merge_rounds` are run independently (single block, no merge).
//!   - The shared round walk drains active bundles into [`merge_active`] (selector-merge
//!     when `t ≥ 2`, pass-through when `t == 1`) and feeds the merged block to the atom.
//!   - The final carrier feeds into the inner `basecase.prove`.

use ark_ff::Field;
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, RngCore};
#[cfg(feature = "tracing")]
use tracing::instrument;
use zeroize::Zeroize;

use crate::{
    algebra::{dot, embedding::Identity},
    buffer::Buffer,
    hash::Hash,
    protocols::{
        irs_commit::Witness as IrsWitness,
        params::{
            batched::{BatchedProtocolConfig, MergeSchedule},
            config::RoundConfig,
        },
        zook::{
            batched::{
                bundle::{build_bundle_claim, WitnessBundle},
                commit::BundleCommittedWitness,
            },
            block::ProverBlock,
            commit::CommittedState,
            round::prove_whir_round,
        },
    },
    transcript::{codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState},
};

impl<F: Field + Default + Zeroize> BatchedProtocolConfig<Identity<F>> {
    /// Prove all per-poly claims across one or more bundles.
    #[cfg_attr(feature = "tracing", instrument(skip_all, name = "zook::batched::prove", fields(num_bundles = bundles.len())))]
    pub fn prove<H, R>(
        &self,
        ps: &mut ProverState<H, R>,
        committed: Vec<BundleCommittedWitness<Identity<F>>>,
        bundles: &[&WitnessBundle<F>],
    ) where
        Standard: Distribution<F>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(
            committed.len(),
            bundles.len(),
            "one committed handle per bundle"
        );
        assert_eq!(
            committed.len(),
            self.bundle_configs().len(),
            "committed bundle count must match the schedule's bundle count",
        );
        assert!(!bundles.is_empty(), "at least one bundle required");

        let claim_counts: Vec<usize> = bundles.iter().map(|b| b.total_claims()).collect();
        self.validate_security_target_met_for_claims(&claim_counts)
            .expect("runtime batched claim counts violate the configured security target");

        let mut bundle_blocks: Vec<Option<ProverBlock<Identity<F>>>> = bundles
            .iter()
            .zip(committed)
            .enumerate()
            .map(|(i, (bundle, cw))| Some(intro_bundle(self, i, bundle, cw)))
            .collect();

        for (i, slot) in bundle_blocks.iter_mut().enumerate() {
            let bcfg = &self.bundle_configs()[i];
            if bcfg.pre_merge_rounds.is_empty() {
                continue;
            }
            let mut block = slot.take().expect("bundle present before pre-merge loop");
            for round in &bcfg.pre_merge_rounds {
                block = prove_whir_round(round, block, ps);
            }
            *slot = Some(block);
        }

        let mut carrier: Option<ProverBlock<Identity<F>>> = None;
        let inner_rounds: Vec<&RoundConfig<Identity<F>>> = self
            .inner()
            .first_round()
            .into_iter()
            .chain(self.inner().tail_rounds())
            .collect();
        for (r, round) in inner_rounds.into_iter().enumerate() {
            let mut active: Vec<ProverBlock<Identity<F>>> = Vec::new();
            if let Some(c) = carrier.take() {
                active.push(c);
            }
            if let Some(join) = self.schedule().join_at(r) {
                for &idx in &join.bundle_indices {
                    let block = bundle_blocks[idx]
                        .take()
                        .expect("each bundle joins exactly once at its scheduled round");
                    active.push(block);
                }
            }
            assert!(
                !active.is_empty(),
                "round {r}: active list must be non-empty (scheduler invariant)"
            );
            if let Some(join) = self.schedule().join_at(r) {
                debug_assert_eq!(join.t, active.len());
            } else {
                debug_assert_eq!(1, active.len());
            }

            let merged = merge_active(self.schedule(), r, active, ps);
            carrier = Some(prove_whir_round(round, merged, ps));
        }

        assert!(
            self.schedule().basecase_join().is_none(),
            "basecase joins are not yet wired by the orchestrator; \
             the scheduler should have rejected this configuration"
        );
        debug_assert!(
            bundle_blocks.iter().all(Option::is_none),
            "every bundle must be pulled exactly once during the round walk"
        );

        let final_block = carrier.expect("at least one round must produce a carrier");
        let ProverBlock {
            message,
            covector,
            sum,
            mut witnesses,
            ..
        } = final_block;
        let _ = self.inner().basecase().prove(
            ps,
            Buffer::from(message),
            &witnesses.remove(0),
            Buffer::from(covector),
            sum,
        );
    }
}

/// Selector-merge the active blocks into one virtual block. Pass-through (with
/// per-source `theta` preserved as `[F::ONE]` of length 1) when `active.len() == 1`.
fn merge_active<F, H, R>(
    schedule: &MergeSchedule<F>,
    round_idx: usize,
    mut active: Vec<ProverBlock<Identity<F>>>,
    ps: &mut ProverState<H, R>,
) -> ProverBlock<Identity<F>>
where
    F: Field + Default + Zeroize + Codec<[H::U]>,
    Standard: Distribution<F>,
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
    [u8; 32]: Decoding<[H::U]>,
    U64: Codec<[H::U]>,
{
    if active.len() == 1 {
        return active.pop().expect("len == 1");
    }
    let join = schedule
        .join_at(round_idx)
        .expect("round with t >= 2 must have a schedule entry");
    let mut messages: Vec<Vec<F>> = Vec::with_capacity(active.len());
    let mut covectors: Vec<Vec<F>> = Vec::with_capacity(active.len());
    let mut sums: Vec<F> = Vec::with_capacity(active.len());
    let mut witnesses: Vec<IrsWitness<F>> = Vec::with_capacity(active.len());
    for mut block in active {
        debug_assert_eq!(
            block.witnesses.len(),
            1,
            "merge input must be single-source blocks"
        );
        messages.push(block.message);
        covectors.push(block.covector);
        sums.push(block.sum);
        witnesses.push(block.witnesses.remove(0));
    }
    let merged = join.selector.prove(ps, &messages, &covectors, &sums);
    ProverBlock {
        message: merged.merged_message,
        covector: merged.merged_covector,
        sum: merged.sum,
        witnesses,
        theta: merged.theta,
    }
}

/// Produce a single-source [`ProverBlock`] from a committed bundle by applying
/// the intra-bundle γ-RLC. The block carries the bundle's IRS witness and the
/// γ-collapsed `(message, covector, sum)`.
///
/// Panics if `committed.state` is [`CommittedState::Basecase`]; bundles always
/// commit through the first WHIR round (basecase-only plans are unsupported by
/// the batched protocol).
#[must_use]
fn intro_bundle<F: Field>(
    cfg: &BatchedProtocolConfig<Identity<F>>,
    bundle_idx: usize,
    bundle: &WitnessBundle<F>,
    committed: BundleCommittedWitness<Identity<F>>,
) -> ProverBlock<Identity<F>> {
    bundle.assert_well_formed();
    let bcfg = &cfg.bundle_configs()[bundle_idx];
    assert_eq!(
        bundle.num_polys(),
        bcfg.spec.num_polys,
        "bundles[{bundle_idx}].num_polys mismatches scheduled BundleSpec",
    );
    assert_eq!(
        bundle.poly_len(),
        bcfg.spec.length,
        "bundles[{bundle_idx}].poly_len mismatches scheduled BundleSpec",
    );

    let reduced_claim = build_bundle_claim(bundle, committed.batching_challenge);

    let (message, irs_witness) = match committed.state {
        CommittedState::Round {
            message,
            irs_witness,
        } => (message, irs_witness),
        CommittedState::Basecase { .. } => panic!(
            "commit_bundle always produces Round state; basecase-only plans are not \
             supported by the batched protocol",
        ),
    };
    debug_assert_eq!(message.len(), reduced_claim.covector.len());
    debug_assert_eq!(dot(&message, &reduced_claim.covector), reduced_claim.sum);

    ProverBlock::single_source(
        message,
        reduced_claim.covector,
        reduced_claim.sum,
        irs_witness,
    )
}
