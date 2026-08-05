//! Per-bundle commitment for the batched protocol.
//!
//! Each bundle is committed independently via [`BatchedProtocolConfig::commit_bundle`]:
//! its polys are flattened into one vector, committed under the bundle's IRS config,
//! and its intra-bundle `batching_challenge` is sampled at commit time. Reuses the shared
//! [`CommittedState`] from [`super::super::commit`].

use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, RngCore};
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::{
    algebra::embedding::Embedding,
    buffer::{Buffer, BufferOps},
    hash::Hash,
    protocols::{
        irs_commit::Commitment as IrsCommitment,
        params::batched::BatchedProtocolConfig,
        zook::{batched::bundle::WitnessBundle, commit::CommittedState},
    },
    transcript::{
        Codec, DuplexSpongeInterface, ProverMessage, ProverState, VerificationResult,
        VerifierMessage, VerifierState,
    },
};

/// Prover handle from [`BatchedProtocolConfig::commit_bundle`]; carries the per-bundle `batching_challenge`.
#[must_use]
#[derive(Clone, Debug)]
pub struct BundleCommittedWitness<M: Embedding> {
    pub(crate) state: CommittedState<M>,
    pub(crate) batching_challenge: M::Target,
}

/// Verifier handle from [`BatchedProtocolConfig::receive_bundle_commitment`]; carries `batching_challenge`.
#[must_use]
#[derive(Clone, Debug)]
pub struct BundleCommitment<F> {
    pub(crate) irs_commitment: IrsCommitment,
    pub(crate) batching_challenge: F,
}

impl<M: Embedding + Default> BatchedProtocolConfig<M> {
    /// Commit one bundle and sample its intra-bundle `batching_challenge`. Pass one handle per bundle to
    /// [`BatchedProtocolConfig::prove`] in bundle-config order.
    #[cfg_attr(
        feature = "tracing",
        instrument(skip_all, name = "zook::commit_bundle")
    )]
    pub fn commit_bundle<H, R>(
        &self,
        ps: &mut ProverState<H, R>,
        bundle_idx: usize,
        bundle: &WitnessBundle<M::Source>,
    ) -> BundleCommittedWitness<M>
    where
        Standard: Distribution<M::Source> + Distribution<M::Target>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        M::Target: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let bundle_cfg = &self.bundle_configs()[bundle_idx];
        assert_eq!(
            bundle.num_polys(),
            bundle_cfg.spec.num_polys,
            "bundle.num_polys does not match the scheduled BundleSpec.num_polys",
        );
        assert_eq!(
            bundle.poly_len(),
            bundle_cfg.spec.length,
            "bundle.poly_len does not match the scheduled BundleSpec.length",
        );

        let mut flat: Vec<M::Source> = Vec::with_capacity(bundle.num_polys() * bundle.poly_len());
        for p in &bundle.polys {
            flat.extend_from_slice(p);
        }
        let flat = Buffer::from(flat);

        let irs_witness = bundle_cfg.irs_config.commit(ps, &[&flat]);
        let batching_challenge: M::Target = ps.verifier_message();

        BundleCommittedWitness {
            state: CommittedState::Round {
                message: flat.into_vec(),
                irs_witness,
            },
            batching_challenge,
        }
    }

    /// Verifier mirror of [`Self::commit_bundle`].
    #[cfg_attr(
        feature = "tracing",
        instrument(skip_all, name = "zook::receive_bundle_commitment")
    )]
    pub fn receive_bundle_commitment<H>(
        &self,
        vs: &mut VerifierState<H>,
        bundle_idx: usize,
    ) -> VerificationResult<BundleCommitment<M::Target>>
    where
        H: DuplexSpongeInterface,
        M::Target: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let bundle_cfg = &self.bundle_configs()[bundle_idx];
        let irs_commitment = bundle_cfg.irs_config.receive_commitment(vs)?;
        let batching_challenge: M::Target = vs.verifier_message();
        Ok(BundleCommitment {
            irs_commitment,
            batching_challenge,
        })
    }
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    use crate::{
        algebra::{embedding::Embedding, random_vector},
        hash,
        protocols::{
            params::{
                batched::{BatchedProtocolConfig, BatchedTuningSpec, BundleSpec},
                spec::{
                    DecodingRegime, FoldingFactor, Mode, PowBudget, RateSchedule, SecuritySpec,
                },
                test_utils::TestEmbedding,
            },
            zook::{
                batched::bundle::{BundleClaim, WitnessBundle},
                commit::CommittedState,
            },
        },
        transcript::{codecs::Empty, DomainSeparator, ProverState, VerifierMessage, VerifierState},
    };

    type F = <TestEmbedding as Embedding>::Source;

    const TEST_TARGET_BITS: u32 = 40;

    fn batched_test_spec() -> SecuritySpec {
        SecuritySpec {
            mode: Mode::ZeroKnowledge,
            decoding_regime: DecodingRegime::Johnson,
            target_security_bits: TEST_TARGET_BITS,
            pow_budget: PowBudget::per_slot(20),
            hash_id: hash::BLAKE3,
        }
    }

    fn batched_tuning(bundles: Vec<BundleSpec>) -> BatchedTuningSpec {
        BatchedTuningSpec {
            bundles,
            starting_log_inv_rate: 1,
            folding_factor: FoldingFactor::Constant(2),
            rate_schedule: RateSchedule::Stepping,
        }
    }

    fn no_claim_bundle(polys: &[Vec<F>]) -> WitnessBundle<'_, F> {
        WitnessBundle {
            polys: polys.iter().map(Vec::as_slice).collect(),
            per_poly_claims: polys.iter().map(|_| Vec::<BundleClaim<F>>::new()).collect(),
        }
    }

    #[test]
    fn commit_bundle_single_bundle_single_poly_roundtrip() {
        let cfg = BatchedProtocolConfig::<TestEmbedding>::derive(
            batched_test_spec(),
            batched_tuning(vec![BundleSpec {
                length: 1 << 8,
                num_polys: 1,
            }]),
        )
        .unwrap();
        let mut rng = StdRng::seed_from_u64(0);
        let polys = vec![random_vector::<F>(&mut rng, 1 << 8)];
        let bundle = no_claim_bundle(&polys);

        let ds = DomainSeparator::protocol(&"zook-commit-bundle-test")
            .session(&format!("single bundle {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut ps = ProverState::new_std(&ds);
        let committed = cfg.commit_bundle(&mut ps, 0, &bundle);
        let proof = ps.proof();

        let mut vs = VerifierState::new_std(&ds, &proof);
        let _commitment = cfg.receive_bundle_commitment(&mut vs, 0).unwrap();
        vs.check_eof().unwrap();
        assert!(matches!(committed.state, CommittedState::Round { .. }));
    }

    #[test]
    fn commit_bundle_multi_poly_roundtrip() {
        let cfg = BatchedProtocolConfig::<TestEmbedding>::derive(
            batched_test_spec(),
            batched_tuning(vec![BundleSpec {
                length: 1 << 6,
                num_polys: 4,
            }]),
        )
        .unwrap();
        let mut rng = StdRng::seed_from_u64(1);
        let polys: Vec<Vec<F>> = (0..4)
            .map(|_| random_vector::<F>(&mut rng, 1 << 6))
            .collect();
        let bundle = no_claim_bundle(&polys);

        let ds = DomainSeparator::protocol(&"zook-commit-bundle-test")
            .session(&format!("multi-poly {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut ps = ProverState::new_std(&ds);
        let _ = cfg.commit_bundle(&mut ps, 0, &bundle);
        let proof = ps.proof();

        let mut vs = VerifierState::new_std(&ds, &proof);
        let _commitment = cfg.receive_bundle_commitment(&mut vs, 0).unwrap();
        vs.check_eof().unwrap();
    }

    #[test]
    fn multiple_bundles_bind_batching_challenge_at_commit_time() {
        let cfg = BatchedProtocolConfig::<TestEmbedding>::derive(
            batched_test_spec(),
            batched_tuning(vec![
                BundleSpec {
                    length: 1 << 6,
                    num_polys: 1,
                };
                3
            ]),
        )
        .unwrap();
        let mut rng = StdRng::seed_from_u64(2);
        let polys_a = vec![random_vector::<F>(&mut rng, 1 << 6)];
        let polys_b = vec![random_vector::<F>(&mut rng, 1 << 6)];
        let polys_c = vec![random_vector::<F>(&mut rng, 1 << 6)];
        let bundle_a = no_claim_bundle(&polys_a);
        let bundle_b = no_claim_bundle(&polys_b);
        let bundle_c = no_claim_bundle(&polys_c);

        let ds = DomainSeparator::protocol(&"zook-commit-bundle-test")
            .session(&format!("interleaved {}:{}", file!(), line!()))
            .instance(&Empty);

        let mut ps = ProverState::new_std(&ds);
        let committed_a = cfg.commit_bundle(&mut ps, 0, &bundle_a);
        let _extra_a: F = ps.verifier_message();
        let committed_b = cfg.commit_bundle(&mut ps, 1, &bundle_b);
        let _extra_b: F = ps.verifier_message();
        let committed_c = cfg.commit_bundle(&mut ps, 2, &bundle_c);
        let proof = ps.proof();

        let mut vs = VerifierState::new_std(&ds, &proof);
        let commitment_a = cfg.receive_bundle_commitment(&mut vs, 0).unwrap();
        let _extra_a_verifier: F = vs.verifier_message();
        let commitment_b = cfg.receive_bundle_commitment(&mut vs, 1).unwrap();
        let _extra_b_verifier: F = vs.verifier_message();
        let commitment_c = cfg.receive_bundle_commitment(&mut vs, 2).unwrap();
        vs.check_eof().unwrap();

        assert_eq!(
            committed_a.batching_challenge,
            commitment_a.batching_challenge
        );
        assert_eq!(
            committed_b.batching_challenge,
            commitment_b.batching_challenge
        );
        assert_eq!(
            committed_c.batching_challenge,
            commitment_c.batching_challenge
        );
    }

    #[test]
    #[should_panic(expected = "bundle.num_polys does not match")]
    fn commit_bundle_rejects_wrong_num_polys() {
        let cfg = BatchedProtocolConfig::<TestEmbedding>::derive(
            batched_test_spec(),
            batched_tuning(vec![BundleSpec {
                length: 1 << 6,
                num_polys: 2,
            }]),
        )
        .unwrap();
        let mut rng = StdRng::seed_from_u64(3);
        let polys = vec![random_vector::<F>(&mut rng, 1 << 6)];
        let bundle = no_claim_bundle(&polys);

        let ds = DomainSeparator::protocol(&"zook-commit-bundle-test")
            .session(&format!("wrong num_polys {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut ps = ProverState::new_std(&ds);
        let _ = cfg.commit_bundle(&mut ps, 0, &bundle);
    }
}
