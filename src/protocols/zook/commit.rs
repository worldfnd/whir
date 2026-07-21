//! Initial witness commitment for zook.
//!
//! Goes through `rounds[0].code_switch.source` when the plan has rounds;
//! through `basecase.commit` when it doesn't.

use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, RngCore};
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::{
    algebra::{embedding::Embedding, lift},
    buffer::Buffer,
    hash::Hash,
    protocols::{
        irs_commit::{Commitment as IrsCommitment, Witness as IrsWitness},
        params::protocol_config::ProtocolConfig,
    },
    transcript::{
        Codec, DuplexSpongeInterface, ProverMessage, ProverState, VerificationResult, VerifierState,
    },
};

/// Prover handle from [`ProtocolConfig::commit`]; consumed by `prove`.
#[must_use]
#[derive(Clone, Debug)]
pub struct CommittedWitness<M: Embedding> {
    pub(crate) state: CommittedState<M>,
}

/// Internal: the two branches differ in their IRS witness field type.
#[derive(Clone, Debug)]
pub(crate) enum CommittedState<M: Embedding> {
    /// Plan has ≥ 1 round; committed through `rounds[0].code_switch.source`.
    Round {
        message: Vec<M::Target>,
        irs_witness: IrsWitness<M::Source>,
    },
    /// Basecase-only plan; witness was lifted into `M::Target` first.
    Basecase {
        message: Vec<M::Target>,
        irs_witness: IrsWitness<M::Target>,
    },
}

/// Verifier handle from [`ProtocolConfig::receive_commitment`]; consumed by `verify`.
#[must_use]
#[derive(Clone, Debug)]
pub struct Commitment {
    pub(crate) irs_commitment: IrsCommitment,
}

impl<M: Embedding + Default> ProtocolConfig<M> {
    /// Commit the initial witness to the protocol's first IRS codeword.
    #[cfg_attr(feature = "tracing", instrument(skip_all, name = "zook::commit", fields(vector_size = self.tuning().vector_size, num_rounds = self.rounds().len())))]
    pub fn commit<H, R>(
        &self,
        ps: &mut ProverState<H, R>,
        witness: &[M::Source],
    ) -> CommittedWitness<M>
    where
        Standard: Distribution<M::Source> + Distribution<M::Target>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        M::Target: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(
            witness.len(),
            self.tuning().vector_size,
            "zook witness length",
        );

        let state = if let Some(round) = self.rounds().first() {
            let witness_buffer = Buffer::from(witness);
            let irs_witness = round
                .code_switch()
                .source()
                .commit(ps, &[&witness_buffer]);
            let message = lift(round.code_switch().source().embedding(), witness);
            CommittedState::Round {
                message,
                irs_witness,
            }
        } else {
            // Basecase IRS is over `M::Target`; lift before committing.
            let embedding = M::default();
            let message = lift(&embedding, witness);
            let message_buffer = Buffer::from(message.as_slice());
            let irs_witness = self.basecase().commit().commit(ps, &[&message_buffer]);
            CommittedState::Basecase {
                message,
                irs_witness,
            }
        };
        CommittedWitness { state }
    }

    /// Verifier mirror of [`Self::commit`].
    #[cfg_attr(feature = "tracing", instrument(skip_all, name = "zook::receive_commitment", fields(vector_size = self.tuning().vector_size, num_rounds = self.rounds().len())))]
    pub fn receive_commitment<H>(&self, vs: &mut VerifierState<H>) -> VerificationResult<Commitment>
    where
        H: DuplexSpongeInterface,
        M::Target: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let irs_commitment = match self.rounds().first() {
            Some(round) => round.code_switch().source().receive_commitment(vs)?,
            None => self.basecase().commit().receive_commitment(vs)?,
        };
        Ok(Commitment { irs_commitment })
    }
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::{
        algebra::random_vector,
        hash,
        protocols::params::{
            spec::{
                DecodingRegime, FoldingFactor, Mode, PowBudget, RateSchedule, SecuritySpec,
                TuningSpec,
            },
            test_utils::TestEmbedding,
        },
        transcript::{codecs::Empty, DomainSeparator},
    };

    type F = <TestEmbedding as Embedding>::Source;

    /// Keep PoW below the 60-bit cap during `derive` for small test tunings.
    const TEST_TARGET_BITS: u32 = 40;

    fn test_spec(mode: Mode) -> SecuritySpec {
        SecuritySpec {
            mode,
            decoding_regime: DecodingRegime::Johnson,
            target_security_bits: TEST_TARGET_BITS,
            pow_budget: PowBudget::per_slot(10),
            hash_id: hash::BLAKE3,
        }
    }

    /// Vector size large enough for ≥ 1 round under folding_factor 2.
    fn tuning_with_rounds() -> TuningSpec {
        TuningSpec {
            vector_size: 1 << 8,
            starting_log_inv_rate: 1,
            folding_factor: FoldingFactor::Constant(2),
            rate_schedule: RateSchedule::Stepping,
        }
    }

    /// Vector size below `2 · folding_factor`; `round_layout` admits 0 rounds.
    fn tuning_basecase_only() -> TuningSpec {
        TuningSpec {
            vector_size: 1 << 3,
            starting_log_inv_rate: 1,
            folding_factor: FoldingFactor::Constant(2),
            rate_schedule: RateSchedule::Stepping,
        }
    }

    fn roundtrip(
        config: &ProtocolConfig<TestEmbedding>,
        seed: u64,
    ) -> CommittedWitness<TestEmbedding> {
        let mut rng = StdRng::seed_from_u64(seed);
        let witness = random_vector::<F>(&mut rng, config.tuning().vector_size);

        let ds = DomainSeparator::protocol(&"zook-commit-test")
            .session(&format!("commit roundtrip {}:{}", file!(), line!()))
            .instance(&Empty);

        let mut prover_state = ProverState::new_std(&ds);
        let committed = config.commit(&mut prover_state, &witness);
        let proof = prover_state.proof();

        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let _ = config
            .receive_commitment(&mut verifier_state)
            .expect("receive_commitment");
        verifier_state
            .check_eof()
            .expect("transcript fully consumed");

        committed
    }

    #[test]
    fn commit_receive_roundtrip_with_rounds_zk() {
        let config = ProtocolConfig::<TestEmbedding>::derive(
            test_spec(Mode::ZeroKnowledge),
            tuning_with_rounds(),
        )
        .unwrap();
        assert!(!config.rounds().is_empty());
        let committed = roundtrip(&config, 0);
        assert!(matches!(committed.state, CommittedState::Round { .. }));
    }

    #[test]
    fn commit_receive_roundtrip_with_rounds_standard() {
        let config = ProtocolConfig::<TestEmbedding>::derive(
            test_spec(Mode::Standard),
            tuning_with_rounds(),
        )
        .unwrap();
        let committed = roundtrip(&config, 1);
        assert!(matches!(committed.state, CommittedState::Round { .. }));
    }

    #[test]
    fn commit_receive_roundtrip_basecase_only() {
        let config = ProtocolConfig::<TestEmbedding>::derive(
            test_spec(Mode::ZeroKnowledge),
            tuning_basecase_only(),
        )
        .unwrap();
        assert!(config.rounds().is_empty());
        let committed = roundtrip(&config, 2);
        assert!(matches!(committed.state, CommittedState::Basecase { .. }));
    }

    #[test]
    #[should_panic(expected = "zook witness length")]
    fn commit_rejects_wrong_witness_size() {
        let config = ProtocolConfig::<TestEmbedding>::derive(
            test_spec(Mode::ZeroKnowledge),
            tuning_with_rounds(),
        )
        .unwrap();
        let mut rng = StdRng::seed_from_u64(3);
        let too_short = random_vector::<F>(&mut rng, config.tuning().vector_size - 1);

        let ds = DomainSeparator::protocol(&"zook-commit-test")
            .session(&format!("wrong size {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new_std(&ds);
        let _ = config.commit(&mut prover_state, &too_short);
    }
}
