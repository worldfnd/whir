//! Joint openings reject without advancing either transcript while integration is pending.

use ark_ff::{AdditiveGroup, Field};
use ark_std::rand::{distributions::Standard, prelude::Distribution};
use whir::{
    algebra::{
        embedding::{Basefield, Embedding, Identity},
        fields::{Field256, Field64_3},
        linear_form::Covector,
    },
    buffer::Buffer,
    protocols::{
        params::{FoldingFactor, Mode, PowBudget, RateSchedule, SecuritySpec, TuningSpec},
        zook::ProtocolConfig,
    },
    transcript::{
        codecs::Empty, Codec, DomainSeparator, ProverState, VerifierMessage, VerifierState,
    },
};
use zeroize::Zeroize;

fn rejects_pair_without_transcript_changes<M>(mode: Mode)
where
    M: Embedding + Default,
    M::Target: Default + Zeroize + Codec,
    Standard: Distribution<M::Source> + Distribution<M::Target>,
{
    for vector_size in [8, 256] {
        let config = ProtocolConfig::<M>::derive(
            SecuritySpec::new(128)
                .with_mode(mode)
                .with_pow_budget(PowBudget::Forbidden),
            TuningSpec {
                vector_size,
                starting_log_inv_rate: 2,
                folding_factor: FoldingFactor::Constant(2),
                rate_schedule: RateSchedule::Stepping,
            },
        )
        .unwrap();
        assert!(config.validate_pair().is_err());
        let form = Covector::new(vec![M::Target::ZERO; 2 * vector_size]);
        let ds = DomainSeparator::protocol(&config).instance(&Empty);
        let mut ps = ProverState::new_std(&ds);
        let committed = [
            config.commit(&mut ps, Buffer::from(vec![M::Source::ONE; vector_size])),
            config.commit(&mut ps, Buffer::from(vec![M::Source::ONE; vector_size])),
        ];
        assert!(config
            .prove_pair(&mut ps, committed, &[&form], &[M::Target::ZERO])
            .is_err());
        let challenge: M::Target = ps.verifier_message();
        let sentinel = M::Target::from(987);
        ps.prover_message(&sentinel);
        let proof = ps.proof();

        // Replay without verify_pair to detect writes or challenges added by prove_pair.
        let mut control = VerifierState::new_std(&ds, &proof);
        let _ = config.receive_commitment(&mut control).unwrap();
        let _ = config.receive_commitment(&mut control).unwrap();
        assert_eq!(control.verifier_message::<M::Target>(), challenge);
        assert_eq!(control.prover_message::<M::Target>().unwrap(), sentinel);
        control.check_eof().unwrap();

        // verify_pair must leave the same challenge and following message untouched.
        let mut vs = VerifierState::new_std(&ds, &proof);
        let commitments = [
            config.receive_commitment(&mut vs).unwrap(),
            config.receive_commitment(&mut vs).unwrap(),
        ];
        assert!(config
            .verify_pair(&mut vs, commitments, &[&form], &[M::Target::ZERO])
            .is_err());
        assert_eq!(vs.verifier_message::<M::Target>(), challenge);
        assert_eq!(vs.prover_message::<M::Target>().unwrap(), sentinel);
        vs.check_eof().unwrap();
    }
}

#[test]
fn bn254_standard_pair_is_unsupported() {
    rejects_pair_without_transcript_changes::<Identity<Field256>>(Mode::Standard);
}

#[test]
fn bn254_zero_knowledge_pair_is_unsupported() {
    rejects_pair_without_transcript_changes::<Identity<Field256>>(Mode::ZeroKnowledge);
}

#[test]
fn extension_standard_pair_is_unsupported() {
    rejects_pair_without_transcript_changes::<Basefield<Field64_3>>(Mode::Standard);
}

#[test]
fn extension_zero_knowledge_pair_is_unsupported() {
    rejects_pair_without_transcript_changes::<Basefield<Field64_3>>(Mode::ZeroKnowledge);
}
