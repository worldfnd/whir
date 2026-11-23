//! Round loop: prover and verifier reduce linear-form claims through WHIR folding rounds, carrying ZK masks when enabled.

use ark_ff::{AdditiveGroup, Field};
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, RngCore};
use zeroize::Zeroizing;

use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        geometric_sequence,
        linear_form::LinearForm,
    },
    buffer::{Buffer, BufferMath, BufferOps},
    hash::Hash,
    protocols::{
        code_switch::Claim,
        irs_commit::{Commitment as IrsCommitment, Config, Witness as IrsWitness},
    },
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState,
        VerificationResult, VerifierMessage, VerifierState,
    },
    whir::{
        commit::{Commitment, CommittedState, CommittedWitness},
        kernel::Kernel,
        params::{JointMaskConfig, ProtocolConfig, RoundConfig, WitnessMaskConfig},
        zk::{overlay_mask, sumcheck_blinding, MaskClaims, ProverMasks, VerifierMasks},
    },
};

/// The sum always includes the mask contribution; only the witness is folded.
pub(super) struct ProverRoundState<M: Embedding> {
    pub(super) message: Buffer<M::Source>,
    pub(super) irs_witness: IrsWitness<M::Source>,
    pub(super) covector: Buffer<M::Target>,
    pub(super) sum: M::Target,
    pub(super) masks: Option<ProverMasks<M::Target>>,
    pub(super) challenges: Vec<M::Target>,
    pub(super) claim_weights: Vec<M::Target>,
}

pub(super) struct VerifierRoundState<'a, F: Field> {
    pub(super) irs: IrsCommitment,
    pub(super) kernel: Kernel<'a, F>,
    pub(super) sum: F,
    pub(super) masks: Option<VerifierMasks<F>>,
}

/// Reduce the initial evaluation claims to a final masked witness and mask functional.
pub(super) fn reduce_prover<M, H, R>(
    config: &ProtocolConfig<M>,
    ps: &mut ProverState<H, R>,
    committed: CommittedWitness<M>,
    forms: &[&dyn LinearForm<M::Target>],
    evaluations: &[M::Target],
    second: Option<(&Config<M>, IrsWitness<M::Source>)>,
) -> ProverRoundState<Identity<M::Target>>
where
    M: Embedding + Default,
    M::Target: Codec<[H::U]>,
    Standard: Distribution<M::Target>,
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
    u8: Decoding<[H::U]>,
    [u8; 32]: Decoding<[H::U]>,
    U64: Codec<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    assert_eq!(
        forms.len(),
        evaluations.len(),
        "linear_forms.len() != evaluations.len()"
    );
    assert!(!forms.is_empty(), "whir requires ≥ 1 linear form");
    let claim_weights = geometric_sequence(M::Target::ONE, ps.verifier_message(), forms.len());
    let mut covector = vec![M::Target::ZERO; config.padded_vector_size()];
    for (form, &weight) in forms.iter().zip(&claim_weights) {
        assert!(form.size() <= config.params().size);
        form.accumulate(&mut covector[..form.size()], weight);
    }
    let sum = evaluations
        .iter()
        .zip(&claim_weights)
        .map(|(&value, &weight)| value * weight)
        .sum();
    let masks = match (
        config.joint_masks(),
        config.witness_mask(0),
        committed.joint_masks,
        committed.mask_binding,
    ) {
        (Some(layout), Some(mask), Some(joint), Some(binding)) => {
            let mut claims = MaskClaims::new(layout.commit.vector_size());
            claims.subtract_witness_mask(
                layout.witness_masks[0].clone(),
                mask,
                config.padded_vector_size() / mask.num_polynomials,
                |index| covector[index],
            );
            Some(ProverMasks {
                joint,
                binding,
                claims,
            })
        }
        (None, None, None, None) => None,
        _ => panic!("joint mask commitment does not match the configuration"),
    };
    let covector = Buffer::from(covector);
    let mut state = match (committed.state, config.first_round()) {
        (
            CommittedState::Basecase {
                message,
                irs_witness,
            },
            None,
        ) => ProverRoundState {
            message,
            irs_witness,
            covector,
            sum,
            masks,
            challenges: Vec::new(),
            claim_weights,
        },
        (
            CommittedState::Round {
                message,
                irs_witness,
            },
            Some(first),
        ) => prove_round(
            &first,
            second,
            ProverRoundState::<M> {
                message,
                irs_witness,
                covector,
                sum,
                masks,
                challenges: Vec::new(),
                claim_weights,
            },
            config.joint_masks().zip(config.witness_mask(1)),
            ps,
        ),
        _ => panic!("committed witness does not match the round configuration"),
    };
    for round in config.tail_rounds() {
        state = prove_round(
            &round,
            None,
            state,
            config
                .joint_masks()
                .zip(config.witness_mask(round.index + 1)),
            ps,
        );
    }
    state
}

/// Replay the rounds without materializing a witness-sized verifier kernel.
pub(super) fn reduce_verifier<'a, M, H>(
    config: &ProtocolConfig<M>,
    vs: &mut VerifierState<H>,
    commitment: Commitment<M::Target>,
    forms: &'a [&'a dyn LinearForm<M::Target>],
    evaluations: &[M::Target],
    second: Option<(&Config<M>, &IrsCommitment)>,
) -> VerificationResult<VerifierRoundState<'a, M::Target>>
where
    M: Embedding + Default,
    M::Target: Codec<[H::U]>,
    Standard: Distribution<M::Target>,
    H: DuplexSpongeInterface,
    u8: Decoding<[H::U]>,
    [u8; 32]: Decoding<[H::U]>,
    U64: Codec<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    assert_eq!(
        forms.len(),
        evaluations.len(),
        "linear_forms.len() != evaluations.len()"
    );
    assert!(!forms.is_empty(), "whir requires ≥ 1 linear form");
    assert!(forms.iter().all(|form| form.size() <= config.params().size));
    let claim_weights = geometric_sequence(M::Target::ONE, vs.verifier_message(), forms.len());
    let sum = evaluations
        .iter()
        .zip(&claim_weights)
        .map(|(&value, &weight)| value * weight)
        .sum();
    let mut kernel = Kernel::new(forms, claim_weights, config.padded_vector_size());
    let masks = match (
        config.joint_masks(),
        config.witness_mask(0),
        commitment.joint_masks,
        commitment.mask_binding,
    ) {
        (Some(layout), Some(mask), Some(joint), Some(binding)) => {
            let mut claims = MaskClaims::new(layout.commit.vector_size());
            let coefficients = kernel.mask_coefficients(mask);
            let message_length = config.padded_vector_size() / mask.num_polynomials;
            claims.subtract_witness_mask(
                layout.witness_masks[0].clone(),
                mask,
                message_length,
                |index| {
                    coefficients[index / message_length * mask.coefficients_per_polynomial
                        + index % message_length]
                },
            );
            Some(VerifierMasks {
                joint,
                binding,
                claims,
            })
        }
        (None, None, None, None) => None,
        _ => panic!("joint mask commitment does not match the configuration"),
    };
    let mut state = VerifierRoundState {
        irs: commitment.irs,
        kernel,
        sum,
        masks,
    };
    if let Some(first) = config.first_round() {
        verify_round(
            &first,
            second,
            &mut state,
            config.joint_masks().zip(config.witness_mask(1)),
            vs,
        )?;
    }
    for round in config.tail_rounds() {
        verify_round(
            &round,
            None,
            &mut state,
            config
                .joint_masks()
                .zip(config.witness_mask(round.index + 1)),
            vs,
        )?;
    }
    Ok(state)
}

/// Section 5.4, “ZO0K”, Steps 1–4: carry masks, fold, overlay, then aggregate.
fn prove_round<M, H, R>(
    round: &RoundConfig<M>,
    second: Option<(&Config<M>, IrsWitness<M::Source>)>,
    mut state: ProverRoundState<M>,
    mask_plan: Option<(&JointMaskConfig<M::Target>, WitnessMaskConfig)>,
    ps: &mut ProverState<H, R>,
) -> ProverRoundState<Identity<M::Target>>
where
    M: Embedding,
    M::Target: Codec<[H::U]>,
    Standard: Distribution<M::Target>,
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
    u8: Decoding<[H::U]>,
    [u8; 32]: Decoding<[H::U]>,
    U64: Codec<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    assert_eq!(state.masks.is_some(), mask_plan.is_some());
    let index = round.index;
    let remaining_size = round.sumcheck.final_size();
    let carried = state.masks.as_ref().map_or(M::Target::ZERO, |masks| {
        masks.claims.evaluate(masks.joint.coefficients.to_slice())
    });
    let blinding = match state.masks.as_ref().zip(mask_plan) {
        Some((masks, (layout, _))) => sumcheck_blinding(
            &masks.joint.coefficients.to_slice()[layout.sumcheck_masks[index].clone()],
            remaining_size,
        ),
        None => Zeroizing::new(Vec::new()),
    };
    // The generic prover accepts the dot and carried terms separately.
    state.sum -= carried;
    let (mut message, opening) = round.sumcheck.prove_with_carried_sum(
        ps,
        round.code_switch.source().embedding(),
        state.message,
        &mut state.covector,
        &mut state.sum,
        &blinding,
        carried,
    );
    drop(blinding);
    state.covector.scalar_mul(opening.mask_rlc);
    state
        .challenges
        .extend_from_slice(&opening.round_challenges);
    if let Some((masks, (layout, next_mask))) = state.masks.as_mut().zip(mask_plan) {
        masks.claims.fold_sumcheck(
            layout.sumcheck_masks[index].clone(),
            remaining_size,
            &opening,
        );
        message = overlay_mask(
            message,
            &masks.joint.coefficients.to_slice()[layout.witness_masks[index + 1].clone()],
            next_mask,
        );
    }
    let (source, witnesses) = match second {
        Some((source, right)) => (source, vec![state.irs_witness, right]),
        None => (round.code_switch.source(), vec![state.irs_witness]),
    };
    let switched = round.code_switch.prove(
        ps,
        message,
        source,
        witnesses,
        Claim {
            covector: &mut state.covector,
            sum: &mut state.sum,
        },
        &opening.round_challenges,
        &[],
    );
    if let Some((masks, (layout, next_mask))) = state.masks.as_mut().zip(mask_plan) {
        masks.claims.subtract_target_mask(
            layout.witness_masks[index + 1].clone(),
            next_mask,
            round.code_switch.target().message_length(),
            &switched.update,
            |coefficient| state.covector.to_slice()[coefficient],
        );
    }
    debug_assert_eq!(
        state.sum,
        switched.message.dot(&state.covector)
            + state.masks.as_ref().map_or(M::Target::ZERO, |masks| masks
                .claims
                .evaluate(masks.joint.coefficients.to_slice()))
    );
    ProverRoundState {
        message: switched.message,
        irs_witness: switched.target_witness,
        covector: state.covector,
        sum: state.sum,
        masks: state.masks,
        challenges: state.challenges,
        claim_weights: state.claim_weights,
    }
}

fn verify_round<M, H>(
    round: &RoundConfig<M>,
    second: Option<(&Config<M>, &IrsCommitment)>,
    state: &mut VerifierRoundState<M::Target>,
    mask_plan: Option<(&JointMaskConfig<M::Target>, WitnessMaskConfig)>,
    vs: &mut VerifierState<H>,
) -> VerificationResult<()>
where
    M: Embedding,
    M::Target: Codec<[H::U]>,
    Standard: Distribution<M::Target>,
    H: DuplexSpongeInterface,
    u8: Decoding<[H::U]>,
    [u8; 32]: Decoding<[H::U]>,
    U64: Codec<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    assert_eq!(state.masks.is_some(), mask_plan.is_some());
    let index = round.index;
    let opening = round.sumcheck.verify(vs, &mut state.sum)?;
    state.kernel.fold(&opening);
    if let Some((masks, (layout, _))) = state.masks.as_mut().zip(mask_plan) {
        masks.claims.fold_sumcheck(
            layout.sumcheck_masks[index].clone(),
            state.kernel.size,
            &opening,
        );
    }
    let (source, roots) = match second {
        Some((source, right)) => (source, vec![&state.irs, right]),
        None => (round.code_switch.source(), vec![&state.irs]),
    };
    let (irs, update) = round.code_switch.verify_for_implicit(
        vs,
        &mut state.sum,
        &opening.round_challenges,
        source,
        &roots,
    )?;
    state.kernel.aggregate(&update);
    if let Some((masks, (layout, next_mask))) = state.masks.as_mut().zip(mask_plan) {
        let coefficients = state.kernel.mask_coefficients(next_mask);
        let message_length = round.code_switch.target().message_length();
        masks.claims.subtract_target_mask(
            layout.witness_masks[index + 1].clone(),
            next_mask,
            message_length,
            &update,
            |index| {
                coefficients[index / message_length * next_mask.coefficients_per_polynomial
                    + index % message_length]
            },
        );
    }
    state.irs = irs;
    Ok(())
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::{
        algebra::{
            embedding::Basefield,
            fields::Field64_2,
            linear_form::{Evaluate, MultilinearExtension, UnivariateEvaluation},
            random_vector, univariate_evaluate,
        },
        decoding::DecodingRegime,
        transcript::{codecs::Empty, DomainSeparator},
        whir::params::{Mode, Params, RateSchedule},
    };

    fn roundtrip<M: Embedding<Target = Field64_2> + Default>(
        mode: Mode,
        size: usize,
        univariate_size: usize,
    ) where
        Standard: Distribution<M::Source> + Distribution<M::Target>,
    {
        let config = ProtocolConfig::<M>::derive(Params {
            bits: 80,
            mode,
            regime: DecodingRegime::Johnson,
            pow: 0,
            hash: crate::hash::BLAKE3,
            size,
            rate: 2,
            folds: [2, 2],
            schedule: RateSchedule::Stepping,
        })
        .unwrap();
        if size == 4096 {
            assert!(config.num_rounds() >= 2);
        }
        let mut rng = StdRng::seed_from_u64(43);
        let witness = Buffer::<M::Source>::random(&mut rng, size);
        let multilinear = MultilinearExtension::new(random_vector::<Field64_2>(
            &mut rng,
            size.trailing_zeros() as usize,
        ));
        let univariate = UnivariateEvaluation::new(Field64_2::from(7), univariate_size);
        let values = [
            multilinear.evaluate(&M::default(), witness.to_slice()),
            univariate.evaluate(&M::default(), &witness.to_slice()[..univariate_size]),
        ];
        let forms: [&dyn LinearForm<Field64_2>; 2] = [&multilinear, &univariate];
        let ds = DomainSeparator::protocol(&config)
            .session(&"joint mask round reduction")
            .instance(&Empty);
        let mut ps = ProverState::new_std(&ds);
        let committed = config.commit(&mut ps, witness);
        let initial = match &committed.state {
            CommittedState::Round { message, .. } => message.mixed_lift(&M::default()),
            CommittedState::Basecase { message, .. } => message.clone(),
        };
        let state = reduce_prover(&config, &mut ps, committed, &forms, &values, None);
        assert_overlays(&config, initial, &state);
        let _ = config
            .basecase()
            .commit()
            .open(&mut ps, &[&state.irs_witness]);
        if let Some((layout, masks)) = config.joint_masks().zip(state.masks.as_ref()) {
            let _ = layout.commit.open(&mut ps, &[&masks.joint.irs_witness]);
        }
        let challenge: Field64_2 = ps.verifier_message();
        let proof = ps.proof();
        let mut vs = VerifierState::new_std(&ds, &proof);
        let commitment = config.receive_commitment(&mut vs).unwrap();
        let received =
            reduce_verifier(&config, &mut vs, commitment, &forms, &values, None).unwrap();
        assert_eq!(state.sum, received.sum);
        assert_eq!(state.challenges, received.kernel.challenges);
        assert_eq!(state.claim_weights, received.kernel.claim_weights);
        let kernel: Vec<_> = (0..received.kernel.size)
            .map(|i| received.kernel.coefficient(i))
            .collect();
        assert_eq!(state.covector.to_slice(), kernel);
        let final_values = config
            .basecase()
            .commit()
            .verify(&mut vs, &[&received.irs])
            .unwrap();
        let block_size = config.basecase().commit().message_length();
        for (row, &point) in final_values
            .matrix
            .to_slice()
            .chunks_exact(state.message.len() / block_size)
            .zip(&final_values.points)
        {
            for (&value, polynomial) in row
                .iter()
                .zip(state.message.to_slice().chunks_exact(block_size))
            {
                assert_eq!(value, univariate_evaluate(polynomial, point));
            }
        }
        assert_mask_claims(&config, &state, &received, &mut vs);
        assert_eq!(challenge, vs.verifier_message::<Field64_2>());
        vs.check_eof().unwrap();
    }

    fn assert_overlays<M: Embedding<Target = Field64_2>>(
        config: &ProtocolConfig<M>,
        mut expected: Buffer<Field64_2>,
        state: &ProverRoundState<Identity<Field64_2>>,
    ) {
        let folds = config
            .first_round()
            .into_iter()
            .map(|r| r.sumcheck.num_rounds())
            .chain(config.tail_rounds().map(|r| r.sumcheck.num_rounds()));
        let mut offset = 0;
        for (index, count) in folds.enumerate() {
            for &challenge in &state.challenges[offset..offset + count] {
                expected.fold(challenge);
            }
            offset += count;
            if let Some((layout, masks)) = config.joint_masks().zip(state.masks.as_ref()) {
                expected = overlay_mask(
                    expected,
                    &masks.joint.coefficients.to_slice()[layout.witness_masks[index + 1].clone()],
                    config.witness_mask(index + 1).unwrap(),
                );
            }
        }
        assert_eq!(expected, state.message);
    }

    fn assert_mask_claims<M: Embedding<Target = Field64_2>>(
        config: &ProtocolConfig<M>,
        state: &ProverRoundState<Identity<Field64_2>>,
        received: &VerifierRoundState<Field64_2>,
        vs: &mut VerifierState,
    ) {
        match (&state.masks, &received.masks, config.joint_masks()) {
            (Some(masks), Some(checked), Some(layout)) => {
                assert_eq!(masks.claims, checked.claims);
                assert_eq!(masks.binding, checked.binding);
                let coefficients = masks.joint.coefficients.to_slice();
                assert_eq!(
                    state.sum,
                    state.message.dot(&state.covector) + checked.claims.evaluate(coefficients)
                );
                let evaluations = layout.commit.verify(vs, &[&checked.joint]).unwrap();
                assert_eq!(evaluations.matrix.len(), 2 * evaluations.points.len());
                for (&point, row) in evaluations
                    .points
                    .iter()
                    .zip(evaluations.matrix.to_slice().chunks_exact(2))
                {
                    assert_eq!(row[0], univariate_evaluate(coefficients, point));
                    assert_eq!(
                        row[1],
                        univariate_evaluate(masks.joint.opening_blinder.to_slice(), point)
                    );
                }
            }
            (None, None, None) => assert_eq!(state.sum, state.message.dot(&state.covector)),
            _ => panic!("prover and verifier mask states differ"),
        }
    }

    #[test]
    fn joint_mask_rounds_with_basefield_witness() {
        for size in [8, 128, 4096] {
            roundtrip::<Basefield<Field64_2>>(Mode::ZeroKnowledge, size, size);
        }
    }

    #[test]
    fn joint_mask_rounds_with_extension_field_witness() {
        for size in [8, 128, 4096] {
            roundtrip::<Identity<Field64_2>>(Mode::ZeroKnowledge, size, size);
        }
    }

    #[test]
    fn standard_rounds_preserve_the_unmasked_claim() {
        for size in [8, 128, 4096] {
            roundtrip::<Basefield<Field64_2>>(Mode::Standard, size, size);
        }
    }

    #[test]
    fn short_univariate_forms_match_dense_kernels() {
        for size in [8, 4096] {
            roundtrip::<Basefield<Field64_2>>(Mode::ZeroKnowledge, size, 3);
            roundtrip::<Identity<Field64_2>>(Mode::ZeroKnowledge, size, 3);
            roundtrip::<Basefield<Field64_2>>(Mode::Standard, size, 3);
        }
    }
}
