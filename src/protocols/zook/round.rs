//! Joint-mask claim reduction through Zook folding rounds.

use ark_ff::{AdditiveGroup, Field};
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, RngCore};
use zeroize::Zeroizing;

use crate::{
    algebra::{
        embedding::{Embedding, Identity, PublicScalar},
        geometric_accumulate, geometric_sequence,
        linear_form::{LinearForm, UnivariateEvaluation},
    },
    buffer::{Buffer, BufferMath, BufferOps},
    hash::Hash,
    protocols::{
        basecase::Opening,
        code_switch::{Claim, CovectorUpdateParams},
        irs_commit::{Commitment as IrsCommitment, Witness as IrsWitness},
        params::{JointMaskConfig, ProtocolConfig, RoundConfig, WitnessMaskConfig},
        sumcheck::SumcheckOpening,
        zook::{
            binding::MaskBinding,
            claims::{sumcheck_blinding, MaskClaims},
            commit::{
                overlay_mask, Commitment, CommittedState, CommittedWitness, JointMaskWitness,
            },
            evaluate_padded, FinalClaim,
        },
    },
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState,
        VerificationResult, VerifierMessage, VerifierState,
    },
};

pub(super) struct ProverMasks<F: Field> {
    pub(super) joint: JointMaskWitness<F>,
    pub(super) binding: MaskBinding<F>,
    pub(super) claims: MaskClaims<F>,
}

pub(super) struct VerifierMasks<F: Field> {
    pub(super) joint: IrsCommitment,
    pub(super) binding: MaskBinding<F>,
    pub(super) claims: MaskClaims<F>,
}

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

/// Keep the input forms and geometric query terms implicit on the verifier.
pub(super) struct Kernel<'a, F: Field> {
    forms: &'a [&'a dyn LinearForm<F>],
    pub(super) claim_weights: Vec<F>,
    pub(super) challenges: Vec<F>,
    pub(super) initial_scale: F,
    pub(super) size: usize,
    terms: Vec<KernelTerm<F>>,
}

struct KernelTerm<F: Field> {
    form: UnivariateEvaluation<F>,
    weight: F,
    challenge_offset: usize,
}

impl<F: Field> Kernel<'_, F> {
    /// A coefficient of the current folded kernel, in most-significant-bit order.
    pub(super) fn coefficient(&self, index: usize) -> F {
        assert!(index < self.size);
        let point: Vec<_> = (0..self.size.trailing_zeros())
            .rev()
            .map(|bit| F::from(((index >> bit) & 1) as u64))
            .collect();
        self.evaluate(&point)
    }

    pub(super) fn evaluate(&self, suffix: &[F]) -> F {
        assert_eq!(suffix.len(), self.size.trailing_zeros() as usize);
        let point: Vec<_> = self.challenges.iter().chain(suffix).copied().collect();
        self.initial_contribution(&point) + self.query_contribution(&point)
    }

    fn initial_contribution(&self, point: &[F]) -> F {
        let initial: F = self
            .forms
            .iter()
            .zip(&self.claim_weights)
            .map(|(form, &weight)| weight * evaluate_padded(*form, point))
            .sum();
        self.initial_scale * initial
    }

    /// Evaluate only the masked prefix of each interleaved polynomial.
    fn mask_coefficients(&self, mask: WitnessMaskConfig) -> Vec<F> {
        assert_eq!(self.size % mask.num_polynomials, 0);
        let message_length = self.size / mask.num_polynomials;
        assert!(mask.coefficients_per_polynomial <= message_length);
        let dimensions = self.size.trailing_zeros() as usize;
        let mut point = self.challenges.clone();
        point.resize(point.len() + dimensions, F::ZERO);
        let mut coefficients =
            Vec::with_capacity(mask.num_polynomials * mask.coefficients_per_polynomial);
        for polynomial in 0..mask.num_polynomials {
            for coefficient in 0..mask.coefficients_per_polynomial {
                let index = polynomial * message_length + coefficient;
                for (bit, coordinate) in point[self.challenges.len()..].iter_mut().rev().enumerate()
                {
                    *coordinate = F::from(((index >> bit) & 1) as u64);
                }
                coefficients.push(self.initial_contribution(&point));
            }
        }

        // A folded query term has coefficients a*x^i. Evaluate a at the zero
        // suffix once, then advance by x within a prefix and by x^message_length
        // between polynomial starts. No division is needed, including at x = 0.
        point[self.challenges.len()..].fill(F::ZERO);
        let points: Vec<_> = self.terms.iter().map(|term| term.form.point).collect();
        let strides: Vec<_> = points
            .iter()
            .map(|&x| PublicScalar::new(x).pow(message_length as u64))
            .collect();
        let mut scalars: Vec<_> = self
            .terms
            .iter()
            .map(|term| term.weight * term.form.mle_evaluate(&point[term.challenge_offset..]))
            .collect();
        for prefix in coefficients.chunks_exact_mut(mask.coefficients_per_polynomial) {
            geometric_accumulate(prefix, scalars.clone(), &points);
            for (scalar, stride) in scalars.iter_mut().zip(&strides) {
                *scalar = stride.mul(*scalar);
            }
        }
        coefficients
    }

    fn query_contribution(&self, point: &[F]) -> F {
        self.terms
            .iter()
            .map(|term| term.weight * term.form.mle_evaluate(&point[term.challenge_offset..]))
            .sum()
    }

    /// Remove query terms and leading zero-padding coordinates from the final claim.
    pub(super) fn finish(self, opening: &Opening<F>, original_size: usize) -> FinalClaim<F> {
        let point: Vec<_> = self
            .challenges
            .iter()
            .chain(&opening.evaluation_points)
            .copied()
            .collect();
        let contribution = opening.linear_form_evaluation - self.query_contribution(&point);
        let dimensions = original_size.trailing_zeros() as usize;
        let (padding, evaluation_point) = point.split_at(point.len() - dimensions);
        FinalClaim {
            evaluation_point: evaluation_point.to_vec(),
            initial_claim_scale: self.initial_scale
                * padding.iter().map(|&r| F::ONE - r).product::<F>(),
            linear_forms_contribution: contribution,
            rlc_coefficients: self.claim_weights,
        }
    }

    fn scale(&mut self, scalar: F) {
        self.initial_scale *= scalar;
        for term in &mut self.terms {
            term.weight *= scalar;
        }
    }

    fn fold(&mut self, opening: &SumcheckOpening<F>) {
        self.scale(opening.mask_rlc);
        self.challenges.extend_from_slice(&opening.round_challenges);
        self.size >>= opening.round_challenges.len();
    }

    fn aggregate(&mut self, update: &CovectorUpdateParams<F>) {
        self.scale(update.original_sl_coeff);
        for (&weight, &point) in update
            .ood_rlc_coeffs
            .iter()
            .zip(&update.ood_eval_points)
            .chain(
                update
                    .in_domain_rlc_coeffs
                    .iter()
                    .zip(&update.in_domain_eval_points),
            )
        {
            self.terms.push(KernelTerm {
                form: UnivariateEvaluation::new(point, self.size),
                weight,
                challenge_offset: self.challenges.len(),
            });
        }
    }
}

/// Reduce the initial evaluation claims to a final masked witness and mask functional.
pub(super) fn reduce_prover<M, H, R>(
    config: &ProtocolConfig<M>,
    ps: &mut ProverState<H, R>,
    committed: CommittedWitness<M>,
    forms: &[&dyn LinearForm<M::Target>],
    evaluations: &[M::Target],
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
    assert!(!forms.is_empty(), "zook requires ≥ 1 linear form");
    let claim_weights = geometric_sequence(M::Target::ONE, ps.verifier_message(), forms.len());
    let mut covector = vec![M::Target::ZERO; config.padded_vector_size()];
    for (form, &weight) in forms.iter().zip(&claim_weights) {
        assert!(form.size() <= config.tuning().vector_size);
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
            let mut claims = MaskClaims::new(layout.commit().vector_size());
            claims.subtract_witness_mask(
                layout.witness_masks()[0].clone(),
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
            first,
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
            round,
            state,
            config
                .joint_masks()
                .zip(config.witness_mask(round.round_index() + 1)),
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
    assert!(!forms.is_empty(), "zook requires ≥ 1 linear form");
    assert!(forms
        .iter()
        .all(|form| form.size() <= config.tuning().vector_size));
    let claim_weights = geometric_sequence(M::Target::ONE, vs.verifier_message(), forms.len());
    let sum = evaluations
        .iter()
        .zip(&claim_weights)
        .map(|(&value, &weight)| value * weight)
        .sum();
    let kernel = Kernel {
        forms,
        claim_weights,
        challenges: Vec::new(),
        initial_scale: M::Target::ONE,
        size: config.padded_vector_size(),
        terms: Vec::new(),
    };
    let masks = match (
        config.joint_masks(),
        config.witness_mask(0),
        commitment.joint_masks,
        commitment.mask_binding,
    ) {
        (Some(layout), Some(mask), Some(joint), Some(binding)) => {
            let mut claims = MaskClaims::new(layout.commit().vector_size());
            claims.subtract_witness_mask(
                layout.witness_masks()[0].clone(),
                mask,
                config.padded_vector_size() / mask.num_polynomials,
                |index| kernel.coefficient(index),
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
            first,
            &mut state,
            config.joint_masks().zip(config.witness_mask(1)),
            vs,
        )?;
    }
    for round in config.tail_rounds() {
        verify_round(
            round,
            &mut state,
            config
                .joint_masks()
                .zip(config.witness_mask(round.round_index() + 1)),
            vs,
        )?;
    }
    Ok(state)
}

/// Section 5.4, “ZO0K”, Steps 1–4: carry masks, fold, overlay, then aggregate.
fn prove_round<M, H, R>(
    round: &RoundConfig<M>,
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
    let index = round.round_index();
    let remaining_size = round.sumcheck().final_size();
    let carried = state.masks.as_ref().map_or(M::Target::ZERO, |masks| {
        masks.claims.evaluate(masks.joint.coefficients.to_slice())
    });
    let blinding = match state.masks.as_ref().zip(mask_plan) {
        Some((masks, (layout, _))) => sumcheck_blinding(
            &masks.joint.coefficients.to_slice()[layout.sumcheck_masks()[index].clone()],
            remaining_size,
        ),
        None => Zeroizing::new(Vec::new()),
    };
    // The generic prover accepts the dot and carried terms separately.
    state.sum -= carried;
    let (mut message, opening) = round.sumcheck().prove_with_carried_sum(
        ps,
        round.code_switch().source().embedding(),
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
            layout.sumcheck_masks()[index].clone(),
            remaining_size,
            &opening,
        );
        message = overlay_mask(
            message,
            &masks.joint.coefficients.to_slice()[layout.witness_masks()[index + 1].clone()],
            next_mask,
        );
    }
    let switched = round.code_switch().prove(
        ps,
        message,
        state.irs_witness,
        Claim {
            covector: &mut state.covector,
            sum: &mut state.sum,
        },
        &opening.round_challenges,
        &[],
    );
    if let Some((masks, (layout, next_mask))) = state.masks.as_mut().zip(mask_plan) {
        masks.claims.subtract_target_mask(
            layout.witness_masks()[index + 1].clone(),
            next_mask,
            round.code_switch().target().message_length(),
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
    let index = round.round_index();
    let opening = round.sumcheck().verify(vs, &mut state.sum)?;
    state.kernel.fold(&opening);
    if let Some((masks, (layout, _))) = state.masks.as_mut().zip(mask_plan) {
        masks.claims.fold_sumcheck(
            layout.sumcheck_masks()[index].clone(),
            state.kernel.size,
            &opening,
        );
    }
    let (irs, update) = round.code_switch().verify_for_implicit(
        vs,
        &mut state.sum,
        &opening.round_challenges,
        &state.irs,
    )?;
    state.kernel.aggregate(&update);
    if let Some((masks, (layout, next_mask))) = state.masks.as_mut().zip(mask_plan) {
        let coefficients = state.kernel.mask_coefficients(next_mask);
        let message_length = round.code_switch().target().message_length();
        masks.claims.subtract_target_mask(
            layout.witness_masks()[index + 1].clone(),
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
            linear_form::{Covector, Evaluate, MultilinearExtension},
            random_vector, univariate_evaluate,
        },
        protocols::params::{
            FoldingFactor, Mode, PowBudget, RateSchedule, SecuritySpec, TuningSpec,
        },
        transcript::{codecs::Empty, DomainSeparator},
    };

    #[test]
    fn mask_prefixes_match_individual_kernel_coefficients() {
        type F = Field64_2;
        let mut rng = StdRng::seed_from_u64(71);
        let dense = Covector::new(random_vector::<F>(&mut rng, 32));
        let short = UnivariateEvaluation::new(F::from(7), 3);
        let mle = MultilinearExtension::new(random_vector::<F>(&mut rng, 4));
        let forms: [&dyn LinearForm<F>; 3] = [&dense, &short, &mle];
        let points = [
            F::ZERO,
            F::ONE,
            -F::ONE,
            random_vector::<F>(&mut rng, 1)[0],
            F::from(7),
        ];
        for challenges in [
            vec![F::ZERO; 6],
            vec![F::ONE; 6],
            random_vector(&mut rng, 6),
        ] {
            for scale in [F::ZERO, F::ONE, F::from(11)] {
                let mut kernel = Kernel {
                    forms: &forms,
                    claim_weights: vec![F::ONE, F::from(3), F::from(5)],
                    challenges: Vec::new(),
                    initial_scale: F::from(3),
                    size: 64,
                    terms: Vec::new(),
                };
                for stage in 0..=3 {
                    // Check both an empty query set and terms inserted after
                    // different numbers of folds, including zero query weights.
                    assert_mask_prefixes(&kernel);
                    for (index, &point) in points.iter().enumerate() {
                        kernel.terms.push(KernelTerm {
                            form: UnivariateEvaluation::new(point, kernel.size),
                            weight: if index == 4 { F::ZERO } else { F::from(7) },
                            challenge_offset: kernel.challenges.len(),
                        });
                    }
                    assert_mask_prefixes(&kernel);
                    if stage < 3 {
                        kernel.fold(&SumcheckOpening {
                            round_challenges: challenges[2 * stage..2 * stage + 2].to_vec(),
                            mask_rlc: scale,
                        });
                    }
                }
            }
        }
    }

    fn assert_mask_prefixes(kernel: &Kernel<Field64_2>) {
        for log_polynomials in 0..=kernel.size.trailing_zeros() {
            let num_polynomials = 1 << log_polynomials;
            let message_length = kernel.size / num_polynomials;
            for prefix in [1, 3.min(message_length), message_length] {
                let mask = WitnessMaskConfig {
                    num_polynomials,
                    coefficients_per_polynomial: prefix,
                };
                let expected: Vec<_> = (0..num_polynomials)
                    .flat_map(|polynomial| {
                        (0..prefix)
                            .map(move |i| kernel.coefficient(polynomial * message_length + i))
                    })
                    .collect();
                assert_eq!(kernel.mask_coefficients(mask), expected);
            }
        }
    }

    fn roundtrip<M: Embedding<Target = Field64_2> + Default>(
        mode: Mode,
        size: usize,
        univariate_size: usize,
    ) where
        Standard: Distribution<M::Source> + Distribution<M::Target>,
    {
        let config = ProtocolConfig::<M>::derive(
            SecuritySpec::new(80)
                .with_mode(mode)
                .with_pow_budget(PowBudget::Forbidden),
            TuningSpec {
                vector_size: size,
                starting_log_inv_rate: 2,
                folding_factor: FoldingFactor::Constant(2),
                rate_schedule: RateSchedule::Stepping,
            },
        )
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
        let state = reduce_prover(&config, &mut ps, committed, &forms, &values);
        assert_overlays(&config, initial, &state);
        let _ = config
            .basecase()
            .commit()
            .open(&mut ps, &[&state.irs_witness]);
        if let Some((layout, masks)) = config.joint_masks().zip(state.masks.as_ref()) {
            let _ = layout.commit().open(&mut ps, &[&masks.joint.irs_witness]);
        }
        let challenge: Field64_2 = ps.verifier_message();
        let proof = ps.proof();
        let mut vs = VerifierState::new_std(&ds, &proof);
        let commitment = config.receive_commitment(&mut vs).unwrap();
        let received = reduce_verifier(&config, &mut vs, commitment, &forms, &values).unwrap();
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
            .map(|r| r.sumcheck().num_rounds())
            .chain(
                config
                    .tail_rounds()
                    .iter()
                    .map(|r| r.sumcheck().num_rounds()),
            );
        let mut offset = 0;
        for (index, count) in folds.enumerate() {
            for &challenge in &state.challenges[offset..offset + count] {
                expected.fold(challenge);
            }
            offset += count;
            if let Some((layout, masks)) = config.joint_masks().zip(state.masks.as_ref()) {
                expected = overlay_mask(
                    expected,
                    &masks.joint.coefficients.to_slice()[layout.witness_masks()[index + 1].clone()],
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
                let evaluations = layout.commit().verify(vs, &[&checked.joint]).unwrap();
                for (&point, &value) in evaluations.points.iter().zip(evaluations.matrix.to_slice())
                {
                    assert_eq!(value, univariate_evaluate(coefficients, point));
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
