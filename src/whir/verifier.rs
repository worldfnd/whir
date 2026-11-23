//! WHIR verifier: replay joint-mask rounds, then check the final opening.

use ark_ff::{Field, Zero};
use ark_std::rand::{distributions::Standard, prelude::Distribution};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        linear_form::{evaluate_padded, LinearForm},
        multilinear_extend, univariate_evaluate,
    },
    hash::Hash,
    protocols::{
        basecase::Opening,
        irs_commit::{Commitment as IrsCommitment, Config as IrsConfig},
    },
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, VerificationResult,
        VerifierMessage, VerifierState,
    },
    utils::zip_strict,
    verify,
    whir::{
        commit::Commitment,
        params::ProtocolConfig,
        round::{reduce_verifier, VerifierRoundState},
    },
};

/// Output of [`ProtocolConfig::verify`].
///
/// The verifier has completed all round checks. The caller must finish
/// verification by checking that the input forms evaluate to the claimed value:
///
/// ```text
/// initial_claim_scale × Σ_j rlc_coefficients[j] × form_j.mle_evaluate(evaluation_point) == linear_forms_contribution
/// ```
#[must_use]
#[derive(Clone, Debug)]
pub struct FinalClaim<F: ark_ff::Field> {
    /// Sumcheck challenges for the original witness dimensions, excluding leading
    /// padding coordinates. Length = log2(vector_size).
    pub evaluation_point: Vec<F>,
    /// Product of round combination factors and leading zero-padding weights.
    pub initial_claim_scale: F,
    /// `opening.linear_form_evaluation − Σ_c constraint_contributions`.
    /// The portion of the protocol sum attributable to the input linear forms.
    pub linear_forms_contribution: F,
    /// Fiat-Shamir RLC coefficients for each input form.
    /// `rlc_coefficients[0] == F::ONE` always.
    pub rlc_coefficients: Vec<F>,
}

impl<F: ark_ff::Field> FinalClaim<F> {
    /// Complete the verification started by [`ProtocolConfig::verify`].
    ///
    /// Checks `initial_claim_scale × Σ_j rlc_coefficients[j] × form_j.mle_evaluate(evaluation_point) == linear_forms_contribution`.
    ///
    /// For `MultilinearExtension` forms this runs in O(num_forms × log N).
    /// For `Covector` forms the default `mle_evaluate` is O(N).
    pub fn verify(&self, linear_forms: &[&dyn LinearForm<F>]) -> VerificationResult<()>
    where
        F: Default,
    {
        assert_eq!(
            linear_forms.len(),
            self.rlc_coefficients.len(),
            "linear_forms.len() must match rlc_coefficients.len()"
        );
        let form_mle_sum: F = linear_forms
            .iter()
            .zip(&self.rlc_coefficients)
            .map(|(form, &g)| g * evaluate_padded(*form, &self.evaluation_point))
            .sum();
        verify!(self.initial_claim_scale * form_mle_sum == self.linear_forms_contribution);
        Ok(())
    }
}

impl<M: Embedding + Default> ProtocolConfig<M> {
    /// Verify the supplied linear evaluations against the received commitment.
    #[cfg_attr(feature = "tracing", instrument(skip_all, name = "whir::verify"))]
    pub fn verify<H>(
        &self,
        vs: &mut VerifierState<H>,
        commitment: Commitment<M::Target>,
        linear_forms: &[&dyn LinearForm<M::Target>],
        evaluations: &[M::Target],
    ) -> VerificationResult<FinalClaim<M::Target>>
    where
        M::Target: Codec<[H::U]>,
        Standard: Distribution<M::Target>,
        H: DuplexSpongeInterface,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let reduced = reduce_verifier(self, vs, commitment, linear_forms, evaluations, None)?;
        self.verify_final_opening(vs, reduced)
    }

    pub(super) fn verify_final_opening<H>(
        &self,
        vs: &mut VerifierState<H>,
        state: VerifierRoundState<'_, M::Target>,
    ) -> VerificationResult<FinalClaim<M::Target>>
    where
        M::Target: Codec<[H::U]>,
        H: DuplexSpongeInterface,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let basecase = self.basecase();
        let opening = if self.params().mode.is_zk() {
            let (message, mut sum) = self.receive_final_claim(vs, &state)?;
            let point = basecase.sumcheck().verify(vs, &mut sum)?.round_challenges;
            let value = multilinear_extend(&message, &point);
            verify!(!value.is_zero());
            Opening {
                evaluation_points: point,
                linear_form_evaluation: sum / value,
            }
        } else {
            assert!(state.masks.is_none());
            basecase.verify(vs, &state.irs, state.sum)?
        };
        Ok(state.kernel.finish(&opening, self.params().size))
    }

    /// Check both encoding relations and all three deferred Phase 0 equations.
    ///
    /// §5.4, “The final opening protocol”: `L(M') = lambda*L(m) + L(M)`
    /// for each binding form, then `sum = lambda*c + d - B(M')`.
    fn receive_final_claim<H>(
        &self,
        vs: &mut VerifierState<H>,
        state: &VerifierRoundState<'_, M::Target>,
    ) -> VerificationResult<(Vec<M::Target>, M::Target)>
    where
        M::Target: Codec<[H::U]>,
        H: DuplexSpongeInterface,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let (Some(layout), Some(masks)) = (self.joint_masks(), state.masks.as_ref()) else {
            panic!("final opening requires the joint mask commitment");
        };
        let commit = self.final_commit();
        let witness_blind_commit = commit.receive_commitment(vs)?;
        let blind_sum: M::Target = vs.prover_message()?;
        let blind_binding: Vec<M::Target> = vs.prover_messages_vec(3)?;
        let lambda = vs.verifier_message::<M::Target>();
        let message = vs.prover_messages_vec(commit.vector_size())?;
        let mask_message = vs.prover_messages_vec(layout.commit.vector_size())?;
        verify_combination(
            commit,
            vs,
            &[&witness_blind_commit, &state.irs],
            &message,
            &[M::Target::ONE, lambda],
        )?;
        verify_combination(
            &layout.commit,
            vs,
            &[&masks.joint],
            &mask_message,
            &[lambda, M::Target::ONE],
        )?;
        for ((actual, &value), blind) in masks
            .binding
            .evaluate(layout, &mask_message)
            .into_iter()
            .zip(&masks.binding.values)
            .zip(blind_binding)
        {
            verify!(actual == blind + lambda * value);
        }
        let sum = blind_sum + lambda * state.sum - masks.claims.evaluate(&mask_message);
        Ok((message, sum))
    }
}

/// Authenticate a revealed linear combination against the committed columns.
fn verify_combination<F, H>(
    config: &IrsConfig<Identity<F>>,
    vs: &mut VerifierState<H>,
    commitments: &[&IrsCommitment],
    message: &[F],
    weights: &[F],
) -> VerificationResult<()>
where
    F: Field + Codec<[H::U]>,
    H: DuplexSpongeInterface,
    u8: Decoding<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    assert_eq!(config.interleaving_depth(), 1);
    assert_eq!(config.num_messages() * commitments.len(), weights.len());
    assert_eq!(config.mask_length(), 0);
    let evaluations = config.verify(vs, commitments)?;
    let values = evaluate_points(message, &evaluations.points);
    for (actual, expected) in zip_strict(values, evaluations.values(weights)) {
        verify!(actual == expected);
    }
    Ok(())
}

/// Evaluate independent queries in one batch, preserving their order.
fn evaluate_points<F: Field>(message: &[F], points: &[F]) -> Vec<F> {
    #[cfg(feature = "parallel")]
    let points = points.par_iter();
    #[cfg(not(feature = "parallel"))]
    let points = points.iter();
    points
        .map(|&point| univariate_evaluate(message, point))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use ark_ff::AdditiveGroup;
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::{
        algebra::{
            embedding::Basefield,
            fields::{Field64_2, Field64_3},
            linear_form::{Evaluate, MultilinearExtension, UnivariateEvaluation},
            random_vector,
        },
        buffer::{Buffer, BufferMath, BufferOps},
        decoding::DecodingRegime,
        transcript::{codecs::Empty, DomainSeparator, ProverState},
        whir::{
            commit::CommittedWitness,
            params::{Mode, Params, RateSchedule},
        },
    };

    #[test]
    fn batched_evaluations_preserve_query_order_and_duplicates() {
        type F = Field64_3;
        let mut rng = StdRng::seed_from_u64(121);
        let mut points = random_vector::<F>(&mut rng, 67);
        points[0] = F::ZERO;
        points[1] = F::ONE;
        points[2] = points[4];
        for size in [0, 1, 3, 8192] {
            let message = random_vector::<F>(&mut rng, size);
            let expected: Vec<_> = points
                .iter()
                .map(|point| {
                    message
                        .iter()
                        .rev()
                        .fold(F::ZERO, |value, coefficient| value * point + coefficient)
                })
                .collect();
            assert_eq!(evaluate_points(&message, &points), expected);
            assert_eq!(evaluate_points(&message, &[]), Vec::<F>::new());
        }
    }

    fn check_final_opening(mutate: impl Fn(&mut VerifierRoundState<'_, Field64_2>), accept: bool) {
        check_final_opening_with_witness(|_| {}, mutate, accept);
    }

    fn check_final_opening_with_witness(
        mutate_witness: impl Fn(&mut CommittedWitness<Basefield<Field64_2>>),
        mutate: impl Fn(&mut VerifierRoundState<'_, Field64_2>),
        accept: bool,
    ) {
        for size in [8_usize, 4096] {
            let config = ProtocolConfig::<Basefield<Field64_2>>::derive(Params {
                bits: 80,
                mode: Mode::ZeroKnowledge,
                regime: DecodingRegime::Johnson,
                pow: 0,
                hash: crate::hash::BLAKE3,
                size,
                rate: 2,
                folds: [2, 2],
                schedule: RateSchedule::Stepping,
            })
            .unwrap();
            let mut rng = StdRng::seed_from_u64(102);
            let witness = Buffer::random(&mut rng, size);
            let multilinear =
                MultilinearExtension::new(random_vector(&mut rng, size.trailing_zeros() as usize));
            let univariate = UnivariateEvaluation::new(Field64_2::from(7), 3);
            let forms: [&dyn LinearForm<Field64_2>; 2] = [&multilinear, &univariate];
            let values = [
                multilinear.evaluate(&Basefield::new(), witness.to_slice()),
                univariate.evaluate(&Basefield::new(), &witness.to_slice()[..3]),
            ];
            let ds = DomainSeparator::protocol(&config)
                .session(&"final-opening")
                .instance(&Empty);
            let mut ps = ProverState::new_std(&ds);
            let mut committed = config.commit(&mut ps, witness);
            mutate_witness(&mut committed);
            let prover_claim = config.prove(&mut ps, committed, &forms, &values);
            let proof = ps.proof();
            let result = catch_unwind(AssertUnwindSafe(|| {
                let mut vs = VerifierState::new_std(&ds, &proof);
                let commitment = config.receive_commitment(&mut vs)?;
                let mut state =
                    reduce_verifier(&config, &mut vs, commitment, &forms, &values, None)?;
                mutate(&mut state);
                let claim = config.verify_final_opening(&mut vs, state)?;
                claim.verify(&forms)?;
                assert_eq!(claim.evaluation_point, prover_claim.evaluation_point);
                assert_eq!(claim.evaluation_point.len(), size.trailing_zeros() as usize);
                assert_eq!(claim.rlc_coefficients, prover_claim.rlc_coefficients);
                vs.check_eof()
            }));
            assert_eq!(matches!(result, Ok(Ok(()))), accept, "witness size {size}");
        }
    }

    #[test]
    fn final_opening_preserves_points_for_padded_and_short_forms() {
        check_final_opening(|_| {}, true);
    }

    #[test]
    fn final_opening_rejects_changed_precommitted_blinder() {
        check_final_opening_with_witness(
            |committed| {
                let joint = committed.joint_masks.as_mut().unwrap();
                let mut changed = joint.opening_blinder.to_slice().to_vec();
                changed[0] += Field64_2::ONE;
                joint.opening_blinder.wipe();
                joint.opening_blinder = Buffer::from(changed);
            },
            |_| {},
            false,
        );
    }

    #[test]
    fn final_opening_checks_each_deferred_binding_equation() {
        for index in 0..3 {
            check_final_opening(
                |state| {
                    state.masks.as_mut().unwrap().binding.values[index] += Field64_2::ONE;
                },
                false,
            );
        }
    }

    #[test]
    fn final_opening_checks_accumulated_mask_claim_and_sum() {
        check_final_opening(
            |state| {
                state.masks.as_mut().unwrap().claims.covector.vector[0] += Field64_2::ONE;
            },
            false,
        );
        check_final_opening(|state| state.sum += Field64_2::ONE, false);
    }

    #[test]
    fn final_opening_authenticates_both_original_commitments() {
        check_final_opening(
            |state| state.irs = state.masks.as_ref().unwrap().joint.clone(),
            false,
        );
        check_final_opening(
            |state| state.masks.as_mut().unwrap().joint = state.irs.clone(),
            false,
        );
    }
}
