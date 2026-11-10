//! Zook verifier: replay joint-mask rounds, then check the final opening.

use ark_ff::{Field, Zero};
use ark_std::rand::{distributions::Standard, prelude::Distribution};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        linear_form::LinearForm,
        multilinear_extend, univariate_evaluate,
    },
    buffer::BufferOps,
    hash::Hash,
    protocols::{
        basecase::Opening,
        irs_commit::{Commitment as IrsCommitment, Config as IrsConfig},
        params::ProtocolConfig,
        zook::{
            commit::Commitment,
            round::{reduce_verifier, VerifierRoundState},
            FinalClaim,
        },
    },
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, VerificationResult,
        VerifierMessage, VerifierState,
    },
    utils::zip_strict,
    verify,
};

impl<M: Embedding + Default> ProtocolConfig<M> {
    /// Verify the supplied linear evaluations against the received commitment.
    #[cfg_attr(feature = "tracing", instrument(skip_all, name = "zook::verify"))]
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
        let reduced = reduce_verifier(self, vs, commitment, None, None, linear_forms, evaluations)?;
        self.verify_final_opening(vs, reduced, self.params().size)
    }

    pub(super) fn verify_final_opening<H>(
        &self,
        vs: &mut VerifierState<H>,
        state: VerifierRoundState<'_, M::Target>,
        original_size: usize,
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
        let opening = if self.params().zk {
            let (message, mut sum) = self.receive_final_claim(vs, &state)?;
            let point = basecase.sumcheck().verify(vs, &mut sum)?.round_challenges;
            let value = multilinear_extend(&message, &point);
            verify!(!value.is_zero());
            Opening {
                evaluation_points: point,
                linear_form_evaluation: sum / value,
            }
        } else {
            assert!(state.masks.is_empty());
            basecase.verify(vs, &state.irs, state.sum)?
        };
        Ok(state.kernel.finish(&opening, original_size))
    }

    /// Authenticate the witness and each mask collection, then check every retained binding.
    /// For each collection and binding form, check `L(m_i') = lambda*L(m_i) + L(U_i)`.
    /// The resulting witness claim is `sum = lambda*c + d - sum_i B_i(m_i')`.
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
        let layout = self.joint_masks().expect("final opening mask layout");
        assert!(!state.masks.is_empty());
        let commit = self.final_commit();
        let witness_blind_commit = commit.receive_commitment(vs)?;
        let blind_sum: M::Target = vs.prover_message()?;
        let blind_binding: Vec<M::Target> = vs.prover_messages_vec(3 * state.masks.len())?;
        let lambda = vs.verifier_message::<M::Target>();
        let message = vs.prover_messages_vec(commit.vector_size())?;
        let mut mask_messages = Vec::with_capacity(state.masks.len());
        for _ in &state.masks {
            mask_messages.push(vs.prover_messages_vec(layout.commit.vector_size())?);
        }
        verify_combination(
            commit,
            vs,
            &[&witness_blind_commit, &state.irs],
            &message,
            &[M::Target::ONE, lambda],
        )?;
        // Keep the two columns of every mask root in the same query event.
        let roots: Vec<_> = state.masks.iter().map(|masks| &masks.joint).collect();
        let evaluations = layout.commit.verify(vs, &roots)?;
        for (collection, ((masks, mask_message), blind)) in state
            .masks
            .iter()
            .zip(&mask_messages)
            .zip(blind_binding.chunks_exact(3))
            .enumerate()
        {
            let values = evaluate_points(mask_message, &evaluations.points);
            for (&actual, row) in values.iter().zip(
                evaluations
                    .matrix
                    .to_slice()
                    .chunks_exact(2 * state.masks.len()),
            ) {
                verify!(actual == lambda * row[2 * collection] + row[2 * collection + 1]);
            }
            for ((actual, &value), &blind) in masks
                .binding
                .evaluate(layout, mask_message)
                .into_iter()
                .zip(&masks.binding.values)
                .zip(blind)
            {
                verify!(actual == blind + lambda * value);
            }
        }
        let sum = blind_sum + lambda * state.sum
            - state
                .masks
                .iter()
                .zip(&mask_messages)
                .map(|(masks, message)| masks.claims.evaluate(message))
                .sum::<M::Target>();
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
        protocols::{
            params::{Params, RateSchedule},
            zook::{commit::CommittedWitness, round::reduce_prover},
        },
        transcript::{codecs::Empty, DomainSeparator, ProverState},
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
                zk: true,
                regime: crate::protocols::params::DecodingRegime::Johnson,
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
                    reduce_verifier(&config, &mut vs, commitment, None, None, &forms, &values)?;
                mutate(&mut state);
                let claim = config.verify_final_opening(&mut vs, state, size)?;
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
                    state.masks.first_mut().unwrap().binding.values[index] += Field64_2::ONE;
                },
                false,
            );
        }
    }

    #[test]
    fn final_opening_checks_accumulated_mask_claim_and_sum() {
        check_final_opening(
            |state| {
                state.masks.first_mut().unwrap().claims.covector.vector[0] += Field64_2::ONE;
            },
            false,
        );
        check_final_opening(|state| state.sum += Field64_2::ONE, false);
    }

    #[test]
    fn final_opening_authenticates_both_original_commitments() {
        check_final_opening(
            |state| state.irs = state.masks.first().unwrap().joint.clone(),
            false,
        );
        check_final_opening(
            |state| state.masks.first_mut().unwrap().joint = state.irs.clone(),
            false,
        );
    }

    fn check_pair_final_opening(
        mutate_witness: impl Fn(&mut [CommittedWitness<Basefield<Field64_2>>; 2]),
        mutate: impl Fn(&mut VerifierRoundState<'_, Field64_2>),
        accept: bool,
    ) {
        let size = 4096;
        let config = ProtocolConfig::<Basefield<Field64_2>>::derive(Params {
            bits: 80,
            zk: true,
            regime: crate::protocols::params::DecodingRegime::Johnson,
            pow: 0,
            hash: crate::hash::BLAKE3,
            size,
            rate: 2,
            folds: [2, 2],
            schedule: RateSchedule::Stepping,
        })
        .unwrap();
        let source = config.pair_source().unwrap();
        let first = config.pair_round(&source).unwrap();
        let mut rng = StdRng::seed_from_u64(225);
        let left = Buffer::random(&mut rng, size);
        let right = Buffer::random(&mut rng, size);
        let form = MultilinearExtension::new(random_vector(&mut rng, 13));
        let forms: [&dyn LinearForm<Field64_2>; 1] = [&form];
        let values = [form.evaluate(&Basefield::new(), left.concat(&right).to_slice())];
        let ds = DomainSeparator::protocol(&config)
            .session(&"pair-final-opening")
            .instance(&Empty);
        let mut ps = ProverState::new_std(&ds);
        let mut handles = [config.commit(&mut ps, left), config.commit(&mut ps, right)];
        mutate_witness(&mut handles);
        let [left, right] = handles;
        let state = reduce_prover(
            &config,
            &mut ps,
            left,
            Some(right),
            Some(&first),
            &forms,
            &values,
        );
        let prover_claim = config.prove_final_opening(&mut ps, state, 2 * size);
        let proof = ps.proof();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut vs = VerifierState::new_std(&ds, &proof);
            let left = config.receive_commitment(&mut vs)?;
            let right = config.receive_commitment(&mut vs)?;
            let mut state = reduce_verifier(
                &config,
                &mut vs,
                left,
                Some(right),
                Some(&first),
                &forms,
                &values,
            )?;
            mutate(&mut state);
            let claim = config.verify_final_opening(&mut vs, state, 2 * size)?;
            claim.verify(&forms)?;
            assert_eq!(claim.evaluation_point, prover_claim.evaluation_point);
            vs.check_eof()
        }));
        assert_eq!(matches!(result, Ok(Ok(()))), accept);
    }

    #[test]
    fn pair_final_opening_checks_both_mask_roots_claims_and_all_six_bindings() {
        check_pair_final_opening(|_| {}, |_| {}, true);
        for collection in 0..2 {
            for binding in 0..3 {
                check_pair_final_opening(
                    |_| {},
                    |state| state.masks[collection].binding.values[binding] += Field64_2::ONE,
                    false,
                );
            }
            check_pair_final_opening(
                |_| {},
                |state| state.masks[collection].claims.covector.vector[0] += Field64_2::ONE,
                false,
            );
            check_pair_final_opening(
                |_| {},
                |state| state.masks[collection].joint = state.masks[1 - collection].joint.clone(),
                false,
            );
            check_pair_final_opening(
                |handles| {
                    let masks = handles[collection].joint_masks.as_mut().unwrap();
                    let mut changed = masks.opening_blinder.to_slice().to_vec();
                    changed[0] += Field64_2::ONE;
                    masks.opening_blinder.wipe();
                    masks.opening_blinder = Buffer::from(changed);
                },
                |_| {},
                false,
            );
        }
    }
}
