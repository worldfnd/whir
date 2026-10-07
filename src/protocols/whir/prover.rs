use std::{any::Any, borrow::Cow, mem};

use ark_ff::{AdditiveGroup, FftField, Field};
use ark_std::rand::{CryptoRng, RngCore};
#[cfg(feature = "tracing")]
use tracing::instrument;

use super::{Config, Witness};
use crate::{
    algebra::{
        dot,
        embedding::Embedding,
        lift,
        linear_form::{Covector, Evaluate, LinearForm, UnivariateEvaluation},
        mixed_scalar_mul_add,
        sumcheck::fold,
        tensor_product, MultilinearPoint,
    },
    hash::Hash,
    protocols::{geometric_challenge::geometric_challenge, whir::FinalClaim},
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState,
        VerifierMessage,
    },
    utils::zip_strict,
};

impl<M: Embedding> Config<M>
where
    M::Source: FftField,
    M::Target: FftField,
{
    /// Prove a WHIR opening.
    ///
    /// * `prover_state` the mutable transcript to write the proof to.
    /// * `vectors` all the vectors we are opening.
    /// * `witnesses` witnesses corresponding to the `vectors`, in the same
    ///   order. Multiple vectors may share the same witness, in which case
    ///   only one witness should be provided.
    /// * `linear_forms` the covectors (if any) to evaluate each vector at.
    /// * `evaluations` a matrix of each vector evaluated at each linear form.
    ///
    /// The `evaluations` matrix is in row-major order with the number of rows
    /// equal to the `linear_forms.len()` and the number of columns equal to
    /// `vectors.len()`.
    ///
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    #[allow(
        clippy::too_many_lines,
        clippy::cognitive_complexity,
        clippy::needless_pass_by_value
    )]
    pub fn prove<'a, H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        vectors: Vec<Cow<'a, [M::Source]>>,
        witnesses: Vec<Cow<'a, Witness<M::Target, M>>>,
        linear_forms: Vec<Box<dyn LinearForm<M::Target>>>,
        evaluations: Cow<'a, [M::Target]>,
    ) -> FinalClaim<M::Target>
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        M::Target: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let num_vectors = vectors.len();

        // Input validation
        assert_eq!(
            num_vectors,
            witnesses.len() * self.initial_committer.num_vectors
        );
        assert_eq!(evaluations.len(), num_vectors * linear_forms.len());
        for vector in &vectors {
            assert_eq!(vector.len(), self.initial_size());
        }
        for linear_form in &linear_forms {
            assert_eq!(linear_form.size(), self.initial_size());
        }
        #[cfg(debug_assertions)]
        for (linear_form, evaluations) in
            zip_strict(linear_forms.iter(), evaluations.chunks_exact(num_vectors))
        {
            use crate::algebra::linear_form::Covector;
            let covector = Covector::from(&**linear_form);
            for (vector, evaluation) in zip_strict(&vectors, evaluations) {
                debug_assert_eq!(covector.evaluate(self.embedding(), vector), *evaluation);
            }
        }
        if vectors.is_empty() {
            // TODO: Should we draw a random evaluation point of the right size?
            return FinalClaim::default();
        }

        // Complete evaluations of EVERY vector at EVERY linear form.
        let (oods_evals, oods_matrix) = {
            let mut oods_evals = Vec::new();
            let mut oods_matrix = Vec::new();

            // Out of domain samples. Compute missing cross-terms and send to verifier.
            let mut vector_offset = 0;
            for witness in &witnesses {
                for (oods_eval, oods_row) in zip_strict(
                    witness.out_of_domain().evaluators(self.initial_size()),
                    witness.out_of_domain().rows(),
                ) {
                    for (j, vector) in vectors.iter().enumerate() {
                        if j >= vector_offset && j < oods_row.len() + vector_offset {
                            debug_assert_eq!(
                                oods_row[j - vector_offset],
                                oods_eval.evaluate(self.embedding(), vector)
                            );

                            oods_matrix.push(oods_row[j - vector_offset]);
                        } else {
                            let eval = oods_eval.evaluate(self.embedding(), vector);
                            prover_state.prover_message(&eval);
                            oods_matrix.push(eval);
                        }
                    }
                    oods_evals.push(oods_eval);
                }
                vector_offset += witness.num_vectors();
            }
            (oods_evals, oods_matrix)
        };

        // Random linear combination of the vectors.
        let vector_rlc_coeffs: Vec<M::Target> = geometric_challenge(prover_state, num_vectors);
        assert_eq!(vector_rlc_coeffs[0], M::Target::ONE);
        // Recycle the first input as the accumulator (its coefficient is always ONE).
        let mut vectors = vectors.into_iter();
        let first = vectors.next().expect("non-empty");
        let mut vector = match first {
            Cow::Borrowed(slice) => lift(self.embedding(), slice),
            Cow::Owned(vec) => self.embedding().map_vec(vec),
        };
        for (rlc_coeff, input_vector) in zip_strict(&vector_rlc_coeffs[1..], vectors) {
            mixed_scalar_mul_add(self.embedding(), &mut vector, *rlc_coeff, &input_vector);
        }

        // Random linear combination of the constraints.
        let constraint_rlc_coeffs: Vec<M::Target> =
            geometric_challenge(prover_state, linear_forms.len() + oods_evals.len());
        let has_constraints = !constraint_rlc_coeffs.is_empty();
        let (initial_forms_rlc_coeffs, oods_rlc_coeffs) =
            constraint_rlc_coeffs.split_at(linear_forms.len());
        // Try to recycle the first linear form as Covector.
        let mut covector = vec![];
        let mut linear_forms = linear_forms;
        if let Some((first, linear_forms)) = linear_forms.split_first_mut() {
            debug_assert_eq!(initial_forms_rlc_coeffs[0], M::Target::ONE);
            if let Some(covector_form) =
                (first.as_mut() as &mut dyn Any).downcast_mut::<Covector<M::Target>>()
            {
                mem::swap(&mut covector, &mut covector_form.vector);
            } else {
                covector.resize(self.initial_size(), M::Target::ZERO);
                first.accumulate(&mut covector, M::Target::ONE);
            }
            for (rlc_coeff, linear_form) in zip_strict(&initial_forms_rlc_coeffs[1..], linear_forms)
            {
                linear_form.accumulate(&mut covector, *rlc_coeff);
            }
        } else if has_constraints {
            covector.resize(self.initial_size(), M::Target::ZERO);
        }
        drop(linear_forms);

        // Compute "The Sum"
        let mut the_sum: M::Target = zip_strict(
            initial_forms_rlc_coeffs,
            evaluations.chunks_exact(num_vectors),
        )
        .map(|(poly_coeff, row)| *poly_coeff * dot(&vector_rlc_coeffs, row))
        .sum();
        drop(evaluations);

        debug_assert!(!has_constraints || dot(&vector, &covector) == the_sum);

        // Add OODS constraints
        UnivariateEvaluation::accumulate_many(&oods_evals, &mut covector, oods_rlc_coeffs);
        the_sum += zip_strict(oods_rlc_coeffs, oods_matrix.chunks_exact(num_vectors))
            .map(|(poly_coeff, row)| *poly_coeff * dot(&vector_rlc_coeffs, row))
            .sum::<M::Target>();
        drop(oods_evals);
        drop(oods_matrix);

        debug_assert!(!has_constraints || dot(&vector, &covector) == the_sum);

        // Run initial sumcheck on batched vectors with combined statement
        let mut folding_randomness = if has_constraints {
            self.initial_sumcheck
                .prove(prover_state, &mut vector, &mut covector, &mut the_sum)
        } else {
            // There are no constraints yet, so we can skip the sumcheck.
            // (If we did run it, all sumcheck vectors would be constant zero)
            // TODO: Don't compute evaluations and constraints in the first place.
            let folding_randomness = (0..self.initial_sumcheck.num_rounds)
                .map(|_| prover_state.verifier_message())
                .collect();
            self.initial_skip_pow.prove(prover_state);
            // Fold vector
            for &f in &folding_randomness {
                fold(&mut vector, f);
            }
            // Covector must be all zeros.
            covector = vec![M::Target::ZERO; self.initial_sumcheck.final_size()];
            MultilinearPoint(folding_randomness)
        };
        let mut evaluation_point = folding_randomness.0.clone();

        debug_assert_eq!(dot(&vector, &covector), the_sum);

        if self.round_configs.is_empty() {
            // 0-rounds case: open initial commitment and run final sumcheck directly.
            assert_eq!(vector.len(), self.final_sumcheck.initial_size);
            for coeff in &vector {
                prover_state.prover_message(coeff);
            }
            self.final_pow.prove(prover_state);
            let witness_refs: Vec<&_> = witnesses.iter().map(|c| &**c).collect();
            let _in_domain = self.initial_committer.open(prover_state, &witness_refs);
            let final_folding =
                self.final_sumcheck
                    .prove(prover_state, &mut vector, &mut covector, &mut the_sum);
            evaluation_point.extend(final_folding.0.iter().copied());
        } else {
            // Round 0: open initial witnesses with embedding lift and tensor_product.
            let round0_config = &self.round_configs[0];
            let round0_witness = round0_config.irs_committer.commit(prover_state, &[&vector]);
            round0_config.pow.prove(prover_state);

            let witness_refs: Vec<&_> = witnesses.iter().map(|c| &**c).collect();
            let in_domain = self
                .initial_committer
                .open(prover_state, &witness_refs)
                .lift(self.embedding());

            let stir_challenges = round0_witness
                .out_of_domain()
                .evaluators(round0_config.initial_size())
                .chain(in_domain.evaluators(round0_config.initial_size()))
                .collect::<Vec<_>>();
            let stir_evaluations = round0_witness
                .out_of_domain()
                .values(&[M::Target::ONE])
                .chain(in_domain.values(&tensor_product(
                    &vector_rlc_coeffs,
                    &folding_randomness.eq_weights(),
                )))
                .collect::<Vec<_>>();
            let stir_rlc_coeffs =
                geometric_challenge(prover_state, 1 + stir_challenges.len())[1..].to_vec();
            UnivariateEvaluation::accumulate_many(
                &stir_challenges,
                &mut covector,
                &stir_rlc_coeffs,
            );
            the_sum += dot(&stir_rlc_coeffs, &stir_evaluations);
            debug_assert_eq!(dot(&vector, &covector), the_sum);

            folding_randomness = round0_config.sumcheck.prove(
                prover_state,
                &mut vector,
                &mut covector,
                &mut the_sum,
            );
            evaluation_point.extend(folding_randomness.0.iter().copied());
            debug_assert_eq!(dot(&vector, &covector), the_sum);

            // Rounds 1..N + final round.
            let result = super::rounds::prove_remaining_rounds(
                &self.round_configs,
                &super::rounds::FinalRoundConfig {
                    sumcheck: &self.final_sumcheck,
                    pow: &self.final_pow,
                },
                prover_state,
                &mut super::rounds::SumcheckState {
                    vector: &mut vector,
                    covector: &mut covector,
                    the_sum: &mut the_sum,
                },
                round0_witness,
                &folding_randomness,
            );
            for fr in &result.round_folding_randomness {
                evaluation_point.extend(fr.0.iter().copied());
            }
        }

        FinalClaim {
            evaluation_point,
            rlc_coefficients: initial_forms_rlc_coeffs.to_vec(),
            linear_form_rlc: M::Target::ZERO,
        }
    }
}
