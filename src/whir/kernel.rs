//! The verifier's implicit kernel: input forms, fold challenges, and geometric
//! query terms, evaluated on demand instead of materialized at witness size.

use std::any::Any;

use ark_ff::Field;

use crate::{
    algebra::{
        eq_weights, geometric_accumulate_prefixes,
        linear_form::{evaluate_padded, LinearForm, MultilinearExtension, UnivariateEvaluation},
        scalar_mul_add,
    },
    protocols::{basecase::Opening, code_switch::CovectorUpdateParams, sumcheck::SumcheckOpening},
    whir::{params::WitnessMaskConfig, FinalClaim},
};

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
    folded_multiplier: F,
    folded_through: usize,
    challenge_offset: usize,
}

impl<'a, F: Field> Kernel<'a, F> {
    pub(super) fn new(
        forms: &'a [&'a dyn LinearForm<F>],
        claim_weights: Vec<F>,
        size: usize,
    ) -> Self {
        Self {
            forms,
            claim_weights,
            challenges: Vec::new(),
            initial_scale: F::ONE,
            size,
            terms: Vec::new(),
        }
    }
}

impl<F: Field> Kernel<'_, F> {
    /// A coefficient of the current folded kernel, in most-significant-bit order.
    #[cfg(test)]
    pub(super) fn coefficient(&self, index: usize) -> F {
        assert!(index < self.size);
        let point: Vec<_> = self
            .challenges
            .iter()
            .copied()
            .chain(
                (0..self.size.trailing_zeros())
                    .rev()
                    .map(|bit| F::from(((index >> bit) & 1) as u64)),
            )
            .collect();
        self.initial_contribution(&point) + self.query_contribution(&point)
    }

    #[cfg(test)]
    fn initial_contribution(&self, point: &[F]) -> F {
        let initial: F = self
            .forms
            .iter()
            .zip(&self.claim_weights)
            .map(|(form, &weight)| weight * evaluate_padded(*form, point))
            .sum();
        self.initial_scale * initial
    }

    /// Accumulate a multilinear form using the tensor product of polynomial and prefix weights.
    ///
    /// For polynomial index `p` and coefficient index `c`, the weight factors as
    /// `scalar * eq(folded, self.challenges) * eq(polynomial, bits(p)) * eq(message, bits(c))`,
    /// where `bits` lists bits most significant first and `eq(a, b) = ∏(a_i*b_i + (1-a_i)*(1-b_i))`.
    fn accumulate_multilinear_mask(
        &self,
        form: &MultilinearExtension<F>,
        mask: WitnessMaskConfig,
        scalar: F,
        coefficients: &mut [F],
    ) {
        let dimensions = self.challenges.len() + self.size.trailing_zeros() as usize;
        // Zero coordinates encode the form's implicit leading padding dimensions.
        let mut point = vec![F::ZERO; dimensions - form.point.len()];
        point.extend_from_slice(&form.point);
        let (folded, remaining) = point.split_at(self.challenges.len());
        let scalar =
            scalar * MultilinearExtension::new(folded.to_vec()).mle_evaluate(&self.challenges);
        let (polynomial, message) =
            remaining.split_at(mask.num_polynomials.trailing_zeros() as usize);
        let prefix_dimensions = mask.coefficients_per_polynomial.next_power_of_two().ilog2();
        let (zero_prefix, varying) = message.split_at(message.len() - prefix_dimensions as usize);
        let scalar = zero_prefix
            .iter()
            .fold(scalar, |weight, &r| weight * (F::ONE - r));
        // Indices in the masked prefix have zero high bits. Expand only its low
        // coordinates, allocating fewer than twice the requested coefficients.
        let prefix = eq_weights(varying);
        let polynomial_weights = eq_weights(polynomial);
        for (chunk, weight) in coefficients
            .chunks_exact_mut(mask.coefficients_per_polynomial)
            .zip(polynomial_weights)
        {
            scalar_mul_add(chunk, scalar * weight, &prefix[..chunk.len()]);
        }
    }

    fn initial_mask_coefficients(&self, mask: WitnessMaskConfig) -> Vec<F> {
        let message_length = self.size / mask.num_polynomials;
        let dimensions = self.size.trailing_zeros() as usize;
        let mut point = self.challenges.clone();
        point.resize(point.len() + dimensions, F::ZERO);
        let mut coefficients =
            vec![F::ZERO; mask.num_polynomials * mask.coefficients_per_polynomial];
        for (&form, &weight) in self.forms.iter().zip(&self.claim_weights) {
            let scalar = self.initial_scale * weight;
            if let Some(form) = (form as &dyn Any).downcast_ref::<MultilinearExtension<F>>() {
                self.accumulate_multilinear_mask(form, mask, scalar, &mut coefficients);
                continue;
            }
            for (polynomial, chunk) in coefficients
                .chunks_exact_mut(mask.coefficients_per_polynomial)
                .enumerate()
            {
                for (coefficient, value) in chunk.iter_mut().enumerate() {
                    let index = polynomial * message_length + coefficient;
                    for (bit, coordinate) in
                        point[self.challenges.len()..].iter_mut().rev().enumerate()
                    {
                        *coordinate = F::from(((index >> bit) & 1) as u64);
                    }
                    *value += scalar * evaluate_padded(form, &point);
                }
            }
        }
        coefficients
    }

    /// Evaluate only the masked prefix of each interleaved polynomial.
    pub(super) fn mask_coefficients(&mut self, mask: WitnessMaskConfig) -> Vec<F> {
        assert_eq!(self.size % mask.num_polynomials, 0);
        let message_length = self.size / mask.num_polynomials;
        assert!(mask.coefficients_per_polynomial <= message_length);
        let mut coefficients = self.initial_mask_coefficients(mask);
        for term in &mut self.terms {
            let challenges = &self.challenges[term.folded_through..];
            if challenges.is_empty() {
                continue;
            }
            // For k new challenges r_j, a query point x gains the factor
            // product_j (1 - r_j + r_j * x^(self.size * 2^(k-1-j))).
            let mut power = term.form.point.pow([self.size as u64]);
            for &challenge in challenges.iter().rev() {
                term.folded_multiplier *= F::ONE - challenge + power * challenge;
                power.square_in_place();
            }
            term.folded_through = self.challenges.len();
        }
        let points: Vec<_> = self.terms.iter().map(|term| term.form.point).collect();
        let mut scalars: Vec<_> = self
            .terms
            .iter()
            .map(|term| term.weight * term.folded_multiplier)
            .collect();
        geometric_accumulate_prefixes(
            &mut coefficients,
            &mut scalars,
            &points,
            mask.coefficients_per_polynomial,
            message_length,
        );
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

    pub(super) fn fold(&mut self, opening: &SumcheckOpening<F>) {
        self.scale(opening.mask_rlc);
        self.challenges.extend_from_slice(&opening.round_challenges);
        self.size >>= opening.round_challenges.len();
    }

    pub(super) fn aggregate(&mut self, update: &CovectorUpdateParams<F>) {
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
                folded_multiplier: F::ONE,
                folded_through: self.challenges.len(),
                challenge_offset: self.challenges.len(),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{
        distributions::Standard, prelude::Distribution, rngs::StdRng, SeedableRng,
    };

    use super::*;
    use crate::algebra::{
        fields::{Field64, Field64_2, Field64_3},
        linear_form::{Covector, MultilinearExtension},
        random_vector,
    };

    #[test]
    fn mask_prefixes_match_individual_kernel_coefficients() {
        check_mask_prefixes::<Field64>();
        check_mask_prefixes::<Field64_2>();
        check_mask_prefixes::<Field64_3>();
    }

    fn check_mask_prefixes<F: Field>()
    where
        Standard: Distribution<F>,
    {
        let mut rng = StdRng::seed_from_u64(71);
        let dense = Covector::new(random_vector::<F>(&mut rng, 32));
        let short = UnivariateEvaluation::new(F::from(7), 3);
        let mle = MultilinearExtension::new(random_vector::<F>(&mut rng, 4));
        let zero = MultilinearExtension::new(vec![F::ZERO; 4]);
        let one = MultilinearExtension::new(vec![F::ONE; 4]);
        let constant = MultilinearExtension::new(Vec::new());
        let full = MultilinearExtension::new(random_vector::<F>(&mut rng, 6));
        let empty = Covector::new(Vec::new());
        let forms: [&dyn LinearForm<F>; 8] =
            [&dense, &short, &mle, &zero, &one, &constant, &full, &empty];
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
                for fold_size in [1, 2, 3, 6] {
                    let mut kernel = Kernel {
                        forms: &forms,
                        claim_weights: (1..=forms.len()).map(|i| F::from(i as u64)).collect(),
                        challenges: Vec::new(),
                        initial_scale: F::from(3),
                        size: 64,
                        terms: Vec::new(),
                    };
                    for batch in challenges.chunks(fold_size).map(Some).chain([None]) {
                        // Check both an empty query set and terms inserted after
                        // different numbers of folds, including zero query weights.
                        assert_mask_prefixes(&mut kernel);
                        for (index, &point) in points.iter().enumerate() {
                            kernel.terms.push(KernelTerm {
                                form: UnivariateEvaluation::new(point, kernel.size),
                                weight: if index == 4 { F::ZERO } else { F::from(7) },
                                folded_multiplier: F::ONE,
                                folded_through: kernel.challenges.len(),
                                challenge_offset: kernel.challenges.len(),
                            });
                        }
                        assert_mask_prefixes(&mut kernel);
                        if let Some(batch) = batch {
                            kernel.fold(&SumcheckOpening {
                                round_challenges: batch.to_vec(),
                                mask_rlc: scale,
                            });
                        }
                    }
                }
            }
        }
    }

    fn assert_mask_prefixes<F: Field>(kernel: &mut Kernel<F>) {
        for log_polynomials in 0..=kernel.size.trailing_zeros() {
            let num_polynomials = 1 << log_polynomials;
            let message_length = kernel.size / num_polynomials;
            for prefix in [1, 3.min(message_length), message_length] {
                let mask = WitnessMaskConfig {
                    num_polynomials,
                    coefficients_per_polynomial: prefix,
                };
                let reference = &*kernel;
                let expected: Vec<_> = (0..num_polynomials)
                    .flat_map(|polynomial| {
                        (0..prefix)
                            .map(move |i| reference.coefficient(polynomial * message_length + i))
                    })
                    .collect();
                assert_eq!(kernel.mask_coefficients(mask), expected);
            }
        }
    }
}
