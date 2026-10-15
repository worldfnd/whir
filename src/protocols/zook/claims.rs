//! Linear mask terms carried alongside the folded witness claim.

use std::ops::Range;

use ark_ff::Field;
use zeroize::Zeroizing;

use crate::{
    algebra::{
        embedding::{Identity, PublicScalar},
        linear_form::{Covector, Evaluate},
    },
    protocols::{
        code_switch::CovectorUpdateParams, params::WitnessMaskConfig, sumcheck::SumcheckOpening,
    },
};

/// The public linear functional `B(m)` in `c = <p_hat, K> + B(m)`.
///
/// Section 5.4, “ZO0K”, “Modifications of a WHIR round”: `p_hat` is the
/// masked witness, `K` its current kernel, and `m` the joint mask vector.
/// Mask terms remain in this claim until the final opening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct MaskClaims<F: Field> {
    pub(super) covector: Covector<F>,
}

impl<F: Field> MaskClaims<F> {
    pub(super) fn new(joint_size: usize) -> Self {
        Self {
            covector: Covector::new(vec![F::ZERO; joint_size]),
        }
    }

    /// Evaluate the carried term locally; this value is not a transcript message.
    pub(super) fn evaluate(&self, coefficients: &[F]) -> F {
        self.covector.evaluate(&Identity::new(), coefficients)
    }

    /// Initialize `B(m) = -<mask_0, K>` from each IRS polynomial's prefix.
    ///
    /// `kernel_at` returns a witness-kernel coefficient at its flattened index.
    /// It lets the prover read its dense kernel and the verifier evaluate only
    /// these coefficients without materializing the full witness kernel.
    pub(super) fn subtract_witness_mask(
        &mut self,
        range: Range<usize>,
        mask: WitnessMaskConfig,
        message_length: usize,
        mut kernel_at: impl FnMut(usize) -> F,
    ) {
        assert_eq!(
            range.len(),
            mask.num_polynomials * mask.coefficients_per_polynomial
        );
        assert!(mask.coefficients_per_polynomial <= message_length);
        for (polynomial, weights) in self.covector.vector[range]
            .chunks_exact_mut(mask.coefficients_per_polynomial)
            .enumerate()
        {
            for (coefficient, weight) in weights.iter_mut().enumerate() {
                *weight -= kernel_at(polynomial * message_length + coefficient);
            }
        }
    }

    /// Carry prior mask terms through sumcheck and add its sparse mask evaluation.
    ///
    /// Section 5.4, “ZO0K”, Step 2: after `r` folds, with combination challenge
    /// `lambda`, `B' = lambda * B / 2^r + n * h(beta)`. Here `n` is the remaining
    /// witness size, `beta` the fold challenges, and
    /// `h(beta) = h_0 + sum_j (a_j * beta_j + b_j * beta_j^2)`.
    /// The witness kernel must also be multiplied by `lambda`.
    pub(super) fn fold_sumcheck(
        &mut self,
        range: Range<usize>,
        remaining_size: usize,
        opening: &SumcheckOpening<F>,
    ) {
        let rounds = opening.round_challenges.len();
        assert_eq!(range.len(), 2 * rounds + 1);
        // Sumcheck configuration requires odd characteristic. No inverse of a
        // transcript challenge is needed, including when lambda is zero.
        let divisor = F::BasePrimeField::from(2).pow([rounds as u64]);
        let inverse = F::BasePrimeField::ONE / divisor;
        let scale = opening.mask_rlc.mul_by_base_prime_field(&inverse);
        for weight in &mut self.covector.vector {
            *weight *= scale;
        }
        let scale = F::BasePrimeField::from(remaining_size as u64);
        let weights = &mut self.covector.vector[range];
        weights[0] += F::from_base_prime_field(scale);
        for (pair, &challenge) in weights[1..]
            .chunks_exact_mut(2)
            .zip(&opening.round_challenges)
        {
            pair[0] += challenge.mul_by_base_prime_field(&scale);
            pair[1] += challenge.square().mul_by_base_prime_field(&scale);
        }
    }

    /// Add the next overlaid witness mask after code-switch aggregation.
    ///
    /// Section 5.4, “ZO0K”, Steps 3–4: the folded claim and in-domain queries
    /// constrain `p_hat_next - mask_next`, while DEEP queries constrain
    /// `p_hat_next`. Thus the new term is `-<mask_next, K_next - K_DEEP>`.
    /// `kernel_at` supplies coefficients of the updated kernel `K_next`.
    pub(super) fn subtract_target_mask(
        &mut self,
        range: Range<usize>,
        mask: WitnessMaskConfig,
        message_length: usize,
        update: &CovectorUpdateParams<F>,
        mut kernel_at: impl FnMut(usize) -> F,
    ) {
        assert_eq!(update.ood_rlc_coeffs.len(), update.ood_eval_points.len());
        if !update.original_sl_coeff.is_one() {
            let scale = PublicScalar::new(update.original_sl_coeff);
            for weight in &mut self.covector.vector {
                *weight = scale.mul(*weight);
            }
        }
        self.subtract_witness_mask(range, mask, message_length, |index| {
            let deep: F = update
                .ood_rlc_coeffs
                .iter()
                .zip(&update.ood_eval_points)
                .map(|(&weight, point)| weight * point.pow([index as u64]))
                .sum();
            kernel_at(index) - deep
        });
    }
}

/// Expand one sparse quadratic mask into the existing sumcheck's round masks.
///
/// Section 5.4, “ZO0K”, Step 1: `[h_0, a_0, b_0, ...]` represents
/// `h(X) = h_0 + sum_j (a_j * X_j + b_j * X_j^2)`.
/// For `n` remaining witness entries, sumcheck receives `[n*h_0, n*a_0,
/// n*b_0], [0, n*a_1, n*b_1], ...`. Its existing Boolean-sum recurrence then
/// contributes `n * h(beta)` without revealing a separate mask evaluation.
pub(super) fn sumcheck_blinding<F: Field>(
    coefficients: &[F],
    remaining_size: usize,
) -> Zeroizing<Vec<F>> {
    assert!(coefficients.len() >= 3 && coefficients.len() % 2 == 1);
    let scale = F::BasePrimeField::from(remaining_size as u64);
    let mut expanded = Zeroizing::new(vec![F::ZERO; 3 * (coefficients.len() / 2)]);
    expanded[0] = coefficients[0].mul_by_base_prime_field(&scale);
    for (mask, pair) in expanded
        .chunks_exact_mut(3)
        .zip(coefficients[1..].chunks_exact(2))
    {
        mask[1] = pair[0].mul_by_base_prime_field(&scale);
        mask[2] = pair[1].mul_by_base_prime_field(&scale);
    }
    expanded
}

#[cfg(test)]
mod tests {
    use ark_ff::AdditiveGroup;
    use ark_std::rand::{
        distributions::Standard, prelude::Distribution, rngs::StdRng, SeedableRng,
    };

    use super::*;
    use crate::{
        algebra::{
            dot,
            embedding::{Basefield, Embedding},
            fields::{Field64, Field64_2},
            random_vector, univariate_evaluate,
        },
        buffer::{Buffer, BufferMath, BufferOps},
        protocols::{
            proof_of_work::Config as PowConfig,
            sumcheck::{Config as SumcheckConfig, SumcheckMaskLen, SumcheckMode},
            zook::commit::overlay_mask,
        },
        transcript::{
            codecs::Empty, Codec, DomainSeparator, ProverState, VerifierMessage, VerifierState,
        },
    };

    fn sumcheck_roundtrip<M: Embedding<Target = Field64_2> + Default>()
    where
        Standard: Distribution<M::Source>,
        M::Target: Codec<[u8]>,
    {
        let embedding = M::default();
        let mut rng = StdRng::seed_from_u64(31);
        for rounds in 1..=3 {
            for remaining_size in [1, 8] {
                let initial_size = remaining_size << rounds;
                let mask = WitnessMaskConfig {
                    num_polynomials: 1 << rounds,
                    coefficients_per_polynomial: remaining_size.min(2),
                };
                let sparse = 0..2 * rounds + 1;
                let initial = sparse.end
                    ..sparse.end + mask.num_polynomials * mask.coefficients_per_polynomial;
                let mut coefficients = random_vector::<Field64_2>(&mut rng, initial.end + 7);
                let source_mask: Vec<M::Source> = random_vector(&mut rng, initial.len());
                for (target, &source) in coefficients[initial.clone()].iter_mut().zip(&source_mask)
                {
                    *target = embedding.map(source);
                }
                let witness = Buffer::<M::Source>::random(&mut rng, initial_size);
                let mut kernel = Buffer::<Field64_2>::random(&mut rng, initial_size);
                let original_sum = witness.mixed_dot(&embedding, &kernel);
                let mut claims = MaskClaims::new(coefficients.len());
                claims.subtract_witness_mask(initial, mask, remaining_size, |index| {
                    kernel.to_slice()[index]
                });
                let mut received_claims = claims.clone();
                let witness = overlay_mask(witness, &source_mask, mask);
                let carried = claims.evaluate(&coefficients);
                let mut sum = witness.mixed_dot(&embedding, &kernel);
                assert_eq!(sum + carried, original_sum);

                let config = SumcheckConfig::new(
                    initial_size,
                    PowConfig::none(),
                    rounds,
                    SumcheckMode::ZeroKnowledge {
                        mask_length: SumcheckMaskLen::new(3),
                    },
                );
                let ds = DomainSeparator::protocol(&config)
                    .session(&"joint mask claim")
                    .instance(&Empty);
                let mut ps = ProverState::new_std(&ds);
                let blinding = sumcheck_blinding(&coefficients[sparse.clone()], remaining_size);
                let (folded, opening) = config.prove_with_carried_sum(
                    &mut ps,
                    &embedding,
                    witness,
                    &mut kernel,
                    &mut sum,
                    &blinding,
                    carried,
                );
                let sparse_evaluation = coefficients[sparse.start]
                    + coefficients[sparse.start + 1..sparse.end]
                        .chunks_exact(2)
                        .zip(&opening.round_challenges)
                        .map(|(pair, &challenge)| {
                            pair[0] * challenge + pair[1] * challenge.square()
                        })
                        .sum::<Field64_2>();
                assert_eq!(
                    sum,
                    opening.mask_rlc
                        * (folded.dot(&kernel) + carried / Field64_2::from(1_u64 << rounds))
                        + Field64_2::from(remaining_size as u64) * sparse_evaluation
                );
                kernel.scalar_mul(opening.mask_rlc);
                claims.fold_sumcheck(sparse.clone(), remaining_size, &opening);
                assert_eq!(sum, folded.dot(&kernel) + claims.evaluate(&coefficients));
                let challenge: Field64_2 = ps.verifier_message();
                let proof = ps.proof();

                let mut vs = VerifierState::new_std(&ds, &proof);
                let mut received_sum = original_sum;
                let received = config.verify(&mut vs, &mut received_sum).unwrap();
                received_claims.fold_sumcheck(sparse, remaining_size, &received);
                assert_eq!(received.round_challenges, opening.round_challenges);
                assert_eq!(received.mask_rlc, opening.mask_rlc);
                assert_eq!(received_sum, sum);
                assert_eq!(received_claims, claims);
                assert_eq!(vs.verifier_message::<Field64_2>(), challenge);
                vs.check_eof().unwrap();
            }
        }
    }

    #[test]
    fn sparse_sumcheck_carries_claims_across_a_field_embedding() {
        sumcheck_roundtrip::<Basefield<Field64_2>>();
    }

    #[test]
    fn sparse_sumcheck_carries_claims_in_the_same_field() {
        sumcheck_roundtrip::<Identity<Field64_2>>();
    }

    #[test]
    fn zero_combination_challenge_drops_old_claims_without_an_inverse() {
        let mut claims = MaskClaims::<Field64>::new(9);
        claims.covector.vector.fill(Field64::from(7));
        let opening = SumcheckOpening {
            round_challenges: vec![Field64::ZERO, Field64::ONE],
            mask_rlc: Field64::ZERO,
        };
        claims.fold_sumcheck(4..9, 4, &opening);
        assert_eq!(
            claims.covector.vector,
            [0, 0, 0, 0, 4, 0, 0, 4, 4].map(Field64::from)
        );
    }

    #[test]
    fn target_mask_corrects_folded_and_in_domain_claims_but_not_deep() {
        let coefficients: Vec<Field64> = (1..=9).map(Field64::from).collect();
        let folded = Buffer::from((10..18).map(Field64::from).collect::<Vec<_>>());
        let kernel: Vec<Field64> = (20..28).map(Field64::from).collect();
        let mask = WitnessMaskConfig {
            num_polynomials: 2,
            coefficients_per_polynomial: 2,
        };
        let range = 3..7;
        let mut claims = MaskClaims::new(coefficients.len());
        claims.covector.vector[..3].copy_from_slice(&[2, 3, 5].map(Field64::from));
        let original_sum = dot(folded.to_slice(), &kernel) + claims.evaluate(&coefficients);
        let update = CovectorUpdateParams {
            original_sl_coeff: Field64::from(3),
            ood_rlc_coeffs: vec![Field64::from(7)],
            ood_eval_points: vec![Field64::from(5)],
            in_domain_rlc_coeffs: vec![Field64::from(11), Field64::from(13)],
            in_domain_eval_points: vec![Field64::from(2), Field64::from(3)],
        };
        let masked = overlay_mask(folded.clone(), &coefficients[range.clone()], mask);
        let sum = update.original_sl_coeff * original_sum
            + update
                .ood_rlc_coeffs
                .iter()
                .zip(&update.ood_eval_points)
                .map(|(&weight, &point)| weight * univariate_evaluate(masked.to_slice(), point))
                .sum::<Field64>()
            + update
                .in_domain_rlc_coeffs
                .iter()
                .zip(&update.in_domain_eval_points)
                .map(|(&weight, &point)| weight * univariate_evaluate(folded.to_slice(), point))
                .sum::<Field64>();
        let updated_kernel: Vec<_> = kernel
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                update.original_sl_coeff * value
                    + update
                        .ood_rlc_coeffs
                        .iter()
                        .zip(&update.ood_eval_points)
                        .chain(
                            update
                                .in_domain_rlc_coeffs
                                .iter()
                                .zip(&update.in_domain_eval_points),
                        )
                        .map(|(&weight, point)| weight * point.pow([index as u64]))
                        .sum::<Field64>()
            })
            .collect();
        claims.subtract_target_mask(range, mask, 4, &update, |index| updated_kernel[index]);
        assert_eq!(
            sum,
            dot(masked.to_slice(), &updated_kernel) + claims.evaluate(&coefficients)
        );
        assert_eq!(&claims.covector.vector[7..], &[Field64::ZERO; 2]);
    }
}
