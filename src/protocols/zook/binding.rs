//! Initial joint-mask evaluations retained as constraints for the final opening.

use ark_ff::Field;
use ark_std::rand::{CryptoRng, RngCore};

use crate::{
    algebra::{
        embedding::Identity,
        linear_form::{Covector, Evaluate, UnivariateEvaluation},
    },
    protocols::params::JointMaskConfig,
    transcript::{
        Codec, Decoding, DuplexSpongeInterface, ProverState, VerificationResult, VerifierMessage,
        VerifierState,
    },
};

/// Three deferred equations on the committed joint-mask coefficients.
///
/// Section 6.3, “The zk-IOPP”, Phase 0 requires `m(z) = v`, `V_z(m) = w`,
/// and `V_phi(z)(m) = phi(w)`. Here `V_z` evaluates the concatenation of the
/// initial witness mask and its membership-check blinder, and `phi` fixes the
/// initial witness commitment's field. The final opening checks all three.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct MaskBinding<F: Field> {
    pub(super) forms: [Covector<F>; 3],
    pub(super) values: [F; 3],
}

/// Publish the two Phase 0 evaluations after both commitment roots.
pub(super) fn prove_binding<F, H, R>(
    config: &JointMaskConfig<F>,
    coefficients: &[F],
    source_degree: u64,
    ps: &mut ProverState<H, R>,
) -> MaskBinding<F>
where
    F: Field + Codec<[H::U]>,
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
{
    let point = binding_challenge(ps);
    let forms = binding_forms(config, point, source_degree);
    let identity = Identity::new();
    let evaluation = forms[0].evaluate(&identity, coefficients);
    let membership = forms[1].evaluate(&identity, coefficients);
    ps.prover_message(&evaluation);
    ps.prover_message(&membership);
    MaskBinding {
        forms,
        values: [evaluation, membership, frobenius(membership, source_degree)],
    }
}

/// Receive claimed evaluations; their equations remain deferred to final opening.
pub(super) fn receive_binding<F, H>(
    config: &JointMaskConfig<F>,
    source_degree: u64,
    vs: &mut VerifierState<H>,
) -> VerificationResult<MaskBinding<F>>
where
    F: Field + Codec<[H::U]>,
    H: DuplexSpongeInterface,
{
    let point = binding_challenge(vs);
    let evaluation = vs.prover_message()?;
    let membership = vs.prover_message()?;
    Ok(MaskBinding {
        forms: binding_forms(config, point, source_degree),
        values: [evaluation, membership, frobenius(membership, source_degree)],
    })
}

/// Exclude zero: the shifted membership blinder vanishes there, exposing a mask coefficient.
fn binding_challenge<F, T>(transcript: &mut T) -> F
where
    F: Field + Decoding<[T::U]>,
    T: VerifierMessage,
{
    loop {
        let point: F = transcript.verifier_message();
        if !point.is_zero() {
            return point;
        }
    }
}

fn binding_forms<F: Field>(
    config: &JointMaskConfig<F>,
    point: F,
    source_degree: u64,
) -> [Covector<F>; 3] {
    [
        Covector::from(&UnivariateEvaluation::new(
            point,
            config.commit().vector_size(),
        )),
        membership_form(config, point),
        membership_form(config, frobenius(point, source_degree)),
    ]
}

/// `V_z(m) = initial_mask(z) + z^q * blinder(z)`, with `q` initial-mask coefficients.
///
/// The two ranges need not be adjacent in the joint commitment. Their weights
/// use consecutive powers starting at one, independent of their stored offsets.
fn membership_form<F: Field>(config: &JointMaskConfig<F>, point: F) -> Covector<F> {
    let mut weights = vec![F::ZERO; config.commit().vector_size()];
    let mut power = F::ONE;
    for index in config.witness_masks()[0]
        .clone()
        .chain(config.basefield_blinder().clone())
    {
        weights[index] = power;
        power *= point;
    }
    Covector::new(weights)
}

/// Apply `x -> x^(p^d)`, where `d` is the source field's degree over its prime field.
fn frobenius<F: Field>(mut value: F, source_degree: u64) -> F {
    for _ in 0..source_degree {
        value.frobenius_map_in_place(1);
    }
    value
}

#[cfg(test)]
mod tests {
    use ark_ff::AdditiveGroup;

    use super::*;
    use crate::{
        algebra::{
            embedding::Basefield,
            fields::{Field64, Field64_2},
            univariate_evaluate,
        },
        protocols::params::{
            FoldingFactor, Mode, PowBudget, ProtocolConfig, RateSchedule, SecuritySpec, TuningSpec,
        },
        transcript::{codecs::Empty, DomainSeparator, MockSponge},
    };

    fn config() -> ProtocolConfig<Basefield<Field64_2>> {
        ProtocolConfig::derive(
            SecuritySpec::new(80)
                .with_mode(Mode::ZeroKnowledge)
                .with_pow_budget(PowBudget::Forbidden),
            TuningSpec {
                vector_size: 4096,
                starting_log_inv_rate: 2,
                folding_factor: FoldingFactor::Constant(2),
                rate_schedule: RateSchedule::Stepping,
            },
        )
        .unwrap()
    }

    #[test]
    fn binding_resamples_zero_and_preserves_transcript_agreement() {
        let config = config();
        let layout = config.joint_masks().unwrap();
        let mut coefficients = vec![Field64_2::ZERO; layout.commit().vector_size()];
        coefficients[layout.basefield_blinder().start] = Field64_2::ONE;
        let ds = DomainSeparator::protocol(&"zook-binding-test")
            .session(&"zero challenges")
            .instance(&Empty);
        let mut repr = <Field64_2 as Decoding<[u8]>>::Repr::default();
        let challenge_bytes = repr.as_mut().len();
        for zeros in [0, 1, 3] {
            // Decode little-endian 7 after the rejected zero field elements.
            let mut squeeze = vec![0; (zeros + 1) * challenge_bytes];
            squeeze[zeros * challenge_bytes] = 7;
            squeeze.push(42);
            let sponge = MockSponge {
                absorb: None,
                squeeze: &squeeze,
            };
            let mut ps = ProverState::new(&ds, sponge.clone());
            let binding = prove_binding(layout, &coefficients, 1, &mut ps);
            assert_eq!(binding.forms, binding_forms(layout, Field64_2::from(7), 1));
            assert_ne!(binding.values[1], Field64_2::ZERO);
            assert_eq!(ps.verifier_message::<u8>(), 42);
            let proof = ps.proof();
            let mut vs = VerifierState::new(&ds, &proof, sponge);
            assert_eq!(receive_binding(layout, 1, &mut vs).unwrap(), binding);
            assert_eq!(vs.verifier_message::<u8>(), 42);
            vs.check_eof().unwrap();
        }
    }

    #[test]
    fn membership_uses_logical_concatenation_not_storage_offsets() {
        let config = config();
        let layout = config.joint_masks().unwrap();
        let initial = layout.witness_masks()[0].clone();
        let blinder = layout.basefield_blinder().clone();
        assert!(initial.start > 0);
        assert!(blinder.start > initial.end);
        let coefficients: Vec<_> = (1..=layout.commit().vector_size())
            .map(|index| Field64_2::from(index as u64))
            .collect();
        for point in [Field64_2::ZERO, Field64_2::ONE, Field64_2::from(7)] {
            let expected = univariate_evaluate(&coefficients[initial.clone()], point)
                + point.pow([initial.len() as u64])
                    * univariate_evaluate(&coefficients[blinder.clone()], point);
            let form = membership_form(layout, point);
            assert_eq!(form.evaluate(&Identity::new(), &coefficients), expected);
        }
    }

    #[test]
    fn conjugate_equation_uses_conjugated_value_and_source_degree() {
        let config = config();
        let layout = config.joint_masks().unwrap();
        let point = Field64_2::new(Field64::ONE, Field64::ONE);
        let mut coefficients = vec![Field64_2::ZERO; layout.commit().vector_size()];
        coefficients[layout.witness_masks()[0].start + 1] = Field64_2::ONE;
        let forms = binding_forms(layout, point, 1);
        let value = forms[1].evaluate(&Identity::new(), &coefficients);
        let conjugate = forms[2].evaluate(&Identity::new(), &coefficients);
        assert_ne!(value, conjugate);
        assert_eq!(conjugate, value.frobenius_map(1));

        // A non-source-field constant leaves the first evaluation well-defined
        // but violates the Frobenius equation for the same committed vector.
        coefficients[layout.witness_masks()[0].start] = point;
        let value = forms[1].evaluate(&Identity::new(), &coefficients);
        assert_ne!(
            forms[2].evaluate(&Identity::new(), &coefficients),
            value.frobenius_map(1)
        );

        // When the source is the whole quadratic field, its Frobenius fixes
        // every coefficient, including the non-prime-field constant above.
        let forms = binding_forms(layout, point, Field64_2::extension_degree());
        let value = forms[1].evaluate(&Identity::new(), &coefficients);
        assert_eq!(forms[2].evaluate(&Identity::new(), &coefficients), value);
        assert_eq!(frobenius(value, Field64_2::extension_degree()), value);
    }

    #[test]
    fn false_evaluations_remain_unsatisfied_deferred_equations() {
        let config = config();
        let layout = config.joint_masks().unwrap();
        let coefficients = vec![Field64_2::ONE; layout.commit().vector_size()];
        let ds = DomainSeparator::protocol(&"zook-binding-test")
            .session(&"false evaluations")
            .instance(&Empty);
        for changed in 0..2 {
            let mut ps = ProverState::new_std(&ds);
            let point = binding_challenge(&mut ps);
            let forms = binding_forms(layout, point, 1);
            for (index, form) in forms[..2].iter().enumerate() {
                let mut value = form.evaluate(&Identity::new(), &coefficients);
                if index == changed {
                    value += Field64_2::ONE;
                }
                ps.prover_message(&value);
            }
            let proof = ps.proof();
            let mut vs = VerifierState::new_std(&ds, &proof);
            let binding = receive_binding(layout, 1, &mut vs).unwrap();
            vs.check_eof().unwrap();
            for (index, form) in binding.forms[..2].iter().enumerate() {
                assert_eq!(
                    form.evaluate(&Identity::new(), &coefficients) == binding.values[index],
                    index != changed,
                );
            }
        }
    }

    #[test]
    fn truncated_binding_message_is_rejected() {
        let config = config();
        let layout = config.joint_masks().unwrap();
        let coefficients = vec![Field64_2::ONE; layout.commit().vector_size()];
        let ds = DomainSeparator::protocol(&"zook-binding-test")
            .session(&"truncated evaluation")
            .instance(&Empty);
        let mut ps = ProverState::new_std(&ds);
        let _ = prove_binding(layout, &coefficients, 1, &mut ps);
        let mut proof = ps.proof();
        proof.narg_string.pop();
        let mut vs = VerifierState::new_std(&ds, &proof);
        assert!(receive_binding(layout, 1, &mut vs).is_err());
    }
}
