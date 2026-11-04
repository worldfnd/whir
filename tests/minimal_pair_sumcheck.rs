//! The selector round folds two concatenated witnesses and their linear forms.

use ark_ff::{AdditiveGroup, Field};
use whir::algebra::{
    dot,
    fields::Field64,
    sumcheck::{compute_sumcheck_polynomial, fold},
    univariate_evaluate,
};

#[test]
fn selector_polynomial_is_quadratic_and_starts_from_166() {
    type F = Field64;
    let witness = [1u64, 2, 3, 0, 5, 7, 0, 0].map(F::from);
    let form = [2u64, 3, 4, 0, 11, 13, 0, 0].map(F::from);
    let sum = dot(&witness, &form);
    assert_eq!(sum, F::from(166));
    let (constant, quadratic) = compute_sumcheck_polynomial(&witness, &form);
    let linear = sum - constant.double() - quadratic;
    let coefficients = [constant, linear, quadratic];
    assert_eq!(coefficients, [20u64, 28, 98].map(F::from));
    for s in [F::ZERO, F::ONE, F::from(2), F::from(7)] {
        let mut folded_witness = witness.to_vec();
        let mut folded_form = form.to_vec();
        fold(&mut folded_witness, s);
        fold(&mut folded_form, s);
        assert_eq!(
            dot(&folded_witness, &folded_form),
            univariate_evaluate(&coefficients, s)
        );
    }
    assert_eq!(univariate_evaluate(&coefficients, F::ONE), F::from(146));
    assert_eq!(univariate_evaluate(&coefficients, F::from(2)), F::from(468));
}
