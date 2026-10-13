//! Readable algebra for the extra quadratic selector round.
use ark_ff::{AdditiveGroup, Field};
use whir::algebra::{dot, fields::Field64};

#[test]
fn selector_polynomial_is_quadratic_and_starts_from_166() {
    type F = Field64;
    let left = [1u64, 2, 3, 0].map(F::from);
    let right = [5u64, 7, 0, 0].map(F::from);
    let left_form = [2u64, 3, 4, 0].map(F::from);
    let right_form = [11u64, 13, 0, 0].map(F::from);
    let q = |s: F| {
        let witness: Vec<_> = left
            .iter()
            .zip(right)
            .map(|(&a, b)| (F::ONE - s) * a + s * b)
            .collect();
        let form: Vec<_> = left_form
            .iter()
            .zip(right_form)
            .map(|(&a, b)| (F::ONE - s) * a + s * b)
            .collect();
        dot(&witness, &form)
    };
    assert_eq!(q(F::ZERO), F::from(20));
    assert_eq!(q(F::ONE), F::from(146));
    assert_eq!(q(F::ZERO) + q(F::ONE), F::from(166));
    for s in [F::ZERO, F::ONE, F::from(2), F::from(7)] {
        assert_eq!(q(s), F::from(20) + F::from(28) * s + F::from(98) * s * s);
    }
    assert_eq!(q(F::from(2)), F::from(468));
}
