use ark_ff::Field;
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::algebra::embedding::{Embedding, Identity};

/// Evaluate the multi-linear extension of `evals` in `point`.
#[cfg_attr(feature = "tracing", instrument(skip_all, fields(n = evals.len())))]
pub fn multilinear_extend<F: Field>(evals: &[F], point: &[F]) -> F {
    mixed_multilinear_extend(&Identity::<F>::new(), evals, point)
}

/// Evaluate the multi-linear extension of `evals` in `point`.
pub fn mixed_multilinear_extend<M: Embedding>(
    embedding: &M,
    evals: &[M::Source],
    point: &[M::Target],
) -> M::Target {
    assert_eq!(evals.len(), 1 << point.len());

    // Helper to compute (a + (b - a) * c) efficiently with a, b in source field.
    let mixed = |a, b, c| embedding.mixed_add(embedding.mixed_mul(c, b - a), a);

    match point {
        [] => embedding.map(evals[0]),
        [x] => mixed(evals[0], evals[1], *x),
        [x0, x1] => {
            let a0 = mixed(evals[0], evals[1], *x1);
            let a1 = mixed(evals[2], evals[3], *x1);
            a0 + (a1 - a0) * *x0
        }
        [x0, x1, x2] => {
            let a00 = mixed(evals[0], evals[1], *x2);
            let a01 = mixed(evals[2], evals[3], *x2);
            let a10 = mixed(evals[4], evals[5], *x2);
            let a11 = mixed(evals[6], evals[7], *x2);
            let a0 = a00 + (a01 - a00) * *x1;
            let a1 = a10 + (a11 - a10) * *x1;
            a0 + (a1 - a0) * *x0
        }
        [x0, x1, x2, x3] => {
            let a000 = mixed(evals[0], evals[1], *x3);
            let a001 = mixed(evals[2], evals[3], *x3);
            let a010 = mixed(evals[4], evals[5], *x3);
            let a011 = mixed(evals[6], evals[7], *x3);
            let a100 = mixed(evals[8], evals[9], *x3);
            let a101 = mixed(evals[10], evals[11], *x3);
            let a110 = mixed(evals[12], evals[13], *x3);
            let a111 = mixed(evals[14], evals[15], *x3);
            let a00 = a000 + (a001 - a000) * *x2;
            let a01 = a010 + (a011 - a010) * *x2;
            let a10 = a100 + (a101 - a100) * *x2;
            let a11 = a110 + (a111 - a110) * *x2;
            let a0 = a00 + (a01 - a00) * *x1;
            let a1 = a10 + (a11 - a10) * *x1;
            a0 + (a1 - a0) * *x0
        }
        [x, tail @ ..] => {
            let (f0, f1) = evals.split_at(evals.len() / 2);
            #[cfg(not(feature = "parallel"))]
            let (f0, f1) = (
                mixed_multilinear_extend(embedding, f0, tail),
                mixed_multilinear_extend(embedding, f1, tail),
            );
            #[cfg(feature = "parallel")]
            let (f0, f1) = {
                use crate::utils::workload_size;
                if evals.len() > workload_size::<M::Source>() {
                    rayon::join(
                        || mixed_multilinear_extend(embedding, f0, tail),
                        || mixed_multilinear_extend(embedding, f1, tail),
                    )
                } else {
                    (
                        mixed_multilinear_extend(embedding, f0, tail),
                        mixed_multilinear_extend(embedding, f1, tail),
                    )
                }
            };
            f0 + (f1 - f0) * *x
        }
    }
}

/// Accumulates a scaled evaluation of the equality function.
///
/// Given an evaluation point `point`, the function computes
/// the equality polynomial recursively using the formula:
///
/// ```text
/// eq(X) = ∏ (1 - X_i + 2X_i z_i)
/// ```
///
/// where `z_i` are the  points.
pub fn eval_eq<F: Field>(accumulator: &mut [F], point: &[F], scalar: F) {
    assert_eq!(accumulator.len(), 1 << point.len());
    if let [x0, xs @ ..] = point {
        let (acc_0, acc_1) = accumulator.split_at_mut(1 << xs.len());
        let s1 = scalar * x0; // Contribution when `X_i = 1`
        let s0 = scalar - s1; // Contribution when `X_i = 0`

        #[cfg(feature = "parallel")]
        {
            use crate::utils::workload_size;
            if acc_0.len() > workload_size::<F>() {
                rayon::join(|| eval_eq(acc_0, xs, s0), || eval_eq(acc_1, xs, s1));
                return;
            }
        }
        eval_eq(acc_0, xs, s0);
        eval_eq(acc_1, xs, s1);
    } else {
        accumulator[0] += scalar;
    }
}
