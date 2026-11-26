//! Multilinear extensions, equality-polynomial weights, and eq-weighted chunk folding.

use ark_ff::{AdditiveGroup, Field};

use crate::algebra::embedding::{Embedding, Identity};

/// Evaluate the multi-linear extension of `evals` in `point`.
pub fn multilinear_extend<F: Field>(evals: &[F], point: &[F]) -> F {
    mixed_multilinear_extend(&Identity::<F>::new(), evals, point)
}

/// Evaluate the multi-linear extension of `evals` in `point`.
///
/// Supports implicit zero-padding: when `evals.len() < 1 << point.len()`,
/// the missing tail entries are treated as zeros.
#[allow(clippy::too_many_lines)]
pub fn mixed_multilinear_extend<M: Embedding>(
    embedding: &M,
    evals: &[M::Source],
    point: &[M::Target],
) -> M::Target {
    #[inline]
    fn eval_exact<M: Embedding>(
        embedding: &M,
        evals: &[M::Source],
        point: &[M::Target],
    ) -> M::Target {
        debug_assert_eq!(evals.len(), 1 << point.len());

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
                    eval_exact(embedding, f0, tail),
                    eval_exact(embedding, f1, tail),
                );

                #[cfg(feature = "parallel")]
                let (f0, f1) = {
                    use crate::utils::workload_size;
                    if evals.len() > workload_size::<M::Source>() {
                        rayon::join(
                            || eval_exact(embedding, f0, tail),
                            || eval_exact(embedding, f1, tail),
                        )
                    } else {
                        (
                            eval_exact(embedding, f0, tail),
                            eval_exact(embedding, f1, tail),
                        )
                    }
                };

                f0 + (f1 - f0) * *x
            }
        }
    }

    #[inline]
    fn eval_partial<M: Embedding>(
        embedding: &M,
        evals: &[M::Source],
        point: &[M::Target],
    ) -> M::Target {
        let size = 1 << point.len();
        debug_assert!(evals.len() <= size);
        if evals.is_empty() {
            return M::Target::ZERO;
        }
        if evals.len() == size {
            return eval_exact(embedding, evals, point);
        }

        match point {
            [] => embedding.map(evals[0]),
            [x, tail @ ..] => {
                let half = size / 2;

                // Only low half has data; high half is all implicit zeros.
                if evals.len() <= half {
                    let f0 = eval_partial(embedding, evals, tail);
                    return f0 * (M::Target::ONE - *x);
                }

                // Low subtree is exact/full, high subtree is partial.
                let (low, high) = evals.split_at(half);

                #[cfg(not(feature = "parallel"))]
                let (f0, f1) = (
                    eval_exact(embedding, low, tail),
                    eval_partial(embedding, high, tail),
                );

                #[cfg(feature = "parallel")]
                let (f0, f1) = {
                    use crate::utils::workload_size;
                    if evals.len() > workload_size::<M::Source>() {
                        rayon::join(
                            || eval_exact(embedding, low, tail),
                            || eval_partial(embedding, high, tail),
                        )
                    } else {
                        (
                            eval_exact(embedding, low, tail),
                            eval_partial(embedding, high, tail),
                        )
                    }
                };

                f0 + (f1 - f0) * *x
            }
        }
    }

    eval_partial(embedding, evals, point)
}

/// Computes eq(points, p) on the hypercube for all p ∈ {0,1}^k.
pub fn eq_weights<F: Field>(point: &[F]) -> Vec<F> {
    let mut result = vec![F::ZERO; 1 << point.len()];
    eval_eq(&mut result, point, F::ONE);
    result
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
/// where `z_i` are the points.
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

/// Fold ι parallel chunks of length `chunk_len` into a single chunk.
///
/// Uses `eq_weights(γ)` over the layout
/// `values = [chunk_0; chunk_1; ...; chunk_{ι−1}]` (each chunk of length
/// `chunk_len`) and returns `Σ_l eq_weights(γ)[l] · chunk_l`.
pub fn fold_chunks<F: Field>(values: &[F], chunk_len: usize, folding_randomness: &[F]) -> Vec<F> {
    // The identity case of `mixed_fold_chunks`: for `Identity<F>`, `map` is a
    // no-op, so this monomorphizes to a plain multiply-and-sum with no overhead.
    mixed_fold_chunks(&Identity::<F>::new(), values, chunk_len, folding_randomness)
}

/// Embedding-aware [`fold_chunks`].
///
/// The values live in the source field `M::Source` while the folding randomness
/// lives in the target field `M::Target`, so each output chunk is
/// `Σ_l eq_weights(γ)[l] · φ(chunk_l)` with `φ = embedding.map`. Used to fold a
/// round's base-field IRS masks by the ext-field sumcheck challenges
/// (Construction 9.7 `(r ‖ s)`).
pub fn mixed_fold_chunks<M: Embedding>(
    embedding: &M,
    values: &[M::Source],
    chunk_len: usize,
    folding_randomness: &[M::Target],
) -> Vec<M::Target> {
    let iota = 1 << folding_randomness.len();
    assert_eq!(values.len(), chunk_len * iota);
    if iota == 1 {
        // No folding: each chunk is a single value, lifted into `M::Target`.
        return values.iter().map(|&v| embedding.map(v)).collect();
    }
    let weights = eq_weights(folding_randomness);
    (0..chunk_len)
        .map(|j| {
            (0..iota)
                .map(|l| embedding.mixed_mul(weights[l], values[l * chunk_len + j]))
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use proptest::proptest;

    use super::*;
    use crate::{algebra::random_vector, test_utils::zero_pad};

    pub type F = crate::algebra::fields::Field64;

    #[test]
    fn test_multilinear_zero_extend() {
        proptest!(|(seed:u64, length in 0_usize..(1 << 14))| {
            let mut rng = StdRng::seed_from_u64(seed);
            let vector: Vec<F> = random_vector(&mut rng, length);
            let extended_vector = zero_pad(&vector);
            let num_variables = length.next_power_of_two().trailing_zeros();
            let point = random_vector(&mut rng, num_variables as usize);
            assert_eq!(
                multilinear_extend(&vector, &point),
                multilinear_extend(&extended_vector, &point)
            );
        });
    }

    #[test]
    fn test_multilinear_extra_variables() {
        proptest!(|(seed:u64, length in 0_usize..(1 << 10), excess_variables in 0_usize..3)| {
            let mut rng = StdRng::seed_from_u64(seed);
            let vector: Vec<F> = random_vector(&mut rng, length);
            let num_variables = length.next_power_of_two().trailing_zeros() as usize + excess_variables;
            let point = random_vector(&mut rng, num_variables);
            let mut extended_vector = vector.clone();
            extended_vector.resize(1 << num_variables, F::ZERO);
            assert_eq!(
                multilinear_extend(&vector, &point),
                multilinear_extend(&extended_vector, &point)
            );
        });
    }
}
