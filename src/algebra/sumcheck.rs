use ark_ff::{AdditiveGroup, Field};
#[cfg(feature = "parallel")]
use rayon::{join, prelude::*};

use crate::algebra::{dot, embedding::Embedding, mixed_dot, scalar_mul};
#[cfg(feature = "parallel")]
use crate::utils::workload_size;

/// Computes the constant and quadratic coefficient of the sumcheck polynomial.
///
/// Vectors `a` and `b` are implicitly zero-extended to the next power of two.
pub fn compute_sumcheck_polynomial<F: Field>(a: &[F], b: &[F]) -> (F, F) {
    fn recurse<F: Field>(a0: &[F], a1: &[F], b0: &[F], b1: &[F]) -> (F, F) {
        debug_assert_eq!(a0.len(), b0.len());
        debug_assert_eq!(a1.len(), b1.len());
        debug_assert_eq!(a0.len(), a1.len());

        #[cfg(feature = "parallel")]
        if a0.len() * 4 > workload_size::<F>() {
            let mid = a0.len() / 2;
            let (a0l, a0r) = a0.split_at(mid);
            let (b0l, b0r) = b0.split_at(mid);
            let (a1l, a1r) = a1.split_at(mid);
            let (b1l, b1r) = b1.split_at(mid);
            let (left, right) = join(
                || recurse(a0l, a1l, b0l, b1l),
                || recurse(a0r, a1r, b0r, b1r),
            );
            return (left.0 + right.0, left.1 + right.1);
        }
        let mut acc0 = F::ZERO;
        let mut acc2 = F::ZERO;
        for ((&a0, &a1), (&b0, &b1)) in a0.iter().zip(a1).zip(b0.iter().zip(b1)) {
            acc0 += a0 * b0;
            acc2 += (a1 - a0) * (b1 - b0);
        }
        (acc0, acc2)
    }

    let non_padded = a.len().min(b.len());
    let a = &a[..non_padded];
    let b = &b[..non_padded];
    if a.is_empty() {
        return (F::ZERO, F::ZERO);
    }
    if a.len() == 1 {
        return (a[0] * b[0], F::ZERO);
    }

    let half = a.len().next_power_of_two() >> 1;
    let (a0, a1) = a.split_at(half);
    let (b0, b1) = b.split_at(half);
    debug_assert!(a0.len() >= a1.len());
    let (a0, a0_tail) = a0.split_at(a1.len());
    let (b0, b0_tail) = b0.split_at(a1.len());
    let (acc0, acc2) = recurse(a0, a1, b0, b1);

    // Handle the tail part where a1, b1 is implicit zero padding,
    // When a1, b1 = 0, then acc0 = acc2 = a0 * b0:
    let acc = dot(a0_tail, b0_tail);

    (acc0 + acc, acc2 + acc)
}

/// Folds evaluations by linear interpolation at the given weight, in place.
///
/// The `values` are implicitly zero-padded to the next power of two. On return,
/// the length of `values` will always be a power of two.
pub fn fold<F: Field>(values: &mut Vec<F>, weight: F) {
    fn recurse_both<F: Field>(low: &mut [F], high: &[F], weight: F) {
        #[cfg(feature = "parallel")]
        if low.len() > workload_size::<F>() {
            let split = low.len() / 2;
            let (ll, lr) = low.split_at_mut(split);
            let (hl, hr) = high.split_at(split);
            rayon::join(
                || recurse_both(ll, hl, weight),
                || recurse_both(lr, hr, weight),
            );
            return;
        }

        for (low, high) in low.iter_mut().zip(high) {
            *low += (*high - *low) * weight;
        }
    }

    if values.len() <= 1 {
        return;
    }

    let half = values.len().next_power_of_two() >> 1;
    let (low, high) = values.split_at_mut(half);
    debug_assert!(low.len() >= high.len());
    let (low, tail) = low.split_at_mut(high.len());
    recurse_both(low, high, weight);

    // Tail part where `high` is implicit zero padding
    // When high = 0 we have *low *= 1 - weight.
    scalar_mul(tail, F::ONE - weight);

    values.truncate(half);
    values.shrink_to_fit();
}

pub fn fold_and_compute_polynomial<F: Field>(a: &mut Vec<F>, b: &mut Vec<F>, weight: F) -> (F, F) {
    // TODO: Replace with a single pass implementation.
    fold(a, weight);
    fold(b, weight);
    compute_sumcheck_polynomial(a, b)
}

/// Computes the round polynomial of the cubic sumcheck
///
/// ```text
///   p(X) = Σ_{b ∈ {0,1}^{n-1}}  eq(X, b) · m(X, b) · v(X, b)
/// ```
///
/// where `eq`, `m`, and `v` are multilinear over `n` variables, supplied
/// as their length-`2^n` evaluation tables on the Boolean hypercube. The
/// `X = 0` half occupies the first `2^{n-1}` entries; the `X = 1` half
/// occupies the rest. Returns `[c_0, c_2, c_3]` — the three coefficients
/// the prover sends; `c_1` is recovered by the verifier from
/// `p(0) + p(1) = sum`.
///
/// Implicitly zero-extends `eq`, `m`, `v` to the next power of two when
/// their lengths are equal but not already a power of two. (When the input
/// lengths differ, the shortest length is used and the others are truncated
/// — matching the convention of [`compute_sumcheck_polynomial`].)
pub fn compute_round_poly_degree3<F: Field>(eq: &[F], m: &[F], v: &[F]) -> (F, F, F) {
    let non_padded = eq.len().min(m.len()).min(v.len());
    let eq = &eq[..non_padded];
    let m = &m[..non_padded];
    let v = &v[..non_padded];

    if eq.is_empty() {
        return (F::ZERO, F::ZERO, F::ZERO);
    }
    if eq.len() == 1 {
        // X has no half to split into; the X=1 side is implicit zero. So:
        //   p(X) = eq[0]·(1-X) · m[0]·(1-X) · v[0]·(1-X)
        //        = eq[0]·m[0]·v[0] · (1 - X)^3
        //        = eq[0]·m[0]·v[0] · (1 - 3X + 3X^2 - X^3)
        let t = eq[0] * m[0] * v[0];
        return (t, t + t + t, -t);
    }

    let half = eq.len().next_power_of_two() >> 1;
    let (eq0, eq1) = eq.split_at(half);
    let (m0, m1) = m.split_at(half);
    let (v0, v1) = v.split_at(half);
    debug_assert!(eq0.len() >= eq1.len());
    // Split eq0/m0/v0 into a paired prefix (matching eq1/m1/v1) and a tail
    // where the X=1 side is implicit zero padding.
    let (eq0_paired, eq0_tail) = eq0.split_at(eq1.len());
    let (m0_paired, m0_tail) = m0.split_at(v1.len());
    let (v0_paired, v0_tail) = v0.split_at(v1.len());

    // p(X) = (e0 + (e1-e0)·X) · (m0 + (m1-m0)·X) · (v0 + (v1-v0)·X)
    //      = a0 + a1·X + a2·X^2 + a3·X^3
    // a0 = e0·m0·v0
    // a2 = e0·dm·dv + m0·dv·de + v0·dm·de      where d* = (*_1 - *_0)
    // a3 = de·dm·dv
    let mut acc0 = F::ZERO;
    let mut acc2 = F::ZERO;
    let mut acc3 = F::ZERO;
    for ((((&e0, &e1), (&mm0, &mm1)), &vv0), &vv1) in eq0_paired
        .iter()
        .zip(eq1.iter())
        .zip(m0_paired.iter().zip(m1.iter()))
        .zip(v0_paired.iter())
        .zip(v1.iter())
    {
        let de = e1 - e0;
        let dm = mm1 - mm0;
        let dv = vv1 - vv0;

        acc0 += e0 * mm0 * vv0;
        acc2 += e0 * dm * dv + mm0 * dv * de + vv0 * dm * de;
        acc3 += de * dm * dv;
    }

    // Tail (`X=1` side implicit zero): p(X) = e0·m0·v0 · (1-X)^3, contributes
    //   c0 += t,  c2 += 3·t,  c3 -= t.
    let mut tail = F::ZERO;
    for ((&e0, &mm0), &vv0) in eq0_tail.iter().zip(m0_tail.iter()).zip(v0_tail.iter()) {
        tail += e0 * mm0 * vv0;
    }
    acc0 += tail;
    acc2 += tail + tail + tail;
    acc3 -= tail;

    (acc0, acc2, acc3)
}

/// Embedding-aware [`compute_sumcheck_polynomial`]: `a` lives in the source
/// field, `b` in the target field.
///
/// Computes the same `(c0, c2)` as `compute_sumcheck_polynomial(lift(a), b)`
/// (the embedding is a ring homomorphism) without materializing the lift.
pub fn mixed_compute_sumcheck_polynomial<M: Embedding>(
    embedding: &M,
    a: &[M::Source],
    b: &[M::Target],
) -> (M::Target, M::Target) {
    fn recurse<M: Embedding>(
        embedding: &M,
        a0: &[M::Source],
        a1: &[M::Source],
        b0: &[M::Target],
        b1: &[M::Target],
    ) -> (M::Target, M::Target) {
        debug_assert_eq!(a0.len(), b0.len());
        debug_assert_eq!(a1.len(), b1.len());
        debug_assert_eq!(a0.len(), a1.len());

        #[cfg(feature = "parallel")]
        if a0.len() * 4 > workload_size::<M::Target>() {
            let mid = a0.len() / 2;
            let (a0l, a0r) = a0.split_at(mid);
            let (b0l, b0r) = b0.split_at(mid);
            let (a1l, a1r) = a1.split_at(mid);
            let (b1l, b1r) = b1.split_at(mid);
            let (left, right) = join(
                || recurse(embedding, a0l, a1l, b0l, b1l),
                || recurse(embedding, a0r, a1r, b0r, b1r),
            );
            return (left.0 + right.0, left.1 + right.1);
        }
        let mut acc0 = M::Target::ZERO;
        let mut acc2 = M::Target::ZERO;
        for ((&a0, &a1), (&b0, &b1)) in a0.iter().zip(a1).zip(b0.iter().zip(b1)) {
            acc0 += embedding.mixed_mul(b0, a0);
            acc2 += embedding.mixed_mul(b1 - b0, a1 - a0);
        }
        (acc0, acc2)
    }

    let non_padded = a.len().min(b.len());
    let a = &a[..non_padded];
    let b = &b[..non_padded];
    if a.is_empty() {
        return (M::Target::ZERO, M::Target::ZERO);
    }
    if a.len() == 1 {
        return (embedding.mixed_mul(b[0], a[0]), M::Target::ZERO);
    }

    let half = a.len().next_power_of_two() >> 1;
    let (a0, a1) = a.split_at(half);
    let (b0, b1) = b.split_at(half);
    debug_assert!(a0.len() >= a1.len());
    let (a0, a0_tail) = a0.split_at(a1.len());
    let (b0, b0_tail) = b0.split_at(a1.len());
    let (acc0, acc2) = recurse(embedding, a0, a1, b0, b1);

    // Handle the tail part where a1, b1 is implicit zero padding,
    // When a1, b1 = 0, then acc0 = acc2 = a0 * b0:
    let acc = mixed_dot(embedding, b0_tail, a0_tail);

    (acc0 + acc, acc2 + acc)
}

/// Embedding-aware [`fold`]: folds source-field `values` at a target-field
/// `weight`, lifting the result into the target field.
///
/// Returns the same vector as `fold(lift(values), weight)` (the embedding is a
/// ring homomorphism) without materializing the lift. Like [`fold`], `values`
/// is implicitly zero-padded to the next power of two and the output length is
/// always a power of two (or zero).
pub fn mixed_fold<M: Embedding>(
    embedding: &M,
    values: &[M::Source],
    weight: M::Target,
) -> Vec<M::Target> {
    if values.len() <= 1 {
        return values.iter().map(|&v| embedding.map(v)).collect();
    }

    let half = values.len().next_power_of_two() >> 1;
    let (low, high) = values.split_at(half);
    debug_assert!(low.len() >= high.len());
    let (low, tail) = low.split_at(high.len());

    let folded = |(&low, &high): (&M::Source, &M::Source)| {
        embedding.mixed_add(embedding.mixed_mul(weight, high - low), low)
    };
    // Tail part where `high` is implicit zero padding: low * (1 - weight).
    let tail_folded = |&low: &M::Source| embedding.mixed_mul(M::Target::ONE - weight, low);

    #[cfg(feature = "parallel")]
    if half > workload_size::<M::Target>() {
        let mut out: Vec<M::Target> = low.par_iter().zip(high).map(folded).collect();
        out.par_extend(tail.par_iter().map(tail_folded));
        return out;
    }

    let mut out: Vec<M::Target> = low.iter().zip(high).map(folded).collect();
    out.extend(tail.iter().map(tail_folded));
    out
}

/// Evaluate a coefficient vector at a multilinear point in the target field.
pub fn mixed_eval<M: Embedding>(
    embedding: &M,
    coeff: &[M::Source],
    eval: &[M::Target],
    scalar: M::Target,
) -> M::Target {
    debug_assert_eq!(coeff.len(), 1 << eval.len());

    if let Some((&x, tail)) = eval.split_first() {
        let (low, high) = coeff.split_at(coeff.len() / 2);

        #[cfg(feature = "parallel")]
        if low.len() > workload_size::<M::Source>() {
            let (a, b) = join(
                || mixed_eval(embedding, low, tail, scalar),
                || mixed_eval(embedding, high, tail, scalar * x),
            );
            return a + b;
        }

        mixed_eval(embedding, low, tail, scalar) + mixed_eval(embedding, high, tail, scalar * x)
    } else {
        embedding.mixed_mul(scalar, coeff[0])
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use ark_std::rand::{rngs::StdRng, Rng, SeedableRng};
    use proptest::proptest;

    use super::*;
    use crate::algebra::{fields::Field64, random_vector};

    type F = Field64;

    /// Zero-pad to the next power of two.
    pub fn zero_pad<F: Field>(values: &[F]) -> Vec<F> {
        if values.is_empty() {
            return Vec::new();
        }
        let mut vec = values.to_vec();
        vec.resize(vec.len().next_power_of_two(), F::ZERO);
        vec
    }

    #[test]
    fn sumcheck_poly_zero_extend() {
        proptest!(|(seed:u64, length in 0_usize..(1 << 14))| {
            let mut rng = StdRng::seed_from_u64(seed);
            let vector: Vec<F> = random_vector(&mut rng, length);
            let covector: Vec<F> = random_vector(&mut rng, length);
            let extended_vector = zero_pad(&vector);
            let extended_covector = zero_pad(&covector);
            let expected = compute_sumcheck_polynomial(&extended_vector, &extended_covector);
            assert_eq!(compute_sumcheck_polynomial(&vector, &covector), expected);
            assert_eq!(compute_sumcheck_polynomial(&extended_vector, &covector), expected);
            assert_eq!(compute_sumcheck_polynomial(&vector, &extended_covector), expected);
        });
    }

    /// Naive reference: enumerate every Boolean hypercube assignment and
    /// evaluate `p(X) = Σ_b eq(X, b) · m(X, b) · v(X, b)` at four points,
    /// then interpolate. Slow but obviously correct.
    fn naive_round_poly_degree3<F: Field>(eq: &[F], m: &[F], v: &[F]) -> (F, F, F) {
        let n = eq.len().max(m.len()).max(v.len()).next_power_of_two();
        let eq = {
            let mut e = eq.to_vec();
            e.resize(n, F::ZERO);
            e
        };
        let m = {
            let mut x = m.to_vec();
            x.resize(n, F::ZERO);
            x
        };
        let v = {
            let mut x = v.to_vec();
            x.resize(n, F::ZERO);
            x
        };
        if n <= 1 {
            // Trivial case: single value, X=1 side implicit zero.
            // p(X) = e[0]·m[0]·v[0]·(1-X)^3 = t·(1 - 3X + 3X² - X³)
            let e0 = *eq.first().unwrap_or(&F::ZERO);
            let m0 = *m.first().unwrap_or(&F::ZERO);
            let v0 = *v.first().unwrap_or(&F::ZERO);
            let t = e0 * m0 * v0;
            return (t, t + t + t, -t);
        }
        let half = n / 2;
        // Evaluate p at X = 0, 1, 2, 3 (four points uniquely determine a cubic).
        let eval_at = |x: F| -> F {
            let one_minus_x = F::ONE - x;
            let mut acc = F::ZERO;
            for b in 0..half {
                let e_at_x = eq[b] * one_minus_x + eq[b + half] * x;
                let m_at_x = m[b] * one_minus_x + m[b + half] * x;
                let v_at_x = v[b] * one_minus_x + v[b + half] * x;
                acc += e_at_x * m_at_x * v_at_x;
            }
            acc
        };
        let p0 = eval_at(F::ZERO);
        let p1 = eval_at(F::ONE);
        let two = F::ONE + F::ONE;
        let three = two + F::ONE;
        let p2 = eval_at(two);
        let p3 = eval_at(three);
        // Solve for [c0, c1, c2, c3] from p(0..3).
        //   c0 = p0
        //   c0 + c1 + c2 + c3 = p1
        //   c0 + 2c1 + 4c2 + 8c3 = p2
        //   c0 + 3c1 + 9c2 + 27c3 = p3
        // Forward-difference (finite-difference inversion):
        //   c3 = (p3 - 3p2 + 3p1 - p0) / 6
        //   c2 = (p2 - 2p1 + p0)/2 - 3·c3
        //   c1 = p1 - p0 - c2 - c3
        let six = three + three;
        let c3 = (p3 - p2 - p2 - p2 + p1 + p1 + p1 - p0) * six.inverse().unwrap();
        let c2 = (p2 - p1 - p1 + p0) * two.inverse().unwrap() - (c3 + c3 + c3);
        let c0 = p0;
        (c0, c2, c3)
    }

    #[test]
    fn compute_round_poly_degree3_matches_naive() {
        proptest!(|(seed: u64, log_len in 0_usize..6)| {
            let mut rng = StdRng::seed_from_u64(seed);
            let len = 1 << log_len;
            let eq: Vec<F> = random_vector(&mut rng, len);
            let m: Vec<F> = random_vector(&mut rng, len);
            let v: Vec<F> = random_vector(&mut rng, len);
            let expected = naive_round_poly_degree3(&eq, &m, &v);
            let got = compute_round_poly_degree3(&eq, &m, &v);
            assert_eq!(got, expected, "len = {len}");
        });
    }

    #[test]
    fn compute_round_poly_degree3_zero_extend() {
        // Lengths that aren't powers of two should match the zero-extended
        // versions (the tail path is exercised).
        proptest!(|(seed: u64, len in 0_usize..(1 << 6))| {
            let mut rng = StdRng::seed_from_u64(seed);
            let eq: Vec<F> = random_vector(&mut rng, len);
            let m: Vec<F> = random_vector(&mut rng, len);
            let v: Vec<F> = random_vector(&mut rng, len);
            let extended_eq = zero_pad(&eq);
            let extended_m = zero_pad(&m);
            let extended_v = zero_pad(&v);
            let expected = compute_round_poly_degree3(&extended_eq, &extended_m, &extended_v);
            assert_eq!(compute_round_poly_degree3(&eq, &m, &v), expected);
        });
    }

    #[test]
    fn compute_round_poly_degree3_singleton() {
        // n = 1 (single value, X=1 side implicit zero). p(X) = e·m·v·(1-X)^3.
        proptest!(|(seed: u64)| {
            let mut rng = StdRng::seed_from_u64(seed);
            let e: F = rng.gen();
            let m: F = rng.gen();
            let v: F = rng.gen();
            let (c0, c2, c3) = compute_round_poly_degree3(&[e], &[m], &[v]);
            let t = e * m * v;
            assert_eq!(c0, t);
            assert_eq!(c2, t + t + t);
            assert_eq!(c3, -t);
        });
    }

    #[test]
    fn mixed_sumcheck_poly_matches_lifted() {
        use crate::algebra::{embedding::Basefield, fields::Field64_3, lift};
        let embedding = Basefield::<Field64_3>::new();
        proptest!(|(seed: u64, length in 0_usize..(1 << 10))| {
            let mut rng = StdRng::seed_from_u64(seed);
            let a: Vec<F> = random_vector(&mut rng, length);
            let b: Vec<Field64_3> = random_vector(&mut rng, length);
            let expected = compute_sumcheck_polynomial(&lift(&embedding, &a), &b);
            assert_eq!(
                mixed_compute_sumcheck_polynomial(&embedding, &a, &b),
                expected
            );
        });
    }

    #[test]
    fn mixed_fold_matches_lifted() {
        use crate::algebra::{embedding::Basefield, fields::Field64_3, lift};
        let embedding = Basefield::<Field64_3>::new();
        proptest!(|(seed: u64, length in 0_usize..(1 << 10))| {
            let mut rng = StdRng::seed_from_u64(seed);
            let a: Vec<F> = random_vector(&mut rng, length);
            let weight = rng.gen::<Field64_3>();
            let mut lifted = lift(&embedding, &a);
            fold(&mut lifted, weight);
            assert_eq!(mixed_fold(&embedding, &a, weight), lifted);
        });
    }

    #[test]
    fn fold_zero_extend() {
        proptest!(|(seed:u64, length in 0_usize..(1 << 14))| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut vector: Vec<F> = random_vector(&mut rng, length);
            let mut extended_vector = zero_pad(&vector);
            let weight = rng.gen::<F>();

            fold(&mut vector, weight);
            assert!(vector.is_empty() || vector.len().is_power_of_two());
            fold(&mut extended_vector, weight);
            assert_eq!(vector, extended_vector);
        });
    }
}
