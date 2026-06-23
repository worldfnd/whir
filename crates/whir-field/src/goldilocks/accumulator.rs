//! The delayed-reduction accumulator.
//!
//! Instead of reducing after every multiply in a dot-product, sum the raw
//! products in a wide integer and reduce once at the end. A length `N` dot
//! is then `N` multiplies + a single reduction, versus `N` reductions for the
//! naive `map(*).sum()` path.

use crate::goldilocks::{reduce128, Goldilocks, R2};

/// A 192-bit unsigned accumulator for sums of Goldilocks products.
///
/// It covers any realistic vector length before the final reduction.
#[derive(Clone, Copy, Default, Debug)]
pub struct GoldilocksAcc {
    lo: u128,
    hi: u64,
}

impl GoldilocksAcc {
    #[inline(always)]
    #[must_use]
    pub const fn zero() -> Self {
        Self { lo: 0, hi: 0 }
    }

    /// self += a · b, with no reduction.
    #[inline(always)]
    pub fn mul_add(&mut self, a: Goldilocks, b: Goldilocks) {
        let product = u128::from(a.0) * u128::from(b.0);
        let (lo, carry) = self.lo.overflowing_add(product);
        self.lo = lo;
        self.hi += u64::from(carry);
    }

    /// `self += a`, with no reduction.
    #[inline(always)]
    pub fn add_elem(&mut self, a: Goldilocks) {
        let (lo, carry) = self.lo.overflowing_add(u128::from(a.0));
        self.lo = lo;
        self.hi += u64::from(carry);
    }

    /// Reduce the 192-bit sum to a field element
    ///
    /// lo + hi·2^128 ≡ reduce128(lo) + reduce128(hi · R2) (mod p), since
    /// 2^128 ≡ R2 (mod p) and hi · R2 < 2^128 fits a u128.
    #[inline]
    #[must_use]
    pub fn reduce(self) -> Goldilocks {
        let low = Goldilocks(reduce128(self.lo));
        let high = Goldilocks(reduce128(u128::from(self.hi) * u128::from(R2)));
        low + high
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::goldilocks::P;

    const PP: u128 = P as u128;

    /// Reference for the 192-bit reduction, computed via `2^64 ≡ ε` (so
    /// `2^128 ≡ ε²`) using a hardcoded `ε` — deliberately **not** the crate's
    /// `R2`, so this independently checks that `R2` is correct.
    fn reduce192_ref(lo: u128, hi: u64) -> u64 {
        const E: u128 = 0xFFFF_FFFF; // ε = 2^32 - 1
        let hi_term = (((u128::from(hi) % PP) * E % PP) * E) % PP; // hi · 2^128 mod p
        ((lo % PP + hi_term) % PP) as u64
    }

    /// Eager reference dot: reduce every product and every partial sum mod p.
    fn ref_dot(a: &[u64], b: &[u64]) -> u64 {
        let mut acc: u128 = 0;
        for (&x, &y) in a.iter().zip(b) {
            let prod = (u128::from(x) % PP) * (u128::from(y) % PP) % PP; // < p
            acc = (acc + prod) % PP; // acc < p, prod < p => < 2p < 2^128
        }
        acc as u64
    }

    fn acc_dot(a: &[u64], b: &[u64]) -> u64 {
        let mut acc = GoldilocksAcc::zero();
        for (&x, &y) in a.iter().zip(b) {
            acc.mul_add(Goldilocks(x), Goldilocks(y));
        }
        acc.reduce().as_canonical_u64()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4096))]

        /// The 192-bit reduction matches the independent (R2-free) reference.
        #[test]
        fn reduce192_matches_reference(lo: u128, hi: u64) {
            let got = (GoldilocksAcc { lo, hi }).reduce().as_canonical_u64();
            prop_assert_eq!(got, reduce192_ref(lo, hi));
        }

        /// The deferred-reduction dot equals the eager one, bit-for-bit —
        /// including loose inputs and lengths that drive the high carry limb.
        #[test]
        fn accumulator_equals_eager(
            pairs in proptest::collection::vec((any::<u64>(), any::<u64>()), 0..=300)
        ) {
            let a: Vec<u64> = pairs.iter().map(|&(x, _)| x).collect();
            let b: Vec<u64> = pairs.iter().map(|&(_, y)| y).collect();
            prop_assert_eq!(acc_dot(&a, &b), ref_dot(&a, &b));
        }

        /// `add_elem` accumulates a plain sum correctly.
        #[test]
        fn add_elem_equals_eager(xs in proptest::collection::vec(any::<u64>(), 0..=300)) {
            let mut acc = GoldilocksAcc::zero();
            for &x in &xs {
                acc.add_elem(Goldilocks(x));
            }
            let got = acc.reduce().as_canonical_u64();
            let want = xs.iter().fold(0u128, |s, &x| (s + u128::from(x) % PP) % PP) as u64;
            prop_assert_eq!(got, want);
        }
    }

    /// Many maximal products drive the high limb `hi` well past 0.
    #[test]
    fn high_limb_stress() {
        let n = 100_000;
        let max = P - 1;
        let a = vec![max; n];
        let b = vec![max; n];
        assert_eq!(acc_dot(&a, &b), ref_dot(&a, &b));
    }
}
