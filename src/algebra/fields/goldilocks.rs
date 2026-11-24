mod ark_compat;

use core::ops::{Add, AddAssign};

/// Goldilocks prime modulus `p = 2^64 - 2^32 + 1`.
pub const MODULUS: u64 = 0xffff_ffff_0000_0001;

/// `EPSILON = 2^32 - 1 = 2^64 - p` (so `2^64 ≡ EPSILON (mod p)`).
pub const EPSILON: u64 = 0xffff_ffff;

/// Representation-parameterized Goldilocks prime field element (`p = 2^64 - 2^32 + 1`).
///
/// - `Goldilocks<u64>` ([`Field64`]): non-canonical `[0, 2^64)` storage form; boundaries
///   that require uniqueness lazily canonicalize via [`GoldilocksReduce::reduce_fully`].
/// - `Goldilocks<u128>`: unreduced 128-bit intermediate (`a * b`, with room for up to
///   two `u64` additions before a single [`reduce128`]: `(2^64-1)^2 + 2(2^64-1) = 2^128-1`).
#[derive(Copy, Clone, Default)]
#[repr(transparent)]
pub struct Goldilocks<T = u64>(pub T);

pub type Field64 = Goldilocks<u64>;

/// Delayed-reduction accumulator for sums of up to `2^31` 128-bit products.
///
/// Splits each product into its top 32 bits (`val >> 96`, accumulated in `hi`) and
/// low 96 bits (`lo_plus_hi - (hi << 96)`), avoiding a 3-limb carry chain and reducing
/// in a single [`reduce128`] via `2^96 ≡ -1 (mod p)`.
#[derive(Copy, Clone, Debug, Default)]
pub struct Acc {
    lo_plus_hi: u128,
    hi: u64,
}

/// Reduction from any [`Goldilocks<T>`] (or backing integer) down to 64-bit storage
/// or canonical `[0, p)` form.
pub trait GoldilocksReduce {
    /// Reduce into non-canonical 64-bit storage form `Goldilocks<u64>` (`[0, 2^64)`).
    fn reduce_u64(self) -> Goldilocks<u64>;

    /// Fully reduce into canonical `[0, p)`.
    fn reduce_fully(self) -> u64;
}

impl<T: GoldilocksReduce> GoldilocksReduce for Goldilocks<T> {
    #[inline]
    fn reduce_u64(self) -> Goldilocks<u64> {
        self.0.reduce_u64()
    }

    #[inline]
    fn reduce_fully(self) -> u64 {
        self.0.reduce_fully()
    }
}

impl GoldilocksReduce for u64 {
    #[inline]
    fn reduce_u64(self) -> Goldilocks<u64> {
        Goldilocks(self)
    }

    #[inline]
    fn reduce_fully(self) -> u64 {
        if self >= MODULUS {
            self - MODULUS
        } else {
            self
        }
    }
}

impl GoldilocksReduce for u128 {
    #[inline]
    fn reduce_u64(self) -> Goldilocks<u64> {
        Goldilocks(reduce128(self))
    }

    #[inline]
    fn reduce_fully(self) -> u64 {
        reduce128(self).reduce_fully()
    }
}

impl GoldilocksReduce for Acc {
    #[inline]
    fn reduce_u64(self) -> Goldilocks<u64> {
        // Recover the low 96-bit sum and fold `hi * 2^96 ≡ -hi (mod p)` in one `reduce128`.
        // For N <= 2^31 products: `lo < 2^127` and `hi < 2^63 < MODULUS`, so no over/underflow.
        let lo = self.lo_plus_hi.wrapping_sub(u128::from(self.hi) << 96);
        Goldilocks(reduce128(lo + u128::from(MODULUS - self.hi)))
    }

    #[inline]
    fn reduce_fully(self) -> u64 {
        self.reduce_u64().reduce_fully()
    }
}

// ── Core Goldilocks arithmetic ──────────────────────────────────────────────

#[inline]
#[cold]
const fn cold_path() {}

/// Manual 64x64 -> 128-bit widening multiplication for `wasm32` (avoids `__multi3`).
#[inline]
#[cfg_attr(not(any(target_arch = "wasm32", test)), allow(dead_code))]
const fn widening_mul_manual(a: u64, b: u64) -> u128 {
    let a_lo = a & 0xffff_ffff;
    let a_hi = a >> 32;
    let b_lo = b & 0xffff_ffff;
    let b_hi = b >> 32;

    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;

    let mid = (ll >> 32) + (lh & 0xffff_ffff) + (hl & 0xffff_ffff);
    let lo = (ll & 0xffff_ffff) | (mid << 32);
    let hi = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
    (lo as u128) | ((hi as u128) << 64)
}

/// Manual 64-bit squaring -> 128-bit for `wasm32` (3 `u64` multiplies instead of 4).
#[inline]
#[cfg_attr(not(any(target_arch = "wasm32", test)), allow(dead_code))]
const fn widening_square_manual(a: u64) -> u128 {
    let lo = a & 0xffff_ffff;
    let hi = a >> 32;
    let ll = lo * lo;
    let lh = lo * hi;
    let hh = hi * hi;

    let middle = lh + (ll >> 33);
    let prod_hi = hh + (middle >> 31);
    let prod_lo = ll.wrapping_add(lh << 33);
    (prod_lo as u128) | ((prod_hi as u128) << 64)
}

#[inline]
pub const fn widening_mul(a: u64, b: u64) -> u128 {
    #[cfg(target_arch = "wasm32")]
    {
        widening_mul_manual(a, b)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        (a as u128) * (b as u128)
    }
}

#[inline]
pub const fn widening_square(a: u64) -> u128 {
    #[cfg(target_arch = "wasm32")]
    {
        widening_square_manual(a)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        (a as u128) * (a as u128)
    }
}

/// Multiplication-free reduction of any `u128` into non-canonical `[0, 2^64)`.
///
/// Splits `x = lo + hi_lo * 2^64 + hi_hi * 2^96` and applies:
/// - `2^96 ≡ -1 (mod p)` → `t0 = lo - hi_hi` (borrow prob < 2^-33)
/// - `2^64 ≡ 2^32 - 1 (mod p)` → `t1 = (hi_lo << 32) - hi_lo` (shift + sub, no mul)
#[inline]
pub const fn reduce128(x: u128) -> u64 {
    let lo = x as u64;
    let hi = (x >> 64) as u64;
    let hi_lo = hi & 0xffff_ffff;
    let hi_hi = hi >> 32;

    let (mut t0, borrow) = lo.overflowing_sub(hi_hi);
    if borrow {
        cold_path();
        t0 -= EPSILON; // Safe: borrow implies lo < hi_hi < 2^32, so t0 >= p.
    }

    let t1 = (hi_lo << 32) - hi_lo;
    let (res, carry) = t0.overflowing_add(t1);
    // Max t0 + t1 = 2^65 - 2^33, so on carry `res <= 2^64 - 2^33` and `res + EPSILON < p`.
    if carry {
        res + EPSILON
    } else {
        res
    }
}

#[inline]
const fn sq_n(mut x: u64, n: u32) -> u64 {
    let mut i = 0;
    while i < n {
        x = reduce128(widening_square(x));
        i += 1;
    }
    x
}

impl Goldilocks<u64> {
    /// Wrap any `u64` (may be non-canonical in `[0, 2^64)`).
    #[inline]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Widening multiply into an unreduced `Goldilocks<u128>`, leaving room
    /// for up to two `u64` additions before a single [`reduce128`].
    #[inline]
    pub const fn mul_wide(self, rhs: Self) -> Goldilocks<u128> {
        Goldilocks(widening_mul(self.0, rhs.0))
    }

    /// Widening square into an unreduced `Goldilocks<u128>`.
    #[inline]
    pub const fn square_wide(self) -> Goldilocks<u128> {
        Goldilocks(widening_square(self.0))
    }

    #[inline]
    pub(crate) const fn add_impl(self, rhs: Self) -> Self {
        let (sum, over) = self.0.overflowing_add(rhs.0);
        let (mut sum, over) = sum.overflowing_add(if over { EPSILON } else { 0 });
        if over {
            cold_path();
            sum += EPSILON;
        }
        Self(sum)
    }

    #[inline]
    pub(crate) const fn sub_impl(self, rhs: Self) -> Self {
        let (diff, under) = self.0.overflowing_sub(rhs.0);
        let (mut diff, under) = diff.overflowing_sub(if under { EPSILON } else { 0 });
        if under {
            cold_path();
            diff -= EPSILON;
        }
        Self(diff)
    }

    #[inline]
    pub(crate) const fn neg_impl(self) -> Self {
        // When canonical == 0 this returns MODULUS, which is valid non-canonical 0.
        let c = if self.0 >= MODULUS {
            self.0 - MODULUS
        } else {
            self.0
        };
        Self(MODULUS - c)
    }

    #[inline]
    pub(crate) const fn mul_impl(self, rhs: Self) -> Self {
        Self(reduce128(widening_mul(self.0, rhs.0)))
    }

    #[inline]
    pub(crate) const fn square_impl(self) -> Self {
        Self(reduce128(widening_square(self.0)))
    }

    #[inline]
    pub(crate) fn sum_of_products_impl<const T: usize>(a: &[Self; T], b: &[Self; T]) -> Self {
        match T {
            0 => Self(0),
            1 => a[0].mul_impl(b[0]),
            _ => {
                let mut acc = Goldilocks::<Acc>::ZERO;
                for i in 0..T {
                    acc += a[i].mul_wide(b[i]);
                }
                acc.reduce_u64()
            }
        }
    }

    #[inline]
    pub(crate) const fn inv_impl(self) -> Option<Self> {
        let x = if self.0 >= MODULUS {
            self.0 - MODULUS
        } else {
            self.0
        };
        if x == 0 {
            return None;
        }
        // Addition chain for `x^(p - 2)` where `p - 2 = 2^64 - 2^32 - 1`
        // (63 squarings + 9 multiplications).
        let t2 = reduce128(widening_mul(sq_n(x, 1), x));
        let t3 = reduce128(widening_mul(sq_n(t2, 1), x));
        let t6 = reduce128(widening_mul(sq_n(t3, 3), t3));
        let t12 = reduce128(widening_mul(sq_n(t6, 6), t6));
        let t24 = reduce128(widening_mul(sq_n(t12, 12), t12));
        let t30 = reduce128(widening_mul(sq_n(t24, 6), t6));
        let t31 = reduce128(widening_mul(sq_n(t30, 1), x));
        let t63 = reduce128(widening_mul(sq_n(t31, 32), t31));
        let inv = reduce128(widening_mul(sq_n(t63, 1), x));
        Some(Self(inv))
    }
}

impl Goldilocks<Acc> {
    pub const ZERO: Self = Self(Acc {
        lo_plus_hi: 0,
        hi: 0,
    });
}

impl Add<Goldilocks<u64>> for Goldilocks<u128> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Goldilocks<u64>) -> Self {
        Self(self.0 + u128::from(rhs.0))
    }
}

impl AddAssign<Goldilocks<u64>> for Goldilocks<u128> {
    #[inline]
    fn add_assign(&mut self, rhs: Goldilocks<u64>) {
        self.0 += u128::from(rhs.0);
    }
}

impl AddAssign<Goldilocks<u128>> for Goldilocks<Acc> {
    #[inline]
    #[allow(clippy::suspicious_op_assign_impl)]
    fn add_assign(&mut self, rhs: Goldilocks<u128>) {
        let val_hi = (rhs.0 >> 96) as u64;
        self.0.lo_plus_hi = self.0.lo_plus_hi.wrapping_add(rhs.0);
        self.0.hi = self.0.hi.wrapping_add(val_hi);
    }
}

impl Add<Goldilocks<u128>> for Goldilocks<Acc> {
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: Goldilocks<u128>) -> Self {
        self += rhs;
        self
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::{AdditiveGroup, FftField, Field, Fp64, MontBackend, MontConfig, PrimeField};
    use proptest::prelude::*;

    use super::*;

    #[derive(MontConfig)]
    #[modulus = "18446744069414584321"]
    #[generator = "7"]
    #[small_subgroup_base = "15"]
    #[small_subgroup_power = "1"]
    struct ArkGoldilocksConfig;
    type ArkField64 = Fp64<MontBackend<ArkGoldilocksConfig, 1>>;

    fn to_ark(x: Field64) -> ArkField64 {
        ArkField64::from(x.reduce_fully())
    }

    #[test]
    fn constants_and_edge_cases_match_arkworks() {
        assert_eq!(Field64::MODULUS, ArkField64::MODULUS);
        assert_eq!(
            Field64::GENERATOR.into_bigint(),
            ArkField64::GENERATOR.into_bigint()
        );
        assert_eq!(
            Field64::TWO_ADIC_ROOT_OF_UNITY.into_bigint(),
            ArkField64::TWO_ADIC_ROOT_OF_UNITY.into_bigint()
        );
        assert_eq!(
            Field64::LARGE_SUBGROUP_ROOT_OF_UNITY.map(PrimeField::into_bigint),
            ArkField64::LARGE_SUBGROUP_ROOT_OF_UNITY.map(PrimeField::into_bigint)
        );

        // Include non-canonical inputs in `[MODULUS, u64::MAX]` to test lazy canonicalization!
        let edges = [
            0,
            1,
            2,
            7,
            EPSILON - 1,
            EPSILON,
            EPSILON + 1,
            1 << 32,
            1 << 63,
            MODULUS - 2,
            MODULUS - 1,
            MODULUS,
            MODULUS + 1,
            u64::MAX - 1,
            u64::MAX,
        ];

        for &a_raw in &edges {
            let a = Goldilocks::new(a_raw);
            let aa = ArkField64::from(a_raw % MODULUS);
            assert_eq!(to_ark(-a), -aa);
            assert_eq!(to_ark(a.double()), aa.double());
            assert_eq!(to_ark(a.square()), aa.square());
            assert_eq!(a.inverse().map(to_ark), aa.inverse());
            assert_eq!(widening_square_manual(a_raw), widening_square(a_raw));

            for &b_raw in &edges {
                let b = Goldilocks::new(b_raw);
                let ab = ArkField64::from(b_raw % MODULUS);
                assert_eq!(to_ark(a + b), aa + ab);
                assert_eq!(to_ark(a - b), aa - ab);
                assert_eq!(to_ark(a * b), aa * ab);
                assert_eq!(
                    widening_mul_manual(a_raw, b_raw),
                    widening_mul(a_raw, b_raw)
                );

                // Up to two deferred additions after a widening multiply (even for u64::MAX!).
                for &c_raw in &[0, 1, EPSILON, MODULUS - 1, u64::MAX] {
                    let c = Goldilocks::new(c_raw);
                    let ac = ArkField64::from(c_raw % MODULUS);
                    assert_eq!(to_ark((a.mul_wide(b) + c).reduce_u64()), aa * ab + ac);
                    assert_eq!(
                        to_ark((a.mul_wide(b) + c + b).reduce_u64()),
                        aa * ab + ac + ab
                    );
                }
            }
        }
    }

    proptest! {
        #[test]
        fn proptest_ops_including_non_canonical(
            a_raw in any::<u64>(),
            b_raw in any::<u64>(),
            c_raw in any::<u64>(),
            d_raw in any::<u64>(),
        ) {
            let (a, b, c, d) = (
                Goldilocks::new(a_raw),
                Goldilocks::new(b_raw),
                Goldilocks::new(c_raw),
                Goldilocks::new(d_raw),
            );
            let (aa, ab, ac, ad) = (
                ArkField64::from(a_raw % MODULUS),
                ArkField64::from(b_raw % MODULUS),
                ArkField64::from(c_raw % MODULUS),
                ArkField64::from(d_raw % MODULUS),
            );

            prop_assert_eq!(to_ark(a + b), aa + ab);
            prop_assert_eq!(to_ark(a - b), aa - ab);
            prop_assert_eq!(to_ark(a * b), aa * ab);
            prop_assert_eq!(to_ark(-a), -aa);
            prop_assert_eq!(to_ark(a.square()), aa.square());
            prop_assert_eq!(a.inverse().map(to_ark), aa.inverse());
            prop_assert_eq!(to_ark((a.mul_wide(b) + c + d).reduce_u64()), aa * ab + ac + ad);

            let sop = Field64::sum_of_products(&[a, b, c, d], &[d, c, b, a]);
            let ark_sop = ArkField64::sum_of_products(&[aa, ab, ac, ad], &[ad, ac, ab, aa]);
            prop_assert_eq!(to_ark(sop), ark_sop);

            prop_assert_eq!(widening_mul_manual(a_raw, b_raw), widening_mul(a_raw, b_raw));
            prop_assert_eq!(widening_square_manual(a_raw), widening_square(a_raw));
        }

        #[test]
        fn proptest_acc(pairs in prop::collection::vec((any::<u64>(), any::<u64>()), 0..64)) {
            let mut acc = Goldilocks::<Acc>::ZERO;
            let mut expected = ArkField64::ZERO;
            for (x_raw, y_raw) in pairs {
                let x = Goldilocks::new(x_raw);
                let y = Goldilocks::new(y_raw);
                acc += x.mul_wide(y);
                expected += ArkField64::from(x_raw % MODULUS) * ArkField64::from(y_raw % MODULUS);
            }
            prop_assert_eq!(to_ark(acc.reduce_u64()), expected);
        }
    }
}
