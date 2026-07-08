//! The Goldilocks prime field `p = 2^64 - 2^32 + 1`, in loose representation.
//!
//! Elements are stored loose: the inner `u64` lies in `[0, 2^64)` and represents
//! `value mod p`, canonicalized to the unique `[0, p)` form only at
//! compare / hash / serialize boundaries.

use core::{
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

mod accumulator;
pub use accumulator::GoldilocksAcc;

/// The Goldilocks prime, `p = 2^64 - 2^32 + 1`.
pub(crate) const P: u64 = 0xFFFF_FFFF_0000_0001;

/// `ε = 2^64 mod p = 2^32 - 1`. A lost `2^64` is repaid by `+ε`, a borrow by `−ε`.
const EPSILON: u64 = 0xFFFF_FFFF;

/// `R2 = 2^128 mod p = ε² = 2^64 - 2^33 + 1`. Folds the high 64-bit limb of the
/// 192-bit accumulator.
pub(crate) const R2: u64 = 0xFFFF_FFFE_0000_0001;

/// A Goldilocks field element in loose representation: the stored `u64` lies in
/// `[0, 2^64)` and represents `value mod p`, not necessarily the canonical
/// `[0, p)` form (see [`Self::as_canonical_u64`]).
#[derive(Clone, Copy, Default)]
#[repr(transparent)]
pub struct Goldilocks(pub(crate) u64);

impl Goldilocks {
    /// The field order `p`.
    pub const ORDER: u64 = P;

    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1);

    /// Multiplicative generator of the field.
    pub const GENERATOR: Self = Self(7);
    /// Two-adicity of `p - 1` (`p - 1 = 2^32 · 3 · 5 · 17 · 257 · 65537`).
    pub const TWO_ADICITY: u32 = 32;
    /// A primitive `2^32`-th root of unity, `GENERATOR^((p-1)/2^32)`.
    pub const TWO_ADIC_GENERATOR: Self = Self(1_753_635_133_440_165_772);
    /// A primitive `(3·2^32)`-th root of unity, `GENERATOR^((p-1)/(3·2^32))`;
    /// lets the NTT build mixed radix-2/3 domains.
    pub const LARGE_SUBGROUP_GENERATOR: Self = Self(14_159_254_819_154_955_796);

    /// Wrap an integer already known to be in `[0, p)`. Debug-checked.
    #[inline(always)]
    pub const fn from_canonical_u64(n: u64) -> Self {
        debug_assert!(n < P);
        Self(n)
    }

    /// Reduce an arbitrary `u64` into the field.
    #[inline(always)]
    pub const fn from_wrapped_u64(n: u64) -> Self {
        Self(canonicalize(n))
    }

    /// The unique representative in `[0, p)`. The only reduction to canonical
    /// form, called solely at compare / hash / serialize boundaries.
    #[inline(always)]
    pub const fn as_canonical_u64(self) -> u64 {
        canonicalize(self.0)
    }

    /// `2·self`.
    #[must_use]
    #[inline(always)]
    pub const fn double(self) -> Self {
        Self(add(self.0, self.0))
    }

    /// `self·self`.
    #[must_use]
    #[inline(always)]
    pub const fn square(self) -> Self {
        Self(mul(self.0, self.0))
    }

    /// Whether this element is the additive identity (canonical check).
    #[inline]
    pub const fn is_zero(self) -> bool {
        self.as_canonical_u64() == 0
    }

    /// The multiplicative inverse, or `None` for zero. Fermat: `a^(p-2)`.
    #[inline]
    #[must_use]
    pub const fn inverse(self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            Some(Self(inv_nonzero(self.0)))
        }
    }

    /// `self^(2^log_n)` by repeated squaring.
    #[must_use]
    pub fn exp_power_of_2(self, log_n: usize) -> Self {
        let mut x = self;
        for _ in 0..log_n {
            x = x.square();
        }
        x
    }

    /// `self^e` by square-and-multiply (cold path).
    #[must_use]
    pub fn pow_u64(self, mut e: u64) -> Self {
        let mut base = self;
        let mut acc = Self::ONE;
        while e > 0 {
            if e & 1 == 1 {
                acc = Self(mul(acc.0, base.0));
            }
            base = base.square();
            e >>= 1;
        }
        acc
    }

    /// A primitive `n`-th root of unity for `n = 2^i · 3^j` (`i ≤ 32`, `j ≤ 1`),
    /// else `None`. Mirrors `ark_ff::FftField::get_root_of_unity`.
    #[must_use]
    pub fn get_root_of_unity(n: u64) -> Option<Self> {
        let two_adicity = k_adicity(2, n);
        let two_part = 2u64.checked_pow(two_adicity)?;
        let three_adicity = k_adicity(3, n);
        let three_part = 3u64.checked_pow(three_adicity)?;
        if n != two_part * three_part || two_adicity > Self::TWO_ADICITY || three_adicity > 1 {
            return None;
        }
        // Start from the (3·2^32)-th root; drop the factor 3 if `n` has none,
        // then square down to the requested two-adic order.
        let mut omega = Self::LARGE_SUBGROUP_GENERATOR;
        for _ in three_adicity..1 {
            omega = omega.pow_u64(3);
        }
        for _ in two_adicity..Self::TWO_ADICITY {
            omega = omega.square();
        }
        Some(omega)
    }
}

// Operators (by value).

impl Add for Goldilocks {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(add(self.0, rhs.0))
    }
}
impl Sub for Goldilocks {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(sub(self.0, rhs.0))
    }
}
impl Mul for Goldilocks {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(mul(self.0, rhs.0))
    }
}
impl Neg for Goldilocks {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(neg(self.0))
    }
}
impl AddAssign for Goldilocks {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 = add(self.0, rhs.0);
    }
}
impl SubAssign for Goldilocks {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = sub(self.0, rhs.0);
    }
}
impl MulAssign for Goldilocks {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 = mul(self.0, rhs.0);
    }
}

// Boundary surface: equality / ordering / hashing canonicalize first, so the two
// loose representatives of an element behave identically.

impl PartialEq for Goldilocks {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_canonical_u64() == other.as_canonical_u64()
    }
}
impl Eq for Goldilocks {}

impl PartialOrd for Goldilocks {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Goldilocks {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_canonical_u64().cmp(&other.as_canonical_u64())
    }
}

impl Hash for Goldilocks {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_canonical_u64().hash(state);
    }
}

impl fmt::Debug for Goldilocks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_canonical_u64())
    }
}
impl fmt::Display for Goldilocks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_canonical_u64())
    }
}

// Raw loose arithmetic on `u64` (value in `[0, 2^64)`, representing v mod p.

/// Bring a loose `u64` into `[0, p)` with a single conditional subtract.
#[inline(always)]
const fn canonicalize(x: u64) -> u64 {
    if x >= P {
        x - P
    } else {
        x
    }
}

/// `a + b mod p`, loose.
///
/// The first fold repays a lost `2^64` with `+ε`; the rare second fold covers the edge cases.
#[inline(always)]
const fn add(a: u64, b: u64) -> u64 {
    let (s1, c1) = a.overflowing_add(b);
    let (s2, c2) = s1.overflowing_add(EPSILON * c1 as u64);
    s2.wrapping_add(EPSILON * c2 as u64)
}

/// `a - b mod p`, loose.
///
/// A borrow's lost `2^64` is repaid with `−ε`; the rare second fold mirrors [`add`].
#[inline(always)]
const fn sub(a: u64, b: u64) -> u64 {
    let (d1, b1) = a.overflowing_sub(b);
    let (d2, b2) = d1.overflowing_sub(EPSILON * b1 as u64);
    d2.wrapping_sub(EPSILON * b2 as u64)
}

/// `-a mod p`, loose.
#[inline(always)]
const fn neg(a: u64) -> u64 {
    sub(0, a)
}

/// `x + y` folding a single carry with `+ε`, assuming no second carry. Used by
/// [`reduce128`], where `y ≤ ε² < 2^64 − 2^33` guarantees the bound.
#[inline(always)]
const fn add_no_double_carry(x: u64, y: u64) -> u64 {
    let (res, carry) = x.overflowing_add(y);
    res.wrapping_add(EPSILON * carry as u64)
}

/// Reduce a full 128-bit value to a loose element in `[0, 2^64)`.
///
/// Split `x = lo + 2^64·mid + 2^96·hi` (mid, hi each 32 bits). Using
/// `2^64 ≡ ε` and `2^96 ≡ −1 (mod p)`: `x ≡ lo − hi + ε·mid (mod p)`.
#[inline(always)]
pub(crate) const fn reduce128(x: u128) -> u64 {
    let lo = x as u64;
    let hi64 = (x >> 64) as u64;
    let hi = hi64 >> 32;
    let mid = hi64 & EPSILON;

    // lo − hi: on borrow the lost 2^64 is repaid with −ε; cannot underflow
    // twice because the borrow forces lo < hi < 2^32.
    let (mut t, borrow) = lo.overflowing_sub(hi);
    if borrow {
        t = t.wrapping_sub(EPSILON);
    }
    // + ε·mid: ε·mid ≤ ε² < 2^64 − 2^33, so only one fold is needed.
    add_no_double_carry(t, mid * EPSILON)
}

/// `a * b mod p`, loose. Inputs loose (`< 2^64`), product `< 2^128`.
#[inline(always)]
const fn mul(a: u64, b: u64) -> u64 {
    reduce128(a as u128 * b as u128)
}

/// `a^(p-2) mod p` (Fermat inverse), loose. `a ≢ 0`.
///
/// Addition chain: 63 squarings + 9 multiplications (vs ~125 for naive
/// square-and-multiply). Builds `a^(2^k − 1)` blocks via the doubling trick
/// `a^(2^(m+n) − 1) = (a^(2^m − 1))^(2^n) · a^(2^n − 1)`.
const fn inv_nonzero(a: u64) -> u64 {
    /// `x^(2^n)` by repeated squaring.
    const fn square_n(mut x: u64, n: u32) -> u64 {
        let mut i = 0;
        while i < n {
            x = mul(x, x);
            i += 1;
        }
        x
    }
    debug_assert!(canonicalize(a) != 0, "inv_nonzero called on zero");
    let t2 = mul(square_n(a, 1), a); // a^(2^2 - 1)
    let t3 = mul(square_n(t2, 1), a); // a^(2^3 - 1)
    let t6 = mul(square_n(t3, 3), t3); // a^(2^6 - 1)
    let t12 = mul(square_n(t6, 6), t6); // a^(2^12 - 1)
    let t24 = mul(square_n(t12, 12), t12); // a^(2^24 - 1)
    let t30 = mul(square_n(t24, 6), t6); // a^(2^30 - 1)
    let t31 = mul(square_n(t30, 1), a); // a^(2^31 - 1)
    let t63 = mul(square_n(t31, 32), t31); // a^(2^63 - 2^32 + 2^31 - 1)
    mul(square_n(t63, 1), a) // a^(p - 2)
}

/// The k-adic valuation of `n`: how many times `k` divides `n`.
fn k_adicity(k: u64, mut n: u64) -> u32 {
    if n == 0 {
        return 0;
    }
    let mut count = 0;
    while n % k == 0 {
        n /= k;
        count += 1;
    }
    count
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    const PP: u128 = P as u128;

    // The independent reference: plain `mod p` in u128, obviously correct.
    fn r(a: u64) -> u128 {
        a as u128 % PP
    }
    fn ref_add(a: u64, b: u64) -> u64 {
        ((r(a) + r(b)) % PP) as u64
    }
    fn ref_sub(a: u64, b: u64) -> u64 {
        ((r(a) + PP - r(b)) % PP) as u64
    }
    fn ref_mul(a: u64, b: u64) -> u64 {
        ((r(a) * r(b)) % PP) as u64
    }
    fn ref_neg(a: u64) -> u64 {
        ((PP - r(a)) % PP) as u64
    }
    fn ref_pow(a: u64, mut e: u64) -> u64 {
        let mut base = r(a);
        let mut acc: u128 = 1;
        while e > 0 {
            if e & 1 == 1 {
                acc = acc * base % PP;
            }
            base = base * base % PP;
            e >>= 1;
        }
        acc as u64
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4096))]

        // Representation.
        #[test]
        fn canonicalize_matches_mod_p(x: u64) {
            prop_assert_eq!(u128::from(canonicalize(x)), u128::from(x) % PP);
        }
        #[test]
        fn wrapped_is_canonical(x: u64) {
            prop_assert_eq!(u128::from(Goldilocks::from_wrapped_u64(x).as_canonical_u64()), u128::from(x) % PP);
        }
        #[test]
        fn loose_representatives_are_equal(a in 0u64..0xFFFF_FFFF) {
            let canonical = Goldilocks::from_canonical_u64(a);
            let loose = Goldilocks(a + P);
            prop_assert_eq!(canonical, loose);
            prop_assert_eq!(loose.as_canonical_u64(), a);
        }

        // Arithmetic vs the reference (random u64s exercise loose inputs).
        #[test]
        fn diff_add(a: u64, b: u64) {
            prop_assert_eq!((Goldilocks(a) + Goldilocks(b)).as_canonical_u64(), ref_add(a, b));
        }
        #[test]
        fn diff_sub(a: u64, b: u64) {
            prop_assert_eq!((Goldilocks(a) - Goldilocks(b)).as_canonical_u64(), ref_sub(a, b));
        }
        #[test]
        fn diff_mul(a: u64, b: u64) {
            prop_assert_eq!((Goldilocks(a) * Goldilocks(b)).as_canonical_u64(), ref_mul(a, b));
        }
        #[test]
        fn diff_neg(a: u64) {
            prop_assert_eq!((-Goldilocks(a)).as_canonical_u64(), ref_neg(a));
        }
        #[test]
        fn diff_double(a: u64) {
            prop_assert_eq!(Goldilocks(a).double().as_canonical_u64(), ref_add(a, a));
        }
        #[test]
        fn diff_square(a: u64) {
            prop_assert_eq!(Goldilocks(a).square().as_canonical_u64(), ref_mul(a, a));
        }
        #[test]
        fn diff_reduce128(x: u128) {
            prop_assert_eq!(u128::from(canonicalize(reduce128(x))), x % PP);
        }

        // Cold surface (inverse, pow).
        #[test]
        fn diff_pow(a: u64, e: u64) {
            prop_assert_eq!(Goldilocks(a).pow_u64(e).as_canonical_u64(), ref_pow(a, e));
        }
        #[test]
        fn inverse_roundtrip(a: u64) {
            let x = Goldilocks(a);
            match x.inverse() {
                Some(inv) => prop_assert_eq!((x * inv).as_canonical_u64(), 1),
                None => prop_assert!(x.is_zero()),
            }
        }
    }

    /// Edge cases the random sampler almost never hits — including the
    /// non-canonical representatives (`P`, `P+1`, `u64::MAX`) that drive the
    /// double-carry / double-borrow paths in `add`/`sub`.
    #[test]
    fn boundary_values() {
        let edges = [
            0u64,
            1,
            2,
            EPSILON - 1,
            EPSILON,
            EPSILON + 1,
            1 << 63,
            P - 2,
            P - 1,
            P,
            P + 1,
            u64::MAX,
        ];
        for &a in &edges {
            assert_eq!((-Goldilocks(a)).as_canonical_u64(), ref_neg(a), "neg {a}");
            assert_eq!(
                Goldilocks(a).double().as_canonical_u64(),
                ref_add(a, a),
                "double {a}"
            );
            assert_eq!(
                Goldilocks(a).square().as_canonical_u64(),
                ref_mul(a, a),
                "square {a}"
            );
            for &b in &edges {
                assert_eq!(
                    (Goldilocks(a) + Goldilocks(b)).as_canonical_u64(),
                    ref_add(a, b),
                    "add {a} {b}"
                );
                assert_eq!(
                    (Goldilocks(a) - Goldilocks(b)).as_canonical_u64(),
                    ref_sub(a, b),
                    "sub {a} {b}"
                );
                assert_eq!(
                    (Goldilocks(a) * Goldilocks(b)).as_canonical_u64(),
                    ref_mul(a, b),
                    "mul {a} {b}"
                );
            }
        }
    }

    #[test]
    fn constants() {
        assert_eq!(Goldilocks::ZERO.as_canonical_u64(), 0);
        assert_eq!(Goldilocks::ONE.as_canonical_u64(), 1);
        assert_eq!((Goldilocks::ONE + (-Goldilocks::ONE)).as_canonical_u64(), 0);
        assert_eq!(Goldilocks::ORDER, 0xFFFF_FFFF_0000_0001);
    }

    // FFT constants & root orders.

    #[test]
    fn inverse_of_zero_is_none() {
        assert_eq!(Goldilocks::ZERO.inverse(), None);
        // A non-canonical representative of zero is also rejected.
        assert_eq!(Goldilocks(P).inverse(), None);
    }

    #[test]
    fn two_adic_generator_has_order_2_pow_32() {
        let g = Goldilocks::TWO_ADIC_GENERATOR;
        assert_eq!(g.exp_power_of_2(32).as_canonical_u64(), 1, "g^(2^32) = 1");
        assert_ne!(
            g.exp_power_of_2(31).as_canonical_u64(),
            1,
            "g^(2^31) ≠ 1 (primitive)"
        );
    }

    #[test]
    fn generator_derives_the_two_adic_root() {
        // TWO_ADIC_GENERATOR = GENERATOR^((p-1)/2^32) = 7^(2^32 - 1).
        let derived = Goldilocks::GENERATOR.pow_u64((1u64 << 32) - 1);
        assert_eq!(derived, Goldilocks::TWO_ADIC_GENERATOR);
    }

    #[test]
    fn large_subgroup_generator_has_order_3x2_pow_32() {
        let w = Goldilocks::LARGE_SUBGROUP_GENERATOR;
        let w3 = w.pow_u64(3);
        // Order 3·2^32: w^(3·2^32) = 1, but removing either prime factor is not.
        assert_eq!(
            w3.exp_power_of_2(32).as_canonical_u64(),
            1,
            "w^(3·2^32) = 1"
        );
        assert_ne!(
            w3.exp_power_of_2(31).as_canonical_u64(),
            1,
            "2-adic part full"
        );
        assert_ne!(
            w.exp_power_of_2(32).as_canonical_u64(),
            1,
            "factor 3 present"
        );
    }

    #[test]
    fn roots_of_unity_have_exact_order() {
        for log_n in 0..=20u32 {
            let n = 1u64 << log_n;
            let w = Goldilocks::get_root_of_unity(n).expect("power-of-two root exists");
            assert_eq!(
                w.exp_power_of_2(log_n as usize).as_canonical_u64(),
                1,
                "w^n = 1"
            );
            if log_n > 0 {
                assert_ne!(
                    w.exp_power_of_2(log_n as usize - 1).as_canonical_u64(),
                    1,
                    "primitive"
                );
            }
        }
        // A mixed 3·2^k order (here 24) exists and is primitive.
        let w = Goldilocks::get_root_of_unity(24).expect("3·8 root exists");
        assert_eq!(w.pow_u64(24).as_canonical_u64(), 1, "w^24 = 1");
        assert_ne!(w.pow_u64(8).as_canonical_u64(), 1, "factor 3 present");
        assert_ne!(w.pow_u64(12).as_canonical_u64(), 1, "2-adic part full");
        // Non-smooth / out-of-range orders are rejected.
        assert_eq!(
            Goldilocks::get_root_of_unity(5),
            None,
            "5 is not 2,3-smooth"
        );
        assert_eq!(
            Goldilocks::get_root_of_unity(9),
            None,
            "3^2 exceeds adicity 1"
        );
    }
}
