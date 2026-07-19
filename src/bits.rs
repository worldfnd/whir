use std::{
    cmp::Ordering,
    fmt::{self, Display},
    hash::Hash,
    iter::Sum,
    ops::{Add, Sub},
};

use serde::{Deserialize, Serialize};

/// Wrapper for `bits` value types.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bits(f64);

impl Hash for Bits {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl Eq for Bits {}

impl Bits {
    pub fn new(bits: f64) -> Self {
        assert!(bits.is_finite());
        Self(bits)
    }

    pub fn is_zero(&self) -> bool {
        self.0 == 0.0
    }

    /// Absolute difference, e.g. for tolerance comparisons.
    #[must_use]
    pub fn abs_diff(self, other: Self) -> Self {
        Self::new((self.0 - other.0).abs())
    }
}

impl Add for Bits {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.0 + rhs.0)
    }
}

impl Sub for Bits {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.0 - rhs.0)
    }
}

impl Sum for Bits {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self(0.0), Add::add)
    }
}

impl From<f64> for Bits {
    fn from(bits: f64) -> Self {
        Self::new(bits)
    }
}

impl From<Bits> for f64 {
    fn from(bits: Bits) -> Self {
        bits.0
    }
}

impl Display for Bits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <f64 as Display>::fmt(&self.0, f)
    }
}

impl PartialOrd for Bits {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Bits {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}
