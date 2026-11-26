//! Reed–Solomon decoding bounds shared by the protocol and its sub-protocols.
//! Mathematical references use the pinned [ProveKit whitepaper], §6.4 and Appendix A.
//!
//! [ProveKit whitepaper]: https://github.com/worldfnd/provekit/tree/6605cf5242567e17899a26c27db829c7ce9eb478/whitepaper

use std::{
    f64::consts::LOG2_10,
    fmt::{self, Display, Formatter},
};

use serde::{Deserialize, Serialize};

/// Unique, Johnson, and conjectured capacity decoding bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecodingRegime {
    Unique,
    Johnson,
    Capacity,
}

impl Display for DecodingRegime {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Return log2(1 - distance) for the given decoding regime and code rate.
pub fn decoding_distance(policy: DecodingRegime, rho: f64) -> f64 {
    match policy {
        DecodingRegime::Unique => f64::midpoint(1.0, rho),
        DecodingRegime::Johnson => rho.sqrt() + rho.sqrt() / 20.0,
        DecodingRegime::Capacity => rho + rho / 20.0,
    }
    .log2()
}

/// Canonical [list size, log2(1-distance), log2(MCA error)].
/// Johnson uses BCSS25 Theorem 1.5; Capacity uses STIR Conjecture 5.6.
pub fn decoding_bounds(policy: DecodingRegime, rho: f64, length: usize, field: f64) -> [f64; 3] {
    use DecodingRegime::{Capacity, Johnson, Unique};
    let r = -rho.log2();
    let d = (length as f64).log2();
    let slack = if policy == Johnson {
        rho.sqrt() / 20.0
    } else {
        rho / 20.0
    };
    let list = match policy {
        Unique => 0.0,
        Johnson => -1.0 - slack.log2() + 0.5 * r,
        Capacity => d + r - slack.log2(),
    };
    let mca = match policy {
        Unique => d + r,
        Johnson => {
            assert!(slack.log2() >= -(0.5 * r + LOG2_10 + 1.0) - 1e-6);
            (2.0 * 10.5_f64.powi(5) / 3.0).log2() + d + 2.5 * r
        }
        Capacity => {
            assert!(slack.log2() >= -(r + LOG2_10 + 1.0) - 1e-6);
            d + 3.0 * r + LOG2_10 + 1.0
        }
    };
    [
        2_f64.powf(list),
        decoding_distance(policy, rho),
        mca - field,
    ]
}

pub fn rate(r: f64) -> f64 {
    2_f64.powf(-r)
}
