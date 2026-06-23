//! A loose-representation Goldilocks field (`p = 2^64 - 2^32 + 1`).
//!
//! [`Goldilocks`] wraps a `u64` that is *not necessarily canonical*: it lies in
//! `[0, 2^64)` and represents `value mod p`. Reduction to the unique `[0, p)`
//! form happens only at compare / hash / serialize boundaries, never in the
//! arithmetic.

mod goldilocks;

pub use goldilocks::{Goldilocks, GoldilocksAcc};
