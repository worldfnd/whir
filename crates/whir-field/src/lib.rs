//! `whir-field` — a WHIR-owned Goldilocks field, performance-first.
//!
//! A scalar Goldilocks field, proven bit-for-bit correct against an independent
//! reference **in isolation** before it ever touches WHIR.
//!
//! # Representation
//!
//! [`Goldilocks`] is a `#[repr(transparent)]` `u64` in **loose** form: the
//! stored value lies in `[0, 2^64)` and represents `value mod p`, where
//! `p = 2^64 - 2^32 + 1`. Two different `u64`s can denote the same field
//! element. Canonicalization to the unique `[0, p)` form happens *only* at
//! boundaries — equality, hashing, ordering, serialization — never on the
//! arithmetic hot path.
//!
//! Layout: each field is a self-contained module (`goldilocks/`) that owns its
//! element, arithmetic, and accumulator; shared field traits will live at the
//! crate root (`traits.rs`). A second field would be its own `m31/` module.

mod goldilocks;

pub use goldilocks::{Goldilocks, GoldilocksAcc};
