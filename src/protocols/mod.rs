//! Interactive (sub)protocols for WHIR.
//!
//! These interact through the [`spongefish`] Fiat–Shamir transformation.
//!
//! Protocols are parameterized through `Config` structs. These implement serde `Serialize` and
//! `Deserialize` and importantly all generics are *also* serialized so the serialization captures
//! all necessary information to uniquely identify a concrete protocol. The intention is for the
//! hash of the Config serialization to serve as protocol domain separator for Spongefish.
//!

pub mod basecase;
pub mod challenge_indices;
pub mod code_switch;
pub mod geometric_challenge;
pub mod irs_commit;
pub mod mask_proximity;
pub mod matrix_commit;
pub mod merkle_tree;
pub mod params;
pub mod proof_of_work;
pub mod sumcheck;
pub mod whir;
pub mod whir_zk;
