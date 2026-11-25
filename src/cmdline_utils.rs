use std::str::FromStr;

use crate::{engines::EngineId, hash};

#[derive(Debug, Clone, Copy)]
pub enum AvailableFields {
    Goldilocks1, // Just Goldilocks
    Goldilocks2, // Quadratic extension of Goldilocks
    Goldilocks3, // Cubic extension of Goldilocks
    Field128,    // 128-bit prime field
    Field192,    // 192-bit prime field
    Field256,    // 256-bit prime field
}

impl FromStr for AvailableFields {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Field128" => Ok(Self::Field128),
            "Field192" => Ok(Self::Field192),
            "Field256" => Ok(Self::Field256),
            "Goldilocks1" => Ok(Self::Goldilocks1),
            "Goldilocks2" => Ok(Self::Goldilocks2),
            "Goldilocks3" => Ok(Self::Goldilocks3),
            _ => Err(format!("Invalid field: {s}")),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum AvailableHash {
    Sha2,
    Sha3,
    Keccak,
    Blake3,
}

impl AvailableHash {
    pub const fn hash_id(&self) -> EngineId {
        match self {
            Self::Sha2 => hash::SHA2,
            Self::Sha3 => hash::SHA3,
            Self::Keccak => hash::KECCAK,
            Self::Blake3 => hash::BLAKE3,
        }
    }
}

impl FromStr for AvailableHash {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Sha2" => Ok(Self::Sha2),
            "Sha3" => Ok(Self::Sha3),
            "Keccak" => Ok(Self::Keccak),
            "Blake3" => Ok(Self::Blake3),
            _ => Err(format!(
                "Invalid hash: {s}, options are: Sha2, Sha3, Keccak, Blake3"
            )),
        }
    }
}
