//! `Copy`: a no-op hash engine for inputs of at most 32 bytes.

use std::borrow::Cow;

use const_oid::ObjectIdentifier;
use hex_literal::hex;

use super::{Hash, HashEngine};
use crate::engines::EngineId;

pub const COPY: EngineId = EngineId::new(hex!(
    "09459020f451874a1b399819d079632cc0f9263b1486c423173c6e15d8e2d61d"
));

/// No-op hash engine that copies the input data without hashing it.
///
/// Requires the input data to be at most 32 bytes long.
#[derive(Clone, Copy, Debug, Default)]
pub struct Copy;

impl Copy {
    pub const fn new() -> Self {
        Self
    }
}

impl HashEngine for Copy {
    fn name(&self) -> Cow<'_, str> {
        "copy".into()
    }

    fn oid(&self) -> Option<ObjectIdentifier> {
        None
    }

    fn supports_size(&self, size: usize) -> bool {
        size <= 32
    }

    fn preferred_batch_size(&self) -> usize {
        1
    }

    fn hash_many(&self, size: usize, input: &[u8], output: &mut [Hash]) {
        assert!(size <= 32, "Copy engine only supports sizes up to 32 bytes");
        assert_eq!(
            input.len(),
            size * output.len(),
            "Input length should be size * output.len() = {size} * {}",
            output.len()
        );
        if size == 0 {
            output.fill(Hash([0; 32]));
            return;
        }
        for (input, out) in input.chunks_exact(size).zip(output.iter_mut()) {
            let mut bytes = [0; 32];
            bytes[..size].copy_from_slice(input);
            *out = Hash(bytes);
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::engines::Engine;

    #[test]
    fn test_protocol_ids() {
        assert_eq!(Copy::new().engine_id(), COPY);
    }
}
