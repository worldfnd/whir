//! The prover's private RNG, exposed through the `rand` 0.8 traits arkworks uses.

use ark_std::rand::{CryptoRng, Error, RngCore};
use spongefish::{DuplexSpongeInit, PrivateRng};

/// Borrows the prover's [`PrivateRng`] as a `rand` 0.8 [`RngCore`].
pub struct ProverRng<'a, R: DuplexSpongeInit<U = u8>>(pub(super) &'a mut PrivateRng<R>);

impl<R: DuplexSpongeInit<U = u8>> RngCore for ProverRng<'_, R> {
    fn next_u32(&mut self) -> u32 {
        let mut bytes = [0; 4];
        self.fill_bytes(&mut bytes);
        u32::from_le_bytes(bytes)
    }

    fn next_u64(&mut self) -> u64 {
        let mut bytes = [0; 8];
        self.fill_bytes(&mut bytes);
        u64::from_le_bytes(bytes)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.fill_bytes(dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl<R: DuplexSpongeInit<U = u8>> CryptoRng for ProverRng<'_, R> {}
