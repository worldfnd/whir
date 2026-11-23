//! Test-only sponge that replays fixed bytes.
#![cfg(test)]

use super::DuplexSpongeInterface;

#[derive(Clone, Debug)]
pub struct MockSponge<'a> {
    pub absorb: Option<&'a [u8]>,
    pub squeeze: &'a [u8],
}

impl DuplexSpongeInterface for MockSponge<'_> {
    type U = u8;

    fn absorb(&mut self, input: &[Self::U]) -> &mut Self {
        if let Some(absorb) = self.absorb.as_mut() {
            assert_eq!(&absorb[..input.len()], input);
            *absorb = &absorb[input.len()..];
        }
        self
    }

    fn squeeze(&mut self, output: &mut [Self::U]) -> &mut Self {
        output.copy_from_slice(&self.squeeze[..output.len()]);
        self.squeeze = &self.squeeze[output.len()..];
        self
    }

    fn ratchet(&mut self) -> &mut Self {
        if let Some(absorb) = self.absorb.as_mut() {
            assert_eq!(&absorb[..7], b"RATCHET");
            *absorb = &absorb[7..];
        }
        assert_eq!(&self.squeeze[..7], b"RATCHET");
        self.squeeze = &self.squeeze[7..];
        self
    }
}
