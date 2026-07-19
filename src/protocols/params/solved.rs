//! Solver-recorded analytic floors.

use std::ops::Deref;

use crate::bits::Bits;

/// A sub-protocol config paired with the analytic-error floor the params
/// solver consumed when grinding its PoW slot.
///
/// Being wrapped in `Solved` is a type-level guarantee that the config came
/// out of a params solver; ad-hoc construction paths only produce the bare
/// config. Drift checks in
/// [`super::protocol_config::ProtocolConfig::validate`] compare the recorded
/// floor against a fresh recompute.
#[derive(Clone, Debug)]
pub struct Solved<C> {
    config: C,
    analytic: Bits,
}

impl<C> Solved<C> {
    pub(crate) const fn new(config: C, analytic: Bits) -> Self {
        Self { config, analytic }
    }

    /// Analytic-error floor recorded at solve time.
    pub const fn analytic(&self) -> Bits {
        self.analytic
    }

    pub const fn config(&self) -> &C {
        &self.config
    }

    pub fn into_config(self) -> C {
        self.config
    }
}

#[cfg(test)]
impl<C> Solved<C> {
    pub(crate) const fn config_mut_for_test(&mut self) -> &mut C {
        &mut self.config
    }

    pub(crate) const fn set_analytic_for_test(&mut self, analytic: Bits) {
        self.analytic = analytic;
    }
}

impl<C> Deref for Solved<C> {
    type Target = C;

    fn deref(&self) -> &C {
        &self.config
    }
}
