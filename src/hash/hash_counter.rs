//! Global counter of hash invocations, reported by the binaries.

use std::sync::atomic::{AtomicUsize, Ordering};

pub static HASH_COUNTER: HashCounter = HashCounter::new();

#[derive(Debug)]
pub struct HashCounter(AtomicUsize);

impl Default for HashCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl HashCounter {
    pub const fn new() -> Self {
        Self(AtomicUsize::new(0))
    }

    pub(crate) fn add(&self, count: usize) {
        self.0.fetch_add(count, Ordering::Relaxed);
    }

    pub fn reset(&self) {
        self.0.store(0, Ordering::SeqCst);
    }

    pub fn get(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }
}
