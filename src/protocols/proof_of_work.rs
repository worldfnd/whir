//! Protocol for grinding and verifying proof of work.

use core::slice;

use ark_std::rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
#[cfg(feature = "tracing")]
use tracing::instrument;
use zerocopy::IntoBytes;

use crate::{
    bits::Bits,
    engines::EngineId,
    hash::{Hash, BLAKE3, ENGINES},
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverState, VerificationResult,
        VerifierMessage, VerifierState,
    },
    utils::zip_strict,
    verify,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Config {
    pub hash_id: EngineId,
    pub threshold: u64,
}

pub fn threshold(difficulty: Bits) -> u64 {
    assert!((0.0..=60.0).contains(&difficulty.into()));

    let threshold = (64.0 - f64::from(difficulty)).exp2().ceil();
    #[allow(clippy::cast_sign_loss)]
    if threshold >= u64::MAX as f64 {
        u64::MAX
    } else {
        threshold as u64
    }
}

pub fn difficulty(threshold: u64) -> Bits {
    Bits::from(64.0 - (threshold as f64).log2())
}

impl Config {
    pub const fn none() -> Self {
        Self {
            hash_id: BLAKE3,
            threshold: u64::MAX,
        }
    }

    /// Creates a new configuration from a difficulty.
    ///
    /// Defaults to Blake3 as the hash function.
    pub fn from_difficulty(difficulty: Bits) -> Self {
        Self {
            hash_id: BLAKE3,
            threshold: threshold(difficulty),
        }
    }

    pub fn difficulty(&self) -> Bits {
        difficulty(self.threshold)
    }

    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(engine)))]
    pub fn prove<H, R>(&self, prover_state: &mut ProverState<H, R>)
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
    {
        if self.threshold == u64::MAX {
            // If the difficulty is zero, do nothing (also produce no transcript)
            return;
        }

        // Retrieve the engine
        let engine = ENGINES
            .retrieve(self.hash_id)
            .expect("Hash Engine not found");
        #[cfg(feature = "tracing")]
        tracing::Span::current().record("engine", engine.name().as_ref());
        let batch_size = engine.preferred_batch_size();

        let challenge: [u8; 32] = prover_state.verifier_message();

        #[cfg(not(feature = "parallel"))]
        let nonce = (0_u64..)
            .step_by(batch_size)
            .find_map({
                let mut inputs = vec![[0u8; 64]; batch_size];
                for input in &mut inputs {
                    input[..32].copy_from_slice(&challenge);
                }
                let mut outputs = vec![Hash::default(); batch_size];
                move |nonce| {
                    let input_len = inputs.len();
                    for (input, nonce) in zip_strict(inputs.iter_mut(), (nonce..).take(input_len)) {
                        input[32..40].copy_from_slice(&nonce.to_le_bytes());
                    }
                    engine.hash_many(64, inputs.as_bytes(), &mut outputs);
                    let output_len = outputs.len();
                    for (output, nonce) in zip_strict(outputs.iter(), (nonce..).take(output_len)) {
                        let value = u64::from_le_bytes(output.0[..8].try_into().unwrap());
                        if value <= self.threshold {
                            return Some(nonce);
                        }
                    }
                    None
                }
            })
            .expect("Proof of Work failed to solve.");

        #[cfg(feature = "parallel")]
        let nonce = {
            use std::sync::atomic::{AtomicU64, Ordering};

            // Split the work across all available threads.
            // Use atomics to find the unique deterministic lowest satisfying nonce.
            let global_min = AtomicU64::new(u64::MAX);
            rayon::broadcast(|ctx| {
                let thread_nonces =
                    ((batch_size * ctx.index()) as u64..).step_by(batch_size * ctx.num_threads());
                let mut inputs = vec![[0u8; 64]; batch_size];
                for input in &mut inputs {
                    input[..32].copy_from_slice(&challenge);
                }
                let mut outputs = vec![Hash::default(); batch_size];
                for batch_start in thread_nonces {
                    // Stop work if another thread already found a lower valid nonce.
                    if batch_start >= global_min.load(Ordering::Relaxed) {
                        break;
                    }
                    let input_len = inputs.len();
                    for (input, nonce) in
                        zip_strict(inputs.iter_mut(), (batch_start..).take(input_len))
                    {
                        input[32..40].copy_from_slice(&nonce.to_le_bytes());
                    }
                    engine.hash_many(64, inputs.as_bytes(), &mut outputs);
                    let output_len = outputs.len();
                    for (output, nonce) in
                        zip_strict(outputs.iter(), (batch_start..).take(output_len))
                    {
                        let value = u64::from_le_bytes(output.0[..8].try_into().unwrap());
                        if value <= self.threshold {
                            // We found a solution, store it in the global_min.
                            // Use fetch_min to solve race condition with simultaneous solutions.
                            global_min.fetch_min(nonce, Ordering::SeqCst);
                            break;
                        }
                    }
                }
            });

            // Return the best found nonce, or fallback check on `u64::MAX`.
            let nonce = global_min.load(Ordering::SeqCst);
            assert!(nonce != u64::MAX, "Proof of Work failed to solve.");
            nonce
        };

        prover_state.prover_message(&U64(nonce));
    }

    pub fn verify<H>(&self, verifier_state: &mut VerifierState<H>) -> VerificationResult<()>
    where
        H: DuplexSpongeInterface,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
    {
        if self.threshold == u64::MAX {
            return Ok(());
        }
        let engine = ENGINES.retrieve(self.hash_id);
        verify!(engine.is_some());
        let engine = engine.unwrap();
        let challenge: [u8; 32] = verifier_state.verifier_message();
        let nonce: U64 = verifier_state.prover_message()?;

        let mut input = [0u8; 64];
        input[..32].copy_from_slice(&challenge);
        input[32..40].copy_from_slice(&nonce.0.to_le_bytes());
        let mut output = Hash::default();
        engine.hash_many(64, &input, slice::from_mut(&mut output));
        let value = u64::from_le_bytes(output.0[..8].try_into().unwrap());
        verify!(value <= self.threshold);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use proptest::{proptest, strategy::Strategy};
    #[cfg(feature = "tracing")]
    use tracing::instrument;

    use super::*;
    use crate::{
        hash::tests::hash_for_size,
        transcript::{codecs::Empty, DomainSeparator},
    };

    impl Config {
        pub fn arbitrary() -> impl Strategy<Value = Self> {
            (hash_for_size(64), (u64::MAX >> 6)..)
                .prop_map(|(hash_id, threshold)| Self { hash_id, threshold })
        }
    }

    #[cfg_attr(feature = "tracing", instrument)]
    fn test_config(config: &Config) {
        let ds = DomainSeparator::protocol(&config)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&Empty);

        // Prover
        let mut prover_state = ProverState::new_std(&ds);
        config.prove(&mut prover_state);
        let proof = prover_state.proof();

        // Verifier
        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        config.verify(&mut verifier_state).unwrap();
        verifier_state.check_eof().unwrap();
    }

    #[test]
    fn test_pow() {
        crate::tests::init();
        proptest!(|(config in Config::arbitrary())| {
            test_config(&config);
        });
    }

    #[test]
    fn test_threshold_integer() {
        assert_eq!(threshold(Bits::new(0.0)), u64::MAX);
        assert_eq!(threshold(Bits::new(60.0)), 1 << 4);
        proptest!(|(bits in 1_u64..=60)| {
            assert_eq!(threshold(Bits::new(bits as f64)), 1 << (64 - bits));
        });
    }

    #[test]
    fn test_threshold_fractional() {
        proptest!(|(bits in 0.0..=60.0)| {
            let t = threshold(Bits::new(bits));
            let min = threshold(Bits::new(bits.ceil()));
            let max = threshold(Bits::new(bits.floor()));
            assert!((min..=max).contains(&t));
        });
    }

    #[test]
    fn test_threshold_monotonic() {
        proptest!(|(bits in 0.0..=59.0, delta in 0.0..=1.0)| {
            let low = threshold(Bits::new(bits + delta));
            let high = threshold(Bits::new(bits));
            assert!(low <= high);
        });
    }

    #[test]
    fn test_difficulty_integer() {
        assert_eq!(difficulty(u64::MAX), Bits::new(0.0));
        assert_eq!(difficulty(1 << 4), Bits::new(60.0));
        proptest!(|(bits in 1_u64..=60)| {
            assert_eq!(difficulty(1 << (64 - bits)), Bits::new(bits as f64));
        });
    }

    #[test]
    fn test_difficulty_fractional() {
        proptest!(|(threshold in 16_u64..)| {
            let d = difficulty(threshold);
            let min = difficulty(threshold.checked_next_power_of_two().unwrap_or(u64::MAX));
            let max = Bits::new(f64::from(min) + 1.0);
            assert!((min..=max).contains(&d));
        });
    }

    #[test]
    fn test_difficulty_monotonic() {
        proptest!(|(threshold in 16_u64.., delta: u64)| {
            let high = difficulty(threshold);
            let low = difficulty(threshold.saturating_add(delta));
            assert!(low <= high);
        });
    }
}
