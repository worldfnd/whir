//! Produce challenge indices from a transcript.

use crate::transcript::{Decoding, VerifierMessage};

/// Generate a set of indices for challenges.
pub fn challenge_indices<T>(
    transcript: &mut T,
    num_leaves: usize,
    count: usize,
    deduplicate: bool,
) -> Vec<usize>
where
    T: VerifierMessage,
    u8: Decoding<[T::U]>,
{
    if count == 0 {
        return Vec::new();
    }
    if num_leaves <= 1 {
        // `size_bytes` would be zero; short-circuit before the entropy loop.
        return if deduplicate { vec![0] } else { vec![0; count] };
    }

    let domain = IndexDomain::new(num_leaves);
    let mut indices = Vec::with_capacity(count);
    for _ in 0..count {
        indices.push(domain.sample(transcript));
    }

    if deduplicate {
        indices.sort_unstable();
        indices.dedup();
    }
    indices
}

/// Rejection-sampling domain for transcript-derived row indices.
///
/// For power-of-two domains, `entropy_space` is an exact multiple of
/// `num_leaves`, so rejection never triggers and sampling is bit-identical to
/// the legacy `% num_leaves` implementation. For non-power-of-two domains,
/// candidates in the biased tail are rejected and redrawn.
#[derive(Clone, Copy, Debug)]
struct IndexDomain {
    num_leaves: usize,
    size_bytes: usize,
    threshold: u128,
}

impl IndexDomain {
    fn new(num_leaves: usize) -> Self {
        debug_assert!(num_leaves > 1);
        let bits_needed = ceil_log2(num_leaves);
        let size_bytes = bits_needed.div_ceil(8);
        let entropy_bits = 8 * size_bytes;
        debug_assert!(entropy_bits < u128::BITS as usize);

        let entropy_space = 1u128 << entropy_bits;
        let num_leaves_u = num_leaves as u128;
        let threshold = (entropy_space / num_leaves_u) * num_leaves_u;

        Self {
            num_leaves,
            size_bytes,
            threshold,
        }
    }

    fn sample<T>(&self, transcript: &mut T) -> usize
    where
        T: VerifierMessage,
        u8: Decoding<[T::U]>,
    {
        loop {
            let candidate = self.draw_candidate(transcript);
            if candidate < self.threshold {
                return (candidate % self.num_leaves as u128) as usize;
            }
        }
    }

    fn draw_candidate<T>(&self, transcript: &mut T) -> u128
    where
        T: VerifierMessage,
        u8: Decoding<[T::U]>,
    {
        let mut candidate = 0u128;
        for _ in 0..self.size_bytes {
            candidate = (candidate << 8) | u128::from(transcript.verifier_message::<u8>());
        }
        candidate
    }
}

fn ceil_log2(n: usize) -> usize {
    debug_assert!(n > 1);
    usize::BITS as usize - (n - 1).leading_zeros() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::{codecs::Empty, DomainSeparator, MockSponge, ProverState};

    #[test]
    fn test_challenge_stir_queries_single_byte_indices() {
        let num_leaves = 1 << 7;
        let num_queries = 5;

        let ds = DomainSeparator::protocol(&module_path!())
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&Empty);
        // Mock transcript with fixed bytes (ensuring reproducibility)
        let sponge = MockSponge {
            absorb: None, // Anything is fine
            squeeze: &[
                0x01, 0x23, 0x45, 0x67, 0x89, // Query 1
                0xAB, 0xCD, 0xEF, 0x12, 0x34, // Query 2
                0x56, 0x78, 0x9A, 0xBC, 0xDE, // Query 3
                0xF0, 0x11, 0x22, 0x33, 0x44, // Query 4
                0x55, 0x66, 0x77, 0x88, 0x99, // Query 5
            ],
        };
        let mut prover_state = ProverState::new(&ds, sponge);

        let result = challenge_indices(&mut prover_state, num_leaves, num_queries, true);

        // Manually computed expected indices
        let index_0 = 0x01 % num_leaves;
        let index_1 = 0x23 % num_leaves;
        let index_2 = 0x45 % num_leaves;
        let index_3 = 0x67 % num_leaves;
        let index_4 = 0x89 % num_leaves;

        let mut expected_indices = vec![index_0, index_1, index_2, index_3, index_4];
        expected_indices.sort_unstable();
        expected_indices.dedup();

        assert_eq!(
            result, expected_indices,
            "Mismatch in computed indices for domain_size_bytes = 1"
        );
    }

    #[test]
    fn test_challenge_stir_queries_two_byte_indices() {
        let num_leaves = 1 << 13;
        let num_queries = 5;

        let ds = DomainSeparator::protocol(&module_path!())
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&Empty);
        // Expected `folded_domain_size = 65536 / 8 = 8192`
        let sponge = MockSponge {
            absorb: None,
            squeeze: &[
                0x01, 0x23, 0x45, 0x67, 0x89, // Query 1
                0xAB, 0xCD, 0xEF, 0x12, 0x34, // Query 2
                0x56, 0x78, 0x9A, 0xBC, 0xDE, // Query 3
                0xF0, 0x11, 0x22, 0x33, 0x44, // Query 4
                0x55, 0x66, 0x77, 0x88, 0x99, // Query 5
            ],
        };
        let mut prover_state = ProverState::new(&ds, sponge);

        let result = challenge_indices(&mut prover_state, num_leaves, num_queries, true);

        // Manually computed expected indices using two bytes per index
        let index_0 = ((0x01 << 8) | 0x23) % num_leaves;
        let index_1 = ((0x45 << 8) | 0x67) % num_leaves;
        let index_2 = ((0x89 << 8) | 0xAB) % num_leaves;
        let index_3 = ((0xCD << 8) | 0xEF) % num_leaves;
        let index_4 = ((0x12 << 8) | 0x34) % num_leaves;

        let mut expected_indices = vec![index_0, index_1, index_2, index_3, index_4];
        expected_indices.sort_unstable();

        assert_eq!(
            result, expected_indices,
            "Mismatch in computed indices for domain_size_bytes = 2"
        );
    }

    #[test]
    fn test_challenge_stir_queries_three_byte_indices() {
        let num_leaves = 1 << 20;
        let num_queries = 4;

        // Expected `folded_domain_size = 2^24 / 16 = 2^20 = 1,048,576`
        let ds = DomainSeparator::protocol(&module_path!())
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&Empty);
        let sponge = MockSponge {
            absorb: None,
            squeeze: &[
                0x12, 0x34, 0x56, // Query 1
                0x78, 0x9A, 0xBC, // Query 2
                0xDE, 0xF0, 0x11, // Query 3
                0x22, 0x33, 0x44, // Query 4
            ],
        };
        let mut prover_state = ProverState::new(&ds, sponge);

        let result = challenge_indices(&mut prover_state, num_leaves, num_queries, true);

        // Manually computed expected indices using three bytes per index
        let index_0 = ((0x12 << 16) | (0x34 << 8) | 0x56) % num_leaves;
        let index_1 = ((0x78 << 16) | (0x9A << 8) | 0xBC) % num_leaves;
        let index_2 = ((0xDE << 16) | (0xF0 << 8) | 0x11) % num_leaves;
        let index_3 = ((0x22 << 16) | (0x33 << 8) | 0x44) % num_leaves;

        let mut expected_indices = vec![index_0, index_1, index_2, index_3];
        expected_indices.sort_unstable();

        assert_eq!(
            result, expected_indices,
            "Mismatch in computed indices for domain_size_bytes = 3"
        );
    }

    #[test]
    fn test_challenge_stir_queries_duplicate_indices() {
        // Case where the function should deduplicate indices
        let num_leaves = 128;
        let num_queries = 5;

        // Mock narg_string where some indices will collide
        let ds = DomainSeparator::protocol(&module_path!())
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&Empty);
        let mut prover_state = ProverState::new(
            &ds,
            MockSponge {
                absorb: None,
                squeeze: &[
                    0x20, 0x40, 0x20, 0x60, 0x40, // Duplicate indices 0x20 and 0x40
                ],
            },
        );

        let result = challenge_indices(&mut prover_state, num_leaves, num_queries, true);

        // Manually computed expected indices, ensuring duplicates are removed
        let mut expected_indices = vec![0x20 % num_leaves, 0x40 % num_leaves, 0x60 % num_leaves];
        expected_indices.sort_unstable();

        assert_eq!(
            result, expected_indices,
            "Mismatch in computed indices for deduplication test"
        );
    }

    /// Non-pow2 `num_leaves`. With `num_leaves = 589824 = 2^16 · 9`:
    ///   `bits_needed = 20`, `size_bytes = 3`, entropy_space = 2^24 = 16_777_216,
    ///   `threshold = floor(2^24 / 589824) · 589824 = 28 · 589824 = 16_515_072`.
    ///
    /// The first 3-byte chunk decodes to 0xFFFFFF = 16_777_215, which is ≥
    /// threshold and must be rejected. The second chunk decodes to a small
    /// value < threshold and is returned.
    #[test]
    fn test_challenge_indices_non_pow2_rejection_retry() {
        let num_leaves: usize = 589_824;
        let ds = DomainSeparator::protocol(&module_path!())
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&Empty);
        let sponge = MockSponge {
            absorb: None,
            squeeze: &[
                0xFF, 0xFF, 0xFF, // candidate = 16_777_215 ≥ threshold → reject
                0x00, 0x00, 0x05, // candidate = 5 < threshold → accept
            ],
        };
        let mut prover_state = ProverState::new(&ds, sponge);

        let result = challenge_indices(&mut prover_state, num_leaves, 1, false);
        assert_eq!(result, vec![5]);
    }

    /// All indices returned for a non-pow2 `num_leaves` lie in `[0, num_leaves)`,
    /// and the function is deterministic (same sponge bytes → same result).
    #[test]
    fn test_challenge_indices_non_pow2_in_range_and_deterministic() {
        let num_leaves: usize = 589_824;
        let count = 5;
        let bytes: &[u8] = &[
            0x12, 0x34, 0x56, // candidate 1_193_046 < threshold
            0x78, 0x9A, 0xBC, // candidate 7_904_956 < threshold
            0xDE, 0xF0, 0x11, // candidate 14_610_449 < threshold
            0x22, 0x33, 0x44, // candidate 2_241_348 < threshold
            0x55, 0x66, 0x77, // candidate 5_596_791 < threshold
        ];
        let make_state = || {
            let ds = DomainSeparator::protocol(&module_path!())
                .session(&format!("Test at {}:{}", file!(), line!()))
                .instance(&Empty);
            ProverState::new(
                &ds,
                MockSponge {
                    absorb: None,
                    squeeze: bytes,
                },
            )
        };

        let mut first = make_state();
        let mut second = make_state();
        let r1 = challenge_indices(&mut first, num_leaves, count, false);
        let r2 = challenge_indices(&mut second, num_leaves, count, false);

        assert_eq!(r1, r2, "challenge_indices must be deterministic");
        assert_eq!(r1.len(), count);
        assert!(
            r1.iter().all(|&i| i < num_leaves),
            "all indices must lie in [0, {num_leaves}): got {r1:?}",
        );

        // Spot-check the first index: 0x123456 = 1_193_046 < threshold 16_515_072
        // → accepted, index = 1_193_046 % 589_824 = 13_398.
        assert_eq!(r1[0], 1_193_046 % num_leaves);
    }
}
