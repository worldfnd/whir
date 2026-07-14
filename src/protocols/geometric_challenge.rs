//! Produce challenge indices from a transcript.

use ark_ff::Field;

use crate::{
    algebra::geometric_sequence,
    buffer::{ActiveBuffer, Buffer},
    transcript::{Decoding, VerifierMessage},
};

/// Draw a geometric challenge `[1, x, x², …]` of length `count` as a host
/// `Vec`. Used by the (host-side) verifiers; the prover uses the buffer
/// variants below.
pub fn geometric_challenge<T, F>(transcript: &mut T, count: usize) -> Vec<F>
where
    T: VerifierMessage,
    F: Field + Decoding<[T::U]>,
{
    match count {
        0 => Vec::new(),
        1 => vec![F::ONE],
        _ => {
            // Only source entropy when required
            let x = transcript.verifier_message();
            geometric_sequence(F::ONE, x, count)
        }
    }
}

/// Buffer-native equivalent of [`geometric_challenge`]: the sequence is
/// generated on the backend so it never touches the host.
pub fn geometric_challenge_buffer<T, F>(transcript: &mut T, count: usize) -> ActiveBuffer<F>
where
    T: VerifierMessage,
    F: Field + Decoding<[T::U]>,
{
    geometric_challenge_groups(transcript, &[count])
        .into_iter()
        .next()
        .unwrap()
}

/// Sample a single geometric challenge base `x`.
///
/// Split the sequence `[1, x, x², …]` into consecutive groups of the given
/// `lengths`, each returned as its own on-device buffer. Group `k` starts at
/// `x^(lengths[0] + … + lengths[k-1])`.
///
/// This is the buffer-native equivalent of drawing one
/// [`geometric_challenge`] of length `lengths.iter().sum()` and slicing it
/// into consecutive runs — but nothing is read back to the host. Entropy is
/// sourced on exactly the same condition (total length `> 1`), so a buffer
/// prover and a host verifier drawing the same total stay in agreement.
pub fn geometric_challenge_groups<T, F>(
    transcript: &mut T,
    lengths: &[usize],
) -> Vec<ActiveBuffer<F>>
where
    T: VerifierMessage,
    F: Field + Decoding<[T::U]>,
{
    let total: usize = lengths.iter().sum();
    let base = if total > 1 {
        transcript.verifier_message()
    } else {
        F::ONE
    };
    let mut current = F::ONE;
    lengths
        .iter()
        .map(|&len| {
            let group = ActiveBuffer::<F>::geometric_challenge(current, base, len);
            current *= base.pow([len as u64]);
            group
        })
        .collect()
}
