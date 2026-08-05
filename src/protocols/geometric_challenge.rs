//! Produce challenge indices from a transcript.

use ark_ff::Field;

use crate::{
    algebra::geometric_sequence,
    buffer::{Buffer, BufferMath},
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
    let base = geometric_challenge_base(transcript, count);
    geometric_sequence(F::ONE, base, count)
}

/// Draw the base `x` for a geometric challenge of the given total length.
///
/// No entropy is needed for an empty challenge or the singleton `[1]`, so
/// those cases return one without touching the transcript.
pub fn geometric_challenge_base<T, F>(transcript: &mut T, count: usize) -> F
where
    T: VerifierMessage,
    F: Field + Decoding<[T::U]>,
{
    if count > 1 {
        transcript.verifier_message()
    } else {
        F::ONE
    }
}

/// Buffer-native equivalent of [`geometric_challenge`]: the sequence is
/// generated on the backend so it never touches the host.
pub fn geometric_challenge_buffer<T, F>(transcript: &mut T, count: usize) -> Buffer<F>
where
    T: VerifierMessage,
    F: Field + Decoding<[T::U]>,
{
    geometric_challenge_groups(transcript, &[count])
        .into_iter()
        .next()
        .unwrap()
}

/// Split the sequence `[1, x, x², …]` into consecutive groups of the given
/// `lengths`, each returned as its own on-device buffer. Group `k` starts at
/// `x^(lengths[0] + … + lengths[k-1])`.
///
/// This is the buffer-native equivalent of drawing one
/// [`geometric_challenge`] of length `lengths.iter().sum()` and slicing it
/// into consecutive runs — but nothing is read back to the host. Entropy is
/// sourced on exactly the same condition (total length `> 1`), so a buffer
/// prover and a host verifier drawing the same total stay in agreement.
pub fn geometric_challenge_groups<T, F>(transcript: &mut T, lengths: &[usize]) -> Vec<Buffer<F>>
where
    T: VerifierMessage,
    F: Field + Decoding<[T::U]>,
{
    geometric_challenge_groups_with_offset(transcript, 0, lengths).1
}

/// Draw one base and return consecutive resident groups beginning at
/// `x^offset`.
///
/// The entropy condition includes the omitted prefix, so the result matches a
/// host [`geometric_challenge`] of length `offset + lengths.iter().sum()`.
/// Returning the base lets callers combine resident groups with
/// transcript-sized host values without transferring either one.
pub fn geometric_challenge_groups_with_offset<T, F>(
    transcript: &mut T,
    offset: usize,
    lengths: &[usize],
) -> (F, Vec<Buffer<F>>)
where
    T: VerifierMessage,
    F: Field + Decoding<[T::U]>,
{
    let total = offset + lengths.iter().sum::<usize>();
    let base: F = geometric_challenge_base(transcript, total);
    let mut current = base.pow([offset as u64]);
    let groups = lengths
        .iter()
        .map(|&len| {
            let group = Buffer::<F>::geometric_challenge(current, base, len);
            current *= base.pow([len as u64]);
            group
        })
        .collect();
    (base, groups)
}
