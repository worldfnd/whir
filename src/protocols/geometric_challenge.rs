//! Produce challenge indices from a transcript.

use ark_ff::Field;

use crate::{
    algebra::geometric_sequence,
    transcript::{Decoding, VerifierMessage},
};

/// Draw `count` consecutive powers of a transcript challenge, beginning at
/// exponent `offset`. An initial claim uses exponent zero; constraints added
/// to an existing claim begin at exponent one.
pub fn geometric_challenge<T, F>(transcript: &mut T, offset: usize, count: usize) -> Vec<F>
where
    T: VerifierMessage,
    F: Field + Decoding<[T::U]>,
{
    if count == 0 {
        return Vec::new();
    }

    let x = if offset + count > 1 {
        transcript.verifier_message()
    } else {
        F::ONE
    };
    geometric_sequence(x, offset + count)
        .into_iter()
        .skip(offset)
        .collect()
}

#[cfg(test)]
mod tests {
    use ark_ff::Field;

    use super::geometric_challenge;
    use crate::{
        algebra::fields::Field64,
        transcript::{codecs::Empty, DomainSeparator, ProverState, VerifierMessage},
    };

    #[test]
    fn fresh_constraints_start_after_existing_claim() {
        let ds = DomainSeparator::protocol(&"fresh constraints").instance(&Empty);
        let mut reference = ProverState::new_std(&ds);
        let gamma: Field64 = reference.verifier_message();
        assert_ne!(gamma, Field64::ONE);

        for (count, expected) in [
            (1, vec![gamma]),
            (3, vec![gamma, gamma.square(), gamma.square() * gamma]),
        ] {
            let mut transcript = ProverState::new_std(&ds);
            let actual: Vec<Field64> = geometric_challenge(&mut transcript, 1, count);
            assert_eq!(actual, expected);
        }
    }
}
