//! Produce challenge indices from a transcript.

use ark_ff::Field;

use crate::{
    algebra::geometric_sequence,
    transcript::{Decoding, VerifierMessage},
};

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
            geometric_sequence(x, count)
        }
    }
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
            (0, vec![]),
            (1, vec![gamma]),
            (3, vec![gamma, gamma.square(), gamma.square() * gamma]),
        ] {
            let mut transcript = ProverState::new_std(&ds);
            let actual: Vec<Field64> =
                geometric_challenge(&mut transcript, 1 + count)[1..].to_vec();
            assert_eq!(actual, expected);
        }
    }
}
