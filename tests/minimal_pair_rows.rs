//! Opening both roots and folding their rows equals encoding the folded message.

use std::num::NonZeroUsize;

use ark_ff::{AdditiveGroup, Field};
use whir::{
    algebra::{
        embedding::{Basefield, Embedding},
        eq_weights,
        fields::{Field64, Field64_3},
        univariate_evaluate,
    },
    buffer::{Buffer, BufferOps},
    hash,
    protocols::{
        code_switch::mixed_fold_chunks,
        irs_commit::{Config, IrsMode, IrsParams},
        params::DecodingRegime,
    },
    transcript::{codecs::Empty, DomainSeparator, ProverState},
};

#[test]
fn extension_field_row_fold_equals_encoding_the_folded_pair() {
    let embedding = Basefield::<Field64_3>::default();
    for mode in [
        IrsMode::Standard,
        IrsMode::ZeroKnowledge {
            mask_length: NonZeroUsize::new(3).unwrap(),
        },
    ] {
        let config = Config::<Basefield<Field64_3>>::new(IrsParams {
            security_target: 128.0,
            decoding_regime: DecodingRegime::Johnson,
            hash_id: hash::BLAKE3,
            num_vectors: 1,
            vector_size: 32,
            interleaving_depth: 4,
            rate: 0.25,
            mode,
        });
        let ds = DomainSeparator::protocol(&config).instance(&Empty);
        let mut prover = ProverState::new_std(&ds);
        let left = Buffer::from((1u64..=32).map(Field64::from).collect::<Vec<_>>());
        let right = Buffer::from(
            (0u64..32)
                .map(|i| Field64::from(2 * i + 11))
                .collect::<Vec<_>>(),
        );
        let commitments = [
            config.commit(&mut prover, &[&left]),
            config.commit(&mut prover, &[&right]),
        ];
        let opened = config.open(&mut prover, &[&commitments[0], &commitments[1]]);

        // Two roots contribute four columns each, at the same query points.
        assert_eq!(opened.num_columns(), 8);
        assert!(!opened.points.is_empty());
        let selector = Field64_3::new(Field64::ZERO, Field64::ONE, Field64::ZERO);
        let point = [selector, Field64_3::from(2), Field64_3::from(3)];
        let weights = Buffer::from(eq_weights(&point));
        let actual = opened.values_buffer(&embedding, &weights);

        // Each polynomial has eight coefficients. Fold all eight chunks of [left | right].
        let pair = left.concat(&right);
        let mut folded = mixed_fold_chunks(&embedding, pair.to_slice(), 8, &point);
        assert_eq!(folded.len(), 8);
        // Independent source masks follow the same left-then-right column order.
        let masks = commitments[0].masks.concat(&commitments[1].masks);
        if config.mask_length() != 0 {
            assert_ne!(
                commitments[0].masks.to_slice(),
                commitments[1].masks.to_slice()
            );
        }
        folded.extend(mixed_fold_chunks(
            &embedding,
            masks.to_slice(),
            config.mask_length(),
            &point,
        ));
        for (&x, &value) in opened.points.iter().zip(actual.to_slice()) {
            assert_eq!(value, univariate_evaluate(&folded, embedding.map(x)));
        }
    }
}
