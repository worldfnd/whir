//! Zero-knowledge mask machinery: sampling, overlay onto witnesses, the joint
//! mask witness, mask-binding equations (`binding`), and the linear mask terms
//! carried through rounds (`claims`). Mask dimensions come from `params::masks`.

mod binding;
mod claims;

use std::iter::once;

use ark_ff::{AdditiveGroup, Field};
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, Rng, RngCore};
pub(in crate::whir) use binding::{prove_binding, receive_binding, MaskBinding};
pub(in crate::whir) use claims::{sumcheck_blinding, MaskClaims};
use zeroize::Zeroizing;

use crate::{
    algebra::{embedding::Embedding, random_vector},
    buffer::{Buffer, BufferOps},
    protocols::irs_commit::{Commitment as IrsCommitment, Witness as IrsWitness},
    whir::params::{JointMaskConfig, WitnessMaskConfig},
};

/// Joint masks and their final-opening blinder, committed as two columns in one tree.
#[derive(Clone, Debug)]
pub struct JointMaskWitness<F: Field> {
    pub(crate) coefficients: Buffer<F>,
    pub(crate) opening_blinder: Buffer<F>,
    pub(crate) irs_witness: IrsWitness<F>,
}

impl<F: Field> Drop for JointMaskWitness<F> {
    fn drop(&mut self) {
        self.coefficients.wipe();
        self.opening_blinder.wipe();
        self.irs_witness.masks.wipe();
        self.irs_witness.matrix.wipe();
    }
}

pub(in crate::whir) struct ProverMasks<F: Field> {
    pub(in crate::whir) joint: JointMaskWitness<F>,
    pub(in crate::whir) binding: MaskBinding<F>,
    pub(in crate::whir) claims: MaskClaims<F>,
}

pub(in crate::whir) struct VerifierMasks<F: Field> {
    pub(in crate::whir) joint: IrsCommitment,
    pub(in crate::whir) binding: MaskBinding<F>,
    pub(in crate::whir) claims: MaskClaims<F>,
}

/// Sample the configured coefficient ranges; unused encoding padding stays zero.
///
/// Section 5.4, “ZO0K”, “The required masks”: the initial mask and Frobenius-check
/// blinder are sampled in the source field. All other masks use the target field.
/// A basecase-only witness is lifted first, so all its masks use the target field.
pub(in crate::whir) fn sample_masks<M, R>(
    config: &JointMaskConfig<M::Target>,
    has_rounds: bool,
    rng: &mut R,
) -> (Buffer<M::Target>, Zeroizing<Vec<M::Source>>)
where
    M: Embedding + Default,
    R: RngCore + CryptoRng,
    Standard: Distribution<M::Source> + Distribution<M::Target>,
{
    let mut coefficients = vec![M::Target::ZERO; config.commit.vector_size()];
    for range in config
        .sumcheck_masks
        .iter()
        .chain(config.witness_masks.iter().skip(usize::from(has_rounds)))
        .chain(once(&config.blinder))
    {
        for coefficient in &mut coefficients[range.clone()] {
            *coefficient = rng.gen();
        }
    }
    let embedding = M::default();
    let initial_mask = if has_rounds {
        let range = config.witness_masks[0].clone();
        let mask = Zeroizing::new(random_vector::<M::Source>(&mut *rng, range.len()));
        for (coefficient, &value) in coefficients[range].iter_mut().zip(mask.iter()) {
            *coefficient = embedding.map(value);
        }
        mask
    } else {
        Zeroizing::new(Vec::new())
    };
    for coefficient in &mut coefficients[config.basefield_blinder.clone()] {
        *coefficient = if has_rounds {
            embedding.map(rng.gen())
        } else {
            rng.gen()
        };
    }
    (Buffer::from(coefficients), initial_mask)
}

/// Add an independent prefix to each contiguous IRS polynomial block.
///
/// For polynomial `j` and coefficient `i < q`, replace `p[j * m + i]` with
/// `p[j * m + i] + mask[j * q + i]`; all remaining coefficients are unchanged.
/// Here `m` is the block length and `q` the configured prefix length.
pub(in crate::whir) fn overlay_mask<F: Field>(
    message: Buffer<F>,
    mask: &[F],
    config: WitnessMaskConfig,
) -> Buffer<F> {
    assert!(config.num_polynomials > 0);
    assert!(message.len().is_multiple_of(config.num_polynomials));
    assert_eq!(
        config
            .num_polynomials
            .checked_mul(config.coefficients_per_polynomial),
        Some(mask.len()),
    );
    let block_len = message.len() / config.num_polynomials;
    let prefix = config.coefficients_per_polynomial;
    assert!(prefix > 0 && prefix <= block_len);
    let mut coefficients = message.into_vec();
    for (polynomial, mask) in coefficients
        .chunks_exact_mut(block_len)
        .zip(mask.chunks_exact(prefix))
    {
        for (coefficient, &randomness) in polynomial[..prefix].iter_mut().zip(mask) {
            *coefficient += randomness;
        }
    }
    Buffer::from(coefficients)
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::{
        algebra::{
            embedding::Basefield,
            fields::{Field64, Field64_2},
        },
        soundness::DecodingRegime,
        whir::params::{Mode, Params, ProtocolConfig, RateSchedule},
    };

    type M = Basefield<Field64_2>;

    #[test]
    fn overlay_changes_only_each_polynomial_prefix() {
        let message = Buffer::from((1..=8).map(Field64::from).collect::<Vec<_>>());
        let mask = [10, 20, 30, 40].map(Field64::from);
        let masked = overlay_mask(
            message,
            &mask,
            WitnessMaskConfig {
                num_polynomials: 2,
                coefficients_per_polynomial: 2,
            },
        );
        assert_eq!(
            masked.to_slice(),
            &[11, 22, 3, 4, 35, 46, 7, 8].map(Field64::from)
        );
    }

    #[test]
    fn initial_mask_and_membership_blinder_use_the_source_field() {
        let config = ProtocolConfig::<M>::derive(Params {
            bits: 80,
            mode: Mode::ZeroKnowledge,
            regime: DecodingRegime::Johnson,
            pow: 0,
            hash: crate::hash::BLAKE3,
            size: 4096,
            rate: 2,
            folds: [2, 2],
            schedule: RateSchedule::Stepping,
        })
        .unwrap();
        let layout = config.joint_masks().unwrap();
        let mut rng = StdRng::seed_from_u64(17);
        let (mut coefficients, initial) = sample_masks::<M, _>(layout, true, &mut rng);
        let first = layout.witness_masks[0].clone();
        for (&value, &source) in coefficients.to_slice()[first.clone()]
            .iter()
            .zip(initial.iter())
        {
            assert_eq!(value, M::default().map(source));
        }
        for range in [first, layout.basefield_blinder.clone()] {
            assert!(coefficients.to_slice()[range]
                .iter()
                .all(|value| value.frobenius_map(1) == *value));
        }
        for range in layout
            .sumcheck_masks
            .iter()
            .chain(layout.witness_masks.iter().skip(1))
            .chain(once(&layout.blinder))
        {
            assert!(coefficients.to_slice()[range.clone()]
                .iter()
                .any(|value| value.frobenius_map(1) != *value));
        }
        assert!(coefficients.to_slice()[layout.basefield_blinder.end..]
            .iter()
            .all(|value| *value == Field64_2::ZERO));
        coefficients.wipe();
    }
}
