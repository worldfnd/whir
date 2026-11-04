//! IRS parameter selection; all masks are supplied by Zook.

use crate::{
    algebra::embedding::Embedding,
    protocols::{
        irs_commit::{Config as IrsConfig, IrsMode, IrsParams},
        params::{
            bounds::{rate, usize_to_f64},
            error::{DeriveError, ZookSecurityError},
            regime::DecodingRegimeParams,
            spec::{Mode, RoundContext, SecuritySpec},
        },
    },
};

/// Encode the supplied vector without appending randomness.
pub fn solve<M: Embedding + Default>(
    spec: &SecuritySpec,
    ctx: &RoundContext,
) -> Result<IrsConfig<M>, DeriveError> {
    solve_with_num_vectors(spec, ctx, 1)
}

/// Encode equal-length vectors in one commitment with shared query paths.
pub(super) fn solve_with_num_vectors<M: Embedding + Default>(
    spec: &SecuritySpec,
    ctx: &RoundContext,
    num_vectors: usize,
) -> Result<IrsConfig<M>, DeriveError> {
    let depth = 1_usize
        .checked_shl(ctx.folding_factor)
        .ok_or(DeriveError::SizeOverflow)?;
    if ctx.vector_size == 0 || !ctx.vector_size.is_multiple_of(depth) {
        return Err(DeriveError::InvalidMaskLayout);
    }
    let rate = rate(f64::from(ctx.log_inv_rate));
    if rate == 0.0 {
        return Err(DeriveError::SizeOverflow);
    }
    // Zook query steps sum two error terms; reserve half the error budget
    // for the in-domain term. No grinding credit is used for these IOP bounds.
    let security_target = match spec.mode {
        Mode::Standard => f64::from(spec.protocol_security_target_bits()),
        Mode::ZeroKnowledge => f64::from(spec.target_security_bits) + 1.0,
    };
    Ok(IrsConfig::try_new(IrsParams {
        security_target,
        decoding_regime: spec.decoding_regime,
        hash_id: spec.hash_id,
        num_vectors,
        vector_size: ctx.vector_size,
        interleaving_depth: depth,
        rate,
        mode: IrsMode::Standard,
    })?)
}

/// Require degree-preserving encoding and `delta/2 <= theta < 1 - sqrt(1-delta)`.
/// Here `delta = (n - m + 1)/n` is the relative Reed–Solomon distance,
/// `n = codeword_length()`, `m = message_length()`, and `theta` is the
/// configured proximity radius.
///
/// ProveKit Prover Whitepaper Appendix A, “Correlated agreement”, constrained
/// correlated-agreement theorem. Section 5.3, “Overlaid masks instead of
/// zk-encoding”, defines the degree-preserving commitment to `p + mask`.
pub(super) fn validate_zook_code<M: Embedding>(
    code: &IrsConfig<M>,
) -> Result<(), ZookSecurityError> {
    if !matches!(code.mode(), IrsMode::Standard) {
        return Err(ZookSecurityError::AppendedMasks);
    }
    let DecodingRegimeParams::Johnson { .. } = code.regime() else {
        return Err(ZookSecurityError::UnsupportedRegime);
    };
    let rho = code.rate();
    let theta = 1.0 - 2.0_f64.powf(code.regime().one_minus_distance_log2(-rho.log2()));
    // RS minimum distance is (n - k + 1) / n, including the constant term.
    let distance = 1.0 - rho + 1.0 / usize_to_f64(code.codeword_length());
    if theta < distance / 2.0 || theta >= 1.0 - (1.0 - distance).sqrt() {
        return Err(ZookSecurityError::InvalidProximity);
    }
    Ok(())
}

/// Restrict the final witness bound to one polynomial per commitment.
pub(super) fn validate_single_polynomial<M: Embedding>(
    code: &IrsConfig<M>,
) -> Result<(), ZookSecurityError> {
    validate_zook_code(code)?;
    if code.num_vectors() != 1 || code.interleaving_depth() != 1 {
        return Err(ZookSecurityError::ExpectedSinglePolynomial);
    }
    Ok(())
}

/// Require two full-length columns: joint masks and their final-opening blinder.
pub(super) fn validate_joint_mask_code<M: Embedding>(
    code: &IrsConfig<M>,
) -> Result<(), ZookSecurityError> {
    validate_zook_code(code)?;
    if code.num_vectors() != 2 || code.interleaving_depth() != 1 {
        return Err(ZookSecurityError::ExpectedMaskAndBlinder);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::*;
    use crate::{
        algebra::{
            embedding::Identity,
            fields::{Field64, Field64_2},
        },
        hash::BLAKE3,
        protocols::params::{
            spec::DecodingRegime,
            test_utils::{johnson_code, johnson_code_with_vectors},
        },
    };
    type F = Field64_2;

    #[test]
    fn joint_masks_require_two_full_length_columns() {
        for vectors in [1, 2, 3] {
            for depth in [1, 2] {
                let code = johnson_code_with_vectors::<Identity<F>>(128, depth, 0.25, vectors);
                let expected = if vectors == 2 && depth == 1 {
                    Ok(())
                } else {
                    Err(ZookSecurityError::ExpectedMaskAndBlinder)
                };
                assert_eq!(validate_joint_mask_code(&code), expected);
                assert_eq!(
                    validate_single_polynomial(&code).is_ok(),
                    vectors == 1 && depth == 1
                );
            }
        }
    }

    #[test]
    fn rejects_appended_masks_and_unsupported_regimes() {
        for (mode, regime, expected) in [
            (
                IrsMode::ZeroKnowledge {
                    mask_length: NonZeroUsize::new(16).unwrap(),
                },
                DecodingRegime::Johnson,
                ZookSecurityError::AppendedMasks,
            ),
            (
                IrsMode::Standard,
                DecodingRegime::Unique,
                ZookSecurityError::UnsupportedRegime,
            ),
            (
                IrsMode::Standard,
                DecodingRegime::Capacity,
                ZookSecurityError::UnsupportedRegime,
            ),
        ] {
            let config = IrsConfig::<Identity<F>>::new(IrsParams {
                security_target: 80.0,
                decoding_regime: regime,
                hash_id: BLAKE3,
                num_vectors: 1,
                vector_size: 128,
                interleaving_depth: 1,
                rate: 0.25,
                mode,
            });
            assert_eq!(validate_zook_code(&config), Err(expected));
        }
    }

    #[test]
    fn rejects_proximity_below_half_distance() {
        let small = johnson_code::<Identity<Field64>>(2, 1, 0.5);
        assert_eq!(
            validate_zook_code(&small),
            Err(ZookSecurityError::InvalidProximity)
        );
    }
}
