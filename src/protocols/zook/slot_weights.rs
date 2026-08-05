//! Slot-weight builder for virtual-source code-switch.
//!
//! After [`super::batched::selector::merge`] reduces `t` active blocks to a single
//! virtual block, the downstream code-switch must open every active block's
//! IRS commitment at the same in-domain query rows and combine those rows
//! into one virtual row per query. The combination weights are the tensor
//! product
//!
//! ```text
//!   slot_weights[b · num_cols + s] = θ_b · eq_weights(γ)[s]
//! ```
//!
//! where `θ_b = eq_ℓ(δ, b)` is the per-block selector weight and `γ` is the
//! WHIR sumcheck folding randomness for the round.

use ark_ff::Field;

use crate::algebra::eq_weights;

/// Build the flat `slot_weights` vector that [`crate::protocols::code_switch::Config::prove_virtual`]
/// and [`crate::protocols::code_switch::Config::verify_virtual_for_implicit`]
/// expect. Returns a vector of length `theta.len() · 2^folding_randomness.len()`
/// with block index as the outer axis and per-block columns as the inner axis.
pub fn build<F: Field>(theta: &[F], folding_randomness: &[F]) -> Vec<F> {
    let collapse = eq_weights(folding_randomness);
    let num_cols = collapse.len();
    if theta.is_empty() || num_cols == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(theta.len() * num_cols);
    for &theta_b in theta {
        for &col_weight in &collapse {
            out.push(theta_b * col_weight);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use ark_ff::AdditiveGroup;
    use ark_std::rand::{rngs::StdRng, Rng, SeedableRng};

    use super::*;
    use crate::algebra::{dot, embedding::Identity, fields::Field64, mixed_dot, random_vector};

    type F = Field64;

    #[test]
    fn single_block_slot_weights_equal_eq_weights() {
        // t = 1, θ = [1]: slot_weights collapses to eq_weights(γ).
        let mut rng = StdRng::seed_from_u64(0);
        let folding: Vec<F> = random_vector(&mut rng, 3);
        let collapse = eq_weights(&folding);
        let slot = build(&[F::ONE], &folding);
        assert_eq!(slot, collapse);
    }

    #[test]
    fn empty_theta_yields_empty() {
        let theta: Vec<F> = Vec::new();
        let folding: Vec<F> = vec![F::from(2u64), F::from(3u64)];
        assert!(build(&theta, &folding).is_empty());
    }

    #[test]
    fn tensor_layout_block_outer_slot_inner() {
        // For t = 2, num_cols = 4: slot_weights[block * 4 + slot] = θ_block · collapse_slot.
        let mut rng = StdRng::seed_from_u64(7);
        let folding: Vec<F> = random_vector(&mut rng, 2); // num_cols = 4
        let collapse = eq_weights(&folding);
        let theta: Vec<F> = vec![rng.gen(), rng.gen()];
        let slot = build(&theta, &folding);
        assert_eq!(slot.len(), 2 * 4);
        for b in 0..2 {
            for s in 0..4 {
                assert_eq!(slot[b * 4 + s], theta[b] * collapse[s]);
            }
        }
    }

    #[test]
    fn slot_weights_match_per_block_then_combine_reduction() {
        // For a synthetic 2-block, 4-col-per-block matrix row, doing the
        // reduction as a single `mixed_dot(slot_weights, row)` must equal
        // the two-step reduction:
        //   per_block_collapse(b) = Σ_s collapse[s] · row[b · 4 + s]
        //   combined              = Σ_b θ_b · per_block_collapse(b)
        let mut rng = StdRng::seed_from_u64(11);
        let folding: Vec<F> = random_vector(&mut rng, 2);
        let collapse = eq_weights(&folding);
        let theta: Vec<F> = vec![rng.gen(), rng.gen()];

        // Synthetic IRS row (8 source-field elements per query).
        let row: Vec<F> = random_vector(&mut rng, 8);

        let slot = build(&theta, &folding);
        let one_step = mixed_dot(&Identity::<F>::default(), &slot, &row);

        let mut two_step = F::ZERO;
        for b in 0..2 {
            let per_block: F = (0..4).map(|s| collapse[s] * row[b * 4 + s]).sum();
            two_step += theta[b] * per_block;
        }
        assert_eq!(one_step, two_step);
        // Sanity check: mixed_dot over Identity = plain dot.
        assert_eq!(one_step, dot(&slot, &row));
    }
}
