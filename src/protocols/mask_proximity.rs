// IGNORE CHANGES TO THIS FILE - NOT FULLY PORTED TO PROPERLY USE BUFFER ABSTRACTION.

//! Mask proximity verification via γ-combination.
//!
//! Implements Construction 7.2 (p.43-44) specialized for zero-constraint mask
//! oracles. Given a shared Merkle tree containing 2n vectors — n original masks
//! and n fresh mask-of-masks — this protocol proves that each original mask is
//! close to a C_zk codeword without revealing the mask polynomials.
//!
//! The tree layout is:
//!   columns 0..n-1:   original masks   ξ_1, ..., ξ_n
//!   columns n..2n-1:  mask-of-masks    s_1, ..., s_n
//!
//! Protocol:
//!   1. Verifier sends γ (combination randomness)
//!   2. Prover sends combined polynomials ξ*_i = s_i + γ·ξ_i and
//!      combined IRS randomness r*_i = r'_i + γ·r_i for each mask pair
//!   3. Shared tree is opened at random positions
//!   4. Verifier checks: Enc(ξ*_i, r*_i)(y_j) = s_i(y_j) + γ·ξ_i(y_j)
//!      at each opened position, using linearity of the RS encoding
//!
//! ZK safety (follows the pattern of Construction 7.2, §7.1):
//!   - Only the combined ξ*_i = s_i + γ·ξ_i is revealed in full. Since s_i
//!     is uniformly random, ξ*_i is uniform regardless of ξ_i.
//!   - The tree is opened at ≤ t_zk positions, revealing ξ_i(y_j) and s_i(y_j)
//!     at each opened position. This is within the ZK query budget per mask.
//!
//! Note: Lemma 7.3 (p.44) derives the ZK bound ζ_C + n·ζ_{C_zk} assuming
//! each mask is a separate oracle. This implementation uses a shared tree
//! for all 2n vectors, so a single opening reveals 2n values. The per-mask
//! query count is unchanged (each ξ_i is opened at t_zk positions), but the
//! shared tree means all masks are opened at the SAME positions.
//!
//! Assumed: the shared-tree variant preserves the ZK bound. The argument
//! is that the simulator can simulate all 2n values at each position
//! independently (each pair (ξ_i, s_i) is simulatable from C_zk's ZK
//! property, and the pairs are independent across i). Formal derivation
//! for the shared-tree case is pending.
//!
//! Soundness: if ξ_i is far from C_zk, the spot-check fails with high
//! probability over γ (Lemma 7.4, p.45).

use std::fmt;

use ark_ff::Field;
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, RngCore};
use serde::{Deserialize, Serialize};

use crate::{
    algebra::{
        buffer::CpuBuffer, embedding::Identity, random_vector, scalar_mul_add_new,
        univariate_evaluate,
    },
    hash::Hash,
    protocols::irs_commit::{
        Commitment as IrsCommitment, Config as IrsConfig, Witness as IrsWitness,
    },
    transcript::{
        Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState, VerificationResult,
        VerifierMessage, VerifierState,
    },
    utils::zip_strict,
    verify,
};

/// Mask proximity configuration.
///
/// Wraps an IRS config for the shared mask tree and the number of mask pairs.
#[derive(Clone, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Config<F: Field> {
    pub c_zk_commit: IrsConfig<Identity<F>>,
    pub num_masks: usize,
}

/// Prover output from the commit phase.
#[must_use]
pub struct Witness<F: Field> {
    pub(crate) mask_witness: IrsWitness<F>,
    pub(crate) fresh_msgs: Vec<Vec<F>>,
}

/// Verifier output from the commit phase.
pub type Commitment<F> = IrsCommitment<F>;

impl<F: Field> Config<F> {
    pub fn new(c_zk_commit: IrsConfig<Identity<F>>, num_masks: usize) -> Self {
        assert_eq!(
            c_zk_commit.num_vectors,
            2 * num_masks,
            "c_zk.num_vectors must be 2 * num_masks"
        );
        assert_eq!(
            c_zk_commit.interleaving_depth, 1,
            "mask proximity requires interleaving_depth = 1"
        );
        // OOD evaluations are sent in the clear during IRS commit/receive,
        // which would leak raw mask values before the γ-combination and
        // break the ZK contract. The OOD path in irs_commit is slated for
        // removal in the new construction; until then, enforce zero here.
        assert_eq!(
            c_zk_commit.out_domain_samples, 0,
            "mask proximity requires out_domain_samples = 0 (OOD openings would leak raw mask evaluations)"
        );
        Self {
            c_zk_commit,
            num_masks,
        }
    }

    /// Commit all masks and their mask-of-masks in a single shared tree.
    ///
    /// Samples n fresh mask-of-mask polynomials, combines them with the
    /// provided original masks into a 2n-vector tree, and commits via IRS.
    pub fn commit<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        original_msgs: &[Vec<F>],
    ) -> Witness<F>
    where
        F: Codec<[H::U]>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        Standard: Distribution<F>,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(original_msgs.len(), self.num_masks);
        for msg in original_msgs {
            assert_eq!(msg.len(), self.c_zk_commit.vector_size);
        }

        // Sample fresh mask-of-masks
        let fresh_msgs: Vec<Vec<F>> = (0..self.num_masks)
            .map(|_| random_vector(prover_state.rng(), self.c_zk_commit.vector_size))
            .collect();

        let original_buffers = original_msgs
            .iter()
            .map(|msg| CpuBuffer::from_slice(msg))
            .collect::<Vec<_>>();
        let fresh_buffers = fresh_msgs
            .iter()
            .map(|msg| CpuBuffer::from_slice(msg))
            .collect::<Vec<_>>();

        // Tree layout: [originals..., freshes...]
        let all_vectors: Vec<&CpuBuffer<F>> = original_buffers
            .iter()
            .chain(fresh_buffers.iter())
            .collect();

        let mask_witness = self.c_zk_commit.commit(prover_state, &all_vectors);

        Witness {
            mask_witness,
            fresh_msgs,
        }
    }

    /// Receive a mask proximity commitment
    pub fn receive_commitment<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
    ) -> VerificationResult<Commitment<F>>
    where
        F: Codec<[H::U]>,
        H: DuplexSpongeInterface,
        Hash: ProverMessage<[H::U]>,
    {
        self.c_zk_commit.receive_commitment(verifier_state)
    }

    /// Prove that each original mask is close to a C_zk codeword.
    pub fn prove<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        witness: &Witness<F>,
        original_msgs: &[Vec<F>],
    ) where
        F: Codec<[H::U]>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        Standard: Distribution<F>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(original_msgs.len(), self.num_masks);
        assert_eq!(witness.fresh_msgs.len(), self.num_masks);

        // Step 1: receive combination randomness γ
        let gamma: F = prover_state.verifier_message();

        // Step 2: compute and send combined polynomials + IRS randomness
        let irs_masks_per_vector =
            self.c_zk_commit.mask_length * self.c_zk_commit.interleaving_depth;
        let irs_masks = witness.mask_witness.masks.as_slice();
        assert_eq!(irs_masks.len(), 2 * self.num_masks * irs_masks_per_vector);
        for (i, (orig_msg, fresh_msg)) in original_msgs
            .iter()
            .zip(witness.fresh_msgs.iter())
            .enumerate()
        {
            // ξ*_i = s_i + γ · ξ_i
            let combined_msg = scalar_mul_add_new(fresh_msg, gamma, orig_msg);
            prover_state.prover_messages(&combined_msg);

            // r*_i = r'_i + γ · r_i
            if irs_masks_per_vector > 0 {
                let orig_r = &irs_masks[i * irs_masks_per_vector..(i + 1) * irs_masks_per_vector];
                let fresh_r = &irs_masks[(self.num_masks + i) * irs_masks_per_vector
                    ..(self.num_masks + i + 1) * irs_masks_per_vector];
                let combined_r = scalar_mul_add_new(fresh_r, gamma, orig_r);
                prover_state.prover_messages(&combined_r);
            }
        }

        // Step 3: open the shared tree at random in-domain positions
        self.c_zk_commit
            .open(prover_state, &[&witness.mask_witness]);
    }

    /// Verify that each original mask is close to a C_zk codeword.
    pub fn verify<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
        commitment: &Commitment<F>,
    ) -> VerificationResult<()>
    where
        F: Codec<[H::U]>,
        H: DuplexSpongeInterface,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        // Step 1: send combination randomness γ
        let gamma: F = verifier_state.verifier_message();

        // Step 2: read combined polynomials + IRS randomness
        let msg_len = self.c_zk_commit.message_length();
        let irs_masks_per_vector =
            self.c_zk_commit.mask_length * self.c_zk_commit.interleaving_depth;
        let has_irs_masks = irs_masks_per_vector > 0;
        let mut combined_msgs = Vec::with_capacity(self.num_masks);
        let mut combined_rs: Option<Vec<Vec<F>>> =
            has_irs_masks.then(|| Vec::with_capacity(self.num_masks));
        for _ in 0..self.num_masks {
            combined_msgs.push(verifier_state.prover_messages_vec(msg_len)?);
            if let Some(ref mut rs) = combined_rs {
                rs.push(verifier_state.prover_messages_vec(irs_masks_per_vector)?);
            }
        }

        // Step 3: verify tree openings and get codeword values at opened positions
        let evaluations = self.c_zk_commit.verify(verifier_state, &[commitment])?;

        // Step 4: spot-check γ-combination at each opened position
        let num_cols = self.c_zk_commit.num_cols();
        for (row, &point) in zip_strict(
            evaluations.matrix.chunks_exact(num_cols),
            &evaluations.points,
        ) {
            let shift = combined_rs.as_ref().map(|_| point.pow([msg_len as u64]));
            // Direct indexing: c_zk_commit.interleaving_depth == 1 by Config::new.
            for i in 0..self.num_masks {
                let original_val = row[i];
                let fresh_val = row[self.num_masks + i];

                // Enc(ξ*_i, r*_i)(point) = ξ*_i(point) + point^msg_len · r*_i(point)
                let mut expected = univariate_evaluate(&combined_msgs[i], point);
                if let Some((rs, shift)) = combined_rs.as_ref().zip(shift) {
                    expected += shift * univariate_evaluate(&rs[i], point);
                }

                let actual = fresh_val + gamma * original_val;
                verify!(expected == actual);
            }
        }

        Ok(())
    }
}

impl<F: Field> fmt::Display for Config<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "masks {} tree {}", self.num_masks, self.c_zk_commit)
    }
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{
        distributions::{Distribution, Standard},
        rngs::StdRng,
        SeedableRng,
    };
    use proptest::{prelude::Strategy, prop_assume, prop_oneof, proptest, strategy::Just};

    use super::*;
    use crate::{
        algebra::{fields, ntt, random_vector},
        transcript::{codecs::U64, DomainSeparator},
    };

    impl<F: Field + 'static> Config<F>
    where
        Standard: Distribution<F>,
    {
        pub fn arbitrary() -> impl Strategy<Value = Self> {
            let valid_sizes = (1..=256)
                .filter(|&n| ntt::next_order::<F>(n) == Some(n))
                .collect::<Vec<_>>();

            // mask_length=0: tests non-ZK mask tree (no IRS masking on the committed vectors).
            let mask_length = prop_oneof![
                3 => Just(0_usize),
                7 => 1_usize..=4,
            ];
            (
                1_usize..=6,
                proptest::sample::select(valid_sizes),
                mask_length,
            )
                .prop_flat_map(|(num_masks, vector_size, mask_length)| {
                    let c_zk = IrsConfig::<Identity<F>>::arbitrary(
                        Identity::new(),
                        2 * num_masks,
                        vector_size,
                        mask_length,
                        1,
                    );
                    (Just(num_masks), c_zk)
                })
                .prop_filter(
                    "mask proximity requires out_domain_samples = 0",
                    |(_, c_zk)| c_zk.out_domain_samples == 0,
                )
                .prop_map(|(num_masks, c_zk)| Self::new(c_zk, num_masks))
        }
    }

    fn test_config<F>(seed: u64, config: &Config<F>)
    where
        F: Field + Codec<[u8]> + 'static,
        Standard: Distribution<F>,
        Hash: crate::transcript::ProverMessage<[u8]>,
    {
        let instance = U64(seed);
        let ds = DomainSeparator::protocol(config)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&instance);
        let mut rng = StdRng::seed_from_u64(seed);

        let original_msgs: Vec<Vec<F>> = (0..config.num_masks)
            .map(|_| random_vector(&mut rng, config.c_zk_commit.vector_size))
            .collect();

        let mut prover_state = ProverState::new_std(&ds);
        let witness = config.commit(&mut prover_state, &original_msgs);
        config.prove(&mut prover_state, &witness, &original_msgs);
        let proof = prover_state.proof();

        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let commitment = config.receive_commitment(&mut verifier_state).unwrap();
        config.verify(&mut verifier_state, &commitment).unwrap();
        verifier_state.check_eof().unwrap();
    }

    fn test<F: Field + Codec<[u8]> + 'static>()
    where
        Standard: Distribution<F>,
        Hash: crate::transcript::ProverMessage<[u8]>,
    {
        crate::tests::init();
        proptest!(|(seed: u64, config in Config::arbitrary())| {
            test_config(seed, &config);
        });
    }

    #[test]
    fn test_field64_1() {
        test::<fields::Field64>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field64_2() {
        test::<fields::Field64_2>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field64_3() {
        test::<fields::Field64_3>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field128() {
        test::<fields::Field128>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field192() {
        test::<fields::Field192>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field256() {
        test::<fields::Field256>();
    }

    /// Pre-γ tamper: commit honestly, then prove with different original masks.
    fn test_tampered_mask_config<F>(seed: u64, config: &Config<F>)
    where
        F: Field + Codec<[u8]> + 'static,
        Standard: Distribution<F>,
        Hash: crate::transcript::ProverMessage<[u8]>,
    {
        let instance = U64(seed);
        let ds = DomainSeparator::protocol(config)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&instance);
        let mut rng = StdRng::seed_from_u64(seed);

        let original_msgs: Vec<Vec<F>> = (0..config.num_masks)
            .map(|_| random_vector(&mut rng, config.c_zk_commit.vector_size))
            .collect();

        let mut prover_state = ProverState::new_std(&ds);
        let witness = config.commit(&mut prover_state, &original_msgs);

        let mut tampered_msgs = original_msgs;
        tampered_msgs[0][0] += F::ONE;
        config.prove(&mut prover_state, &witness, &tampered_msgs);
        let proof = prover_state.proof();

        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let commitment = config.receive_commitment(&mut verifier_state).unwrap();
        assert_rejected(|| config.verify(&mut verifier_state, &commitment));
    }

    /// Post-γ tamper: commit honestly, then corrupt the combined message to
    /// exercise the verifier's spot-check (step 4).
    fn test_tampered_combined_msg_config<F>(seed: u64, config: &Config<F>)
    where
        F: Field + Codec<[u8]> + 'static,
        Standard: Distribution<F>,
        Hash: crate::transcript::ProverMessage<[u8]>,
    {
        let instance = U64(seed);
        let ds = DomainSeparator::protocol(config)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&instance);
        let mut rng = StdRng::seed_from_u64(seed);

        let original_msgs: Vec<Vec<F>> = (0..config.num_masks)
            .map(|_| random_vector(&mut rng, config.c_zk_commit.vector_size))
            .collect();

        let mut prover_state = ProverState::new_std(&ds);
        let witness = config.commit(&mut prover_state, &original_msgs);

        let gamma: F = prover_state.verifier_message();
        let irs_masks_per_vector =
            config.c_zk_commit.mask_length * config.c_zk_commit.interleaving_depth;
        let irs_masks = witness.mask_witness.masks.as_slice();

        for (i, (orig_msg, fresh_msg)) in original_msgs
            .iter()
            .zip(witness.fresh_msgs.iter())
            .enumerate()
        {
            let mut combined_msg = scalar_mul_add_new(fresh_msg, gamma, orig_msg);
            if i == 0 {
                combined_msg[0] += F::ONE;
            }
            prover_state.prover_messages(&combined_msg);

            if irs_masks_per_vector > 0 {
                let orig_r = &irs_masks[i * irs_masks_per_vector..(i + 1) * irs_masks_per_vector];
                let fresh_r = &irs_masks[(config.num_masks + i) * irs_masks_per_vector
                    ..(config.num_masks + i + 1) * irs_masks_per_vector];
                let combined_r = scalar_mul_add_new(fresh_r, gamma, orig_r);
                prover_state.prover_messages(&combined_r);
            }
        }

        config
            .c_zk_commit
            .open(&mut prover_state, &[&witness.mask_witness]);
        let proof = prover_state.proof();

        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let commitment = config.receive_commitment(&mut verifier_state).unwrap();
        assert_rejected(|| config.verify(&mut verifier_state, &commitment));
    }

    fn assert_rejected<T>(verify: impl FnOnce() -> VerificationResult<T>) {
        #[cfg(feature = "verifier_panics")]
        use std::panic::AssertUnwindSafe;
        #[cfg(feature = "verifier_panics")]
        {
            let result = std::panic::catch_unwind(AssertUnwindSafe(verify));
            assert!(
                !matches!(result, Ok(Ok(_))),
                "verification unexpectedly succeeded"
            );
        }
        #[cfg(not(feature = "verifier_panics"))]
        {
            assert!(verify().is_err(), "verification unexpectedly succeeded");
        }
    }

    #[test]
    fn test_tampered_mask_rejected() {
        crate::tests::init();
        proptest!(|(seed: u64, config in Config::<fields::Field64>::arbitrary())| {
            prop_assume!(config.c_zk_commit.in_domain_samples > 0);
            test_tampered_mask_config(seed, &config);
        });
    }

    #[test]
    fn test_tampered_combined_msg_rejected() {
        crate::tests::init();
        proptest!(|(seed: u64, config in Config::<fields::Field64>::arbitrary())| {
            prop_assume!(config.c_zk_commit.in_domain_samples > 0);
            test_tampered_combined_msg_config(seed, &config);
        });
    }
}
