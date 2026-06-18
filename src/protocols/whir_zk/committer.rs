// IGNORE CHANGES TO THIS FILE - NOT FULLY PORTED TO PROPERLY USE BUFFER ABSTRACTION.

use ark_ff::Field;
use ark_std::rand::{distributions::Standard, prelude::Distribution};
#[cfg(feature = "tracing")]
use tracing::instrument;

use super::{utils::BlindingPolynomials, Config};
use crate::buffer::BufferOps;
use crate::{
    algebra::buffer::ActiveBuffer,
    hash::Hash,
    protocols::{irs_commit, whir},
    transcript::{
        Codec, DuplexSpongeInterface, ProverMessage, ProverState, VerificationResult, VerifierState,
    },
};

/// zkWHIR commitment: one Merkle root per masked polynomial plus one for blinding.
#[derive(Debug)]
pub struct Commitment<F: Field> {
    pub f_hat: Vec<whir::Commitment<F>>,
    pub blinding: whir::Commitment<F>,
}

/// zkWHIR witness produced by [`Config::commit`].
///
/// Contains the masked polynomial witnesses, their coefficient vectors,
/// the blinding polynomial family, and the single blinding commitment witness.
#[derive(Clone, Debug)]
pub struct Witness<F: Field> {
    pub f_hat_vectors: Vec<Vec<F>>,
    pub f_hat_witnesses: Vec<irs_commit::Witness<F, F>>,
    pub blinding_polynomials: Vec<BlindingPolynomials<F>>,
    pub blinding_vectors: Vec<Vec<F>>,
    pub blinding_witness: irs_commit::Witness<F, F>,
}

impl<F: Field> Config<F> {
    /// Commit to one or more polynomials with zero-knowledge blinding.
    ///
    /// For each polynomial, samples fresh blinding coefficients from the
    /// prover's private transcript-bound RNG, constructs the
    /// masked polynomial `f_hat = f + m_poly`, and commits both the masked
    /// polynomials and the blinding vectors to the transcript.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn commit<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        polynomials: &[&ActiveBuffer<F>],
    ) -> Witness<F>
    where
        Standard: Distribution<F>,
        H: DuplexSpongeInterface,
        R: ark_std::rand::RngCore + ark_std::rand::CryptoRng,
        F: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(
            self.blinded_commitment.initial_committer.num_vectors, 1,
            "zkWHIR currently expects one vector per commitment"
        );

        let mut f_hat_vectors = Vec::with_capacity(polynomials.len());
        let mut f_hat_witnesses = Vec::with_capacity(polynomials.len());
        let mut blinding_polynomials = Vec::with_capacity(polynomials.len());
        let num_blinding_variables = self.num_blinding_variables();
        let num_witness_variables = self.num_witness_variables();
        for &poly in polynomials {
            let poly = poly.as_slice();
            let blinding = BlindingPolynomials::sample(
                prover_state.rng(),
                num_blinding_variables,
                num_witness_variables,
            );
            let mask = &blinding.m_poly;
            let witness_size = self.blinded_commitment.initial_size();
            assert!(!mask.is_empty(), "blinding mask vector must be non-empty");
            assert!(
                witness_size >= mask.len(),
                "witness vector smaller than blinding mask vector"
            );
            assert_eq!(
                witness_size % mask.len(),
                0,
                "witness size must be multiple of blinding mask size"
            );
            // Safe to .cycle() because witness_size % mask.len() == 0 (asserted above).
            let f_hat_vec = poly
                .iter()
                .zip(mask.iter().cycle())
                .map(|(&coeff, &m)| coeff + m)
                .collect::<Vec<_>>();
            let f_hat_buffer = ActiveBuffer::from_slice(&f_hat_vec);
            let witness = self
                .blinded_commitment
                .commit(prover_state, &[&f_hat_buffer]);
            f_hat_vectors.push(f_hat_vec);
            f_hat_witnesses.push(witness);
            blinding_polynomials.push(blinding);
        }

        let blinding_num_vectors = self.blinding_commitment.initial_committer.num_vectors;
        assert_eq!(
            blinding_num_vectors,
            polynomials.len() * (num_witness_variables + 1),
            "blinding commitment layout mismatch: expected n*(mu+1) vectors"
        );
        let mut blinding_vectors = Vec::with_capacity(blinding_num_vectors);
        for poly in &blinding_polynomials {
            let layout = poly.layout_vectors();
            debug_assert_eq!(
                layout.len(),
                num_witness_variables + 1,
                "layout_vectors must produce mu+1 vectors"
            );
            blinding_vectors.extend(layout);
        }
        let blinding_buffers = blinding_vectors
            .iter()
            .map(|v| ActiveBuffer::from_slice(v))
            .collect::<Vec<_>>();
        let blinding_vector_refs = blinding_buffers.iter().collect::<Vec<_>>();
        let blinding_witness = self
            .blinding_commitment
            .commit(prover_state, &blinding_vector_refs);

        Witness {
            f_hat_vectors,
            f_hat_witnesses,
            blinding_polynomials,
            blinding_vectors,
            blinding_witness,
        }
    }

    /// Receive `num_polynomials` masked-polynomial commitments and one blinding
    /// commitment from the verifier transcript.
    pub fn receive_commitments<H>(
        &self,
        verifier_state: &mut VerifierState<'_, H>,
        num_polynomials: usize,
    ) -> VerificationResult<Commitment<F>>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let f_hat = (0..num_polynomials)
            .map(|_| self.blinded_commitment.receive_commitment(verifier_state))
            .collect::<Result<Vec<_>, _>>()?;
        let blinding = self
            .blinding_commitment
            .receive_commitment(verifier_state)?;
        Ok(Commitment { f_hat, blinding })
    }
}
