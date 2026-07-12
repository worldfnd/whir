//! Interleaved Reed-Solomon Commitment Protocol
//!
//! Commits to a `num_vectors` by `vector_size` matrix over `F`.
//!
//! This will be reshaped into a `vector_size / interleaving_depth` by
//! `num_vectors * interleaving_depth` matrix. Then each row is encoded
//! using an NTT friendly Reed-Solomon code to produce a `num_vectors * interleaving_depth`
//! by `codeword_size` matrix. This matrix is committed using the [`matrix_commit`] protocol.
//!
//! On opening the commitment, the protocol randomly selects `in_domain_samples` rows and opens
//! them using the [`matrix_commit`] protocol. Sampling is done with replacement, so may produce
//! fewer than `in_domain_samples` distinct rows.
//!
use std::{f64, fmt, num::NonZeroUsize};

use ark_ff::{AdditiveGroup, Field};
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use thiserror::Error;
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::{
    algebra::{
        dot, embedding::Embedding, fields::FieldWithSize, lift, linear_form::UnivariateEvaluation,
        ntt,
    },
    buffer::{ActiveBuffer, Buffer, BufferOps},
    engines::EngineId,
    hash::Hash,
    protocols::{
        challenge_indices::challenge_indices,
        matrix_commit,
        params::{bounds::ood_per_sample_log2, regime::DecodingRegimeParams, spec::DecodingRegime},
    },
    transcript::{
        Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState, VerificationResult,
        VerifierMessage, VerifierState,
    },
    type_info::Typed,
    utils::zip_strict,
};

#[derive(Clone, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
pub enum IrsMode {
    Standard,
    ZeroKnowledge { mask_length: NonZeroUsize },
}

impl IrsMode {
    /// Per-polynomial IRS randomness length. Returns 0 in Standard mode.
    pub const fn mask_length(&self) -> usize {
        match self {
            Self::Standard => 0,
            Self::ZeroKnowledge { mask_length } => mask_length.get(),
        }
    }
}

/// Commit to vectors over an fft-friendly field F
#[must_use]
#[derive(Clone, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Config<M: Embedding> {
    /// Embedding into a (larger) field used for weights and drawing challenges.
    embedding: Typed<M>,

    /// The number of vectors to commit to in one operation.
    num_vectors: usize,

    /// The number of coefficients in each vector.
    vector_size: usize,

    /// The number of Reed-Solomon evaluation points.
    codeword_length: usize,

    /// The number of independent codewords that are interleaved together.
    interleaving_depth: usize,

    /// The matrix commitment configuration.
    matrix_commit: matrix_commit::Config<M::Source>,

    /// Materialized Reed–Solomon decoding regime (Unique / Johnson w/ slack).
    regime: DecodingRegimeParams,

    /// The number of in-domain samples.
    in_domain_samples: usize,

    /// Whether to sort and deduplicate the in-domain samples.
    ///
    /// Deduplication can slightly reduce proof size and prover/verifier
    /// complexity, but it makes transcript pattern and control flow
    /// non-deterministic.
    deduplicate_in_domain: bool,

    /// Standard / ZeroKnowledge.
    mode: IrsMode,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash, Default, Serialize, Deserialize)]
#[must_use]
pub struct Witness<F: Field> {
    pub masks: ActiveBuffer<F>,
    pub matrix: ActiveBuffer<F>,
    pub matrix_witness: matrix_commit::Witness,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash, Default, Serialize, Deserialize)]
#[must_use]
pub struct Commitment {
    pub matrix_commitment: matrix_commit::Commitment,
}

/// Interleaved Reed-Solomon code.
///
/// Used for out- and in-domain samples.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize, Default)]
pub struct Evaluations<F> {
    /// Evaluation points for the RS code.
    pub points: Vec<F>,

    /// Matrix of codewords for each row.
    pub matrix: Vec<F>,
}

/// Named-field inputs to [`Config::new`] / [`Config::try_new`].
///
/// `num_vectors`, `vector_size`, and `interleaving_depth` share a primitive
/// type; named construction keeps call sites swap-proof.
#[derive(Debug, Clone)]
pub struct IrsParams {
    pub security_target: f64,
    pub decoding_regime: DecodingRegime,
    pub hash_id: EngineId,
    pub num_vectors: usize,
    pub vector_size: usize,
    pub interleaving_depth: usize,
    pub rate: f64,
    pub mode: IrsMode,
}

/// The computed codeword length exceeds the NTT engine's supported order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("codeword length {length} exceeds the NTT engine's supported order")]
pub struct CodewordLengthError {
    pub length: usize,
}

impl<M: Embedding> Config<M> {
    /// Panicking version of [`Config::try_new`] for call sites that construct
    /// from already-vetted parameters.
    ///
    /// # Panics
    ///
    /// If the codeword length exceeds the NTT engine's supported order.
    pub fn new(params: IrsParams) -> Self
    where
        M: Default,
    {
        Self::try_new(params).expect("IRS config construction failed")
    }

    /// # Errors
    ///
    /// [`CodewordLengthError`] when `masked_message_length / rate` exceeds the
    /// NTT engine's supported order.
    pub fn try_new(params: IrsParams) -> Result<Self, CodewordLengthError>
    where
        M: Default,
    {
        let IrsParams {
            security_target,
            decoding_regime,
            hash_id,
            num_vectors,
            vector_size,
            interleaving_depth,
            rate,
            mode,
        } = params;
        assert!(vector_size.is_multiple_of(interleaving_depth));
        assert!(rate > 0. && rate <= 1.);
        let masked_message_length = vector_size / interleaving_depth + mode.mask_length();
        // `interleaved_encode` requires `codeword_length` to divide the NTT root
        // order. `masked_message_length` is allowed to be arbitrary (the coset
        // NTT zero-extends internally), so we only round the codeword side here.
        #[allow(clippy::cast_sign_loss)]
        let raw_codeword_length = (masked_message_length as f64 / rate).ceil() as usize;
        let codeword_length =
            ntt::next_order::<M::Source>(raw_codeword_length).ok_or(CodewordLengthError {
                length: raw_codeword_length,
            })?;
        let rate = masked_message_length as f64 / codeword_length as f64;

        let regime = DecodingRegimeParams::from_policy(decoding_regime, rate);
        let in_domain_samples = num_in_domain_queries(decoding_regime, security_target, rate).get();

        Ok(Self {
            embedding: Typed::<M>::default(),
            num_vectors,
            vector_size,
            codeword_length,
            interleaving_depth,
            matrix_commit: matrix_commit::Config::with_hash(
                hash_id,
                codeword_length,
                interleaving_depth * num_vectors,
            ),
            regime,
            in_domain_samples,
            deduplicate_in_domain: false,
            mode,
        })
    }

    pub const fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    pub const fn vector_size(&self) -> usize {
        self.vector_size
    }

    pub const fn codeword_length(&self) -> usize {
        self.codeword_length
    }

    pub const fn interleaving_depth(&self) -> usize {
        self.interleaving_depth
    }

    pub const fn matrix_commit(&self) -> &matrix_commit::Config<M::Source> {
        &self.matrix_commit
    }

    pub const fn regime(&self) -> DecodingRegimeParams {
        self.regime
    }

    pub const fn in_domain_samples(&self) -> usize {
        self.in_domain_samples
    }

    pub const fn deduplicate_in_domain(&self) -> bool {
        self.deduplicate_in_domain
    }

    pub const fn mode(&self) -> &IrsMode {
        &self.mode
    }

    #[cfg(test)]
    pub(crate) const fn set_vector_size_for_test(&mut self, vector_size: usize) {
        self.vector_size = vector_size;
    }

    #[cfg(test)]
    pub(crate) const fn set_in_domain_samples_for_test(&mut self, in_domain_samples: usize) {
        self.in_domain_samples = in_domain_samples;
    }

    pub const fn num_cols(&self) -> usize {
        self.matrix_commit.num_cols
    }

    pub const fn size(&self) -> usize {
        self.matrix_commit.size()
    }

    pub fn embedding(&self) -> &M {
        &self.embedding
    }

    pub const fn num_messages(&self) -> usize {
        self.interleaving_depth * self.num_vectors
    }

    pub fn message_length(&self) -> usize {
        assert!(self.vector_size.is_multiple_of(self.interleaving_depth));
        self.vector_size / self.interleaving_depth
    }

    /// Per-polynomial IRS randomness length. Returns 0 in Standard mode.
    pub const fn mask_length(&self) -> usize {
        self.mode.mask_length()
    }

    /// Message length including mask coefficients.
    pub fn masked_message_length(&self) -> usize {
        self.message_length() + self.mask_length()
    }

    pub fn evaluation_points(&self, indices: &[usize]) -> Vec<M::Source> {
        ntt::evaluation_points::<M::Source>(
            self.masked_message_length(),
            self.codeword_length,
            indices,
        )
    }

    pub fn rate(&self) -> f64 {
        self.masked_message_length() as f64 / self.codeword_length as f64
    }

    pub const fn unique_decoding(&self) -> bool {
        self.regime.is_unique()
    }

    fn log_inv_rate(&self) -> f64 {
        -self.rate().log2()
    }

    /// Compute a list size bound.
    pub fn list_size(&self) -> f64 {
        let log_degree = (self.masked_message_length() as f64).log2();
        self.regime.list_size(log_degree, self.log_inv_rate())
    }

    /// Round-by-round soundness of the in-domain queries in bits.
    pub fn rbr_queries(&self) -> f64 {
        // Query error is (1 - δ)^q in bits = -q · log2(1 - δ).
        -(self.in_domain_samples as f64) * self.regime.one_minus_distance_log2(self.log_inv_rate())
    }

    /// Round-by-round soundness of the proximity-gaps fold in bits.
    /// See WHIR Theorem 4.8.
    pub fn rbr_soundness_fold_prox_gaps(&self) -> f64 {
        -self.regime.eps_mca_log2(
            self.log_inv_rate(),
            self.masked_message_length(),
            M::Target::field_size_bits(),
        )
    }

    /// Commit to one or more vectors.
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(self = %self)))]
    pub fn commit<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        vectors: &[&ActiveBuffer<M::Source>],
    ) -> Witness<M::Source>
    where
        Standard: Distribution<M::Source>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        M::Target: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        // Validate config
        assert!((self.vector_size).is_multiple_of(self.interleaving_depth));
        assert_eq!(self.matrix_commit.num_rows(), self.codeword_length);
        assert_eq!(self.matrix_commit.num_cols, self.num_messages());

        // Validate input
        assert_eq!(vectors.len(), self.num_vectors);
        assert!(vectors.iter().all(|p| p.len() == self.vector_size));

        let masks = ActiveBuffer::<M::Source>::random(
            prover_state.rng(),
            self.mask_length() * self.num_messages(),
        );
        let messages = ntt::Messages::new(vectors, self.message_length(), self.interleaving_depth);
        let matrix = ntt::interleaved_rs_encode(messages, &masks, self.codeword_length);

        // Commit to the matrix
        let matrix_witness = self.matrix_commit.commit(prover_state, &matrix);

        Witness {
            masks,
            matrix,
            matrix_witness,
        }
    }

    /// Receive a commitment to one or more vectors.
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(self = %self)))]
    pub fn receive_commitment<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
    ) -> VerificationResult<Commitment>
    where
        H: DuplexSpongeInterface,
        Hash: ProverMessage<[H::U]>,
        M::Target: Codec<[H::U]>,
    {
        let matrix_commitment = self.matrix_commit.receive_commitment(verifier_state)?;
        Ok(Commitment { matrix_commitment })
    }

    /// Commit to vectors and run the legacy WHIR OOD step in one call.
    ///
    /// Layered helper bundling `commit` + the OOD message exchange (sample
    /// `out_domain_samples` random points, send each vector's evaluation at
    /// each point). Used by the legacy WHIR protocol while the OOD step
    /// is still part of the per-commit protocol shape; the new construction
    /// (Construction 9.7) handles OOD at the code-switch level instead.
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(self = %self)))]
    pub fn commit_with_ood<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        vectors: &[&ActiveBuffer<M::Source>],
        out_domain_samples: usize,
    ) -> (Witness<M::Source>, Evaluations<M::Target>)
    where
        Standard: Distribution<M::Source>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        M::Target: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let witness = self.commit(prover_state, vectors);
        let points: Vec<M::Target> = prover_state.verifier_message_vec(out_domain_samples);
        let mut matrix = Vec::with_capacity(out_domain_samples * vectors.len());
        for &point in &points {
            for &vector in vectors {
                let value = vector.mixed_univariate_evaluate(&*self.embedding, point);
                prover_state.prover_message(&value);
                matrix.push(value);
            }
        }
        (witness, Evaluations { points, matrix })
    }

    /// Receive a commitment and the legacy WHIR OOD evaluations in one call.
    /// Verifier mirror of `commit_with_ood`.
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(self = %self)))]
    pub fn receive_commitment_with_ood<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
        out_domain_samples: usize,
    ) -> VerificationResult<(Commitment, Evaluations<M::Target>)>
    where
        H: DuplexSpongeInterface,
        Hash: ProverMessage<[H::U]>,
        M::Target: Codec<[H::U]>,
    {
        let commitment = self.receive_commitment(verifier_state)?;
        let points: Vec<M::Target> = verifier_state.verifier_message_vec(out_domain_samples);
        let matrix = verifier_state.prover_messages_vec(out_domain_samples * self.num_vectors)?;
        Ok((commitment, Evaluations { points, matrix }))
    }

    /// Opens the commitment and returns the evaluations of the vectors.
    ///
    /// Constraints are returned as a pair of evaluation point and values
    /// for each row.
    ///
    /// When there are multiple openings, the evaluation matrices will
    /// be horizontally concatenated.
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(self = %self)))]
    pub fn open<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        witnesses: &[&Witness<M::Source>],
    ) -> Evaluations<M::Source>
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        for witness in witnesses {
            assert_eq!(witness.matrix.len(), self.size());
        }

        // Get in-domain openings
        let (indices, points) = self.in_domain_challenges(prover_state);

        // For each commitment, send the selected rows to the verifier
        // and collect them in the evaluation matrix.
        let stride = witnesses.len() * self.num_cols();
        let mut matrix = vec![M::Source::ZERO; indices.len() * stride];
        let mut matrix_col_offset = 0;
        for witness in witnesses {
            let submatrix = witness.matrix.read_rows(self.num_cols(), &indices);
            if self.num_cols() != 0 {
                for (point_index, row) in submatrix.chunks_exact(self.num_cols()).enumerate() {
                    let matrix_row = &mut matrix[point_index * stride..(point_index + 1) * stride];
                    matrix_row[matrix_col_offset..matrix_col_offset + self.num_cols()]
                        .copy_from_slice(row);
                }
            }
            prover_state.prover_hint_ark(&submatrix);
            self.matrix_commit
                .open(prover_state, &witness.matrix_witness, &indices);
            matrix_col_offset += self.num_cols();
        }

        Evaluations { points, matrix }
    }

    /// Verifies one or more openings and returns the in-domain evaluations.
    ///
    /// **Note.** This does not verify the out-of-domain evaluations.
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(self = %self)))]
    pub fn verify<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
        commitments: &[&Commitment],
    ) -> VerificationResult<Evaluations<M::Source>>
    where
        H: DuplexSpongeInterface,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        // Get in-domain openings
        let (indices, points) = self.in_domain_challenges(verifier_state);

        // Receive (as a hint) a matrix of all the columns of all the commitments
        // corresponding to the in-domain opening rows.
        let stride = commitments.len() * self.num_cols();
        let mut matrix = vec![M::Source::ZERO; indices.len() * stride];
        let mut matrix_col_offset = 0;
        for commitment in commitments {
            let submatrix: Vec<M::Source> = verifier_state.prover_hint_ark()?;
            self.matrix_commit.verify(
                verifier_state,
                &commitment.matrix_commitment,
                &indices,
                &submatrix,
            )?;
            // Horizontally concatenate matrices.
            if stride != 0 && self.num_cols() != 0 {
                for (dst, src) in zip_strict(
                    matrix.chunks_exact_mut(stride),
                    submatrix.chunks_exact(self.num_cols()),
                ) {
                    dst[matrix_col_offset..matrix_col_offset + self.num_cols()]
                        .copy_from_slice(src);
                }
            }
            matrix_col_offset += self.num_cols();
        }
        Ok(Evaluations { points, matrix })
    }

    fn in_domain_challenges<T>(&self, transcript: &mut T) -> (Vec<usize>, Vec<M::Source>)
    where
        T: VerifierMessage,
        u8: Decoding<[T::U]>,
    {
        // Get in-domain openings
        let indices = challenge_indices(
            transcript,
            self.codeword_length,
            self.in_domain_samples,
            self.deduplicate_in_domain,
        );
        let points = self.evaluation_points(&indices);
        (indices, points)
    }
}

impl<F: Field> Evaluations<F> {
    pub const fn num_points(&self) -> usize {
        self.points.len()
    }

    pub fn num_columns(&self) -> usize {
        self.matrix
            .len()
            .checked_div(self.num_points())
            .unwrap_or_default()
    }

    pub fn rows(&self) -> impl Iterator<Item = &[F]> {
        let cols = self.num_columns();
        (0..self.num_points()).map(move |i| &self.matrix[i * cols..(i + 1) * cols])
    }

    pub fn lift<M>(&self, embedding: &M) -> Evaluations<M::Target>
    where
        M: Embedding<Source = F>,
    {
        Evaluations {
            points: lift(embedding, &self.points),
            matrix: lift(embedding, &self.matrix),
        }
    }

    pub fn evaluators(&self, size: usize) -> impl '_ + Iterator<Item = UnivariateEvaluation<F>> {
        self.points
            .iter()
            .map(move |&point| UnivariateEvaluation::new(point, size))
    }

    pub fn values<'a>(&'a self, weights: &'a [F]) -> impl 'a + Iterator<Item = F> {
        self.rows().map(|row| dot(weights, row))
    }
}

impl<M: Embedding> fmt::Display for Config<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "size {}×{}/{}",
            self.num_vectors, self.vector_size, self.interleaving_depth,
        )?;
        write!(f, " rate 2⁻{:.2}", -self.rate().log2())?;
        write!(f, " samples {} in-domain", self.in_domain_samples)
    }
}

/// Return the number of OOD samples needed.
///
/// Solves `(L choose 2) · ((degree - 1) / |F|)^s ≤ 2^{-security_target}`
/// where `L` is the list size and `degree` is the polynomial degree bound.
/// See [STIR] Lemma 4.5.
#[allow(clippy::cast_sign_loss)]
pub fn num_ood_samples(
    decoding_regime: DecodingRegime,
    security_target: f64,
    field_size_bits: f64,
    list_size: f64,
    degree: usize,
) -> usize {
    if matches!(decoding_regime, DecodingRegime::Unique) {
        return 0;
    }
    let log_per_sample = -ood_per_sample_log2(degree, field_size_bits);
    assert!(log_per_sample > 0.);
    let l_choose_2 = list_size * (list_size - 1.) / 2.;
    ((security_target + l_choose_2.log2()) / log_per_sample)
        .ceil()
        .max(1.) as usize
}

/// Return the number of in-domain queries.
///
/// Always ≥ 1 — the type carries that invariant so callers don't need to
/// re-prove it locally.
///
/// This is used by [`whir_zk`].
// TODO: A method with cleaner abstraction.
#[allow(clippy::cast_sign_loss)]
pub(crate) fn num_in_domain_queries(
    decoding_regime: DecodingRegime,
    security_target: f64,
    rate: f64,
) -> NonZeroUsize {
    let regime = DecodingRegimeParams::from_policy(decoding_regime, rate);
    // Query error is (1 - δ)^q in bits = -q · log2(1 - δ).
    let log_one_minus_delta = regime.one_minus_distance_log2(-rate.log2());
    let q = (security_target / -log_one_minus_delta).ceil() as usize;
    NonZeroUsize::new(q).unwrap_or(NonZeroUsize::MIN)
}

#[cfg(test)]
pub(crate) mod tests {
    use std::iter;

    use ark_std::rand::{
        distributions::Standard, prelude::Distribution, rngs::StdRng, SeedableRng,
    };
    use proptest::{bool, prelude::Strategy, proptest, sample::select, strategy::Just};

    use super::*;
    use crate::{
        algebra::{
            embedding::{Basefield, Compose, Frobenius, Identity},
            fields,
            ntt::NTT,
            random_vector, univariate_evaluate,
        },
        transcript::{codecs::U64, DomainSeparator},
    };

    impl<M: Embedding> Config<M> {
        pub fn arbitrary(
            embedding: M,
            num_vectors: usize,
            vector_size: usize,
            mask_length: usize,
            interleaving_depth: usize,
        ) -> impl Strategy<Value = Self> {
            assert!(interleaving_depth != 0);
            assert!(vector_size.is_multiple_of(interleaving_depth));
            let message_length = vector_size / interleaving_depth + mask_length;

            // Compute supported NTT domains for F
            let engine = NTT.get::<M::Source>().expect("Unsupported field");
            let valid_codeword_lengths =
                iter::successors(engine.next_order(message_length), |size| {
                    engine.next_order(*size + 1)
                })
                .filter(|n| n.is_power_of_two()) // TODO: Remove filter.
                .take(4)
                .collect::<Vec<_>>();
            let codeword_length = select(valid_codeword_lengths);

            // Combine with a matrix commitment config
            let codeword_matrix = codeword_length.prop_flat_map(move |codeword_length| {
                (
                    Just(codeword_length),
                    matrix_commit::tests::config::<M::Source>(
                        codeword_length,
                        interleaving_depth * num_vectors,
                    ),
                )
            });

            (codeword_matrix, 0_usize..=10, bool::ANY).prop_map(
                move |(
                    (codeword_length, matrix_commit),
                    in_domain_samples,
                    deduplicate_in_domain,
                )| {
                    let mode = NonZeroUsize::new(mask_length).map_or(IrsMode::Standard, |n| {
                        IrsMode::ZeroKnowledge { mask_length: n }
                    });
                    Self {
                        embedding: Typed::new(embedding.clone()),
                        num_vectors,
                        vector_size,
                        codeword_length,
                        interleaving_depth,
                        matrix_commit,
                        regime: DecodingRegimeParams::Unique,
                        in_domain_samples,
                        deduplicate_in_domain,
                        mode,
                    }
                },
            )
        }
    }

    fn test<M: Embedding>(seed: u64, config: &Config<M>)
    where
        M::Source: ProverMessage,
        M::Target: Codec,
        Standard: Distribution<M::Source> + Distribution<M::Target>,
    {
        crate::tests::init();

        // Pseudo-random Instance
        let instance = U64(seed);
        let ds = DomainSeparator::protocol(config)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&instance);
        let mut rng = StdRng::seed_from_u64(seed);
        let vectors = (0..config.num_vectors)
            .map(|_| random_vector(&mut rng, config.vector_size))
            .collect::<Vec<_>>();

        // TODO: Multiple commitments and openings.

        // Prover
        let mut prover_state = ProverState::new_std(&ds);
        let vector_buffers = vectors
            .iter()
            .map(|v| ActiveBuffer::from_slice(v))
            .collect::<Vec<_>>();
        let vector_refs = vector_buffers.iter().collect::<Vec<_>>();
        let witness = config.commit(&mut prover_state, &vector_refs);
        let in_domain_evals = config.open(&mut prover_state, &[&witness]);
        if config.deduplicate_in_domain {
            // Sorting is over index order, not points
            assert!(in_domain_evals.points.len() <= config.in_domain_samples);
            assert!({
                let mut unique = in_domain_evals.points.clone();
                unique.sort_unstable();
                unique.dedup();
                unique.len() == in_domain_evals.points.len()
            });
        } else {
            assert_eq!(in_domain_evals.points.len(), config.in_domain_samples);
        }
        assert_eq!(
            in_domain_evals.matrix.len(),
            in_domain_evals.points.len() * config.num_vectors * config.interleaving_depth
        );
        // Value-correctness assertion only valid in non-ZK mode: in ZK the
        // encoding is `Enc(f, r) = f(x) + x^ℓ · r(x)`, so opened values
        // include the mask term. The lifecycle round-trip (open/verify
        // agreement below) covers both modes.
        if config.num_vectors > 0 && config.mask_length() == 0 {
            let base = config.vector_size / config.interleaving_depth;
            for (point, evals) in zip_strict(
                &in_domain_evals.points,
                in_domain_evals
                    .matrix
                    .chunks_exact(config.num_vectors * config.interleaving_depth),
            ) {
                let expected_iter = vectors.iter().flat_map(|poly| {
                    (0..config.interleaving_depth).map(|j| {
                        // coefficients in the contiguous block for this interleaving index
                        let start = j * base;
                        let coeffs: Vec<_> = poly.iter().copied().skip(start).take(base).collect();
                        univariate_evaluate(&coeffs, *point)
                    })
                });
                for (expected, got) in zip_strict(expected_iter, evals.iter()) {
                    assert_eq!(expected, *got);
                }
            }
        }
        let proof = prover_state.proof();

        // Verifier
        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let commitment = config.receive_commitment(&mut verifier_state).unwrap();
        let verifier_in_domain_evals = config.verify(&mut verifier_state, &[&commitment]).unwrap();
        assert_eq!(&verifier_in_domain_evals, &in_domain_evals);
        verifier_state.check_eof().unwrap();
    }

    fn proptest<M: Embedding>(embedding: &M)
    where
        M::Source: ProverMessage,
        M::Target: Codec,
        Standard: Distribution<M::Source> + Distribution<M::Target>,
    {
        let valid_sizes = (1..=1024)
            .filter(|&n| ntt::next_order::<M::Source>(n) == Some(n))
            .collect::<Vec<_>>();
        let size = select(valid_sizes);

        let config = (0_usize..=3, size, 1_usize..=10, 0_usize..=8).prop_flat_map(
            |(num_vectors, size, interleaving_depth, mask_length)| {
                Config::arbitrary(
                    embedding.clone(),
                    num_vectors,
                    size * interleaving_depth,
                    mask_length,
                    interleaving_depth,
                )
            },
        );
        proptest!(|(
            seed: u64,
            config in config,
        )| {
            test(seed, &config);
        });
    }

    #[test]
    fn test_field64_1() {
        proptest(&Identity::<fields::Field64>::new());
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field64_2() {
        proptest(&Identity::<fields::Field64_2>::new());
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field64_3() {
        proptest(&Identity::<fields::Field64_3>::new());
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field128() {
        proptest(&Identity::<fields::Field128>::new());
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field192() {
        proptest(&Identity::<fields::Field192>::new());
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field256() {
        proptest(&Identity::<fields::Field256>::new());
    }

    #[test]
    fn test_basefield_field64_2() {
        proptest(&Basefield::<fields::Field64_2>::new());
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_basefield_field64_3() {
        proptest(&Basefield::<fields::Field64_3>::new());
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_base_frob_field64_3() {
        let embedding = Compose::new(Basefield::<fields::Field64_3>::new(), Frobenius::new(2));
        proptest(&embedding);
    }
}
