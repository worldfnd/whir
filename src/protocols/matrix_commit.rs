//! Protocol for committing to rows of a matrix of some type <code>T: [Encodable]</code>.

use std::fmt;

use ark_ff::{Field, PrimeField};
use ark_std::rand::{CryptoRng, RngCore};
use derive_where::derive_where;
use serde::{Deserialize, Serialize};
use static_assertions::assert_obj_safe;
#[cfg(feature = "tracing")]
use tracing::instrument;
use zerocopy::{Immutable, IntoBytes};

use crate::{
    buffer::ActiveBuffer,
    buffer::BufferOps,
    engines::EngineId,
    hash::{self, Hash},
    protocols::merkle_tree,
    transcript::{
        DuplexSpongeInterface, ProverMessage, ProverState, VerificationError, VerificationResult,
        VerifierState,
    },
    type_info::{Type, TypeInfo},
    utils::{workload_size, zip_strict},
    verify,
};

/// Trait for types that can be encoded into a byte slice.
pub trait Encodable {
    /// Exact size in bytes this type will encode in.
    fn encoded_size() -> usize;

    /// Encoder for this type.
    ///
    /// During leaf hashing, an encoder will be created per thread and re-used
    /// as much as possible.
    fn encoder() -> Box<dyn Encoder<Self>>;
}

/// Object-safe encoder for types that implement [`Encodable`].
pub trait Encoder<T> {
    /// Returns true if the encoder uses an internal buffer.
    ///
    /// When true the input sizes will be chunked to limit the
    /// required buffer size.
    fn is_buffered(&self) -> bool;

    /// Encodes a slice of values into a byte slice.
    ///
    /// The byte slice must be exactly `values.len()` times `T::encoded_size()` bytes long.
    ///
    /// It can be a reference to an internal buffer in the encoder, or a reference to the values.
    fn encode<'e, 'd>(&'e mut self, values: &'d [T]) -> &'e [u8]
    where
        'd: 'e;
}

assert_obj_safe!(Encoder<()>);

/// Encoder for [`ark_ff::Field`]s.
///
/// It encodes reduced values in little-endian byte order using the minimum number
/// of bytes that fit all field elements. For extensions,  the coefficients are
/// encoded in order of increasing degree.
pub struct ArkFieldEncoder(Vec<u8>);

/// Encoder for types that implement [`zerocopy`]'s [`Immutable`] and [`IntoBytes`].
///
/// This is the fastest encoder available for types that support it. To use it you
/// need to implement [`Encodable`] as follows:
///
/// ```
/// # use zerocopy::{Immutable, IntoBytes};
/// # use whir::protocols::matrix_commit::{Encoder, Encodable, ZeroCopyEncoder};
/// #[derive(Immutable, IntoBytes)]
/// pub struct MyType(u64);
///
/// impl Encodable for MyType {
///   fn encoded_size() -> usize {
///       std::mem::size_of::<Self>()
///   }
///
///   fn encoder() -> Box<dyn Encoder<Self>> {
///       Box::new(ZeroCopyEncoder)
///   }
/// }
///
/// ```
///
pub struct ZeroCopyEncoder;

/// Configuration for the matrix commit protocol.
///
/// Commits row-wise to a matrix of field elements using a merkle tree.
#[derive_where(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
#[derive(Serialize, Deserialize)]
#[serde(bound = "T: TypeInfo")]
pub struct Config<T>
where
    T: TypeInfo + Encodable + Send + Sync,
{
    /// Field used for the matrix elements.
    pub element_type: Type<T>,

    /// The number of columns in the matrix.
    pub num_cols: usize,

    /// The hash function used for rows of the matrix.
    pub leaf_hash_id: EngineId,

    /// Merkle tree configuration for the matrix.
    pub merkle_tree: merkle_tree::Config,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash, Default, Serialize, Deserialize)]
pub struct BufferWitness {
    pub nodes: ActiveBuffer<Hash>,
}

pub type Witness = BufferWitness;

pub type Commitment = merkle_tree::Commitment;

/// Encode [`ark_ff::Field`]s using [`ArkFieldEncoder`].
impl<F: Field> Encodable for F {
    fn encoded_size() -> usize {
        let base_bytes = (F::BasePrimeField::MODULUS_BIT_SIZE as usize).div_ceil(8);
        base_bytes * F::extension_degree() as usize
    }

    fn encoder() -> Box<dyn Encoder<Self>> {
        Box::new(ArkFieldEncoder(Vec::new()))
    }
}

impl<F: Field> Encoder<F> for ArkFieldEncoder {
    fn is_buffered(&self) -> bool {
        true
    }

    fn encode<'e, 'd>(&'e mut self, values: &'d [F]) -> &'e [u8]
    where
        'd: 'e,
    {
        let base_bytes = (F::BasePrimeField::MODULUS_BIT_SIZE as usize).div_ceil(8);
        self.0.clear();
        for value in values {
            for coeff in value.to_base_prime_field_elements() {
                // Convert to regular reduced form (e.g. not Montgomery encoded).
                let bigint = coeff.into_bigint();
                // Get the limbs of the bigint in little-endian order.
                let limbs = bigint.as_ref();
                // Get the bytes of the limbs in little-endian order, and strip trailing zeros.
                #[cfg(not(target_endian = "little"))]
                compile_error!("This crate requires a little-endian target.");
                let bytes = &limbs.as_bytes()[..base_bytes];
                self.0.extend_from_slice(bytes);
            }
        }
        self.0.as_ref()
    }
}

impl<T: Immutable + IntoBytes> Encoder<T> for ZeroCopyEncoder {
    fn is_buffered(&self) -> bool {
        false
    }

    fn encode<'e, 'd>(&'e mut self, values: &'d [T]) -> &'e [u8]
    where
        'd: 'e,
    {
        values.as_bytes()
    }
}

impl<T: TypeInfo + Encodable + Send + Sync + Copy> Config<T> {
    /// Create a new matrix commit configuration with the recommended hash function.
    pub fn new(num_rows: usize, num_cols: usize) -> Self {
        // Select a leaf hash function.
        let leaf_size = T::encoded_size() * num_cols;
        let leaf_hash_id = if leaf_size <= 32 {
            hash::COPY
        } else if hash::Blake3::supports_size(leaf_size) {
            hash::BLAKE3
        } else {
            hash::SHA2
        };

        Self {
            element_type: Type::new(),
            num_cols,
            leaf_hash_id,
            merkle_tree: merkle_tree::Config::new(num_rows),
        }
    }

    pub fn with_hash(hash_id: EngineId, num_rows: usize, num_cols: usize) -> Self {
        Self {
            element_type: Type::new(),
            num_cols,
            leaf_hash_id: hash_id,
            merkle_tree: merkle_tree::Config::with_hash(hash_id, num_rows),
        }
    }

    pub const fn num_rows(&self) -> usize {
        self.merkle_tree.num_leaves
    }

    pub const fn size(&self) -> usize {
        self.num_rows() * self.num_cols
    }

    /// Commit the matrix (in row-major order).
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(self = %self, size = matrix.len(), engine)))]
    pub fn commit<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        matrix: &ActiveBuffer<T>,
    ) -> Witness
    where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        Hash: ProverMessage<[H::U]>,
        ActiveBuffer<T>: BufferOps<T, Nodes = ActiveBuffer<Hash>>,
    {
        assert_eq!(matrix.len(), self.num_rows() * self.num_cols);

        let (nodes, root) = matrix.merklize(self.num_cols, self.leaf_hash_id, &self.merkle_tree);
        prover_state.prover_message(&root);
        BufferWitness { nodes }
    }

    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(self = %self)))]
    pub fn receive_commitment<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
    ) -> VerificationResult<Commitment>
    where
        H: DuplexSpongeInterface,
        Hash: ProverMessage<[H::U]>,
    {
        self.merkle_tree.receive_commitment(verifier_state)
    }

    /// Opens the commitment at the provided row indices.
    ///
    /// Indices can be in any order and may contain duplicates. The row values are not provided by
    /// this protocol, it is up to the caller to provide them to the verifier.
    #[cfg_attr(feature = "tracing", instrument(skip(prover_state, witness, indices), fields(self = %self, num_indices = indices.len())))]
    pub fn open<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        witness: &Witness,
        indices: &[usize],
    ) where
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        Hash: ProverMessage<[H::U]>,
    {
        let node_indices = merkle_tree::opening_sibling_indices(
            self.merkle_tree.num_leaves,
            self.merkle_tree.layers.len(),
            indices,
        );
        for hint in witness.nodes.gather_at_indices(&node_indices) {
            prover_state.prover_hint(&hint);
        }
    }

    /// Verifies the commitment at the provided row indices.
    ///
    /// Indices can be in any order and may contain duplicates. The row values are not provided by
    /// this protocol, it is up to the caller to provide them to the verifier.
    #[cfg_attr(feature = "tracing", instrument(skip_all, fields(self = %self, engine, num_indices = indices.len())))]
    pub fn verify<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
        commitment: &Commitment,
        indices: &[usize],
        matrix: &[T],
    ) -> VerificationResult<()>
    where
        H: DuplexSpongeInterface,
        Hash: ProverMessage<[H::U]>,
    {
        verify!(matrix.len() == self.num_cols * indices.len());

        let engine = hash::ENGINES
            .retrieve(self.leaf_hash_id)
            .ok_or(VerificationError)?;
        #[cfg(feature = "tracing")]
        tracing::Span::current().record("engine", engine.name().as_ref());

        let mut leaf_hashes = vec![Hash::default(); indices.len()];
        hash_rows(&*engine, matrix, &mut leaf_hashes);
        self.merkle_tree
            .verify(verifier_state, commitment, indices, &leaf_hashes)
    }
}

impl<T: TypeInfo + Encodable + Send + Sync + Copy> fmt::Display for Config<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MatrixCommit({} x {})", self.num_rows(), self.num_cols)
    }
}

#[cfg(not(feature = "parallel"))]
fn hash_rows<T: Encodable + Send + Sync>(
    engine: &dyn hash::HashEngine,
    matrix: &[T],
    out: &mut [Hash],
) {
    hash_rows_serial(engine, matrix, out);
}

#[cfg(feature = "parallel")]
pub fn hash_rows<T: Encodable + Send + Sync>(
    engine: &dyn hash::HashEngine,
    matrix: &[T],
    out: &mut [Hash],
) {
    let Some(cols) = matrix.len().checked_div(out.len()) else {
        // Empty input and output
        return;
    };
    if matrix.len() > workload_size::<T>() && out.len() > engine.preferred_batch_size() {
        // Large task, split into parallel threads.
        let split = (out.len() / 2).next_multiple_of(engine.preferred_batch_size());
        let (mat_a, mat_b) = matrix.split_at(split * cols);
        let (out_a, out_b) = out.split_at_mut(split);
        rayon::join(
            || hash_rows(engine, mat_a, out_a),
            || hash_rows(engine, mat_b, out_b),
        );
    } else {
        hash_rows_serial(engine, matrix, out);
    }
}

fn hash_rows_serial<T: Encodable + Send + Sync>(
    engine: &dyn hash::HashEngine,
    matrix: &[T],
    out: &mut [Hash],
) {
    assert!(matrix.len().is_multiple_of(out.len()));
    let Some(cols) = matrix.len().checked_div(out.len()) else {
        // Empty input and output
        return;
    };
    let message_size = T::encoded_size() * cols;
    if message_size == 0 {
        // Empty message
        engine.hash_many(0, &[], out);
        return;
    }
    let mut encoder = T::encoder();
    if encoder.is_buffered() {
        // Buffered encoder, find some optimal size.
        let target = workload_size::<u8>() / 8;
        let batch_size = (target / message_size)
            .max(1)
            .next_multiple_of(engine.preferred_batch_size());
        for (matrix, out) in
            zip_strict(matrix.chunks(batch_size * cols), out.chunks_mut(batch_size))
        {
            let bytes = encoder.encode(matrix);
            engine.hash_many(message_size, bytes, out);
        }
    } else {
        // Unbuffered, encode everything in one go
        let bytes = encoder.encode(matrix);
        engine.hash_many(message_size, bytes, out);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use ark_std::rand::{
        distributions::{Distribution, Standard},
        rngs::StdRng,
        Rng, SeedableRng,
    };
    use proptest::{prop_assume, proptest, strategy::Strategy};

    use super::*;
    use crate::{
        algebra::fields,
        hash::{self, tests::hash_for_size},
        transcript::{codecs::Empty, DomainSeparator},
    };

    pub fn config<T>(num_rows: usize, num_cols: usize) -> impl Strategy<Value = Config<T>>
    where
        T: TypeInfo + Encodable + Send + Sync,
    {
        let leaf_hash = hash_for_size(T::encoded_size() * num_cols);
        let merkle_tree = merkle_tree::tests::config(num_rows);
        (leaf_hash, merkle_tree).prop_map(move |(leaf_hash_id, merkle_tree)| Config {
            element_type: Type::new(),
            num_cols,
            leaf_hash_id,
            merkle_tree,
        })
    }

    fn test<T>(
        mut rng: impl RngCore,
        leaf_hash: EngineId,
        node_hash: EngineId,
        layers: usize,
        num_rows: usize,
        num_cols: usize,
        indices: &[usize],
    ) where
        T: Copy + TypeInfo + Encodable + Send + Sync,
        Standard: Distribution<T>,
    {
        crate::tests::init();
        assert!(layers >= merkle_tree::layers_for_size(num_rows));
        assert!(indices.iter().all(|&index| index < num_rows));

        // Config
        let config = Config {
            element_type: Type::<T>::new(),
            num_cols,
            leaf_hash_id: leaf_hash,
            merkle_tree: merkle_tree::Config {
                num_leaves: num_rows,
                layers: vec![merkle_tree::LayerConfig { hash_id: node_hash }; layers],
            },
        };
        let ds = DomainSeparator::protocol(&config)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&Empty);

        // Instance
        let matrix: ActiveBuffer<T> =
            ActiveBuffer::from_vec((0..config.size()).map(|_| rng.gen()).collect());
        let submatrix: Vec<T> = matrix.read_rows(num_cols, indices);

        // Prover
        let mut prover_state = ProverState::new_std(&ds);
        let tree = config.commit(&mut prover_state, &matrix);
        config.open(&mut prover_state, &tree, indices);
        let proof = prover_state.proof();

        // Verifier
        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let commitment = config.receive_commitment(&mut verifier_state).unwrap();
        config
            .verify(&mut verifier_state, &commitment, indices, &submatrix)
            .unwrap();
        verifier_state.check_eof().unwrap();
    }

    fn proptest<T>()
    where
        T: Copy + TypeInfo + Encodable + Send + Sync,
        Standard: Distribution<T>,
    {
        let hashes = [hash::COPY, hash::SHA2, hash::SHA3, hash::BLAKE3];
        proptest!(|(
            seed: u64,
            leaf_hash in 0_usize..hashes.len(),
            node_hash in 1_usize..hashes.len(),
            layers in 0_usize..10,
            num_rows in 0_usize..100,
            num_cols in 0_usize..100,
            num_indices in 0_usize..100,
        )| {
            // There are no valid indices without rows.
            let num_indices = if num_rows == 0 { 0 } else { num_indices };

            // We need at least enough layers to cover the number of rows.
            let layers = layers + merkle_tree::layers_for_size(num_rows);

            let leaf_hash = hashes[leaf_hash];
            let node_hash = hashes[node_hash];
            prop_assume!(hash::ENGINES.retrieve(leaf_hash).unwrap().supports_size(T::encoded_size() * num_cols));
            prop_assume!(hash::ENGINES.retrieve(node_hash).unwrap().supports_size(64));

            let mut rng = StdRng::seed_from_u64(seed);
            let indices = (0..num_indices).map(|_| rng.gen_range(0..num_rows)).collect::<Vec<_>>();

            test::<T>(rng, leaf_hash, node_hash, layers, num_rows, num_cols, &indices);
        });
    }

    #[test]
    fn test_field64() {
        proptest::<fields::Field64>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field128() {
        proptest::<fields::Field128>();
    }

    #[test]
    fn test_field64_3() {
        // A non-power of two sized type.
        proptest::<fields::Field64_3>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field256() {
        proptest::<fields::Field256>();
    }
}
