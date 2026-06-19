// NOTE: 100% AI GENERATED

use std::{
    any::type_name,
    cell::OnceCell,
    cmp::Ordering,
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::RangeBounds,
};

use ark_ff::{AdditiveGroup, Field};
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, Rng, RngCore};
use metal::Buffer;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        fields::Field256,
        linear_form::{Covector, LinearForm, UnivariateEvaluation},
    },
    buffer::{BufferOps, BufferRead, BufferWrite},
    engines::EngineId,
    hash::{self, Hash as Digest},
    protocols::{
        matrix_commit::{hash_rows, Encodable},
        merkle_tree,
    },
};

use super::runtime::{
    as_field256_slice, assert_bn254, copy_field_buffer, copy_field_buffer_at, download_field,
    download_hash_indices,
    encode_field_rows_le, field256_to_f, field256_to_target, f_to_field256,
    geometric_accumulate_at_offset, geometric_accumulate_chunk_size, maybe_upload_bn254,
    parallel_dot, parallel_dot_at, parallel_fold_pair_sumcheck, parallel_geometric_accumulate_point_blocks,
    parallel_geometric_accumulate_point_blocks_batched, parallel_multilinear_extend_at,
    parallel_sumcheck, parallel_sumcheck_at, parallel_univariate_evaluate,
    parallel_univariate_evaluate_at, read_bn254_rows, run_in_place, runtime, scalar_mul_add_at,
    scalar_mul_at_offset, should_use_geometric_point_blocks, should_use_geometric_point_blocks_batched,
    target_to_field256, upload_field, zeroed_field_buffer,
};
use super::sha2::MetalSha2;

#[derive(Clone, Debug)]
pub(crate) struct MetalFieldBuffer {
    pub(crate) limbs: Buffer,
}

#[derive(Clone, Debug)]
pub(crate) struct MetalHashBuffer {
    pub(crate) bytes: Buffer,
}

#[derive(Clone, Debug)]
pub struct MetalBuffer<T> {
    len: usize,
    host_cache: OnceCell<Vec<T>>,
    field: Option<MetalFieldBuffer>,
    hash: Option<MetalHashBuffer>,
    _marker: PhantomData<T>,
}

impl<T: Clone> PartialEq for MetalBuffer<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Clone + Eq> Eq for MetalBuffer<T> {}

impl<T: Clone + PartialOrd> PartialOrd for MetalBuffer<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<T: Clone + Ord> Ord for MetalBuffer<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<T: Clone + Hash> Hash for MetalBuffer<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

impl<T: Clone + Default> Default for MetalBuffer<T> {
    fn default() -> Self {
        Self {
            len: 0,
            host_cache: OnceCell::new(),
            field: None,
            hash: None,
            _marker: PhantomData,
        }
    }
}

impl<T: Clone + Serialize> Serialize for MetalBuffer<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.as_slice().serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for MetalBuffer<T>
where
    T: Clone + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let data = Vec::<T>::deserialize(deserializer)?;
        Ok(BufferOps::from_vec(data))
    }
}

impl<T: Clone> MetalBuffer<T> {
    pub fn warmup() {
        super::runtime::init();
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        self.host_cache
            .get_or_init(|| self.download_host_cache())
            .as_slice()
    }

    pub(crate) fn hash_bn254_rows_sha2(&self, num_cols: usize, out: &mut [Digest]) -> bool {
        if type_name::<T>() != type_name::<Field256>() || self.field.is_none() {
            return false;
        }
        assert_eq!(self.len(), num_cols * out.len());
        let message_size = num_cols * size_of::<Field256>();
        let encoded = encode_field_rows_le(
            &self
                .field
                .as_ref()
                .expect("missing Metal field buffer")
                .limbs,
            out.len(),
            num_cols,
        );
        MetalSha2::new().hash_many_buffer(message_size, &encoded, out.len(), out);
        true
    }

    pub(crate) fn commit_bn254_rows_sha2_merkle(
        &self,
        num_cols: usize,
        num_rows: usize,
        layers: usize,
    ) -> Option<MetalBuffer<Digest>> {
        if type_name::<T>() != type_name::<Field256>() || self.field.is_none() {
            return None;
        }
        if num_rows != (1usize << layers) {
            return None;
        }
        assert_eq!(self.len(), num_cols * num_rows);
        let message_size = num_cols * size_of::<Field256>();
        let encoded = encode_field_rows_le(
            &self
                .field
                .as_ref()
                .expect("missing Metal field buffer")
                .limbs,
            num_rows,
            num_cols,
        );
        let sha = MetalSha2::new();
        let nodes = sha.build_merkle_tree_buffer_from_messages_buffer(
            message_size,
            &encoded,
            num_rows,
            layers,
        );
        Some(MetalBuffer::<Digest>::from_digest_buffer(
            nodes,
            (1usize << (layers + 1)) - 1,
        ))
    }

    pub(crate) fn read_hash_indices(&self, indices: &[usize]) -> Option<Vec<Digest>> {
        let buffer = self.hash.as_ref()?;
        Some(download_hash_indices(&buffer.bytes, self.len, indices))
    }
}

impl MetalBuffer<Digest> {
    pub(crate) fn from_digest_buffer(bytes: Buffer, len: usize) -> Self {
        Self {
            len,
            host_cache: OnceCell::new(),
            field: None,
            hash: Some(MetalHashBuffer { bytes }),
            _marker: PhantomData,
        }
    }

    pub(crate) fn read_hash_at(&self, index: usize) -> Option<Digest> {
        self.read_hash_indices(&[index])
            .map(|mut values| values.pop().expect("missing hash"))
    }
}

impl<T: Clone> BufferOps<T> for MetalBuffer<T> {
    type Nodes = MetalBuffer<Digest>;

    fn as_slice(&self) -> &[T] {
        self.as_slice()
    }

    fn at_index(&self, index: usize) -> Option<T> {
        self.as_slice().get(index).cloned()
    }

    fn gather_at_indices(&self, indices: &[usize]) -> Vec<T>
    where
        T: Copy,
    {
        if let Some(values) = self.read_hash_indices(indices) {
            return values
                .into_iter()
                .map(|value| unsafe { std::mem::transmute_copy(&value) })
                .collect();
        }
        indices
            .iter()
            .map(|&i| self.at_index(i).expect("index out of bounds"))
            .collect()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn read_rows(&self, num_cols: usize, indices: &[usize]) -> Vec<T> {
        if type_name::<T>() == type_name::<Field256>() && self.field.is_some() {
            return read_bn254_rows(
                self.field.as_ref().expect("missing Metal field buffer"),
                num_cols,
                indices,
            )
            .into_iter()
            .map(|value| unsafe { std::mem::transmute_copy(&value) })
            .collect();
        }
        let data = self.as_slice();
        let mut result = Vec::with_capacity(indices.len() * num_cols);
        for i in indices {
            result.extend_from_slice(&data[i * num_cols..(i + 1) * num_cols]);
        }
        result
    }

    fn from_vec(source: Vec<T>) -> Self {
        let len = source.len();
        let field = maybe_upload_bn254(&source);
        let host_cache = if field.is_some() {
            OnceCell::new()
        } else {
            OnceCell::from(source)
        };
        Self {
            len,
            host_cache,
            field,
            hash: None,
            _marker: PhantomData,
        }
    }

    fn from_slice(source: &[T]) -> Self {
        Self::from_vec(Vec::from(source))
    }

    fn merklize(
        &self,
        num_cols: usize,
        leaf_hash: EngineId,
        merkle: &merkle_tree::Config,
    ) -> (Self::Nodes, Digest)
    where
        T: Encodable + Send + Sync,
    {
        let num_rows = merkle.num_leaves;
        let layers = merkle.layers.len();
        assert_eq!(self.len(), num_cols * num_rows);

        // Fast path: build the whole tree on the GPU when the leaf and every node layer is SHA2.
        if leaf_hash == hash::SHA2
            && merkle
                .layers
                .iter()
                .all(|layer| layer.hash_id == hash::SHA2)
        {
            if let Some(nodes) = self.commit_bn254_rows_sha2_merkle(num_cols, num_rows, layers) {
                let root = nodes
                    .read_hash_at(merkle.num_nodes() - 1)
                    .expect("missing Metal Merkle root");
                return (nodes, root);
            }
        }

        let cpu_nodes = || {
            let engine = hash::ENGINES
                .retrieve(leaf_hash)
                .expect("Failed to retrieve hash engine");
            let mut leaves = vec![Digest::default(); num_rows];
            hash_rows(&*engine, self.as_slice(), &mut leaves);
            merkle.build_nodes(leaves)
        };

        let nodes = if leaf_hash == hash::SHA2 {
            let mut leaves = vec![Digest::default(); num_rows];
            if self.hash_bn254_rows_sha2(num_cols, &mut leaves) {
                merkle.build_nodes(leaves)
            } else {
                cpu_nodes()
            }
        } else {
            cpu_nodes()
        };
        let root = nodes[merkle.num_nodes() - 1];
        (BufferOps::from_vec(nodes), root)
    }
}

impl<F: Field + Clone> crate::buffer::Buffer<F> for MetalBuffer<F> {
    fn zeros(length: usize) -> Self {
        assert_bn254::<F>();
        // Montgomery zero is all-zero bytes, so a device-side fill suffices.
        Self {
            len: length,
            host_cache: OnceCell::new(),
            field: Some(zeroed_field_buffer(length)),
            hash: None,
            _marker: PhantomData,
        }
    }

    fn geometric_sequence(base: F, length: usize) -> Self {
        assert_bn254::<F>();
        BufferOps::from_vec(crate::algebra::geometric_sequence(base, length))
    }

    fn random<R>(rng: &mut R, length: usize) -> Self
    where
        R: RngCore + CryptoRng,
        Standard: Distribution<F>,
    {
        assert_bn254::<F>();
        BufferOps::from_vec((0..length).map(|_| rng.gen()).collect())
    }

    fn zero_pad(&mut self) {
        assert_bn254::<F>();
        if !self.is_empty() {
            let mut data = self.as_slice().to_vec();
            data.resize(self.len().next_power_of_two(), F::ZERO);
            *self = BufferOps::from_vec(data);
        }
    }

    fn fold(&mut self, weight: F) {
        assert_bn254::<F>();
        if self.len() <= 1 {
            return;
        }
        let len = self.len();
        let fold_half = len.next_power_of_two() >> 1;
        let weight = upload_field(&[f_to_field256(weight)]);
        let field = self.bn254_buffer();
        run_in_place(
            "bn254_fold",
            &[&field.limbs, &weight.limbs],
            &[len as u32, fold_half as u32],
            fold_half,
        );
        self.len = fold_half;
        self.invalidate_host_cache();
    }

    fn fold_pair(&mut self, other: &mut Self, weight: F) {
        assert_bn254::<F>();
        assert_eq!(self.len(), other.len());
        if self.len() <= 1 {
            return;
        }
        let len = self.len();
        let fold_half = len.next_power_of_two() >> 1;
        let weight = upload_field(&[f_to_field256(weight)]);
        let this = self.bn254_buffer();
        let other_buffer = other.bn254_buffer();
        run_in_place(
            "bn254_fold_pair",
            &[&this.limbs, &other_buffer.limbs, &weight.limbs],
            &[len as u32, fold_half as u32],
            fold_half,
        );
        self.len = fold_half;
        other.len = fold_half;
        self.invalidate_host_cache();
        other.invalidate_host_cache();
    }

    fn fold_pair_sumcheck_polynomial(&mut self, other: &mut Self, weight: F) -> (F, F) {
        assert_bn254::<F>();
        assert_eq!(self.len(), other.len());
        if self.len() <= 1 {
            return self.sumcheck_polynomial(other);
        }
        let len = self.len();
        let fold_half = len.next_power_of_two() >> 1;
        if fold_half == 1 {
            self.fold_pair(other, weight);
            return self.sumcheck_polynomial(other);
        }
        let weight = upload_field(&[f_to_field256(weight)]);
        let this = self.bn254_buffer();
        let other_buffer = other.bn254_buffer();
        let (c0, c2) = parallel_fold_pair_sumcheck(&this, &other_buffer, &weight, len, fold_half);
        self.len = fold_half;
        other.len = fold_half;
        self.invalidate_host_cache();
        other.invalidate_host_cache();
        (field256_to_f::<F>(c0), field256_to_f::<F>(c2))
    }

    fn linear_forms_rlc(
        size: usize,
        linear_forms: &mut [Box<dyn LinearForm<F>>],
        rlc_coeffs: &[F],
    ) -> Self {
        assert_bn254::<F>();
        assert_eq!(linear_forms.len(), rlc_coeffs.len());
        let Some((first, rest)) = linear_forms.split_first_mut() else {
            return Self::zeros(size);
        };
        let first = (first.as_mut() as &mut dyn std::any::Any)
            .downcast_mut::<Covector<F>>()
            .expect("MetalBuffer only supports Covector linear forms for BN254 RLC");
        let mut accumulator = <Self as BufferOps<F>>::from_slice(&first.vector);
        for (coeff, linear_form) in rlc_coeffs[1..].iter().zip(rest) {
            let covector = (linear_form.as_mut() as &mut dyn std::any::Any)
                .downcast_mut::<Covector<F>>()
                .expect("MetalBuffer only supports Covector linear forms for BN254 RLC");
            let vector = <Self as BufferOps<F>>::from_slice(&covector.vector);
            vector.mixed_scalar_mul_add_to(&Identity::new(), &mut accumulator, *coeff);
        }
        accumulator
    }

    fn mixed_linear_combination<M: Embedding<Source = F>>(
        _embedding: &M,
        vectors: &[&Self],
        coeffs: &[M::Target],
    ) -> Self::TargetBuffer<M::Target> {
        assert_bn254::<F>();
        assert_eq!(vectors.len(), coeffs.len());
        let Some((first, vectors)) = vectors.split_first() else {
            return BufferOps::from_vec(Vec::new());
        };
        let mut accumulator = MetalBuffer::<M::Target> {
            len: first.len(),
            host_cache: OnceCell::new(),
            field: Some(copy_field_buffer(&first.bn254_buffer(), first.len())),
            hash: None,
            _marker: PhantomData,
        };
        for (coeff, vector) in coeffs[1..].iter().copied().zip(vectors) {
            vector.mixed_scalar_mul_add_to(_embedding, &mut accumulator, coeff);
        }
        accumulator
    }
}

/// Core geometric accumulate over `field[0..len]` (offset 0). Shared by the
/// owned buffer and a full-range view so the optimized chunk strategies are
/// written once.
fn geometric_accumulate_full<F: Field>(
    field: &MetalFieldBuffer,
    len: usize,
    evaluators: &[UnivariateEvaluation<F>],
    scalars: &[F],
) {
    let points = evaluators
        .iter()
        .map(|e| f_to_field256(e.point))
        .collect::<Vec<_>>();
    let scalars = scalars
        .iter()
        .copied()
        .map(f_to_field256)
        .collect::<Vec<_>>();
    let points = upload_field(&points);
    let scalars = upload_field(&scalars);
    let chunk_size = geometric_accumulate_chunk_size(len);
    if std::env::var_os("WHIR_METAL_TRACE").is_some() {
        eprintln!(
            "metal geometric shape len={} points={} chunk={} chunks={}",
            len,
            evaluators.len(),
            chunk_size,
            len.div_ceil(chunk_size)
        );
    }
    if chunk_size <= 1 {
        run_in_place(
            "bn254_geometric_accumulate",
            &[&field.limbs, &points.limbs, &scalars.limbs],
            &[len as u32, evaluators.len() as u32],
            len,
        );
    } else {
        let point_steps = evaluators
            .iter()
            .map(|e| f_to_field256(e.point.pow([chunk_size as u64])))
            .collect::<Vec<_>>();
        let point_steps = upload_field(&point_steps);
        if should_use_geometric_point_blocks(len, evaluators.len(), chunk_size) {
            parallel_geometric_accumulate_point_blocks(
                field,
                &points,
                &point_steps,
                &scalars,
                len,
                evaluators.len(),
                chunk_size,
            );
        } else if should_use_geometric_point_blocks_batched(len, evaluators.len(), chunk_size) {
            parallel_geometric_accumulate_point_blocks_batched(
                field,
                &points,
                &point_steps,
                &scalars,
                len,
                evaluators.len(),
                chunk_size,
            );
        } else {
            run_in_place(
                "bn254_geometric_accumulate_chunks_strided",
                &[
                    &field.limbs,
                    &points.limbs,
                    &point_steps.limbs,
                    &scalars.limbs,
                ],
                &[len as u32, evaluators.len() as u32, chunk_size as u32],
                len.div_ceil(chunk_size),
            );
        }
    }
}

/// Validate that every evaluator targets `len` entries; returns `false` when
/// there is nothing to accumulate.
fn check_univariate_evaluators<F: Field>(
    evaluators: &[UnivariateEvaluation<F>],
    scalars: &[F],
    len: usize,
) -> bool {
    assert_bn254::<F>();
    assert_eq!(evaluators.len(), scalars.len());
    let Some(size) = evaluators.first().map(|e| e.size) else {
        return false;
    };
    assert_eq!(len, size);
    for evaluator in evaluators {
        assert_eq!(evaluator.size, size);
    }
    true
}

impl<F: Field + Clone> BufferRead<F> for MetalBuffer<F> {
    type TargetBuffer<T: Field> = MetalBuffer<T>;
    type Slice<'a>
        = MetalSlice<'a, F>
    where
        Self: 'a,
        F: 'a;

    fn read_len(&self) -> usize {
        self.len()
    }

    fn dot(&self, other: &Self) -> F {
        assert_bn254::<F>();
        assert_eq!(self.len(), other.len());
        let this = self.bn254_buffer();
        let other = other.bn254_buffer();
        field256_to_f::<F>(parallel_dot(&this, &other, self.len()))
    }

    fn sumcheck_polynomial(&self, other: &Self) -> (F, F) {
        assert_bn254::<F>();
        let len = self.len().min(other.len());
        if len == 0 {
            return (F::ZERO, F::ZERO);
        }
        if len == 1 {
            return (self.as_slice()[0] * other.as_slice()[0], F::ZERO);
        }
        let fold_half = len.next_power_of_two() >> 1;
        let this = self.bn254_buffer();
        let other = other.bn254_buffer();
        let (c0, c2) = parallel_sumcheck(&this, &other, len, fold_half);
        (field256_to_f::<F>(c0), field256_to_f::<F>(c2))
    }

    fn mixed_extend<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        _embedding: &M,
        point: &[M::Target],
    ) -> M::Target {
        assert_bn254::<F>();
        assert_bn254::<T>();
        let num_vars = point.len();
        let point = point
            .iter()
            .copied()
            .map(target_to_field256)
            .collect::<Vec<_>>();
        let point = upload_field(&point);
        let this = self.bn254_buffer();
        let value = parallel_multilinear_extend_at(&this, 0, self.len(), &point, num_vars);
        field256_to_target::<M::Target>(value)
    }

    fn mixed_dot<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        _embedding: &M,
        other: &MetalBuffer<T>,
    ) -> M::Target {
        assert_bn254::<F>();
        assert_bn254::<T>();
        let this = self.bn254_buffer();
        let other = other.bn254_buffer_target();
        let value = field256_to_f::<F>(parallel_dot(&this, &other, self.len()));
        field256_to_target::<M::Target>(f_to_field256(value))
    }

    fn mixed_univariate_evaluate<M: Embedding<Source = F>>(
        &self,
        _embedding: &M,
        point: M::Target,
    ) -> M::Target {
        assert_bn254::<F>();
        let point = target_to_field256(point);
        let point = upload_field(&[point]);
        let this = self.bn254_buffer();
        field256_to_target::<M::Target>(parallel_univariate_evaluate(&this, &point, self.len()))
    }

    fn mixed_scalar_mul_add_to<M: Embedding<Source = F>>(
        &self,
        _embedding: &M,
        accumulator: &mut MetalBuffer<M::Target>,
        weight: M::Target,
    ) {
        assert_bn254::<F>();
        let weight = upload_field(&[target_to_field256(weight)]);
        let vector = self.bn254_buffer();
        let acc = accumulator.bn254_buffer_target();
        scalar_mul_add_at(&acc, 0, &vector, 0, &weight, self.len());
        accumulator.field = Some(acc);
        accumulator.invalidate_host_cache();
    }

    fn slice(&self, range: impl RangeBounds<usize>) -> MetalSlice<'_, F> {
        assert_bn254::<F>();
        let (start, end) = crate::buffer::cpu::resolve_range(range, self.len());
        MetalSlice {
            field: self.bn254_buffer(),
            offset: start,
            len: end - start,
            _parent: PhantomData,
        }
    }

    fn copy_to_owned(&self) -> MetalBuffer<F> {
        assert_bn254::<F>();
        MetalBuffer {
            len: self.len,
            host_cache: OnceCell::new(),
            field: Some(copy_field_buffer(&self.bn254_buffer(), self.len)),
            hash: None,
            _marker: PhantomData,
        }
    }
}

impl<F: Field + Clone> BufferWrite<F> for MetalBuffer<F> {
    type SliceMut<'a>
        = MetalSliceMut<'a, F>
    where
        Self: 'a,
        F: 'a;

    fn scalar_mul(&mut self, weight: F) {
        assert_bn254::<F>();
        if self.is_empty() {
            return;
        }
        let weight = upload_field(&[f_to_field256(weight)]);
        let field = self.bn254_buffer();
        run_in_place(
            "bn254_scalar_mul",
            &[&field.limbs, &weight.limbs],
            &[self.len() as u32],
            self.len(),
        );
        self.field = Some(field);
        self.invalidate_host_cache();
    }

    fn accumulate_univariate_evaluations(
        &mut self,
        evaluators: &[UnivariateEvaluation<F>],
        scalars: &[F],
    ) {
        if !check_univariate_evaluators(evaluators, scalars, self.len()) {
            return;
        }
        let field = self.bn254_buffer();
        geometric_accumulate_full(&field, self.len(), evaluators, scalars);
        self.field = Some(field);
        self.invalidate_host_cache();
    }

    fn slice_mut(&mut self, range: impl RangeBounds<usize>) -> MetalSliceMut<'_, F> {
        assert_bn254::<F>();
        let (start, end) = crate::buffer::cpu::resolve_range(range, self.len());
        MetalSliceMut::new(self, start, end - start)
    }

    fn split_at_mut(&mut self, mid: usize) -> (MetalSliceMut<'_, F>, MetalSliceMut<'_, F>) {
        assert_bn254::<F>();
        let len = self.len();
        assert!(mid <= len, "split_at_mut mid {mid} out of bounds for {len}");
        // Materialize the parent's GPU allocation and invalidate its host
        // cache once up front; the returned views share the handle and write
        // disjoint ranges, so no per-op parent borrow is needed.
        let field = self.bn254_buffer();
        self.field = Some(field.clone());
        self.invalidate_host_cache();
        (
            MetalSliceMut::from_field(field.clone(), 0, mid),
            MetalSliceMut::from_field(field, mid, len - mid),
        )
    }
}
impl<T: Clone> MetalBuffer<T> {
    pub(crate) fn bn254_buffer(&self) -> MetalFieldBuffer {
        self.field
            .clone()
            .unwrap_or_else(|| upload_field(as_field256_slice(self.as_slice())))
    }

    fn invalidate_host_cache(&mut self) {
        let _ = self.host_cache.take();
    }

    fn download_host_cache(&self) -> Vec<T> {
        if self.field.is_some() && type_name::<T>() == type_name::<Field256>() {
            return download_field(
                &self
                    .field
                    .as_ref()
                    .expect("missing Metal field buffer")
                    .limbs,
                self.len,
            )
            .into_iter()
            .map(|value| unsafe { std::mem::transmute_copy(&value) })
            .collect();
        }
        if self.hash.is_some() && type_name::<T>() == type_name::<Digest>() {
            return download_hash_indices(
                &self.hash.as_ref().expect("missing Metal hash buffer").bytes,
                self.len,
                &(0..self.len).collect::<Vec<_>>(),
            )
            .into_iter()
            .map(|value| unsafe { std::mem::transmute_copy(&value) })
            .collect();
        }
        panic!(
            "MetalBuffer<{}> has no host cache and cannot be materialized",
            type_name::<T>()
        );
    }
}

impl<F: Field> MetalBuffer<F> {
    pub(crate) fn from_field_limb_buffer(limbs: Buffer, len: usize) -> Self {
        Self {
            len,
            host_cache: OnceCell::new(),
            field: Some(MetalFieldBuffer { limbs }),
            hash: None,
            _marker: PhantomData,
        }
    }
}

impl<T: Clone + Field> MetalBuffer<T> {
    fn bn254_buffer_target(&self) -> MetalFieldBuffer {
        self.field
            .clone()
            .unwrap_or_else(|| upload_field(as_field256_slice(self.as_slice())))
    }
}

/// Read-only, zero-copy view into a [`MetalBuffer`]'s GPU allocation.
///
/// Holds a clone of the parent's allocation handle plus an `(offset, len)`
/// window. Reductions bind the handle at the byte offset, so they read
/// `parent[offset + i]` without copying.
pub struct MetalSlice<'a, F> {
    field: MetalFieldBuffer,
    offset: usize,
    len: usize,
    _parent: PhantomData<&'a MetalBuffer<F>>,
}

/// Mutable, zero-copy view into a [`MetalBuffer`]'s GPU allocation.
///
/// Like [`MetalSlice`] but exclusive: the parent's host cache is invalidated
/// up front when the view is created, and writes go through the shared handle
/// at the byte offset, landing in the parent's own memory.
pub struct MetalSliceMut<'a, F> {
    field: MetalFieldBuffer,
    offset: usize,
    len: usize,
    _parent: PhantomData<&'a mut MetalBuffer<F>>,
}

impl<'a, F: Field + Clone> MetalSliceMut<'a, F> {
    /// Borrow `parent[offset..offset+len]`, materializing the parent's GPU
    /// allocation and invalidating its host cache once up front.
    fn new(parent: &'a mut MetalBuffer<F>, offset: usize, len: usize) -> Self {
        let field = parent.bn254_buffer();
        parent.field = Some(field.clone());
        parent.invalidate_host_cache();
        Self {
            field,
            offset,
            len,
            _parent: PhantomData,
        }
    }

    /// Build a view directly from an already-materialized handle (used by
    /// `split_at_mut`, where the parent cache was invalidated by the caller).
    fn from_field(field: MetalFieldBuffer, offset: usize, len: usize) -> Self {
        Self {
            field,
            offset,
            len,
            _parent: PhantomData,
        }
    }

    fn as_read(&self) -> MetalSlice<'_, F> {
        MetalSlice {
            field: self.field.clone(),
            offset: self.offset,
            len: self.len,
            _parent: PhantomData,
        }
    }
}

impl<F: Field + Clone> MetalSlice<'_, F> {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<F: Field + Clone> MetalSliceMut<'_, F> {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<F: Field + Clone> BufferRead<F> for MetalSlice<'_, F> {
    type TargetBuffer<T: Field> = MetalBuffer<T>;
    type Slice<'a>
        = MetalSlice<'a, F>
    where
        Self: 'a,
        F: 'a;

    fn read_len(&self) -> usize {
        self.len
    }

    fn dot(&self, other: &Self) -> F {
        assert_bn254::<F>();
        assert_eq!(self.len, other.len);
        field256_to_f::<F>(parallel_dot_at(
            &self.field,
            self.offset,
            &other.field,
            other.offset,
            self.len,
        ))
    }

    fn sumcheck_polynomial(&self, other: &Self) -> (F, F) {
        assert_bn254::<F>();
        let len = self.len.min(other.len);
        if len == 0 {
            return (F::ZERO, F::ZERO);
        }
        let fold_half = len.next_power_of_two() >> 1;
        let (c0, c2) = parallel_sumcheck_at(
            &self.field,
            self.offset,
            &other.field,
            other.offset,
            len,
            fold_half,
        );
        (field256_to_f::<F>(c0), field256_to_f::<F>(c2))
    }

    fn mixed_extend<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        _embedding: &M,
        point: &[M::Target],
    ) -> M::Target {
        assert_bn254::<F>();
        assert_bn254::<T>();
        let num_vars = point.len();
        let point = point
            .iter()
            .copied()
            .map(target_to_field256)
            .collect::<Vec<_>>();
        let point = upload_field(&point);
        let value =
            parallel_multilinear_extend_at(&self.field, self.offset, self.len, &point, num_vars);
        field256_to_target::<M::Target>(value)
    }

    fn mixed_dot<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        _embedding: &M,
        other: &MetalBuffer<T>,
    ) -> M::Target {
        assert_bn254::<F>();
        assert_bn254::<T>();
        let other = other.bn254_buffer_target();
        let value = field256_to_f::<F>(parallel_dot_at(
            &self.field,
            self.offset,
            &other,
            0,
            self.len,
        ));
        field256_to_target::<M::Target>(f_to_field256(value))
    }

    fn mixed_univariate_evaluate<M: Embedding<Source = F>>(
        &self,
        _embedding: &M,
        point: M::Target,
    ) -> M::Target {
        assert_bn254::<F>();
        let point = upload_field(&[target_to_field256(point)]);
        field256_to_target::<M::Target>(parallel_univariate_evaluate_at(
            &self.field,
            self.offset,
            &point,
            self.len,
        ))
    }

    fn mixed_scalar_mul_add_to<M: Embedding<Source = F>>(
        &self,
        _embedding: &M,
        accumulator: &mut MetalBuffer<M::Target>,
        weight: M::Target,
    ) {
        assert_bn254::<F>();
        let weight = upload_field(&[target_to_field256(weight)]);
        let acc = accumulator.bn254_buffer_target();
        scalar_mul_add_at(&acc, 0, &self.field, self.offset, &weight, self.len);
        accumulator.field = Some(acc);
        accumulator.invalidate_host_cache();
    }

    fn slice(&self, range: impl RangeBounds<usize>) -> MetalSlice<'_, F> {
        let (start, end) = crate::buffer::cpu::resolve_range(range, self.len);
        MetalSlice {
            field: self.field.clone(),
            offset: self.offset + start,
            len: end - start,
            _parent: PhantomData,
        }
    }

    fn copy_to_owned(&self) -> MetalBuffer<F> {
        assert_bn254::<F>();
        MetalBuffer {
            len: self.len,
            host_cache: OnceCell::new(),
            field: Some(copy_field_buffer_at(&self.field, self.offset, self.len)),
            hash: None,
            _marker: PhantomData,
        }
    }
}

impl<F: Field + Clone> BufferRead<F> for MetalSliceMut<'_, F> {
    type TargetBuffer<T: Field> = MetalBuffer<T>;
    type Slice<'a>
        = MetalSlice<'a, F>
    where
        Self: 'a,
        F: 'a;

    fn read_len(&self) -> usize {
        self.len
    }

    fn dot(&self, other: &Self) -> F {
        self.as_read().dot(&other.as_read())
    }

    fn sumcheck_polynomial(&self, other: &Self) -> (F, F) {
        self.as_read().sumcheck_polynomial(&other.as_read())
    }

    fn mixed_extend<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        embedding: &M,
        point: &[M::Target],
    ) -> M::Target {
        self.as_read().mixed_extend(embedding, point)
    }

    fn mixed_dot<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        embedding: &M,
        other: &MetalBuffer<T>,
    ) -> M::Target {
        self.as_read().mixed_dot(embedding, other)
    }

    fn mixed_univariate_evaluate<M: Embedding<Source = F>>(
        &self,
        embedding: &M,
        point: M::Target,
    ) -> M::Target {
        self.as_read().mixed_univariate_evaluate(embedding, point)
    }

    fn mixed_scalar_mul_add_to<M: Embedding<Source = F>>(
        &self,
        embedding: &M,
        accumulator: &mut MetalBuffer<M::Target>,
        weight: M::Target,
    ) {
        self.as_read()
            .mixed_scalar_mul_add_to(embedding, accumulator, weight)
    }

    fn slice(&self, range: impl RangeBounds<usize>) -> MetalSlice<'_, F> {
        let (start, end) = crate::buffer::cpu::resolve_range(range, self.len);
        MetalSlice {
            field: self.field.clone(),
            offset: self.offset + start,
            len: end - start,
            _parent: PhantomData,
        }
    }

    fn copy_to_owned(&self) -> MetalBuffer<F> {
        self.as_read().copy_to_owned()
    }
}

impl<F: Field + Clone> BufferWrite<F> for MetalSliceMut<'_, F> {
    type SliceMut<'a>
        = MetalSliceMut<'a, F>
    where
        Self: 'a,
        F: 'a;

    fn scalar_mul(&mut self, weight: F) {
        assert_bn254::<F>();
        if self.len == 0 {
            return;
        }
        let weight = upload_field(&[f_to_field256(weight)]);
        scalar_mul_at_offset(&self.field, self.offset, self.len, &weight);
    }

    fn accumulate_univariate_evaluations(
        &mut self,
        evaluators: &[UnivariateEvaluation<F>],
        scalars: &[F],
    ) {
        if !check_univariate_evaluators(evaluators, scalars, self.len) {
            return;
        }
        if self.offset == 0 {
            // Offset 0: reuse the optimized full-buffer accumulate path.
            geometric_accumulate_full(&self.field, self.len, evaluators, scalars);
        } else {
            // Arbitrary offset: bind the buffer at a byte offset so the
            // kernel's gid-based indexing addresses field[offset + gid].
            let points = evaluators
                .iter()
                .map(|e| f_to_field256(e.point))
                .collect::<Vec<_>>();
            let scalars = scalars
                .iter()
                .copied()
                .map(f_to_field256)
                .collect::<Vec<_>>();
            let points = upload_field(&points);
            let scalars = upload_field(&scalars);
            geometric_accumulate_at_offset(
                &self.field,
                self.offset,
                self.len,
                &points,
                &scalars,
                evaluators.len(),
            );
        }
    }

    fn slice_mut(&mut self, range: impl RangeBounds<usize>) -> MetalSliceMut<'_, F> {
        let (start, end) = crate::buffer::cpu::resolve_range(range, self.len);
        MetalSliceMut::from_field(self.field.clone(), self.offset + start, end - start)
    }

    fn split_at_mut(&mut self, mid: usize) -> (MetalSliceMut<'_, F>, MetalSliceMut<'_, F>) {
        assert!(
            mid <= self.len,
            "split_at_mut mid {mid} out of bounds for {}",
            self.len
        );
        (
            MetalSliceMut::from_field(self.field.clone(), self.offset, mid),
            MetalSliceMut::from_field(self.field.clone(), self.offset + mid, self.len - mid),
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::ntt::{Messages, NttEngine, ReedSolomon};
    use crate::algebra::univariate_evaluate;
    use crate::buffer::metal::rs::rs_transpose_permute;
    use crate::buffer::{Buffer, BufferOps, CpuBuffer, MetalRs};

    fn values(len: usize, offset: u64) -> Vec<Field256> {
        (0..len)
            .map(|i| Field256::from(i as u64 + offset))
            .collect()
    }

    #[test]
    fn metal_bn254_dot_matches_cpu() {
        let a = values(33, 1);
        let b = values(33, 9);
        let cpu_a = CpuBuffer::from_slice(&a);
        let cpu_b = CpuBuffer::from_slice(&b);
        let gpu_a = MetalBuffer::from_slice(&a);
        let gpu_b = MetalBuffer::from_slice(&b);
        assert_eq!(gpu_a.dot(&gpu_b), cpu_a.dot(&cpu_b));
    }

    #[test]
    fn metal_bn254_fold_matches_cpu() {
        let mut cpu = CpuBuffer::from_vec(values(31, 2));
        let mut gpu = MetalBuffer::from_vec(values(31, 2));
        let weight = Field256::from(42);
        cpu.fold(weight);
        gpu.fold(weight);
        assert_eq!(gpu.as_slice(), cpu.as_slice());
    }

    #[test]
    fn metal_bn254_sumcheck_matches_cpu() {
        for len in [1, 2, 27, 64, 65] {
            let a = values(len, 3);
            let b = values(len, 11);
            let cpu_a = CpuBuffer::from_slice(&a);
            let cpu_b = CpuBuffer::from_slice(&b);
            let gpu_a = MetalBuffer::from_slice(&a);
            let gpu_b = MetalBuffer::from_slice(&b);
            assert_eq!(
                gpu_a.sumcheck_polynomial(&gpu_b),
                cpu_a.sumcheck_polynomial(&cpu_b)
            );
        }
    }

    #[test]
    fn metal_bn254_fold_pair_sumcheck_matches_cpu() {
        for len in [2, 3, 27, 64, 65] {
            let mut cpu_a = CpuBuffer::from_vec(values(len, 3));
            let mut cpu_b = CpuBuffer::from_vec(values(len, 11));
            let mut gpu_a = MetalBuffer::from_vec(values(len, 3));
            let mut gpu_b = MetalBuffer::from_vec(values(len, 11));
            let weight = Field256::from(42);
            let cpu_result = cpu_a.fold_pair_sumcheck_polynomial(&mut cpu_b, weight);
            let gpu_result = gpu_a.fold_pair_sumcheck_polynomial(&mut gpu_b, weight);
            assert_eq!(gpu_result, cpu_result);
            assert_eq!(gpu_a.as_slice(), cpu_a.as_slice());
            assert_eq!(gpu_b.as_slice(), cpu_b.as_slice());
        }
    }

    #[test]
    fn metal_bn254_scalar_mul_add_matches_cpu() {
        let mut cpu = CpuBuffer::from_vec(values(19, 1));
        let mut gpu = MetalBuffer::from_vec(values(19, 1));
        let vector = values(19, 5);
        let cpu_vector = CpuBuffer::from_slice(&vector);
        let gpu_vector = MetalBuffer::from_slice(&vector);
        let weight = Field256::from(7);
        cpu_vector.mixed_scalar_mul_add_to(&Identity::new(), &mut cpu, weight);
        gpu_vector.mixed_scalar_mul_add_to(&Identity::new(), &mut gpu, weight);
        assert_eq!(gpu.as_slice(), cpu.as_slice());
    }

    #[test]
    fn metal_bn254_interleaved_rs_encode_matches_cpu() {
        // The GPU encoder and the CPU reference derive the same coset layout from the field.
        let engine = NttEngine::<Field256>::new_from_fftfield();
        let gpu_rs = MetalRs::<Field256>::new_from_fftfield();

        let a = values(8, 1);
        let gpu_a = MetalBuffer::from_slice(&a);
        let gpu_masks = MetalBuffer::from_slice(&[]);

        let messages = a.chunks_exact(4).collect::<Vec<_>>();
        let generator = engine.root(8);
        let mut cpu = Vec::with_capacity(16);
        for index in 0..8 {
            #[cfg(not(feature = "rs_in_order"))]
            let index = rs_transpose_permute(index, 2, 4);
            let point = generator.pow([index as u64]);
            for message in &messages {
                cpu.push(univariate_evaluate(message, point));
            }
        }
        let gpu_vectors = [&gpu_a];
        let gpu_messages = Messages::new(&gpu_vectors, 4, 2);
        let gpu = gpu_rs.interleaved_encode(gpu_messages, &gpu_masks, 8);
        assert_eq!(gpu.as_slice(), cpu.as_slice());
    }

    #[test]
    fn metal_bn254_mixed_extend_matches_cpu() {
        let values = values(8, 3);
        let point = vec![Field256::from(2), Field256::from(5), Field256::from(9)];
        let cpu = CpuBuffer::from_slice(&values);
        let gpu = MetalBuffer::from_slice(&values);
        assert_eq!(
            gpu.mixed_extend(&Identity::new(), &point),
            cpu.mixed_extend(&Identity::new(), &point)
        );
    }
}
