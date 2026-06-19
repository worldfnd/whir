use std::{any::Any, mem};

use super::read_write::{impl_cpu_read, impl_cpu_write};
use ark_ff::Field;
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, Rng, RngCore};

use crate::{
    algebra::{
        embedding::Embedding,
        linear_form::{Covector, LinearForm},
    },
    buffer::{Buffer, BufferOps, BufferRead, BufferWrite},
    engines::EngineId,
    hash::{self, Hash},
    protocols::{
        matrix_commit::{hash_rows, Encodable},
        merkle_tree,
    },
};

#[derive(
    Clone,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Debug,
    Default,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct CpuBuffer<T> {
    data: Vec<T>,
}

/// Read-only view into a [`CpuBuffer`], backed by a borrowed sub-slice.
pub struct CpuSlice<'a, F> {
    data: &'a [F],
}

/// Mutable view into a [`CpuBuffer`], backed by a borrowed sub-slice.
pub struct CpuSliceMut<'a, F> {
    data: &'a mut [F],
}

impl<T: Copy> BufferOps<T> for CpuBuffer<T> {
    type Nodes = CpuBuffer<Hash>;

    fn at_index(&self, index: usize) -> Option<T> {
        if index >= self.len() {
            None
        } else {
            Some(self.data[index])
        }
    }

    fn as_slice(&self) -> &[T] {
        &self.data
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn read_rows(&self, num_cols: usize, indices: &[usize]) -> Vec<T> {
        let mut result = Vec::with_capacity(indices.len() * num_cols);
        for i in indices {
            result.extend_from_slice(&self.data[i * num_cols..(i + 1) * num_cols]);
        }
        result
    }

    fn gather_at_indices(&self, indices: &[usize]) -> Vec<T> {
        indices
            .iter()
            .map(|&i| self.data[i])
            .collect()
    }

    fn from_vec(source: Vec<T>) -> Self {
        Self { data: source }
    }

    fn from_slice(source: &[T]) -> Self {
        Self {
            data: Vec::from(source),
        }
    }

    fn merklize(
        &self,
        num_cols: usize,
        leaf_hash: EngineId,
        merkle: &merkle_tree::Config,
    ) -> (Self::Nodes, Hash)
    where
        T: Encodable + Send + Sync,
    {
        assert_eq!(self.len(), num_cols * merkle.num_leaves);
        let engine = hash::ENGINES
            .retrieve(leaf_hash)
            .expect("Failed to retrieve hash engine");
        #[cfg(feature = "tracing")]
        tracing::Span::current().record("engine", engine.name().as_ref());

        let mut leaves = vec![Hash::default(); merkle.num_leaves];
        hash_rows(&*engine, &self.data, &mut leaves);
        let nodes = merkle.build_nodes(leaves);
        let root = nodes[merkle.num_nodes() - 1];
        (CpuBuffer { data: nodes }, root)
    }
}

impl<F: Field> Buffer<F> for CpuBuffer<F> {
    fn zeros(length: usize) -> Self {
        Self {
            data: vec![F::ZERO; length],
        }
    }

    fn geometric_sequence(base: F, length: usize) -> Self {
        Self {
            data: crate::algebra::geometric_sequence(base, length),
        }
    }

    fn random<R>(rng: &mut R, length: usize) -> Self
    where
        R: RngCore + CryptoRng,
        Standard: Distribution<F>,
    {
        Self {
            data: (0..length).map(|_| rng.gen()).collect(),
        }
    }

    fn linear_forms_rlc(
        size: usize,
        linear_forms: &mut [Box<dyn LinearForm<F>>],
        rlc_coeffs: &[F],
    ) -> Self {
        assert_eq!(linear_forms.len(), rlc_coeffs.len());
        let mut covector = vec![F::ZERO; size];
        if let Some((first, linear_forms)) = linear_forms.split_first_mut() {
            debug_assert_eq!(rlc_coeffs[0], F::ONE);
            if let Some(covector_form) =
                (first.as_mut() as &mut dyn Any).downcast_mut::<Covector<F>>()
            {
                mem::swap(&mut covector, &mut covector_form.vector);
            } else {
                first.accumulate(&mut covector, F::ONE);
            }
            for (rlc_coeff, linear_form) in rlc_coeffs[1..].iter().zip(linear_forms) {
                linear_form.accumulate(&mut covector, *rlc_coeff);
            }
        }
        Self { data: covector }
    }

    fn zero_pad(&mut self) {
        if !self.is_empty() {
            self.data.resize(self.len().next_power_of_two(), F::ZERO);
        }
    }

    fn fold(&mut self, weight: F) {
        crate::algebra::sumcheck::fold(&mut self.data, weight);
    }

    fn fold_pair(&mut self, other: &mut Self, weight: F) {
        self.fold(weight);
        other.fold(weight);
    }

    fn fold_pair_sumcheck_polynomial(&mut self, other: &mut Self, weight: F) -> (F, F) {
        self.fold_pair(other, weight);
        self.sumcheck_polynomial(other)
    }

    fn mixed_linear_combination<M: Embedding<Source = F>>(
        embedding: &M,
        vectors: &[&Self],
        coeffs: &[M::Target],
    ) -> Self::TargetBuffer<M::Target> {
        assert_eq!(vectors.len(), coeffs.len());
        let Some((first, vectors)) = vectors.split_first() else {
            return CpuBuffer { data: Vec::new() };
        };
        debug_assert_eq!(coeffs[0], M::Target::ONE);
        let mut accumulator = crate::algebra::lift(embedding, first.as_slice());
        for (coeff, vector) in coeffs[1..].iter().zip(vectors) {
            crate::algebra::mixed_scalar_mul_add(
                embedding,
                &mut accumulator,
                *coeff,
                vector.as_slice(),
            );
        }
        CpuBuffer { data: accumulator }
    }
}

impl_cpu_read!(CpuSlice<'_, F>);
impl_cpu_read!(CpuSliceMut<'_, F>);
impl_cpu_read!(CpuBuffer<F>);
impl_cpu_write!(CpuSliceMut<'_, F>);
impl_cpu_write!(CpuBuffer<F>);

/// Resolve any range expression (`a..b`, `..b`, `a..`, `..`) against `len`,
/// returning `(start, end)` and bounds-checking like slice indexing.
pub(crate) fn resolve_range(
    range: impl std::ops::RangeBounds<usize>,
    len: usize,
) -> (usize, usize) {
    use std::ops::Bound::{Excluded, Included, Unbounded};
    let start = match range.start_bound() {
        Included(&s) => s,
        Excluded(&s) => s + 1,
        Unbounded => 0,
    };
    let end = match range.end_bound() {
        Included(&e) => e + 1,
        Excluded(&e) => e,
        Unbounded => len,
    };
    assert!(
        start <= end && end <= len,
        "slice range {start}..{end} out of bounds for length {len}"
    );
    (start, end)
}

#[cfg(test)]
mod tests {
    use ark_ff::AdditiveGroup;

    use super::*;
    use crate::algebra::{
        fields::Field64, geometric_accumulate, geometric_sequence as geometric_sequence_vec,
        linear_form::UnivariateEvaluation,
    };

    type F = Field64;

    #[test]
    fn geometric_sequence_matches_free_function() {
        let base = F::from(7u64);
        for length in 0..6 {
            let buffer = CpuBuffer::<F>::geometric_sequence(base, length);
            assert_eq!(
                BufferOps::as_slice(&buffer),
                geometric_sequence_vec(base, length).as_slice(),
                "geometric_sequence mismatch at length {length}"
            );
        }
    }

    #[test]
    fn scale_multiplies_in_place() {
        let values = vec![F::from(1u64), F::from(2u64), F::from(3u64), F::from(4u64)];
        let weight = F::from(5u64);
        let mut buffer = CpuBuffer::from_vec(values.clone());
        buffer.scalar_mul(weight);
        let expected: Vec<F> = values.iter().map(|&v| v * weight).collect();
        assert_eq!(BufferOps::as_slice(&buffer), expected.as_slice());
    }

    #[test]
    fn slice_accumulate_matches_restricted_geometric_accumulate() {
        let len = 8usize;
        let points = vec![F::from(3u64), F::from(5u64)];
        let scalars = vec![F::from(2u64), F::from(9u64)];

        // Try a prefix slice (offset 0) and an interior slice (offset > 0).
        for (offset, window_len) in [(0usize, 5usize), (2usize, 4usize)] {
            let mut buffer = CpuBuffer::from_vec(vec![F::ZERO; len]);
            let evaluators: Vec<_> = points
                .iter()
                .map(|&point| UnivariateEvaluation::new(point, window_len))
                .collect();
            buffer
                .slice_mut(offset..offset + window_len)
                .accumulate_univariate_evaluations(&evaluators, &scalars);

            // Reference: accumulate Σ_j scalars[j]·points[j]^i into the same
            // sub-slice of a plain vector.
            let mut reference = vec![F::ZERO; len];
            geometric_accumulate(
                &mut reference[offset..offset + window_len],
                scalars.clone(),
                &points,
            );

            assert_eq!(
                BufferOps::as_slice(&buffer),
                reference.as_slice(),
                "slice accumulate mismatch at offset {offset}, len {window_len}"
            );
        }
    }

    #[test]
    fn slice_scale_multiplies_only_the_range() {
        let values = vec![F::from(1u64), F::from(2u64), F::from(3u64), F::from(4u64)];
        let weight = F::from(5u64);
        let mut buffer = CpuBuffer::from_vec(values.clone());
        buffer.slice_mut(1..3).scalar_mul(weight);
        let expected = vec![values[0], values[1] * weight, values[2] * weight, values[3]];
        assert_eq!(BufferOps::as_slice(&buffer), expected.as_slice());
    }
}
