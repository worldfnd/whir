use std::{
    collections::{BTreeMap, HashMap},
    iter::from_fn,
    sync::Arc,
};

use ark_ff::{FftField, Field};
use sha3::{Digest, Sha3_256};

use crate::{
    algebra::{
        eval_eq, geometric_accumulate,
        linear_form::LinearForm,
        ntt::{generator, ntt},
        MultilinearPoint,
    },
    hash::Hash,
};

/// A sum of tensor-product coefficient tables at arbitrary offsets.
///
/// Prefix insertion moves tables off binary boundaries. Evaluating the MLE uses
/// binary addition with a carry, without constructing the coefficient vector.
#[must_use]
#[derive(Clone, Debug)]
pub struct TensorLinearForm<F: FftField> {
    size: usize,
    terms: Vec<Term<F>>,
}

#[derive(Clone, Debug)]
struct Term<F: Field> {
    offset: usize,
    scalar: F,
    factors: Arc<[[F; 2]]>,
    first_factor: usize,
    geometric: Option<F>,
    domain: Option<DomainPoint>,
}

#[derive(Clone, Copy, Debug)]
struct DomainPoint {
    size: usize,
    index: usize,
}

#[derive(Hash, PartialEq, Eq)]
enum Factors<'a, F: Field> {
    Geometric { bits: usize, point: F },
    Tensor(&'a [[F; 2]]),
}

impl<F: Field> Term<F> {
    fn factors(&self) -> &[[F; 2]] {
        &self.factors[self.first_factor..]
    }

    fn len(&self) -> usize {
        1 << self.factors().len()
    }

    /// Restrict to an interval, decomposing local indices into binary blocks.
    fn restrict(&self, start: usize, end: usize) -> impl Iterator<Item = Self> + '_ {
        let mut low = start.max(self.offset) - self.offset;
        let high = end.min(self.offset + self.len()) - self.offset;
        let factors = self.factors();
        from_fn(move || {
            if low >= high {
                return None;
            }
            let remaining_bits = (high - low).ilog2() as usize;
            let alignment_bits = if low == 0 {
                factors.len()
            } else {
                low.trailing_zeros() as usize
            };
            let bits = remaining_bits.min(alignment_bits);
            let fixed = factors.len() - bits;
            let scalar = factors[..fixed]
                .iter()
                .enumerate()
                .fold(self.scalar, |acc, (i, factor)| {
                    acc * factor[(low >> (factors.len() - 1 - i)) & 1]
                });
            let piece = Self {
                offset: self.offset + low,
                scalar,
                factors: Arc::clone(&self.factors),
                first_factor: self.first_factor + fixed,
                geometric: self.geometric,
                domain: self.domain,
            };
            low += 1 << bits;
            Some(piece)
        })
    }

    fn mle(&self, point: &[F], prefix_weights: Option<&[Vec<F>]>) -> F {
        let factors = self.factors();
        let mut carry = [self.scalar, F::ZERO];
        let bits = if prefix_weights.is_some() {
            factors.len()
        } else {
            point.len()
        };
        for (bit, &challenge) in point.iter().rev().take(bits).enumerate() {
            let mut next = [F::ZERO; 2];
            let (choices, local_factors) = if bit < factors.len() {
                (2, factors[factors.len() - 1 - bit])
            } else {
                (1, [F::ONE, F::ZERO])
            };
            for (old_carry, &weight) in carry.iter().enumerate() {
                for (local_bit, &factor) in local_factors.iter().enumerate().take(choices) {
                    let total = ((self.offset >> bit) & 1) + local_bit + old_carry;
                    let equality = if total & 1 == 0 {
                        F::ONE - challenge
                    } else {
                        challenge
                    };
                    next[total >> 1] += weight * factor * equality;
                }
            }
            carry = next;
        }
        prefix_weights.map_or(carry[0], |weights| {
            let high = &weights[point.len() - bits];
            let index = self.offset >> bits;
            carry[0] * high[index] + carry[1] * high.get(index + 1).copied().unwrap_or(F::ZERO)
        })
    }

    fn accumulate(&self, accumulator: &mut [F], scalar: F) {
        Self::accumulate_tensor(
            &mut accumulator[self.offset..self.offset + self.len()],
            self.factors(),
            self.scalar * scalar,
        );
    }

    fn accumulate_tensor(accumulator: &mut [F], factors: &[[F; 2]], scalar: F) {
        if let Some((factor, tail)) = factors.split_first() {
            let (low, high) = accumulator.split_at_mut(accumulator.len() / 2);
            Self::accumulate_tensor(low, tail, scalar * factor[0]);
            Self::accumulate_tensor(high, tail, scalar * factor[1]);
        } else {
            accumulator[0] += scalar;
        }
    }
}

impl<F: FftField> TensorLinearForm<F> {
    pub fn zero(size: usize) -> Self {
        assert!(size.is_power_of_two());
        Self {
            size,
            terms: vec![],
        }
    }

    pub fn multilinear(point: &[F]) -> Self {
        Self {
            size: 1 << point.len(),
            terms: vec![Term {
                offset: 0,
                scalar: F::ONE,
                factors: point.iter().map(|r| [F::ONE - r, *r]).collect(),
                first_factor: 0,
                geometric: None,
                domain: None,
            }],
        }
    }

    pub fn add_univariate(&mut self, offset: usize, len: usize, point: F, scalar: F) {
        self.add_geometric(offset, len, point, scalar, None);
    }

    fn add_geometric(
        &mut self,
        offset: usize,
        len: usize,
        point: F,
        scalar: F,
        domain: Option<DomainPoint>,
    ) {
        assert!(offset + len <= self.size);
        if len == 0 || scalar.is_zero() {
            return;
        }
        let padded = len.next_power_of_two();
        let mut power = point;
        let mut factors = Vec::new();
        for _ in 0..padded.ilog2() {
            factors.push([F::ONE, power]);
            power.square_in_place();
        }
        factors.reverse();
        let term = Term {
            offset,
            scalar,
            factors: factors.into(),
            first_factor: 0,
            geometric: Some(point),
            domain,
        };
        for piece in term.restrict(offset, offset + len) {
            self.push(piece);
        }
    }

    pub(super) fn add_in_domain(
        &mut self,
        offset: usize,
        len: usize,
        domain_size: usize,
        index: usize,
        scalar: F,
    ) {
        assert!(index < domain_size && len <= domain_size);
        let point = generator::<F>(domain_size).expect("validated RS domain");
        self.add_geometric(
            offset,
            len,
            point.pow([index as u64]),
            scalar,
            Some(DomainPoint {
                size: domain_size,
                index,
            }),
        );
    }

    pub fn add_coefficients(&mut self, offset: usize, values: &[F]) {
        assert!(offset + values.len() <= self.size);
        let factors: Arc<[[F; 2]]> = Arc::from([]);
        for (i, &scalar) in values.iter().enumerate() {
            self.push(Term {
                offset: offset + i,
                scalar,
                factors: Arc::clone(&factors),
                first_factor: 0,
                geometric: None,
                domain: None,
            });
        }
    }

    pub fn scale(&mut self, scalar: F) {
        for term in &mut self.terms {
            term.scalar *= scalar;
        }
    }

    pub fn add(&mut self, other: Self) {
        assert_eq!(self.size, other.size);
        for term in other.terms {
            self.push(term);
        }
        self.normalize();
    }

    pub(crate) fn digest(&self) -> Hash {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"whir-prefix/public-form/v1");
        bytes.extend_from_slice(&(self.size as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.terms.len() as u64).to_le_bytes());
        for term in &self.terms {
            bytes.extend_from_slice(&(term.offset as u64).to_le_bytes());
            bytes.extend_from_slice(&(term.factors().len() as u64).to_le_bytes());
            term.scalar
                .serialize_compressed(&mut bytes)
                .expect("field serialization");
            for factor in term.factors() {
                for value in factor {
                    value
                        .serialize_compressed(&mut bytes)
                        .expect("field serialization");
                }
            }
        }
        Hash(Sha3_256::digest(bytes).into())
    }

    pub fn resize(&mut self, size: usize) {
        assert!(size.is_power_of_two() && size >= self.size);
        self.size = size;
    }

    pub fn fold(&self, challenges: &[F]) -> Self {
        let len = self.size >> challenges.len();
        assert!(len > 0);
        let weights = MultilinearPoint(challenges.to_vec()).eq_weights();
        let mut result = Self::zero(len);
        for term in &self.terms {
            let first = term.offset / len;
            let last = (term.offset + term.len() - 1) / len;
            for (block, &weight) in weights.iter().enumerate().take(last + 1).skip(first) {
                for mut piece in term.restrict(block * len, (block + 1) * len) {
                    piece.offset -= block * len;
                    piece.scalar *= weight;
                    result.push(piece);
                }
            }
        }
        result.normalize();
        result
    }

    /// Map the first `payload_size` coefficients into ceiling-sized blocks with zero prefix weights.
    /// Coefficients beyond the payload are ignored; each output block has length `encoded_block`.
    pub fn insert_prefixes(
        &self,
        payload_size: usize,
        columns: usize,
        prefix: usize,
        encoded_block: usize,
    ) -> Self {
        assert!(payload_size > 0 && payload_size <= self.size && columns.is_power_of_two());
        let block = payload_size.div_ceil(columns);
        assert!(encoded_block.is_power_of_two() && prefix + block <= encoded_block);
        let mut result = Self::zero(columns * encoded_block);
        for term in &self.terms {
            if term.offset >= payload_size {
                continue;
            }
            let first = term.offset / block;
            let last = ((term.offset + term.len()).min(payload_size) - 1) / block;
            for index in first..=last {
                for mut piece in
                    term.restrict(index * block, ((index + 1) * block).min(payload_size))
                {
                    piece.offset = index * encoded_block + prefix + piece.offset - index * block;
                    result.push(piece);
                }
            }
        }
        result.normalize();
        result
    }

    pub fn coefficients(&self) -> Vec<F> {
        let mut result = vec![F::ZERO; self.size];
        self.accumulate(&mut result, F::ONE);
        result
    }

    fn push(&mut self, term: Term<F>) {
        if term.scalar.is_zero() {
            return;
        }
        if let Some(last) = self.terms.last_mut() {
            if last.offset == term.offset && last.factors() == term.factors() {
                last.scalar += term.scalar;
                last.geometric = last.geometric.or(term.geometric);
                last.domain = last.domain.or(term.domain);
                return;
            }
        }
        self.terms.push(term);
    }

    fn normalize(&mut self) {
        let mut indices = HashMap::with_capacity(self.terms.len());
        let mut combined = vec![(F::ZERO, None, None); self.terms.len()];
        for (index, term) in self.terms.iter().enumerate() {
            let factors = term.geometric.map_or_else(
                || Factors::Tensor(term.factors()),
                |point| Factors::Geometric {
                    bits: term.factors().len(),
                    point: if term.factors().is_empty() {
                        F::ZERO
                    } else {
                        point
                    },
                },
            );
            let first = *indices.entry((term.offset, factors)).or_insert(index);
            combined[first].0 += term.scalar;
            combined[first].1 = combined[first].1.or(term.geometric);
            combined[first].2 = combined[first].2.or(term.domain);
        }
        drop(indices);
        for (term, (scalar, geometric, domain)) in self.terms.iter_mut().zip(combined) {
            term.scalar = scalar;
            term.geometric = geometric;
            term.domain = domain;
        }
        self.terms.retain(|term| !term.scalar.is_zero());
    }
}

impl<F: FftField> LinearForm<F> for TensorLinearForm<F> {
    fn size(&self) -> usize {
        self.size
    }

    fn mle_evaluate(&self, point: &[F]) -> F {
        assert_eq!(1 << point.len(), self.size);
        let fixed_bits: usize = self
            .terms
            .iter()
            .map(|term| point.len() - term.factors().len())
            .sum();
        if fixed_bits <= self.size {
            return self.terms.iter().map(|term| term.mle(point, None)).sum();
        }
        // Many short terms share the same fixed high bits. Compute their
        // equality weights once instead of repeating those products per term.
        let mut weights = vec![vec![]; point.len() + 1];
        weights[point.len()] = vec![F::ZERO; self.size];
        eval_eq(&mut weights[point.len()], point, F::ONE);
        for variables in (0..point.len()).rev() {
            weights[variables] = weights[variables + 1]
                .chunks_exact(2)
                .map(|pair| pair[0] + pair[1])
                .collect();
        }
        self.terms
            .iter()
            .map(|term| term.mle(point, Some(&weights)))
            .sum()
    }

    fn accumulate(&self, accumulator: &mut [F], scalar: F) {
        assert_eq!(accumulator.len(), self.size);
        let mut geometric = BTreeMap::<(usize, usize, Option<usize>), Vec<(F, F, usize)>>::new();
        for term in &self.terms {
            if let Some(point) = term.geometric {
                geometric
                    .entry((
                        term.offset,
                        term.len(),
                        term.domain.map(|domain| domain.size),
                    ))
                    .or_default()
                    .push((
                        term.scalar * scalar,
                        point,
                        term.domain.map_or(0, |domain| domain.index),
                    ));
            } else {
                term.accumulate(accumulator, scalar);
            }
        }
        for ((offset, len, domain_size), terms) in geometric {
            let output = &mut accumulator[offset..offset + len];
            if let Some(size) = domain_size.filter(|&size| {
                len.saturating_mul(terms.len()) > size.saturating_mul(2 * size.ilog2() as usize)
            }) {
                // For z_j = omega^j, FFT(weights)[i] = sum_j weights[j] * z_j^i.
                let mut weights = vec![F::ZERO; size];
                for (scalar, _, index) in terms {
                    weights[index] += scalar;
                }
                ntt(&mut weights);
                for (entry, weight) in output.iter_mut().zip(weights) {
                    *entry += weight;
                }
            } else {
                let (scalars, points): (Vec<_>, Vec<_>) = terms
                    .into_iter()
                    .map(|(scalar, point, _)| (scalar, point))
                    .unzip();
                geometric_accumulate(output, scalars, &points);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::{AdditiveGroup, FftField, Field};

    use super::TensorLinearForm;
    use crate::algebra::{
        fields::{Field128, Field256, Field64},
        linear_form::LinearForm,
        multilinear_extend,
        ntt::generator,
        sumcheck::fold,
    };

    #[test]
    fn offset_forms_match_dense_mles_and_folds() {
        for offset in 0..16 {
            let mut form = TensorLinearForm::<Field64>::zero(32);
            form.add_univariate(offset, 13, Field64::from(7), Field64::from(3));
            form.add_coefficients(29, &[Field64::ONE, Field64::from(9)]);
            for challenge in [Field64::ZERO, Field64::ONE, Field64::from(17)] {
                let point = vec![
                    challenge,
                    Field64::from(2),
                    Field64::from(5),
                    Field64::from(11),
                    Field64::from(19),
                ];
                let dense = form.coefficients();
                assert_eq!(
                    form.mle_evaluate(&point),
                    multilinear_extend(&dense, &point)
                );
                let reduced = form.fold(&point[..2]);
                let mut expected = dense;
                for &c in &point[..2] {
                    fold(&mut expected, c);
                }
                assert_eq!(reduced.coefficients(), expected);
            }
        }
    }

    #[test]
    fn prefix_insertion_preserves_every_coefficient() {
        let form = TensorLinearForm::<Field64>::multilinear(&[
            Field64::from(3),
            Field64::from(5),
            Field64::from(7),
            Field64::from(11),
            Field64::from(13),
        ]);
        let dense = form.coefficients();
        for (payload_size, prefix) in [1usize, 17, 25, 29, 32]
            .into_iter()
            .flat_map(|n| [1, 3, 7].map(|t| (n, t)))
        {
            let block_size = payload_size.div_ceil(4);
            let expanded = form.insert_prefixes(payload_size, 4, prefix, 16);
            let mut expected = vec![Field64::ZERO; 64];
            for (i, block) in dense[..payload_size].chunks(block_size).enumerate() {
                expected[i * 16 + prefix..i * 16 + prefix + block.len()].copy_from_slice(block);
            }
            assert_eq!(expanded.coefficients(), expected);
            let point = [Field64::from(11); 6];
            assert_eq!(
                expanded.mle_evaluate(&point),
                multilinear_extend(&expected, &point)
            );
        }
    }

    #[test]
    fn composed_prefixes_and_folds_match_dense_tables() {
        for base in [Field64::ZERO, Field64::ONE, Field64::from(7)] {
            let mut form = TensorLinearForm::zero(32);
            form.add_univariate(3, 23, base, Field64::from(17));
            let mut dense = form.coefficients();
            for (block, prefix, width) in [(8, 3, 16), (8, 5, 16)] {
                form = form.insert_prefixes(dense.len(), dense.len() / block, prefix, width);
                let mut expanded = vec![Field64::ZERO; dense.len() / block * width];
                for (i, source) in dense.chunks_exact(block).enumerate() {
                    expanded[i * width + prefix..i * width + prefix + block]
                        .copy_from_slice(source);
                }
                let challenges = [Field64::from(3), Field64::from(19)];
                form = form.fold(&challenges);
                for &c in &challenges {
                    fold(&mut expanded, c);
                }
                dense = expanded;
                assert_eq!(form.coefficients(), dense);
                let point: Vec<_> = (0..dense.len().ilog2())
                    .map(|i| Field64::from(u64::from(i) + 2))
                    .collect();
                assert_eq!(
                    form.mle_evaluate(&point),
                    multilinear_extend(&dense, &point)
                );
            }
        }
    }

    #[test]
    fn shared_high_bit_weights_match_dense_mle() {
        let mut form = TensorLinearForm::<Field64>::zero(128);
        for i in 0..256 {
            form.add_univariate(
                i % 98,
                31,
                Field64::from((i % 13) as u64),
                Field64::from(i as u64 + 3),
            );
        }
        let point = [
            Field64::ZERO,
            Field64::ONE,
            Field64::from(2),
            Field64::from(3),
            Field64::from(5),
            Field64::from(7),
            Field64::from(11),
        ];
        assert_eq!(
            form.mle_evaluate(&point),
            multilinear_extend(&form.coefficients(), &point)
        );
    }

    #[test]
    fn domain_covectors_match_direct_sums_after_prefixes_and_folds() {
        fn check<F: FftField>() {
            let mut form = TensorLinearForm::<F>::zero(256);
            let mut expected = vec![F::ZERO; 256];
            for (size, offset, len, count) in [(256, 3, 125, 192), (128, 99, 91, 160)] {
                let root = generator::<F>(size).unwrap();
                for query in 0..count {
                    let index = query * 13 % size;
                    let scalar = F::from(query as u64 + 1);
                    form.add_in_domain(offset, len, size, index, scalar);
                    let point = root.pow([index as u64]);
                    let mut power = scalar;
                    for entry in &mut expected[offset..offset + len] {
                        *entry += power;
                        power *= point;
                    }
                }
            }
            for point in [F::ZERO, F::ONE, F::from(7)] {
                form.add_univariate(5, 137, point, F::from(11));
                let mut power = F::from(11);
                for entry in &mut expected[5..142] {
                    *entry += power;
                    power *= point;
                }
            }
            assert_eq!(form.coefficients(), expected);
            form = form.insert_prefixes(256, 8, 7, 64);
            let mut expanded = vec![F::ZERO; 512];
            for (index, block) in expected.chunks_exact(32).enumerate() {
                expanded[index * 64 + 7..index * 64 + 39].copy_from_slice(block);
            }
            assert_eq!(form.coefficients(), expanded);
            let challenges = [F::ZERO, F::ONE, F::from(19)];
            form = form.fold(&challenges);
            for challenge in challenges {
                fold(&mut expanded, challenge);
            }
            assert_eq!(form.coefficients(), expanded);
            let mut scaled = vec![F::ZERO; expanded.len()];
            form.accumulate(&mut scaled, F::from(17));
            assert_eq!(
                scaled,
                expanded
                    .iter()
                    .map(|value| *value * F::from(17))
                    .collect::<Vec<_>>()
            );
        }
        check::<Field64>();
        check::<Field128>();
        check::<Field256>();
    }
}
