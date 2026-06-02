use std::cmp::max;

use ark_ff::{AdditiveGroup, Field};

use crate::algebra::embedding::{Embedding, Identity};
#[cfg(feature = "parallel")]
use crate::utils::workload_size;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub trait BufferOps<F: Field>: Clone {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    // zero pad to 2**log_m
    fn zero_pad(&mut self, log_m: usize);
    fn mixed_extend<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        embedding: &M,
        point: &[M::Target],
    ) -> M::Target;
    fn mixed_dot<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        embedding: &M,
        other: &impl BufferOps<M::Target>,
    ) -> M::Target;
    fn dot(&self, other: &Self) -> F;
    fn as_slice(&self) -> &[F];
    fn as_ro_buffer(&self, size: usize) -> impl BufferOps<F>;
}

// read-only buffer ops
pub trait ROBufferOps<F: Field> {
    fn split_at(&self, mid: usize) -> (&impl BufferOps<F>, &impl BufferOps<F>);
}

#[derive(Clone)]
pub struct SliceCpuBuffer<'a, F: Field> {
    data: &'a [F],
}

#[derive(Clone)]
pub struct CpuBuffer<F: Field> {
    data: Vec<F>,
    len: usize,
}

impl<F: Field> CpuBuffer<F> {
    pub fn from_vec(source: Vec<F>) -> Self {
        let len = source.len();
        Self { data: source, len }
    }

    pub fn from_slice(source: &[F]) -> Self {
        Self {
            data: Vec::from(source),
            len: source.len(),
        }
    }
}

impl<F: Field> BufferOps<F> for CpuBuffer<F> {
    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn zero_pad(&mut self, log_m: usize) {
        if !self.is_empty() {
            self.data.resize(1 << log_m, F::ZERO);
        }
    }

    fn mixed_extend<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        embedding: &M,
        point: &[M::Target],
    ) -> M::Target {
        #[inline]
        fn eval_exact<M: Embedding>(
            embedding: &M,
            evals: &[M::Source],
            point: &[M::Target],
        ) -> M::Target {
            debug_assert_eq!(evals.len(), 1 << point.len());

            // Helper to compute (a + (b - a) * c) efficiently with a, b in source field.
            let mixed = |a, b, c| embedding.mixed_add(embedding.mixed_mul(c, b - a), a);

            match point {
                [] => embedding.map(evals[0]),
                [x] => mixed(evals[0], evals[1], *x),
                [x0, x1] => {
                    let a0 = mixed(evals[0], evals[1], *x1);
                    let a1 = mixed(evals[2], evals[3], *x1);
                    a0 + (a1 - a0) * *x0
                }
                [x0, x1, x2] => {
                    let a00 = mixed(evals[0], evals[1], *x2);
                    let a01 = mixed(evals[2], evals[3], *x2);
                    let a10 = mixed(evals[4], evals[5], *x2);
                    let a11 = mixed(evals[6], evals[7], *x2);
                    let a0 = a00 + (a01 - a00) * *x1;
                    let a1 = a10 + (a11 - a10) * *x1;
                    a0 + (a1 - a0) * *x0
                }
                [x0, x1, x2, x3] => {
                    let a000 = mixed(evals[0], evals[1], *x3);
                    let a001 = mixed(evals[2], evals[3], *x3);
                    let a010 = mixed(evals[4], evals[5], *x3);
                    let a011 = mixed(evals[6], evals[7], *x3);
                    let a100 = mixed(evals[8], evals[9], *x3);
                    let a101 = mixed(evals[10], evals[11], *x3);
                    let a110 = mixed(evals[12], evals[13], *x3);
                    let a111 = mixed(evals[14], evals[15], *x3);
                    let a00 = a000 + (a001 - a000) * *x2;
                    let a01 = a010 + (a011 - a010) * *x2;
                    let a10 = a100 + (a101 - a100) * *x2;
                    let a11 = a110 + (a111 - a110) * *x2;
                    let a0 = a00 + (a01 - a00) * *x1;
                    let a1 = a10 + (a11 - a10) * *x1;
                    a0 + (a1 - a0) * *x0
                }
                [x, tail @ ..] => {
                    let (f0, f1) = evals.split_at(evals.len() / 2);
                    #[cfg(not(feature = "parallel"))]
                    let (f0, f1) = (
                        eval_exact(embedding, f0, tail),
                        eval_exact(embedding, f1, tail),
                    );

                    #[cfg(feature = "parallel")]
                    let (f0, f1) = {
                        use crate::utils::workload_size;
                        if evals.len() > workload_size::<M::Source>() {
                            rayon::join(
                                || eval_exact(embedding, f0, tail),
                                || eval_exact(embedding, f1, tail),
                            )
                        } else {
                            (
                                eval_exact(embedding, f0, tail),
                                eval_exact(embedding, f1, tail),
                            )
                        }
                    };

                    f0 + (f1 - f0) * *x
                }
            }
        }

        #[inline]
        fn eval_partial<M: Embedding>(
            embedding: &M,
            evals: &[M::Source],
            point: &[M::Target],
        ) -> M::Target {
            let size = 1 << point.len();
            debug_assert!(evals.len() <= size);
            if evals.is_empty() {
                return M::Target::ZERO;
            }
            if evals.len() == size {
                return eval_exact(embedding, evals, point);
            }

            match point {
                [] => embedding.map(evals[0]),
                [x, tail @ ..] => {
                    let half = size / 2;

                    // Only low half has data; high half is all implicit zeros.
                    if evals.len() <= half {
                        let f0 = eval_partial(embedding, evals, tail);
                        return f0 * (M::Target::ONE - *x);
                    }

                    // Low subtree is exact/full, high subtree is partial.
                    let (low, high) = evals.split_at(half);

                    #[cfg(not(feature = "parallel"))]
                    let (f0, f1) = (
                        eval_exact(embedding, low, tail),
                        eval_partial(embedding, high, tail),
                    );

                    #[cfg(feature = "parallel")]
                    let (f0, f1) = {
                        use crate::utils::workload_size;
                        if evals.len() > workload_size::<M::Source>() {
                            rayon::join(
                                || eval_exact(embedding, low, tail),
                                || eval_partial(embedding, high, tail),
                            )
                        } else {
                            (
                                eval_exact(embedding, low, tail),
                                eval_partial(embedding, high, tail),
                            )
                        }
                    };

                    f0 + (f1 - f0) * *x
                }
            }
        }

        eval_partial(embedding, &self.data, point)
    }

    fn mixed_dot<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        embedding: &M,
        other: &impl BufferOps<M::Target>,
    ) -> M::Target {
        assert_eq!(self.len(), other.len());

        let a = other.as_slice();
        let b = self.as_slice();

        #[cfg(feature = "parallel")]
        if a.len() > workload_size::<M::Target>() {
            return a
                .par_iter()
                .zip(b)
                .map(|(a, b)| embedding.mixed_mul(*a, *b))
                .sum();
        }

        a.iter()
            .zip(b)
            .map(|(a, b)| embedding.mixed_mul(*a, *b))
            .sum()
    }

    fn dot(&self, other: &Self) -> F {
        self.mixed_dot(&Identity::new(), other)
    }

    fn as_slice(&self) -> &[F] {
        &self.data[..self.len]
    }

    fn as_ro_buffer(&self, size: usize) -> impl BufferOps<F> {
        SliceCpuBuffer::from_buffer_with_size(self, size)
    }
}

impl<'a, F: Field> SliceCpuBuffer<'a, F> {
    pub fn from_buffer(buffer: &'a CpuBuffer<F>) -> Self {
        Self {
            data: buffer.as_slice(),
        }
    }

    pub fn from_buffer_with_size(buffer: &'a CpuBuffer<F>, size: usize) -> Self {
        Self {
            data: &buffer.data[..max(buffer.len, size)],
        }
    }

    pub fn from_slice_with_size(slice: &'a &[F], size: usize) -> Self {
        assert!(size <= slice.len());
        Self {
            data: &slice[..size],
        }
    }
}

impl<'a, F: Field> BufferOps<F> for SliceCpuBuffer<'a, F> {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn zero_pad(&mut self, log_m: usize) {
        panic!("read only")
    }

    fn mixed_extend<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        embedding: &M,
        point: &[M::Target],
    ) -> M::Target {
        #[inline]
        fn eval_exact<M: Embedding>(
            embedding: &M,
            evals: &[M::Source],
            point: &[M::Target],
        ) -> M::Target {
            debug_assert_eq!(evals.len(), 1 << point.len());

            // Helper to compute (a + (b - a) * c) efficiently with a, b in source field.
            let mixed = |a, b, c| embedding.mixed_add(embedding.mixed_mul(c, b - a), a);

            match point {
                [] => embedding.map(evals[0]),
                [x] => mixed(evals[0], evals[1], *x),
                [x0, x1] => {
                    let a0 = mixed(evals[0], evals[1], *x1);
                    let a1 = mixed(evals[2], evals[3], *x1);
                    a0 + (a1 - a0) * *x0
                }
                [x0, x1, x2] => {
                    let a00 = mixed(evals[0], evals[1], *x2);
                    let a01 = mixed(evals[2], evals[3], *x2);
                    let a10 = mixed(evals[4], evals[5], *x2);
                    let a11 = mixed(evals[6], evals[7], *x2);
                    let a0 = a00 + (a01 - a00) * *x1;
                    let a1 = a10 + (a11 - a10) * *x1;
                    a0 + (a1 - a0) * *x0
                }
                [x0, x1, x2, x3] => {
                    let a000 = mixed(evals[0], evals[1], *x3);
                    let a001 = mixed(evals[2], evals[3], *x3);
                    let a010 = mixed(evals[4], evals[5], *x3);
                    let a011 = mixed(evals[6], evals[7], *x3);
                    let a100 = mixed(evals[8], evals[9], *x3);
                    let a101 = mixed(evals[10], evals[11], *x3);
                    let a110 = mixed(evals[12], evals[13], *x3);
                    let a111 = mixed(evals[14], evals[15], *x3);
                    let a00 = a000 + (a001 - a000) * *x2;
                    let a01 = a010 + (a011 - a010) * *x2;
                    let a10 = a100 + (a101 - a100) * *x2;
                    let a11 = a110 + (a111 - a110) * *x2;
                    let a0 = a00 + (a01 - a00) * *x1;
                    let a1 = a10 + (a11 - a10) * *x1;
                    a0 + (a1 - a0) * *x0
                }
                [x, tail @ ..] => {
                    let (f0, f1) = evals.split_at(evals.len() / 2);
                    #[cfg(not(feature = "parallel"))]
                    let (f0, f1) = (
                        eval_exact(embedding, f0, tail),
                        eval_exact(embedding, f1, tail),
                    );

                    #[cfg(feature = "parallel")]
                    let (f0, f1) = {
                        use crate::utils::workload_size;
                        if evals.len() > workload_size::<M::Source>() {
                            rayon::join(
                                || eval_exact(embedding, f0, tail),
                                || eval_exact(embedding, f1, tail),
                            )
                        } else {
                            (
                                eval_exact(embedding, f0, tail),
                                eval_exact(embedding, f1, tail),
                            )
                        }
                    };

                    f0 + (f1 - f0) * *x
                }
            }
        }

        #[inline]
        fn eval_partial<M: Embedding>(
            embedding: &M,
            evals: &[M::Source],
            point: &[M::Target],
        ) -> M::Target {
            let size = 1 << point.len();
            debug_assert!(evals.len() <= size);
            if evals.is_empty() {
                return M::Target::ZERO;
            }
            if evals.len() == size {
                return eval_exact(embedding, evals, point);
            }

            match point {
                [] => embedding.map(evals[0]),
                [x, tail @ ..] => {
                    let half = size / 2;

                    // Only low half has data; high half is all implicit zeros.
                    if evals.len() <= half {
                        let f0 = eval_partial(embedding, evals, tail);
                        return f0 * (M::Target::ONE - *x);
                    }

                    // Low subtree is exact/full, high subtree is partial.
                    let (low, high) = evals.split_at(half);

                    #[cfg(not(feature = "parallel"))]
                    let (f0, f1) = (
                        eval_exact(embedding, low, tail),
                        eval_partial(embedding, high, tail),
                    );

                    #[cfg(feature = "parallel")]
                    let (f0, f1) = {
                        use crate::utils::workload_size;
                        if evals.len() > workload_size::<M::Source>() {
                            rayon::join(
                                || eval_exact(embedding, low, tail),
                                || eval_partial(embedding, high, tail),
                            )
                        } else {
                            (
                                eval_exact(embedding, low, tail),
                                eval_partial(embedding, high, tail),
                            )
                        }
                    };

                    f0 + (f1 - f0) * *x
                }
            }
        }

        eval_partial(embedding, &self.data, point)
    }

    fn mixed_dot<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        embedding: &M,
        other: &impl BufferOps<M::Target>,
    ) -> M::Target {
        assert_eq!(self.len(), other.len());

        let a = other.as_slice();
        let b = self.as_slice();

        #[cfg(feature = "parallel")]
        if a.len() > workload_size::<M::Target>() {
            return a
                .par_iter()
                .zip(b)
                .map(|(a, b)| embedding.mixed_mul(*a, *b))
                .sum();
        }

        a.iter()
            .zip(b)
            .map(|(a, b)| embedding.mixed_mul(*a, *b))
            .sum()
    }

    fn dot(&self, other: &Self) -> F {
        self.mixed_dot(&Identity::new(), other)
    }

    fn as_slice(&self) -> &[F] {
        &self.data
    }

    fn as_ro_buffer(&self, size: usize) -> impl BufferOps<F> {
        SliceCpuBuffer::from_slice_with_size(&self.data, size)
    }
}
