macro_rules! impl_cpu_read {
    ($ty:ty) => {
        impl<F: Field> BufferRead<F> for $ty {
            type TargetBuffer<T: Field> = CpuBuffer<T>;
            type Slice<'a>
                = CpuSlice<'a, F>
            where
                Self: 'a,
                F: 'a;

            fn read_len(&self) -> usize {
                self.data.len()
            }

            fn dot(&self, other: &Self) -> F {
                crate::algebra::dot(&*self.data, &*other.data)
            }

            fn sumcheck_polynomial(&self, other: &Self) -> (F, F) {
                crate::algebra::sumcheck::compute_sumcheck_polynomial(&*self.data, &*other.data)
            }

            fn mixed_extend<M: Embedding<Source = F, Target = T>, T: Field>(
                &self,
                embedding: &M,
                point: &[M::Target],
            ) -> M::Target {
                crate::algebra::mixed_multilinear_extend(embedding, &*self.data, point)
            }

            fn mixed_dot<M: Embedding<Source = F, Target = T>, T: Field>(
                &self,
                embedding: &M,
                other: &CpuBuffer<T>,
            ) -> M::Target {
                crate::algebra::mixed_dot(embedding, other.as_slice(), &*self.data)
            }

            fn mixed_univariate_evaluate<M: Embedding<Source = F>>(
                &self,
                embedding: &M,
                point: M::Target,
            ) -> M::Target {
                crate::algebra::mixed_univariate_evaluate(embedding, &*self.data, point)
            }

            fn mixed_scalar_mul_add_to<M: Embedding<Source = F>>(
                &self,
                embedding: &M,
                accumulator: &mut CpuBuffer<M::Target>,
                weight: M::Target,
            ) {
                crate::algebra::mixed_scalar_mul_add(
                    embedding,
                    &mut accumulator.data,
                    weight,
                    &*self.data,
                );
            }

            fn slice(&self, range: impl std::ops::RangeBounds<usize>) -> CpuSlice<'_, F> {
                let data = &*self.data;
                let (start, end) = $crate::buffer::cpu::resolve_range(range, data.len());
                CpuSlice {
                    data: &data[start..end],
                }
            }

            fn copy_to_owned(&self) -> CpuBuffer<F> {
                CpuBuffer::from_slice(&*self.data)
            }
        }
    };
}

macro_rules! impl_cpu_write {
    ($ty:ty) => {
        impl<F: Field> BufferWrite<F> for $ty {
            type SliceMut<'a>
                = CpuSliceMut<'a, F>
            where
                Self: 'a,
                F: 'a;

            fn scalar_mul(&mut self, weight: F) {
                crate::algebra::scalar_mul(&mut *self.data, weight);
            }

            fn accumulate_univariate_evaluations(
                &mut self,
                evaluators: &[crate::algebra::linear_form::UnivariateEvaluation<F>],
                scalars: &[F],
            ) {
                crate::algebra::linear_form::UnivariateEvaluation::accumulate_many(
                    evaluators,
                    &mut *self.data,
                    scalars,
                );
            }

            fn slice_mut(
                &mut self,
                range: impl std::ops::RangeBounds<usize>,
            ) -> CpuSliceMut<'_, F> {
                let data = &mut *self.data;
                let (start, end) = $crate::buffer::cpu::resolve_range(range, data.len());
                CpuSliceMut {
                    data: &mut data[start..end],
                }
            }

            fn split_at_mut(&mut self, mid: usize) -> (CpuSliceMut<'_, F>, CpuSliceMut<'_, F>) {
                let (lo, hi) = self.data.split_at_mut(mid);
                (CpuSliceMut { data: lo }, CpuSliceMut { data: hi })
            }
        }
    };
}

pub(crate) use impl_cpu_read;
pub(crate) use impl_cpu_write;
