//! `Covector`: a linear form given explicitly by its coefficient vector.

use ark_ff::Field;

use super::{Evaluate, LinearForm};
use crate::algebra::{embedding::Embedding, mixed_dot, multilinear_extend, scalar_mul_add};

/// Linear form as an explicit covector over the field.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Covector<F: Field> {
    pub vector: Vec<F>,
}

impl<F: Field> LinearForm<F> for Covector<F> {
    fn size(&self) -> usize {
        self.vector.len()
    }

    fn mle_evaluate(&self, point: &[F]) -> F {
        multilinear_extend(&self.vector, point)
    }

    fn accumulate(&self, accumulator: &mut [F], scalar: F) {
        scalar_mul_add(accumulator, scalar, &self.vector);
    }
}

impl<F: Field> Covector<F> {
    pub const fn new(vector: Vec<F>) -> Self {
        Self { vector }
    }

    /// Any [`LinearForm<F>`] can be converted to a [`Covector<F>`].
    pub fn from(linear_form: &dyn LinearForm<F>) -> Self {
        let mut vector = vec![F::ZERO; linear_form.size()];
        linear_form.accumulate(&mut vector, F::ONE);
        Self { vector }
    }
}

impl<M: Embedding> Evaluate<M> for Covector<M::Target> {
    fn evaluate(&self, embedding: &M, vector: &[M::Source]) -> M::Target {
        assert_eq!(self.vector.len(), vector.len());
        mixed_dot(embedding, &self.vector, vector)
    }
}
