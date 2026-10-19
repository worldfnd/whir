//! Traits and types for field morphisms.

use std::fmt::Debug;

use ark_ff::{Field, Zero};
use serde::{Deserialize, Serialize};

use crate::type_info::{Type, TypeInfo, Typed};

/// Trait for a type representing a unital field homomorphism.
///
/// An `Embedding<Source = F, Target = G>` must satisfy
///
/// - `e.map(F::ONE) == G::ONE`
/// - `e.map(a + b) == e.map(a) + e.map(b)`
/// - `e.map(a * b) == e.map(a) * e.map(b)`
/// - `e.mixed_add(a, b) == a + e.map(b)`
/// - `e.mixed_mul(a, b) == a * e.map(b)`
///
pub trait Embedding:
    Clone + PartialEq + Eq + Debug + TypeInfo + Serialize + for<'de> Deserialize<'de> + Send + Sync
{
    type Source: Field;
    type Target: Field;

    fn map(&self, dom: Self::Source) -> Self::Target;

    #[inline]
    fn mixed_mul(&self, cod: Self::Target, dom: Self::Source) -> Self::Target {
        cod * self.map(dom)
    }

    #[inline]
    fn mixed_add(&self, cod: Self::Target, dom: Self::Source) -> Self::Target {
        cod + self.map(dom)
    }

    fn map_vec(&self, source: Vec<Self::Source>) -> Vec<Self::Target> {
        source.into_iter().map(|e| self.map(e)).collect()
    }
}

/// The identity embedding of a field in into itself.
#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default, Serialize, Deserialize,
)]
#[serde(bound = "")]
pub struct Identity<F: Field> {
    field: Type<F>,
}

/// The basefield embedding morphism.
#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default, Serialize, Deserialize,
)]
#[serde(bound = "")]
pub struct Basefield<F: Field> {
    extension: Type<F>,
}

/// The Frobenius automorphism.
///
/// It demonstrates that embeddings are not uniquely defined by their source
/// and target fields alone. Hence they need to be objects in their own right.
#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default, Serialize, Deserialize,
)]
#[serde(bound = "")]
pub struct Frobenius<F: Field> {
    field: Type<F>,
    power: u64,
}

/// The composition of two morphisms.
#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default, Serialize, Deserialize,
)]
#[serde(bound = "")]
pub struct Compose<A: Embedding, B: Embedding<Source = A::Target>> {
    inner: Typed<A>,
    outer: Typed<B>,
}

impl<F: Field> Identity<F> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<F: Field> Basefield<F> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<F: Field> Frobenius<F> {
    pub fn new(power: u64) -> Self {
        Self {
            field: Type::new(),
            power: power % F::extension_degree(),
        }
    }
}

impl<A, B> Compose<A, B>
where
    A: Embedding,
    B: Embedding<Source = A::Target>,
{
    pub const fn new(inner: A, outer: B) -> Self {
        Self {
            inner: Typed::new(inner),
            outer: Typed::new(outer),
        }
    }

    pub fn inner(&self) -> &A {
        &self.inner
    }

    pub fn outer(&self) -> &B {
        &self.outer
    }
}

impl<F: Field> Embedding for Identity<F> {
    type Source = F;
    type Target = F;

    #[inline]
    fn map(&self, dom: Self::Source) -> Self::Target {
        dom
    }

    #[inline]
    fn map_vec(&self, source: Vec<F>) -> Vec<F> {
        source
    }
}

impl<F: Field> Embedding for Basefield<F> {
    type Source = F::BasePrimeField;
    type Target = F;

    #[inline]
    fn map(&self, dom: Self::Source) -> Self::Target {
        Self::Target::from_base_prime_field(dom)
    }

    #[inline]
    fn mixed_mul(&self, cod: Self::Target, dom: Self::Source) -> Self::Target {
        cod.mul_by_base_prime_field(&dom)
    }
}

impl<F: Field> Embedding for Frobenius<F> {
    type Source = F;
    type Target = F;

    #[inline]
    fn map(&self, mut dom: Self::Source) -> Self::Target {
        dom.frobenius_map_in_place(self.power as usize);
        dom
    }
}

impl<A, B> Embedding for Compose<A, B>
where
    A: Embedding,
    B: Embedding<Source = A::Target>,
{
    type Source = A::Source;
    type Target = B::Target;

    #[inline]
    fn map(&self, dom: Self::Source) -> Self::Target {
        self.outer.map(self.inner.map(dom))
    }

    #[inline]
    fn mixed_add(&self, cod: Self::Target, dom: Self::Source) -> Self::Target {
        self.outer.mixed_add(cod, self.inner.map(dom))
    }

    #[inline]
    fn mixed_mul(&self, cod: Self::Target, dom: Self::Source) -> Self::Target {
        self.outer.mixed_mul(cod, self.inner.map(dom))
    }
}

impl<F: Field> TypeInfo for Identity<F> {
    type Info = String;

    fn type_info() -> Self::Info {
        "identity".into()
    }
}

impl<F: Field> TypeInfo for Basefield<F> {
    type Info = String;

    fn type_info() -> Self::Info {
        "basefield".into()
    }
}

impl<F: Field> TypeInfo for Frobenius<F> {
    type Info = String;

    fn type_info() -> Self::Info {
        "frobenius".into()
    }
}

impl<A: Embedding, B: Embedding<Source = A::Target>> TypeInfo for Compose<A, B> {
    type Info = String;

    fn type_info() -> Self::Info {
        "compose".into()
    }
}

/// A public scalar, retaining prime-field arithmetic when it lies in that field.
#[derive(Clone, Copy)]
pub(crate) enum PublicScalar<F: Field> {
    Base(F::BasePrimeField),
    Extension(F),
}

impl<F: Field> PublicScalar<F> {
    pub(crate) fn new(value: F) -> Self {
        project_base_field(value).map_or(Self::Extension(value), Self::Base)
    }

    pub(crate) fn mul(self, value: F) -> F {
        match self {
            Self::Base(scalar) => value.mul_by_base_prime_field(&scalar),
            Self::Extension(scalar) => value * scalar,
        }
    }

    pub(crate) fn pow(self, exponent: u64) -> Self {
        match self {
            Self::Base(value) => Self::Base(value.pow([exponent])),
            Self::Extension(value) => Self::Extension(value.pow([exponent])),
        }
    }

    pub(crate) fn square(&mut self) {
        match self {
            Self::Base(value) => {
                value.square_in_place();
            }
            Self::Extension(value) => {
                value.square_in_place();
            }
        }
    }
}

/// Project a public field element only when all nonconstant coordinates vanish.
pub(crate) fn project_base_field<F: Field>(value: F) -> Option<F::BasePrimeField> {
    let mut coordinates = value.to_base_prime_field_elements();
    let constant = coordinates.next()?;
    coordinates
        .all(|coordinate| coordinate.is_zero())
        .then_some(constant)
}

#[cfg(test)]
pub(crate) mod tests {
    use ark_ff::{AdditiveGroup, Field, PrimeField};
    use proptest::{
        collection,
        prelude::{any, Just, Strategy},
        proptest, strategy,
    };

    use super::*;
    use crate::algebra::{embedding::Embedding, fields};

    pub fn arb_prime_field<F: PrimeField>() -> impl Strategy<Value = F> {
        let nbytes = F::MODULUS_BIT_SIZE.div_ceil(8) as usize;
        let rand = collection::vec(any::<u8>(), nbytes)
            .prop_map(|bytes| F::from_le_bytes_mod_order(&bytes));
        strategy::Union::new_weighted(vec![
            (1, Just(F::ZERO).boxed()),
            (1, Just(F::ONE).boxed()),
            (1, Just(-F::ONE).boxed()),
            (3, rand.boxed()),
        ])
    }

    pub fn arb_field<F: Field>() -> impl Strategy<Value = F> {
        collection::vec(
            arb_prime_field::<F::BasePrimeField>(),
            F::extension_degree() as usize,
        )
        .prop_map(|elements| F::from_base_prime_field_elems(elements).unwrap())
    }

    pub fn test_embedding<E: Embedding>(e: &E) {
        assert_eq!(e.map(E::Source::ZERO), E::Target::ZERO);
        assert_eq!(e.map(E::Source::ONE), E::Target::ONE);
        proptest!(|(a in arb_field(), b in arb_field())| {
            assert_eq!(e.map(a) + e.map(b), e.map(a + b));
        });
        proptest!(|(a in arb_field(), b in arb_field())| {
            assert_eq!(e.map(a) * e.map(b), e.map(a * b));
        });
        proptest!(|(a in arb_field(), b in arb_field())| {
            assert_eq!(e.mixed_add(a, b), a + e.map(b));
        });
        proptest!(|(a in arb_field(), b in arb_field())| {
            assert_eq!(e.mixed_mul(a, b), a * e.map(b));
        });
    }

    #[test]
    fn test_field64_3() {
        test_embedding(&Identity::<fields::Field64_3>::new());
        test_embedding(&Basefield::<fields::Field64_3>::new());
        test_embedding(&Frobenius::<fields::Field64_3>::new(0));
        test_embedding(&Frobenius::<fields::Field64_3>::new(1));
        test_embedding(&Frobenius::<fields::Field64_3>::new(2));
    }
}
