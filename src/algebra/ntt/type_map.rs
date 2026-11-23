use std::{
    any::{Any, TypeId},
    collections::HashMap,
    marker::PhantomData,
    sync::{Arc, RwLock},
};

/// A trait *family*: for each `T`, specifies the corresponding erased dyn-trait object type.
pub trait Family {
    type Dyn<T: 'static>: ?Sized + Send + Sync + 'static;
}

/// A map from types `T` to objects of type `Arc<F::Dyn<T>>`.
pub struct TypeMap<F: Family> {
    inner: RwLock<HashMap<TypeId, Box<dyn Any + Send + Sync>>>,
    _marker: PhantomData<F>,
}

struct Holder<F: Family, T: 'static>(Arc<F::Dyn<T>>);

impl<F: 'static + Family> Default for TypeMap<F> {
    fn default() -> Self {
        Self {
            inner: RwLock::new(HashMap::default()),
            _marker: PhantomData,
        }
    }
}

impl<F: 'static + Family> TypeMap<F> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert<T: 'static>(&self, v: Arc<F::Dyn<T>>) {
        self.inner.write().expect("Lock poisoned").insert(
            TypeId::of::<T>(),
            Box::new(Holder::<F, T>(v)) as Box<dyn Any + Send + Sync>,
        );
    }

    pub fn get<T: 'static>(&self) -> Option<Arc<F::Dyn<T>>> {
        self.inner
            .read()
            .expect("Lock poisoned")
            .get(&TypeId::of::<T>())
            .and_then(|a| a.downcast_ref::<Holder<F, T>>())
            .map(|h| h.0.clone())
    }

    pub fn contains<T: 'static>(&self) -> bool {
        self.inner
            .read()
            .expect("Lock poisoned")
            .contains_key(&TypeId::of::<T>())
    }

    pub fn remove<T: 'static>(&self) -> Option<Arc<F::Dyn<T>>> {
        self.inner
            .write()
            .expect("Lock poisoned")
            .remove(&TypeId::of::<T>())
            .and_then(|a| a.downcast::<Holder<F, T>>().ok())
            .map(|h| h.0)
    }
}
