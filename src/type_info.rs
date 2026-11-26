//! Serializable type information so data contains type information and deserialization checks it.
//!
//! This makes sure `Config` objects can e.g. only be deserialized to instances for the same field.

use std::{
    fmt::{self, Debug, Formatter},
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use derive_where::derive_where;
use serde::{de::Error, Deserialize, Deserializer, Serialize, Serializer};

/// Types that can provide serializable type information for identification.
pub trait TypeInfo {
    type Info: Debug + PartialEq + Eq + Serialize + for<'de> Deserialize<'de>;

    fn type_info() -> Self::Info;
}

/// Zero-sized type that serializes into [`TypeInfo::type_info`].
#[derive_where(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Type<T: TypeInfo>(PhantomData<T>);

/// Wrapper that adds typeinfo when serializing.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct Typed<T: TypeInfo>(pub T);

impl<T: TypeInfo> Type<T> {
    /// Creates a new type instance.
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: TypeInfo> Debug for Type<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        T::type_info().fmt(f)
    }
}

impl<T: TypeInfo> Serialize for Type<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        T::type_info().serialize(serializer)
    }
}

impl<'de, T: TypeInfo> Deserialize<'de> for Type<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let expected = T::type_info();
        let got = T::Info::deserialize(deserializer)?;
        if expected == got {
            Ok(Self(PhantomData))
        } else {
            Err(D::Error::custom(format!(
                "Type mismatch, expected: {expected:?}, got: {got:?}"
            )))
        }
    }
}

impl<T: TypeInfo> Typed<T> {
    /// Creates a new type instance.
    pub const fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T: TypeInfo + Debug> Debug for Typed<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<T: TypeInfo> Deref for Typed<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: TypeInfo> DerefMut for Typed<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: TypeInfo + Serialize> Serialize for Typed<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct TypedValue<'s, T: TypeInfo> {
            #[serde(rename = "type")]
            type_: Type<T>,
            value: &'s T,
        }
        TypedValue {
            type_: Type::new(),
            value: &self.0,
        }
        .serialize(serializer)
    }
}

impl<'de, T: TypeInfo + Deserialize<'de>> Deserialize<'de> for Typed<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct TypedValue<T: TypeInfo> {
            #[serde(rename = "type")]
            #[allow(unused)]
            type_: Type<T>,
            value: T,
        }
        let read = TypedValue::deserialize(deserializer)?;
        Ok(Self(read.value))
    }
}

#[cfg(test)]
mod tests {
    use static_assertions::const_assert_eq;

    use super::*;
    use crate::{
        algebra::fields::{Field128, Field256, Field64_2, Field64_3},
        test_utils::test_serde,
    };

    const_assert_eq!(size_of::<Type<Field256>>(), 0);

    #[test]
    fn test_roundtrip() {
        test_serde(&Type::<Field256>::new());
        test_serde(&Type::<Field64_3>::new());
    }

    #[test]
    fn test_type_mismatch() {
        let value = Type::<Field128>::new();
        assert_eq!(size_of_val(&value), 0);
        let json = serde_json::to_string_pretty(&value).expect("json serialization failed");

        let result: Result<Type<Field64_2>, _> = serde_json::from_str(&json);
        assert!(result.is_err());
    }
}
