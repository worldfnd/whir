#[cfg(test)]
use std::fmt::Debug;
use std::{
    iter::{FusedIterator, RepeatN},
    slice::ChunksExact,
};

#[cfg(test)]
use serde::{Deserialize, Serialize};

pub enum EitherIter<A, B> {
    Left(A),
    Right(B),
}

/// Target single-thread workload size for `T`.
/// Should ideally be a multiple of a cache line (64 bytes)
/// and close to the L1 cache size.
pub const fn workload_size<T: Sized>() -> usize {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    const CACHE_SIZE: usize = 1 << 17; // 128KB for Apple Silicon

    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "ios", target_os = "android", target_os = "linux")
    ))]
    const CACHE_SIZE: usize = 1 << 16; // 64KB for mobile/server ARM

    #[cfg(target_arch = "x86_64")]
    const CACHE_SIZE: usize = 1 << 15; // 32KB for x86-64

    #[cfg(not(any(
        all(target_arch = "aarch64", target_os = "macos"),
        all(
            target_arch = "aarch64",
            any(target_os = "ios", target_os = "android", target_os = "linux")
        ),
        target_arch = "x86_64"
    )))]
    const CACHE_SIZE: usize = 1 << 15; // 32KB default

    CACHE_SIZE / size_of::<T>()
}

/// Like zip, but panics if the iterators are not the same length.
#[inline]
pub fn zip_strict<A, B>(
    a: impl IntoIterator<Item = A>,
    b: impl IntoIterator<Item = B>,
) -> impl Iterator<Item = (A, B)> {
    let mut a = a.into_iter();
    let mut b = b.into_iter();
    std::iter::from_fn(move || match (a.next(), b.next()) {
        (Some(x), Some(y)) => Some((x, y)),
        (None, None) => None,
        _ => panic!("Iterators had different lengths"),
    })
}

impl<T, A, B> Iterator for EitherIter<A, B>
where
    A: Iterator<Item = T>,
    B: Iterator<Item = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Left(a) => a.next(),
            Self::Right(b) => b.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Left(a) => a.size_hint(),
            Self::Right(b) => b.size_hint(),
        }
    }
}

impl<T, A, B> DoubleEndedIterator for EitherIter<A, B>
where
    A: DoubleEndedIterator<Item = T>,
    B: DoubleEndedIterator<Item = T>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            Self::Left(a) => a.next_back(),
            Self::Right(b) => b.next_back(),
        }
    }
}

impl<T, A, B> ExactSizeIterator for EitherIter<A, B>
where
    A: ExactSizeIterator<Item = T>,
    B: ExactSizeIterator<Item = T>,
{
    fn len(&self) -> usize {
        match self {
            Self::Left(a) => a.len(),
            Self::Right(b) => b.len(),
        }
    }
}

impl<T, A, B> FusedIterator for EitherIter<A, B>
where
    A: FusedIterator<Item = T>,
    B: FusedIterator<Item = T>,
{
}

pub fn chunks_exact_or_empty<T>(
    slice: &[T],
    chunk_size: usize,
    count: usize,
) -> EitherIter<RepeatN<&[T]>, ChunksExact<'_, T>> {
    assert_eq!(slice.len(), chunk_size * count);
    if chunk_size == 0 {
        EitherIter::Left(std::iter::repeat_n(&slice[0..0], count))
    } else {
        EitherIter::Right(slice.chunks_exact(chunk_size))
    }
}

#[cfg(test)]
#[track_caller]
pub fn test_serde<T: Debug + PartialEq + Serialize + for<'a> Deserialize<'a>>(value: &T) {
    // Test in human-readable format
    let json = serde_json::to_string_pretty(value).expect("json serialization failed");
    let deserialized = serde_json::from_str(&json).expect("json deserialization failed");
    assert_eq!(value, &deserialized, "json serde roundtrip failed");

    // Test in schemaless binary format

    let bytes = {
        let mut writer = Vec::new();
        ciborium::into_writer(value, &mut writer).expect("ciborium serialization failed");
        writer
    };
    let deserialized =
        ciborium::from_reader(bytes.as_slice()).expect("ciborium deserialization failed");
    assert_eq!(value, &deserialized, "ciborium serde roundtrip failed");
}
