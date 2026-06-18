pub use crate::buffer::cpu::{CpuBuffer, CpuSlice, CpuSliceMut};
pub use crate::buffer::{Buffer, BufferOps, BufferRead, BufferWrite};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub use super::metal_buffer::{MetalBuffer, MetalSlice, MetalSliceMut};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub type ActiveBuffer<T> = MetalBuffer<T>;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub type ActiveSlice<'a, T> = MetalSlice<'a, T>;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub type ActiveSliceMut<'a, T> = MetalSliceMut<'a, T>;

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub type ActiveBuffer<T> = CpuBuffer<T>;
#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub type ActiveSlice<'a, T> = CpuSlice<'a, T>;
#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub type ActiveSliceMut<'a, T> = CpuSliceMut<'a, T>;
