mod buffer;
mod read_write;

pub(crate) use buffer::resolve_range;
pub use buffer::{CpuBuffer, CpuSlice, CpuSliceMut};
