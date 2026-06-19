// NOTE: 100% AI GENERATED

mod buffer;
mod kernels;
mod profile;
mod rs;
mod runtime;
mod sha2;

pub use buffer::{MetalBuffer, MetalSlice, MetalSliceMut};
pub use profile::{
    reset_device_peak as metal_reset_device_peak, snapshot as metal_profile_snapshot,
    MetalProfileSnapshot,
};
pub use rs::MetalRs;
pub use sha2::MetalSha2;
