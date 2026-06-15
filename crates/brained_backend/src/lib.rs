//! Backend abstraction for Brained.

mod backend;
mod cpu;
#[cfg(feature = "wgpu")]
mod wgpu_backend;

pub use backend::*;
pub use cpu::*;
#[cfg(feature = "wgpu")]
pub use wgpu_backend::*;
