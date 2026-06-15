use crate::Backend;

/// CPU backend device.
#[derive(Clone, Default, Debug)]
pub struct CpuDevice;

/// CPU backend implementation.
#[derive(Clone, Debug, Default)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    type Device = CpuDevice;
    type FloatArray = Vec<f32>;
    type IntArray = Vec<u32>;
}
