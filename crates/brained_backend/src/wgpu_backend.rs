use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::Backend;

/// WGPU backend device. Holds the device and queue.
#[derive(Clone)]
pub struct WgpuDevice {
    /// The wgpu Device.
    pub device: Arc<wgpu::Device>,
    /// The wgpu Queue.
    pub queue: Arc<wgpu::Queue>,
    /// Shared extension context to allow backend implementations to store global state.
    pub extensions: Arc<RwLock<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>>,
}

impl std::fmt::Debug for WgpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuDevice").finish()
    }
}

impl WgpuDevice {
    /// Retrieve or initialize a global extension state for this device.
    pub fn get_or_init_extension<T: Any + Send + Sync, F: FnOnce() -> T>(&self, init: F) -> Arc<T> {
        let mut exts = self.extensions.write().unwrap();
        let val = exts.entry(TypeId::of::<T>()).or_insert_with(|| Arc::new(init()));
        val.clone().downcast::<T>().unwrap()
    }
}

impl Default for WgpuDevice {
    fn default() -> Self {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("Failed to find an appropriate adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("Failed to create device");

        Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            extensions: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

/// WGPU array/buffer holding data.
#[derive(Clone, Debug)]
pub struct WgpuArray {
    /// The internal wgpu buffer.
    pub buffer: Arc<wgpu::Buffer>,
}

/// WGPU backend implementation.
#[derive(Clone, Debug, Default)]
pub struct WgpuBackend;

impl Backend for WgpuBackend {
    type Device = WgpuDevice;
    type FloatArray = WgpuArray;
    type IntArray = WgpuArray;
}
