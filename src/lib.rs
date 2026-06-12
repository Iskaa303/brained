//! Brained is a biophysically accurate biological neuron simulation library.
#![no_std]

// Wasm does not support dynamic linking.
#[cfg(all(feature = "dynamic_linking", not(target_family = "wasm")))]
#[expect(
    unused_imports,
    clippy::single_component_path_imports,
    reason = "This causes Brained to be compiled as a dylib when using dynamic linking and therefore cannot be removed or changed without affecting dynamic linking."
)]
use brained_dylib;
pub use brained_internal::*;
