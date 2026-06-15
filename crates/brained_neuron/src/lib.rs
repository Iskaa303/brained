//! Structural root module linking all sub-compartments of the individual neuron together.

/// Traits and structures defining the backend abstraction for neurons.
pub mod backend;
/// CPU-specific backend implementation.
pub mod cpu;
/// WGPU-specific backend implementation, utilizing WGSL compute shaders.
#[cfg(feature = "wgpu")]
pub mod wgpu;

mod axon;
mod dendrite;
mod neuron;
mod soma;
mod synapse;
mod types;

/// The neuron prelude.
///
/// This includes the most common types in this crate, re-exported for your convenience.
pub mod prelude {
    pub use brained_backend::*;

    #[doc(hidden)]
    pub use crate::{
        axon::Axon,
        backend::NeuronBackend,
        dendrite::Dendrite,
        neuron::Neuron,
        soma::Soma,
        synapse::Synapse,
        types::{IonConcentrations, Neurotransmitter},
    };
}
