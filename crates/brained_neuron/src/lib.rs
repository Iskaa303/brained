//! Structural root module linking all sub-compartments of the individual neuron together.

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
    #[doc(hidden)]
    pub use crate::{
        axon::Axon,
        dendrite::Dendrite,
        neuron::Neuron,
        soma::Soma,
        synapse::Synapse,
        types::{IonConcentrations, Neurotransmitter},
    };
}
