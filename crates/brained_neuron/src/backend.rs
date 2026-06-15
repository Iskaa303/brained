use brained_backend::Backend;

use crate::{dendrite::Dendrite, types::Neurotransmitter};

/// A trait extending `Backend` with operations specific to neurons.
pub trait NeuronBackend: Backend {
    /// The internal state of a neuron on this backend.
    type State: Clone + std::fmt::Debug;

    /// Constructs a new neuron state.
    fn new_state(
        device: &Self::Device,
        id: u64,
        soma_surface_area: f64,
        axon_len: f64,
        axon_dia: f64,
        myelinated: bool,
    ) -> Self::State;

    /// Steps the cell forward by time step `dt_ms`.
    /// Returns `true` if an action potential was emitted.
    fn tick(state: &mut Self::State, dt_ms: f64, incoming_signals: &[Neurotransmitter]) -> bool;

    /// Steps a batch of cells forward concurrently.
    /// Returns a vector indicating which neurons spiked.
    fn tick_batch(
        states: &mut [&mut Self::State],
        dt_ms: f64,
        incoming_signals: &[Neurotransmitter],
    ) -> Vec<bool>;

    /// Attaches a custom dendritic arbor branch directly onto the neuron's soma.
    fn add_dendrite(state: &mut Self::State, branch: Dendrite);
}
