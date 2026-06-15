use crate::backend::NeuronBackend;
use crate::dendrite::Dendrite;
use crate::types::Neurotransmitter;

/// Core composition struct managing a high-fidelity biological simulated neuron.
/// Now generic over a `Backend` to allow running on CPU or GPU.
#[derive(Debug, Clone)]
pub struct Neuron<B: NeuronBackend> {
    /// The internal state representation for the selected backend.
    pub state: B::State,
}

impl<B: NeuronBackend> Neuron<B> {
    /// Constructs a new, biophysically accurate biological neuron.
    #[must_use]
    pub fn new(
        device: &B::Device,
        id: u64,
        soma_surface_area: f64,
        axon_len: f64,
        axon_dia: f64,
        myelinated: bool,
    ) -> Self {
        Self { state: B::new_state(device, id, soma_surface_area, axon_len, axon_dia, myelinated) }
    }

    /// Attaches a custom dendritic arbor branch directly onto the neuron's soma.
    pub fn add_dendrite(&mut self, branch: Dendrite) {
        B::add_dendrite(&mut self.state, branch);
    }

    /// Steps the global inner physical state of the cell forward by time step `dt_ms` (milliseconds).
    ///
    /// # Arguments
    /// * `dt_ms` - The discrete delta step slice (recommended 0.025 ms for Hodgkin-Huxley numerical stability).
    /// * `incoming_signals` - Current slice of Neurotransmitter payloads washing over the dendritic arbor.
    ///
    /// # Returns
    /// `true` if an action potential spike was emitted from the axon terminals this clock tick.
    pub fn tick(&mut self, dt_ms: f64, incoming_signals: &[Neurotransmitter]) -> bool {
        B::tick(&mut self.state, dt_ms, incoming_signals)
    }

    /// Ticks a batch of neurons concurrently.
    /// This utilizes parallel processing via CPU thread pools or optimized GPU dispatch.
    pub fn tick_batch(
        neurons: &mut [Self],
        dt_ms: f64,
        incoming_signals: &[Neurotransmitter],
    ) -> Vec<bool> {
        let mut states: Vec<&mut B::State> = neurons.iter_mut().map(|n| &mut n.state).collect();
        B::tick_batch(&mut states, dt_ms, incoming_signals)
    }
}
