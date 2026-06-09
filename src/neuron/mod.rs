//! Structural root module linking all sub-compartments of the individual neuron together.
#![warn(missing_docs)]

pub mod axon;
pub mod dendrite;
pub mod soma;
pub mod synapse;
pub mod types;

use axon::Axon;
use dendrite::Dendrite;
use soma::Soma;
pub use types::Neurotransmitter;

/// Core composition struct managing a high-fidelity biological simulated neuron.
#[derive(Debug, Clone)]
pub struct Neuron {
    /// Unique identifier for structural network topology routing.
    #[allow(dead_code)]
    pub id: u64,
    /// Central cellular computational soma engine.
    pub soma: Soma,
    /// Incoming receptive dendritic arbor branches.
    pub dendrites: Vec<Dendrite>,
    /// Outgoing propagation signaling axon.
    pub axon: Axon,
}

impl Neuron {
    /// Constructs a new, biophysically accurate biological neuron.
    #[must_use]
    pub fn new(
        id: u64,
        soma_surface_area: f64,
        axon_len: f64,
        axon_dia: f64,
        myelinated: bool,
    ) -> Self {
        Self {
            id,
            soma: Soma::new(soma_surface_area),
            dendrites: Vec::new(),
            axon: Axon::new(axon_len, axon_dia, myelinated),
        }
    }

    /// Attaches a custom dendritic arbor branch directly onto the neuron's soma.
    pub fn add_dendrite(&mut self, branch: Dendrite) {
        self.dendrites.push(branch);
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
        let mut total_dendritic_current = 0.0;

        // 1. Gather all incoming chemical stimuli via dendrites using Cable Theory attenuation
        for dendrite in &mut self.dendrites {
            total_dendritic_current +=
                dendrite.process_attenuated_current(incoming_signals, self.soma.v_membrane, dt_ms);
        }

        // 2. Feed integrated current directly to Soma and run Hodgkin-Huxley kinetics
        let is_spiking = self.soma.integrate(total_dendritic_current, dt_ms);

        // 3. If the soma fires an action potential, queue it up down the axon path
        if is_spiking {
            self.axon.queue_spike();
        }

        // 4. Update the structural propagation delays down the length of the axon
        self.axon.update(dt_ms)
    }
}
