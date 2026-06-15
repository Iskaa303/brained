use brained_backend::{CpuBackend, CpuDevice};

use crate::axon::Axon;
use crate::backend::NeuronBackend;
use crate::dendrite::Dendrite;
use crate::soma::Soma;
use crate::types::Neurotransmitter;

/// State of a neuron running on the CPU backend.
#[derive(Debug, Clone)]
pub struct CpuNeuronState {
    /// Unique identifier for structural network topology routing.
    pub id: u64,
    /// Central cellular computational soma engine.
    pub soma: Soma,
    /// Incoming receptive dendritic arbor branches.
    pub dendrites: Vec<Dendrite>,
    /// Outgoing propagation signaling axon.
    pub axon: Axon,
}

impl NeuronBackend for CpuBackend {
    type State = CpuNeuronState;

    fn new_state(
        _device: &CpuDevice,
        id: u64,
        soma_surface_area: f64,
        axon_len: f64,
        axon_dia: f64,
        myelinated: bool,
    ) -> Self::State {
        CpuNeuronState {
            id,
            soma: Soma::new(soma_surface_area),
            dendrites: Vec::new(),
            axon: Axon::new(axon_len, axon_dia, myelinated),
        }
    }

    fn tick(state: &mut Self::State, dt_ms: f64, incoming_signals: &[Neurotransmitter]) -> bool {
        let mut total_dendritic_current = 0.0;

        // 1. Gather all incoming chemical stimuli via dendrites using Cable Theory attenuation
        for dendrite in &mut state.dendrites {
            total_dendritic_current +=
                dendrite.process_attenuated_current(incoming_signals, state.soma.v_membrane, dt_ms);
        }

        // 2. Feed integrated current directly to Soma and run Hodgkin-Huxley kinetics
        let is_spiking = state.soma.integrate(total_dendritic_current, dt_ms);

        // 3. If the soma fires an action potential, queue it up down the axon path
        if is_spiking {
            state.axon.queue_spike();
        }

        // 4. Update the structural propagation delays down the length of the axon
        state.axon.update(dt_ms)
    }

    fn tick_batch(
        states: &mut [&mut Self::State],
        dt_ms: f64,
        incoming_signals: &[Neurotransmitter],
    ) -> Vec<bool> {
        use rayon::prelude::*;
        states.par_iter_mut().map(|state| Self::tick(state, dt_ms, incoming_signals)).collect()
    }

    fn add_dendrite(state: &mut Self::State, branch: Dendrite) {
        state.dendrites.push(branch);
    }
}
