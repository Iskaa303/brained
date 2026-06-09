//! Tests for the cerebrum library.
use cerebrum::neuron::{Neuron, Neurotransmitter, dendrite::Dendrite, synapse::Synapse};

#[test]
fn test_neuron_creation_and_tick() {
    let mut brain_cell = Neuron::new(1, 2000.0, 5000.0, 1.0, true);

    // Initial state check
    assert!((brain_cell.soma.v_membrane - -65.0).abs() < f64::EPSILON);

    let mut branch = Dendrite::new(150.0, 2.0);
    let receptor = Synapse::new(Neurotransmitter::Glutamate, 15.0);
    branch.synapses.push(receptor);
    brain_cell.add_dendrite(branch);

    let active_signals = [Neurotransmitter::Glutamate];

    // First tick
    let spiked = brain_cell.tick(0.02, &active_signals);
    assert!(!spiked, "Should not spike on first tick");
    assert!(
        (brain_cell.soma.v_membrane - -65.0).abs() >= f64::EPSILON,
        "Membrane voltage should change due to Glutamate"
    );
}
