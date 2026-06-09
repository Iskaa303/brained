//! Basic simulation example for the cerebrum library.
use cerebrum::neuron::{Neuron, Neurotransmitter, dendrite::Dendrite, synapse::Synapse};

fn main() {
    // 1. Instantiate the cell
    let mut brain_cell = Neuron::new(1, 2000.0, 5000.0, 1.0, true);

    // 2. Add a dendritic branch with a highly sensitive receptor (efficiency boosted to 15.0)
    let mut branch = Dendrite::new(150.0, 2.0);
    let receptor = Synapse::new(Neurotransmitter::Glutamate, 15.0);
    branch.synapses.push(receptor);
    brain_cell.add_dendrite(branch);

    let dt = 0.02;

    // We will bombard the neuron with Glutamate continuously to force depolarization
    let active_signals = [Neurotransmitter::Glutamate];

    println!("Running neuron simulation loop...");
    println!("Initial Membrane Potential: {:.2} mV", brain_cell.soma.v_membrane);
    println!("--------------------------------------------------");

    // Run for 5000 steps (100 ms of biological time)
    for step in 0..5000 {
        let time_ms = f64::from(step) * dt;

        // Tick the cell forward
        let output_spiked = brain_cell.tick(dt, &active_signals);

        // Every 5 ms of simulated time, print a telemetry report so you can watch the voltage shift
        if step % 250 == 0 || step == 4999 {
            println!(
                "[{:.1} ms] Membrane Voltage: {:.2} mV | Gate states: m={:.2}, h={:.2}, n={:.2}",
                time_ms,
                brain_cell.soma.v_membrane,
                brain_cell.soma.gating.m,
                brain_cell.soma.gating.h,
                brain_cell.soma.gating.n
            );
        }

        if output_spiked {
            println!(
                ">>>> [{time_ms:.2} ms] !!! SPIKE DETECTED !!! Axon Terminal Released Neurotransmitters."
            );
        }
    }
}
