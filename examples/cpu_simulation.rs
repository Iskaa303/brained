//! CPU simulation example for the brained library.
//! Simulates many neurons concurrently using the CPU to compare performance.
use brained::prelude::*;

fn main() {
    let device = CpuDevice;

    let dt = 0.025;
    let num_neurons = 10_000_000;

    println!("Initializing {} neurons on the CPU...", num_neurons);

    // Instantiate the cells
    let mut brain_cells: Vec<Neuron<CpuBackend>> = (0..num_neurons)
        .map(|id| Neuron::<CpuBackend>::new(&device, id as u64, 2000.0, 5000.0, 1.0, true))
        .collect();

    let active_signals = [Neurotransmitter::Glutamate];

    println!("Running simulation loop for 50 steps...");
    println!("--------------------------------------------------");

    let start_time = std::time::Instant::now();

    for step in 0..50 {
        let time_ms = f64::from(step) * dt;

        let spike_results = Neuron::tick_batch(&mut brain_cells, dt, &active_signals);
        let spikes = spike_results.into_iter().filter(|&s| s).count();

        if step % 10 == 0 || step == 49 {
            println!(
                "[{:.2} ms] Processed {} neurons. Spikes this tick: {}",
                time_ms, num_neurons, spikes
            );
        }
    }

    let elapsed = start_time.elapsed();
    println!("Simulation complete in {:.2?}!", elapsed);
    println!("Average time per tick: {:.2?}", elapsed / 50);
}
