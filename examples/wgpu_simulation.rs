//! WGPU simulation example for the brained library.
//! Simulates many neurons concurrently using the GPU.
use brained::prelude::*;

async fn run() {
    // 1. Initialize WGPU device and queue lazily via Default
    let wgpu_device = WgpuDevice::default();

    let dt = 0.025;
    let num_neurons = 30_000;

    println!("Initializing {} neurons on the GPU...", num_neurons);

    // 2. Instantiate the cells
    let mut brain_cells: Vec<Neuron<WgpuBackend>> = (0..num_neurons)
        .map(|id| Neuron::<WgpuBackend>::new(&wgpu_device, id as u64, 2000.0, 5000.0, 1.0, true))
        .collect();

    let active_signals = [Neurotransmitter::Glutamate];

    println!("Running simulation loop for 50 steps...");
    println!("--------------------------------------------------");

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

    println!("Simulation complete!");
}

fn main() {
    pollster::block_on(run());
}
