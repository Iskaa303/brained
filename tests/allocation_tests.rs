//! Allocation tests for scaling limits
use brained::prelude::*;

#[cfg(feature = "wgpu")]
#[test]
fn test_wgpu_allocation_scaling() {
    let device = WgpuDevice::default();
    let dt = 0.025;
    let active_signals = [Neurotransmitter::Glutamate];

    // Test a variety of allocation sizes, from small to very large.
    // 500,000 neurons will require over a gigabyte of VRAM and should trigger buffer chunking
    // if limits dictate, or at least test large contiguous allocation.
    for &num_neurons in &[10, 1000, 100_000, 500_000] {
        let mut brain_cells: Vec<Neuron<WgpuBackend>> = (0..num_neurons)
            .map(|id| Neuron::<WgpuBackend>::new(&device, id as u64, 2000.0, 5000.0, 1.0, true))
            .collect();
        let spikes = Neuron::tick_batch(&mut brain_cells, dt, &active_signals);
        assert_eq!(spikes.len(), num_neurons as usize);
    }
}

#[test]
fn test_cpu_allocation_scaling() {
    let device = CpuDevice;
    let dt = 0.025;
    let active_signals = [Neurotransmitter::Glutamate];

    for &num_neurons in &[10, 1000, 100_000, 500_000] {
        let mut brain_cells: Vec<Neuron<CpuBackend>> = (0..num_neurons)
            .map(|id| Neuron::<CpuBackend>::new(&device, id as u64, 2000.0, 5000.0, 1.0, true))
            .collect();
        let spikes = Neuron::tick_batch(&mut brain_cells, dt, &active_signals);
        assert_eq!(spikes.len(), num_neurons as usize);
    }
}
