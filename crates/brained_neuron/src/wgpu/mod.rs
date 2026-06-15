#![allow(missing_docs, reason = "Internal GPU structures do not require public documentation")]

use std::sync::{Arc, RwLock};

use brained_backend::{WgpuBackend, WgpuDevice};

use crate::axon::Axon;
use crate::backend::NeuronBackend;
use crate::dendrite::Dendrite;
use crate::types::Neurotransmitter;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuNeuronData {
    pub v_membrane: f32,
    pub m: f32,
    pub h: f32,
    pub n: f32,
    pub total_dendritic_current: f32,
    pub is_spiking: u32,
    pub padding1: u32,
    pub padding2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuParams {
    pub dt_ms: f32,
    pub surface_area: f32,
    pub capacitance: f32,
    pub present_glutamate: u32,
    pub present_gaba: u32,
    pub padding1: u32,
    pub padding2: u32,
    pub padding3: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuDendrite {
    pub length: f32,
    pub diameter: f32,
    pub synapse_start: u32,
    pub synapse_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSynapse {
    pub nt_type: u32,
    pub receptor_density: f32,
    pub efficacy: f32,
    pub cleft_width: f32,
    pub open_fraction: f32,
    pub padding1: u32,
    pub padding2: u32,
    pub padding3: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuDendriteBuffer {
    pub num_dendrites: u32,
    pub num_synapses: u32,
    pub padding1: u32,
    pub padding2: u32,
    pub dendrites: [GpuDendrite; 16],
    pub synapses: [GpuSynapse; 64],
}

impl Default for GpuDendriteBuffer {
    fn default() -> Self {
        Self {
            num_dendrites: 0,
            num_synapses: 0,
            padding1: 0,
            padding2: 0,
            dendrites: [GpuDendrite {
                length: 0.0,
                diameter: 0.0,
                synapse_start: 0,
                synapse_count: 0,
            }; 16],
            synapses: [GpuSynapse {
                nt_type: 0,
                receptor_density: 0.0,
                efficacy: 0.0,
                cleft_width: 0.0,
                open_fraction: 0.0,
                padding1: 0,
                padding2: 0,
                padding3: 0,
            }; 64],
        }
    }
}

pub struct WgpuNeuronPoolState {
    pub cpu_neurons: Vec<GpuNeuronData>,
    pub cpu_dendrites: Vec<GpuDendriteBuffer>,
    pub params: GpuParams,
    pub gpu_neuron_buffer: Option<wgpu::Buffer>,
    pub gpu_staging_buffer: Option<wgpu::Buffer>,
    pub gpu_params_buffer: Option<wgpu::Buffer>,
    pub gpu_dendrite_buffer: Option<wgpu::Buffer>,
    pub bind_group: Option<Arc<wgpu::BindGroup>>,
    pub capacity: usize,
    pub dirty_size: bool,
    pub dirty_data: bool, // Set true if we added neurons/dendrites and need to re-upload.
}

pub struct WgpuNeuronPool {
    pub pipeline: Arc<wgpu::ComputePipeline>,
    pub bind_group_layout: Arc<wgpu::BindGroupLayout>,
    pub state: RwLock<WgpuNeuronPoolState>,
}

impl WgpuNeuronPool {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Neuron Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("neuron.wgsl").into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Neuron Compute Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);

        Self {
            pipeline: Arc::new(pipeline),
            bind_group_layout: Arc::new(bind_group_layout),
            state: RwLock::new(WgpuNeuronPoolState {
                cpu_neurons: Vec::new(),
                cpu_dendrites: Vec::new(),
                params: GpuParams {
                    dt_ms: 0.025,
                    surface_area: 5000.0, // default approximation until overridden
                    capacitance: 1.0,
                    present_glutamate: 0,
                    present_gaba: 0,
                    padding1: 0,
                    padding2: 0,
                    padding3: 0,
                },
                gpu_neuron_buffer: None,
                gpu_staging_buffer: None,
                gpu_params_buffer: None,
                gpu_dendrite_buffer: None,
                bind_group: None,
                capacity: 0,
                dirty_size: false,
                dirty_data: false,
            }),
        }
    }
}

#[derive(Debug, Clone)]
pub struct WgpuNeuronState {
    pub id: u64,
    pub index: usize,
    pub device: WgpuDevice,
    pub axon: Axon,
    pub dendrites: Vec<Dendrite>,
    pub soma_surface_area: f32,
}

impl NeuronBackend for WgpuBackend {
    type State = WgpuNeuronState;

    fn new_state(
        device: &WgpuDevice,
        id: u64,
        soma_surface_area: f64,
        axon_len: f64,
        axon_dia: f64,
        myelinated: bool,
    ) -> Self::State {
        let pool = device.get_or_init_extension(|| WgpuNeuronPool::new(&device.device));
        let mut state = pool.state.write().unwrap();

        let index = state.cpu_neurons.len();

        state.cpu_neurons.push(GpuNeuronData {
            v_membrane: -65.0,
            m: 0.05,
            h: 0.6,
            n: 0.32,
            total_dendritic_current: 0.0,
            is_spiking: 0,
            padding1: 0,
            padding2: 0,
        });
        state.cpu_dendrites.push(GpuDendriteBuffer::default());

        state.dirty_size = true;
        state.dirty_data = true;

        WgpuNeuronState {
            id,
            index,
            device: device.clone(),
            axon: Axon::new(axon_len, axon_dia, myelinated),
            dendrites: Vec::new(),
            soma_surface_area: soma_surface_area as f32,
        }
    }

    fn tick(state: &mut Self::State, dt_ms: f64, incoming_signals: &[Neurotransmitter]) -> bool {
        let mut states = vec![state];
        let spikes = Self::tick_batch(&mut states, dt_ms, incoming_signals);
        spikes[0]
    }

    fn tick_batch(
        states: &mut [&mut Self::State],
        dt_ms: f64,
        incoming_signals: &[Neurotransmitter],
    ) -> Vec<bool> {
        if states.is_empty() {
            return Vec::new();
        }

        let device = states[0].device.clone();
        let pool = device.get_or_init_extension(|| WgpuNeuronPool::new(&device.device));

        let present_glutamate =
            if incoming_signals.contains(&Neurotransmitter::Glutamate) { 1 } else { 0 };
        let present_gaba = if incoming_signals.contains(&Neurotransmitter::Gaba) { 1 } else { 0 };

        let mut pool_state = pool.state.write().unwrap();

        // Ensure buffers exist and are large enough
        let count = pool_state.cpu_neurons.len();
        if count == 0 {
            return vec![false; states.len()];
        }

        if pool_state.dirty_size || pool_state.capacity < count {
            let new_capacity = count.next_power_of_two().max(256);

            pool_state.gpu_neuron_buffer =
                Some(device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Global Neuron Buffer"),
                    size: (new_capacity * size_of::<GpuNeuronData>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));

            pool_state.gpu_staging_buffer =
                Some(device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Global Staging Buffer"),
                    size: (new_capacity * size_of::<GpuNeuronData>()) as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));

            pool_state.gpu_dendrite_buffer =
                Some(device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Global Dendrite Buffer"),
                    size: (new_capacity * size_of::<GpuDendriteBuffer>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));

            pool_state.gpu_params_buffer =
                Some(device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Global Params Buffer"),
                    size: size_of::<GpuParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));

            let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Global Neuron Bind Group"),
                layout: &pool.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pool_state
                            .gpu_neuron_buffer
                            .as_ref()
                            .unwrap()
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: pool_state
                            .gpu_params_buffer
                            .as_ref()
                            .unwrap()
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: pool_state
                            .gpu_dendrite_buffer
                            .as_ref()
                            .unwrap()
                            .as_entire_binding(),
                    },
                ],
            });

            pool_state.bind_group = Some(Arc::new(bind_group));
            pool_state.capacity = new_capacity;
            pool_state.dirty_size = false;
            pool_state.dirty_data = true; // force data upload
        }

        // Upload dirty initial state
        if pool_state.dirty_data {
            device.queue.write_buffer(
                pool_state.gpu_neuron_buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&pool_state.cpu_neurons),
            );
            device.queue.write_buffer(
                pool_state.gpu_dendrite_buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&pool_state.cpu_dendrites),
            );
            pool_state.dirty_data = false;
        }

        // Update params
        pool_state.params.dt_ms = dt_ms as f32;
        pool_state.params.present_glutamate = present_glutamate;
        pool_state.params.present_gaba = present_gaba;

        // We use the first neuron's surface area as approximation for the whole batch for simplicity
        // in this implementation, to avoid variable-sized param buffers per neuron.
        pool_state.params.surface_area =
            states.first().map(|s| s.soma_surface_area).unwrap_or(5000.0);

        device.queue.write_buffer(
            pool_state.gpu_params_buffer.as_ref().unwrap(),
            0,
            bytemuck::bytes_of(&pool_state.params),
        );

        // Submit the monolithic compute pass
        let mut encoder =
            device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pool.pipeline);
            cpass.set_bind_group(0, pool_state.bind_group.as_deref().unwrap(), &[]);
            // Dispatch 1 thread per neuron.
            let _workgroups = (count as u32).div_ceil(256);
            // Note: wgsl workgroup_size is 1, but we can change wgsl to use 64 or 256 for better speed.
            // Since wgsl uses @workgroup_size(1), workgroups = count.
            cpass.dispatch_workgroups(count as u32, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            pool_state.gpu_neuron_buffer.as_ref().unwrap(),
            0,
            pool_state.gpu_staging_buffer.as_ref().unwrap(),
            0,
            (count * size_of::<GpuNeuronData>()) as u64,
        );

        device.queue.submit(Some(encoder.finish()));

        // Map and Read
        let buffer_slice = pool_state
            .gpu_staging_buffer
            .as_ref()
            .unwrap()
            .slice(..(count * size_of::<GpuNeuronData>()) as u64);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        device.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result: &[GpuNeuronData] = bytemuck::cast_slice(&data);

        // Extract spike data back to the relevant neuron handles.
        let mut results = vec![false; states.len()];

        // Also update cpu_neurons cache with latest state from GPU so if buffer reallocates, we don't lose state.
        pool_state.cpu_neurons[..count].copy_from_slice(result);

        for (i, state) in states.iter_mut().enumerate() {
            let is_spiking = result[state.index].is_spiking != 0;
            if is_spiking {
                state.axon.queue_spike();
            }
            state.axon.update(dt_ms);
            results[i] = is_spiking;
        }

        drop(data);
        pool_state.gpu_staging_buffer.as_ref().unwrap().unmap();

        results
    }

    fn add_dendrite(state: &mut Self::State, branch: Dendrite) {
        state.dendrites.push(branch);

        let mut buffer_data = GpuDendriteBuffer::default();
        let mut syn_idx = 0;

        buffer_data.num_dendrites = state.dendrites.len().min(16) as u32;

        for (i, d) in state.dendrites.iter().take(16).enumerate() {
            let count = d.synapses.len() as u32;
            buffer_data.dendrites[i] = GpuDendrite {
                length: d.length as f32,
                diameter: d.diameter as f32,
                synapse_start: syn_idx,
                synapse_count: count,
            };

            for syn in &d.synapses {
                if syn_idx < 64 {
                    let nt_type = match syn.neurotransmitter {
                        Neurotransmitter::Glutamate => 0,
                        Neurotransmitter::Gaba => 1,
                        _ => 2,
                    };
                    buffer_data.synapses[syn_idx as usize] = GpuSynapse {
                        nt_type,
                        receptor_density: syn.receptor_density as f32,
                        efficacy: syn.efficacy as f32,
                        cleft_width: syn.cleft_width as f32,
                        open_fraction: syn.open_fraction as f32,
                        padding1: 0,
                        padding2: 0,
                        padding3: 0,
                    };
                    syn_idx += 1;
                }
            }
        }
        buffer_data.num_synapses = syn_idx;

        let pool = state.device.get_or_init_extension(|| WgpuNeuronPool::new(&state.device.device));
        let mut pool_state = pool.state.write().unwrap();
        pool_state.cpu_dendrites[state.index] = buffer_data;
        pool_state.dirty_data = true; // flag upload for next tick
    }
}
