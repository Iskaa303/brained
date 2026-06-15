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
    pub dendrite_start: u32,
    pub dendrite_count: u32,
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

pub struct GpuChunk {
    pub capacity: usize,
    pub gpu_neuron_buffer: wgpu::Buffer,
    pub gpu_staging_buffer: wgpu::Buffer,
    pub bind_group: Arc<wgpu::BindGroup>,
}

pub struct WgpuNeuronPoolState {
    pub cpu_neurons: Vec<GpuNeuronData>,
    pub cpu_dendrites: Vec<GpuDendrite>,
    pub cpu_synapses: Vec<GpuSynapse>,
    pub params: GpuParams,
    pub gpu_params_buffer: Option<wgpu::Buffer>,
    pub gpu_dendrite_buffer: Option<wgpu::Buffer>,
    pub gpu_synapse_buffer: Option<wgpu::Buffer>,
    pub chunks: Vec<GpuChunk>,
    pub chunk_capacity: usize,
    pub dendrite_capacity: usize,
    pub synapse_capacity: usize,
    pub dirty_size: bool,
    pub dirty_data: bool,
    pub dirty_dendrites: bool,
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
                cpu_synapses: Vec::new(),
                params: GpuParams {
                    dt_ms: 0.025,
                    surface_area: 5000.0,
                    capacitance: 1.0,
                    present_glutamate: 0,
                    present_gaba: 0,
                    padding1: 0,
                    padding2: 0,
                    padding3: 0,
                },
                gpu_params_buffer: None,
                gpu_dendrite_buffer: None,
                gpu_synapse_buffer: None,
                chunks: Vec::new(),
                chunk_capacity: 0,
                dendrite_capacity: 0,
                synapse_capacity: 0,
                dirty_size: false,
                dirty_data: false,
                dirty_dendrites: false,
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
            dendrite_start: 0,
            dendrite_count: 0,
        });

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

        let count = pool_state.cpu_neurons.len();
        if count == 0 {
            return vec![false; states.len()];
        }

        if pool_state.chunk_capacity == 0 {
            let max_bytes = device.device.limits().max_storage_buffer_binding_size;
            let neuron_size = size_of::<GpuNeuronData>() as u64;
            let max_elems = max_bytes / neuron_size;
            let capacity = (max_elems as usize * 9 / 10).min(2_000_000);
            pool_state.chunk_capacity = capacity.max(256);
        }

        let chunk_capacity = pool_state.chunk_capacity;
        let required_chunks = count.div_ceil(chunk_capacity);

        // Ensure global dendrite and synapse buffers exist and are large enough
        let num_dendrites = pool_state.cpu_dendrites.len().max(1);
        let num_synapses = pool_state.cpu_synapses.len().max(1);

        let mut rebuild_bind_groups = false;

        if pool_state.gpu_dendrite_buffer.is_none()
            || pool_state.dendrite_capacity < num_dendrites
            || pool_state.gpu_synapse_buffer.is_none()
            || pool_state.synapse_capacity < num_synapses
        {
            let new_d_cap = num_dendrites.next_power_of_two().max(256);
            let new_s_cap = num_synapses.next_power_of_two().max(256);

            pool_state.gpu_dendrite_buffer =
                Some(device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Global Dendrite Buffer"),
                    size: (new_d_cap * size_of::<GpuDendrite>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));

            pool_state.gpu_synapse_buffer =
                Some(device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Global Synapse Buffer"),
                    size: (new_s_cap * size_of::<GpuSynapse>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));

            pool_state.dendrite_capacity = new_d_cap;
            pool_state.synapse_capacity = new_s_cap;
            pool_state.dirty_dendrites = true;
            rebuild_bind_groups = true;
        }

        if pool_state.dirty_size || pool_state.chunks.len() < required_chunks {
            if pool_state.gpu_params_buffer.is_none() {
                pool_state.gpu_params_buffer =
                    Some(device.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Global Params Buffer"),
                        size: size_of::<GpuParams>() as u64,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }));
            }

            while pool_state.chunks.len() < required_chunks {
                let capacity = chunk_capacity;
                let gpu_neuron_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Chunk Neuron Buffer"),
                    size: (capacity * size_of::<GpuNeuronData>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                let gpu_staging_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Chunk Staging Buffer"),
                    size: (capacity * size_of::<GpuNeuronData>()) as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                // Create dummy bind group, will be updated below
                let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Chunk Bind Group"),
                    layout: &pool.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: gpu_neuron_buffer.as_entire_binding(),
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
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: pool_state
                                .gpu_synapse_buffer
                                .as_ref()
                                .unwrap()
                                .as_entire_binding(),
                        },
                    ],
                });

                pool_state.chunks.push(GpuChunk {
                    capacity,
                    gpu_neuron_buffer,
                    gpu_staging_buffer,
                    bind_group: Arc::new(bind_group),
                });
            }

            pool_state.dirty_size = false;
            pool_state.dirty_data = true; // force data upload
        }

        if rebuild_bind_groups {
            let WgpuNeuronPoolState {
                chunks,
                gpu_params_buffer,
                gpu_dendrite_buffer,
                gpu_synapse_buffer,
                ..
            } = &mut *pool_state;

            let params_binding = gpu_params_buffer.as_ref().unwrap().as_entire_binding();
            let dendrite_binding = gpu_dendrite_buffer.as_ref().unwrap().as_entire_binding();
            let synapse_binding = gpu_synapse_buffer.as_ref().unwrap().as_entire_binding();

            for chunk in chunks {
                chunk.bind_group =
                    Arc::new(device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Chunk Bind Group"),
                        layout: &pool.bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: chunk.gpu_neuron_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry { binding: 1, resource: params_binding.clone() },
                            wgpu::BindGroupEntry { binding: 2, resource: dendrite_binding.clone() },
                            wgpu::BindGroupEntry { binding: 3, resource: synapse_binding.clone() },
                        ],
                    }));
            }
        }

        if pool_state.dirty_dendrites {
            if !pool_state.cpu_dendrites.is_empty() {
                device.queue.write_buffer(
                    pool_state.gpu_dendrite_buffer.as_ref().unwrap(),
                    0,
                    bytemuck::cast_slice(&pool_state.cpu_dendrites),
                );
            }
            if !pool_state.cpu_synapses.is_empty() {
                device.queue.write_buffer(
                    pool_state.gpu_synapse_buffer.as_ref().unwrap(),
                    0,
                    bytemuck::cast_slice(&pool_state.cpu_synapses),
                );
            }
            pool_state.dirty_dendrites = false;
        }

        if pool_state.dirty_data {
            for (i, chunk) in pool_state.chunks.iter().enumerate() {
                let start = i * chunk_capacity;
                let end = ((i + 1) * chunk_capacity).min(count);
                if start >= end {
                    break;
                }

                device.queue.write_buffer(
                    &chunk.gpu_neuron_buffer,
                    0,
                    bytemuck::cast_slice(&pool_state.cpu_neurons[start..end]),
                );
            }
            pool_state.dirty_data = false;
        }

        pool_state.params.dt_ms = dt_ms as f32;
        pool_state.params.present_glutamate = present_glutamate;
        pool_state.params.present_gaba = present_gaba;

        pool_state.params.surface_area =
            states.first().map(|s| s.soma_surface_area).unwrap_or(5000.0);

        device.queue.write_buffer(
            pool_state.gpu_params_buffer.as_ref().unwrap(),
            0,
            bytemuck::bytes_of(&pool_state.params),
        );

        let mut encoder =
            device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pool.pipeline);

            for (i, chunk) in pool_state.chunks.iter().enumerate() {
                let start = i * chunk_capacity;
                let end = ((i + 1) * chunk_capacity).min(count);
                if start >= end {
                    break;
                }

                let chunk_count = (end - start) as u32;
                let workgroups = chunk_count.div_ceil(256);
                cpass.set_bind_group(0, chunk.bind_group.as_ref(), &[]);
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }
        }

        for (i, chunk) in pool_state.chunks.iter().enumerate() {
            let start = i * chunk_capacity;
            let end = ((i + 1) * chunk_capacity).min(count);
            if start >= end {
                break;
            }

            let chunk_count = (end - start) as u64;
            encoder.copy_buffer_to_buffer(
                &chunk.gpu_neuron_buffer,
                0,
                &chunk.gpu_staging_buffer,
                0,
                chunk_count * size_of::<GpuNeuronData>() as u64,
            );
        }

        device.queue.submit(Some(encoder.finish()));

        let mut receivers = Vec::new();
        for (i, chunk) in pool_state.chunks.iter().enumerate() {
            let start = i * chunk_capacity;
            let end = ((i + 1) * chunk_capacity).min(count);
            if start >= end {
                break;
            }

            let chunk_count = (end - start) as u64;
            let buffer_slice =
                chunk.gpu_staging_buffer.slice(..chunk_count * size_of::<GpuNeuronData>() as u64);
            let (sender, receiver) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
            receivers.push(receiver);
        }

        device.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        for r in receivers {
            r.recv().unwrap().unwrap();
        }

        {
            let WgpuNeuronPoolState { chunks, cpu_neurons, .. } = &mut *pool_state;
            for (i, chunk) in chunks.iter().enumerate() {
                let start = i * chunk_capacity;
                let end = ((i + 1) * chunk_capacity).min(count);
                if start >= end {
                    break;
                }

                let chunk_count = (end - start) as u64;
                let buffer_slice = chunk
                    .gpu_staging_buffer
                    .slice(..chunk_count * size_of::<GpuNeuronData>() as u64);
                {
                    let data = buffer_slice.get_mapped_range();
                    let result: &[GpuNeuronData] = bytemuck::cast_slice(&data);
                    cpu_neurons[start..end].copy_from_slice(result);
                }
                chunk.gpu_staging_buffer.unmap();
            }
        }

        let mut results = vec![false; states.len()];
        for (i, state) in states.iter_mut().enumerate() {
            let is_spiking = pool_state.cpu_neurons[state.index].is_spiking != 0;
            if is_spiking {
                state.axon.queue_spike();
            }
            state.axon.update(dt_ms);
            results[i] = is_spiking;
        }

        results
    }

    fn add_dendrite(state: &mut Self::State, branch: Dendrite) {
        state.dendrites.push(branch);

        let pool = state.device.get_or_init_extension(|| WgpuNeuronPool::new(&state.device.device));
        let mut pool_state = pool.state.write().unwrap();

        let dendrite_start = pool_state.cpu_dendrites.len() as u32;
        let mut synapse_start = pool_state.cpu_synapses.len() as u32;

        for d in &state.dendrites {
            let syn_count = d.synapses.len() as u32;
            pool_state.cpu_dendrites.push(GpuDendrite {
                length: d.length as f32,
                diameter: d.diameter as f32,
                synapse_start,
                synapse_count: syn_count,
            });

            for syn in &d.synapses {
                let nt_type = match syn.neurotransmitter {
                    Neurotransmitter::Glutamate => 0,
                    Neurotransmitter::Gaba => 1,
                    _ => 2,
                };
                pool_state.cpu_synapses.push(GpuSynapse {
                    nt_type,
                    receptor_density: syn.receptor_density as f32,
                    efficacy: syn.efficacy as f32,
                    cleft_width: syn.cleft_width as f32,
                    open_fraction: syn.open_fraction as f32,
                    padding1: 0,
                    padding2: 0,
                    padding3: 0,
                });
                synapse_start += 1;
            }
        }

        pool_state.cpu_neurons[state.index].dendrite_start = dendrite_start;
        pool_state.cpu_neurons[state.index].dendrite_count = state.dendrites.len() as u32;

        pool_state.dirty_dendrites = true;
        pool_state.dirty_data = true; // neuron data changed
    }
}
