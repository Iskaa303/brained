struct NeuronData {
    v_membrane: f32,
    m: f32,
    h: f32,
    n: f32,
    total_dendritic_current: f32,
    is_spiking: u32,
    dendrite_start: u32,
    dendrite_count: u32,
};

struct Params {
    dt_ms: f32,
    surface_area: f32,
    capacitance: f32,
    present_glutamate: u32,
    present_gaba: u32,
    padding1: u32,
    padding2: u32,
    padding3: u32,
};

struct Dendrite {
    length: f32,
    diameter: f32,
    synapse_start: u32,
    synapse_count: u32,
};

struct Synapse {
    nt_type: u32,
    receptor_density: f32,
    efficacy: f32,
    cleft_width: f32,
    open_fraction: f32,
    padding1: u32,
    padding2: u32,
    padding3: u32,
};

@group(0) @binding(0) var<storage, read_write> neurons: array<NeuronData>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> dendrites: array<Dendrite>;
@group(0) @binding(3) var<storage, read_write> synapses: array<Synapse>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&neurons)) {
        return;
    }

    var data = neurons[index];
    let v = data.v_membrane;
    let dt = params.dt_ms;

    // Process Synapses and Dendrites
    var total_dendritic_current: f32 = 0.0;
    
    for (var d: u32 = 0u; d < data.dendrite_count; d = d + 1u) {
        let dendrite_idx = data.dendrite_start + d;
        let dendrite = dendrites[dendrite_idx];
        let d_cm = dendrite.diameter * 1e-4;
        let r_membrane = 10000.0;
        let axial_resistance = 150.0;
        let lambda = sqrt((d_cm * r_membrane) / (4.0 * axial_resistance));
        
        let length_cm = dendrite.length * 1e-4;
        let attenuation = exp(-length_cm / lambda);
        
        var branch_current: f32 = 0.0;
        
        for (var s: u32 = 0u; s < dendrite.synapse_count; s = s + 1u) {
            let syn_idx = dendrite.synapse_start + s;
            var syn = synapses[syn_idx];
            
            var concentration: f32 = 0.0;
            if (syn.nt_type == 0u && params.present_glutamate == 1u) {
                concentration = 1.0;
            } else if (syn.nt_type == 1u && params.present_gaba == 1u) {
                concentration = 1.0;
            }
            
            let alpha = 5.0;
            let beta_syn = 0.18;
            let dr = (alpha * concentration * (1.0 - syn.open_fraction) - beta_syn * syn.open_fraction) * dt;
            
            syn.open_fraction = clamp(syn.open_fraction + dr, 0.0, 1.0);
            
            var v_reverse: f32 = -20.0;
            if (syn.nt_type == 0u) { v_reverse = 0.0; } // Glutamate
            if (syn.nt_type == 1u) { v_reverse = -70.0; } // GABA
            
            let g_max = 1e-10 * syn.receptor_density * syn.efficacy;
            let syn_current = g_max * syn.open_fraction * (v - v_reverse);
            
            branch_current = branch_current + syn_current;
            synapses[syn_idx] = syn; // Update state
        }
        
        total_dendritic_current = total_dendritic_current + (branch_current * attenuation);
    }
    
    data.total_dendritic_current = total_dendritic_current;

    // Constants
    let g_na_max = 120.0;
    let g_k_max = 36.0;
    let g_l = 0.3;
    let e_na = 50.0;
    let e_k = -77.0;
    let e_l = -54.4;

    // alpha and beta functions
    var alpha_m: f32;
    if (abs(v + 40.0) < 1e-5) { alpha_m = 1.0; } else { alpha_m = 0.1 * (v + 40.0) / (1.0 - exp(-(v + 40.0) / 10.0)); }
    let beta_m = 4.0 * exp(-(v + 65.0) / 18.0);

    let alpha_h = 0.07 * exp(-(v + 65.0) / 20.0);
    let beta_h = 1.0 / (1.0 + exp(-(v + 35.0) / 10.0));

    var alpha_n: f32;
    if (abs(v + 55.0) < 1e-5) { alpha_n = 0.1; } else { alpha_n = 0.01 * (v + 55.0) / (1.0 - exp(-(v + 55.0) / 10.0)); }
    let beta_n = 0.125 * exp(-(v + 65.0) / 80.0);

    // Update gating states
    data.m = data.m + (alpha_m * (1.0 - data.m) - beta_m * data.m) * dt;
    data.h = data.h + (alpha_h * (1.0 - data.h) - beta_h * data.h) * dt;
    data.n = data.n + (alpha_n * (1.0 - data.n) - beta_n * data.n) * dt;

    let m3 = data.m * data.m * data.m;
    let n4 = data.n * data.n * data.n * data.n;

    let i_na = g_na_max * m3 * data.h * (v - e_na);
    let i_k = g_k_max * n4 * (v - e_k);
    let i_leak = g_l * (v - e_l);

    let i_syn = data.total_dendritic_current / (params.surface_area * 1e-8);
    let i_outward = i_na + i_k + i_leak + i_syn;

    let dv = (-i_outward / params.capacitance) * dt;
    let old_v = data.v_membrane;
    data.v_membrane = data.v_membrane + dv;

    if (old_v < 0.0 && data.v_membrane >= 0.0) {
        data.is_spiking = 1u;
    } else {
        data.is_spiking = 0u;
    }

    neurons[index] = data;
}
