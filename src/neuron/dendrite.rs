//! Dendritic branches using Cable Theory.

use crate::neuron::{synapse::Synapse, types::Neurotransmitter};

/// Represents a dendritic cable segment with active synaptic connection inputs.
#[derive(Debug, Clone)]
pub struct Dendrite {
    /// Length of the dendritic segment in micrometers.
    pub length: f64,
    /// Diameter of the dendritic segment in micrometers.
    pub diameter: f64,
    /// Axial resistance of the dendritic segment in Ohm * cm.
    pub axial_resistance: f64,
    /// Membrane capacitance of the dendritic segment in F / cm^2.
    #[allow(dead_code)]
    pub membrane_capacitance: f64,
    /// List of synapses located on this dendritic segment.
    pub synapses: Vec<Synapse>,
}

impl Dendrite {
    /// New dendritic segment.
    #[must_use]
    pub const fn new(length: f64, diameter: f64) -> Self {
        Self {
            length,
            diameter,
            axial_resistance: 150.0,   // Default value in Ohm * cm
            membrane_capacitance: 1.0, // Default value in F / cm^2
            synapses: Vec::new(),
        }
    }

    /// Calculate the space constant (lambda) for the dendritic segment.
    ///
    /// lambda = sqrt((d * Rm) / (4 * Ri))
    /// where d is the diameter in cm, Rm is the membrane resistance in Ohm * cm^2, and Ri is the axial resistance in Ohm * cm.
    /// Determines how far along the dendrite a voltage change will significantly affect the membrane potential.
    #[must_use]
    pub fn space_constant(&self) -> f64 {
        let r_membrane = 10000.0; // Membrane resistance in Ohm * cm^2
        // Convert diameter from micrometers to centimeters for calculation
        let d_cm = self.diameter * 1e-4;
        ((d_cm * r_membrane) / (4.0 * self.axial_resistance)).sqrt()
    }

    /// Process synaptic inputs and calculate the attenuated current reaching the soma.
    pub fn process_attenuated_current(
        &mut self,
        present_transmitters: &[Neurotransmitter],
        v_soma: f64,
        dt_ms: f64,
    ) -> f64 {
        let mut total_current = 0.0;
        let lambda = self.space_constant();

        // Convert length to centimeters to align with lambda
        let length_cm = self.length * 1e-4;
        // Cable theory attenuation factor across the distance: e^(-x / lambda)
        let attenuation = (-length_cm / lambda).exp();

        for synapse in &mut self.synapses {
            let concentration =
                if present_transmitters.contains(&synapse.neurotransmitter) { 1.0 } else { 0.0 };
            synapse.update(concentration, dt_ms);

            // Current driven by local synaptic voltage gradient
            total_current += synapse.get_current(v_soma);
        }

        total_current * attenuation
    }
}
