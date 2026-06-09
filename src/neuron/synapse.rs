//! Chemical synapse simulation.

use crate::neuron::types::Neurotransmitter;

/// Represents a chemical synapse connecting neurons.
#[derive(Debug, Clone)]
pub struct Synapse {
    /// Specific Neurotransmitter released by the synapse.
    pub neurotransmitter: Neurotransmitter,
    /// Density of receptors on the postsynaptic membrane (receptors per square micrometer).
    pub receptor_density: f64,
    /// Synaptic weight, representing connection efficacy (modeled by plasticity rules).
    pub efficacy: f64,
    /// Width of the synaptic cleft (nanometers).
    #[allow(dead_code)]
    pub cleft_width: f64,
    /// Fraction of open ion channels (0.0 to 1.0).
    pub open_fraction: f64,
}

impl Synapse {
    /// New chemical synapse with specified Neurotransmitter and default parameters.
    #[must_use]
    pub const fn new(nt_type: Neurotransmitter, efficacy: f64) -> Self {
        Self {
            neurotransmitter: nt_type,
            receptor_density: 1000.0,
            efficacy,
            cleft_width: 20.0, // Typical synaptic cleft width
            open_fraction: 0.0,
        }
    }

    /// Update the state of the postsynaptic receptors based on Neurotransmitter concentration and time step.
    ///
    /// Uses simple first-order kinetics to model receptor binding and unbinding dynamics.
    /// dr/dt = alpha * \[NT\] * (1 - r) - beta * r
    /// where r is the open fraction, alpha is the binding rate constant, and beta is the unbinding rate constant.
    pub fn update(&mut self, transmitter_concentration: f64, dt_ms: f64) {
        let alpha = 5.0; // Binding rate constant
        let beta = 0.18; // Unbinding rate constant

        let dr = (alpha * transmitter_concentration)
            .mul_add(1.0 - self.open_fraction, -(beta * self.open_fraction))
            * dt_ms;
        self.open_fraction = (self.open_fraction + dr).clamp(0.0, 1.0);
    }

    /// Computes the ionic current passing through this synapse.
    #[must_use]
    pub fn get_current(&self, v_membrane: f64) -> f64 {
        // Reverse potential based on receptor type
        let v_reverse = match self.neurotransmitter {
            Neurotransmitter::Glutamate => 0.0, // Excitatory
            Neurotransmitter::Gaba => -70.0,    // Inhibitory
            _ => -20.0, // For modulatory Neurotransmitters, we can assume a neutral effect
        };

        // Maximal conductance modified by efficacy and open channel fraction.
        // Scaled down to biologically plausible pS/nS ranges to avoid voltage explosion.
        let g_max = 1e-10 * self.receptor_density * self.efficacy;
        g_max * self.open_fraction * (v_membrane - v_reverse)
    }
}
