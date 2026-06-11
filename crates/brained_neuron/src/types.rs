//! Basic types and chemicals.

/// Supported Neurotransmitters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Neurotransmitter {
    /// Excitatory Neurotransmitter.
    Glutamate,
    /// Inhibitory Neurotransmitter.
    Gaba,
    /// Modulatory Neurotransmitter involved in reward and motivation.
    Dopamine,
    /// Modulatory Neurotransmitter involved in mood regulation.
    Serotonin,
}

/// Ion concentrations.
/// Values are in millimoles per liter (mM).
#[derive(Debug, Clone, Copy)]
pub struct IonConcentrations {
    /// Sodium ion concentration.
    pub sodium: f64,
    /// Potassium ion concentration.
    pub potassium: f64,
    /// Calcium ion concentration.
    pub calcium: f64,
    /// Chloride ion concentration.
    pub chloride: f64,
}

impl Default for IonConcentrations {
    /// Standard ion concentrations for mammals.
    fn default() -> Self {
        Self { sodium: 15.0, potassium: 150.0, calcium: 0.0001, chloride: 19.0 }
    }
}

impl IonConcentrations {
    /// Baseline extracellular ion concentrations.
    #[must_use]
    pub const fn extracellular_baseline() -> Self {
        Self { sodium: 145.0, potassium: 5.0, calcium: 2.0, chloride: 125.0 }
    }
}
