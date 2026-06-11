//! Somatic integration cell body utilizing Hodgkin-Huxley equations for action potential generation.

use crate::types::IonConcentrations;

/// Gating particles for the Hodgkin-Huxley ion channel formulations.
#[derive(Debug, Clone, Copy)]
pub struct HodgkinHuxleyGating {
    /// Sodium activation gating variable (m).
    pub m: f64,
    /// Sodium inactivation gating variable (h).
    pub h: f64,
    /// Potassium activation gating variable (n).
    pub n: f64,
}

/// The somatic cell body. Integrates incoming currents and drives spike generation.
#[derive(Debug, Clone)]
pub struct Soma {
    /// Membrane potential in millivolts (mV).
    pub v_membrane: f64,
    /// Total membrane surface area (square micrometers).
    pub surface_area: f64,
    /// Membrane capacitance (`C_m`) in microfarads per square centimeter (uF/cm^2).
    pub capacitance: f64,
    /// Intracellular chemical ion concentration pool.
    pub internal_ions: IonConcentrations,
    /// Extracellular fluid chemical ion concentration pool.
    pub external_ions: IonConcentrations,
    /// Current Hodgkin-Huxley channel gate states.
    pub gating: HodgkinHuxleyGating,
}

impl Soma {
    /// Creates a standard soma with typical mammalian equilibrium states.
    #[must_use]
    pub fn new(surface_area: f64) -> Self {
        Self {
            v_membrane: -65.0, // Rest potential in mV
            surface_area,
            capacitance: 1.0,
            internal_ions: IonConcentrations::default(),
            external_ions: IonConcentrations::extracellular_baseline(),
            gating: HodgkinHuxleyGating { m: 0.05, h: 0.6, n: 0.32 },
        }
    }

    /// Updates membrane potential over time step `dt_ms` using Hodgkin-Huxley equations.
    pub fn integrate(&mut self, injected_current: f64, dt_ms: f64) -> bool {
        let v = self.v_membrane;

        // Maximal conductances (mS / cm^2)
        let g_na_max = 120.0;
        let g_k_max = 36.0;
        let g_l = 0.3; // Leak

        // Reversal potentials (mV) deduced from ion gradients
        let e_na = 50.0;
        let e_k = -77.0;
        let e_l = -54.4;

        // Voltage-dependent transition rates (alpha and beta functions)
        // Handle singularities using L'Hopital's rule when v is near -40 and -55 mV
        let alpha_m = if (v + 40.0).abs() < 1e-5 {
            1.0
        } else {
            0.1 * (v + 40.0) / (1.0 - (-(v + 40.0) / 10.0).exp())
        };
        let beta_m = 4.0 * (-(v + 65.0) / 18.0).exp();

        let alpha_h = 0.07 * (-(v + 65.0) / 20.0).exp();
        let beta_h = 1.0 / (1.0 + (-(v + 35.0) / 10.0).exp());

        let alpha_n = if (v + 55.0).abs() < 1e-5 {
            0.1
        } else {
            0.01 * (v + 55.0) / (1.0 - (-(v + 55.0) / 10.0).exp())
        };
        let beta_n = 0.125 * (-(v + 65.0) / 80.0).exp();

        // Calculate derivatives of gating states
        self.gating.m = (alpha_m * (1.0 - self.gating.m) - beta_m * self.gating.m)
            .mul_add(dt_ms, self.gating.m);
        self.gating.h = (alpha_h * (1.0 - self.gating.h) - beta_h * self.gating.h)
            .mul_add(dt_ms, self.gating.h);
        self.gating.n = (alpha_n * (1.0 - self.gating.n) - beta_n * self.gating.n)
            .mul_add(dt_ms, self.gating.n);

        // Calculate active currents (mS/cm^2 * mV = uA/cm^2)
        let i_na = g_na_max * self.gating.m.powi(3) * self.gating.h * (v - e_na);
        let i_k = g_k_max * self.gating.n.powi(4) * (v - e_k);
        let i_leak = g_l * (v - e_l);

        // Convert synaptic current (inward is negative by convention) to current density
        // injected_current is the sum of I_syn.
        let i_syn = injected_current / (self.surface_area * 1e-8);

        // Total outward current flow
        let i_outward = i_na + i_k + i_leak + i_syn;

        // Update voltage: dV/dt = - I_outward / C_m
        let dv = (-i_outward / self.capacitance) * dt_ms;
        let old_v = self.v_membrane;
        self.v_membrane += dv;

        // Action potential threshold cross verification (Spike output signal detected)
        old_v < 0.0 && self.v_membrane >= 0.0
    }
}
