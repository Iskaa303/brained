//! Axonal action potential propagation and output Neurotransmitter vesicle venting.

/// Models the signaling output structure of a neuron.
#[derive(Debug, Clone)]
pub struct Axon {
    /// Total length of the axon segment (mum).
    pub length: f64,
    /// Diameter of the axon (mum).
    pub diameter: f64,
    /// Toggles whether myelin sheaths protect this axon, enabling saltatory node jumps.
    pub is_myelinated: bool,
    /// Quantity of Neurotransmitters stored in the terminal vesicles ready for release.
    pub vesicle_pool: f64,
    /// Time remaining for the action potential to reach the physical axon terminals (ms).
    pub propagation_delay_buffer: Vec<f64>,
}

impl Axon {
    /// Instantiates an axon segment with configurable structural features.
    #[must_use]
    pub const fn new(length: f64, diameter: f64, is_myelinated: bool) -> Self {
        Self {
            length,
            diameter,
            is_myelinated,
            vesicle_pool: 1.0, // 100% capacity baseline
            propagation_delay_buffer: Vec::new(),
        }
    }

    /// Computes conduction velocity (m / s).
    ///
    /// Myelinated axon speeds scale roughly linearly with diameter (6.0 * d),
    /// whereas unmyelinated paths scale with the square root (sqrt(d)).
    #[must_use]
    pub fn conduction_velocity(&self) -> f64 {
        if self.is_myelinated { 6.0 * self.diameter } else { 1.2 * self.diameter.sqrt() }
    }

    /// Queues an action potential tracking buffer delay for propagation latency.
    pub fn queue_spike(&mut self) {
        let velocity = self.conduction_velocity(); // meters per second
        let length_meters = self.length * 1e-6;
        let delay_ms = (length_meters / velocity) * 1000.0;

        self.propagation_delay_buffer.push(delay_ms);
    }

    /// Updates propagation timers and returns `true` if Neurotransmitter vesicles are vented this tick.
    pub fn update(&mut self, dt_ms: f64) -> bool {
        let mut spike_reached_terminal = false;

        // Age out active propagation signals
        self.propagation_delay_buffer.iter_mut().for_each(|timer| *timer -= dt_ms);

        // Check if any spike reached the end of the line
        if let Some(&first_timer) = self.propagation_delay_buffer.first()
            && first_timer <= 0.0
        {
            self.propagation_delay_buffer.remove(0);
            if self.vesicle_pool > 0.1 {
                self.vesicle_pool -= 0.05; // Deplete pooled vesicles slightly
                spike_reached_terminal = true;
            }
        }

        // Slow metabolic baseline vesicle replenishment
        if self.vesicle_pool < 1.0 {
            self.vesicle_pool = 0.01f64.mul_add(dt_ms, self.vesicle_pool).min(1.0);
        }

        spike_reached_terminal
    }
}
