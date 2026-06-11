//! This module is separated into its own crate to enable simple dynamic linking for Brained, and should not be used directly

/// `use brained::prelude::*;` to import common components.
pub mod prelude;

pub use brained_neuron as neuron;
