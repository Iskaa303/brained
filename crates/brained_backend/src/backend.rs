/// The core Backend trait.
/// Defines the associated types for the backend's primitive data structures.
pub trait Backend: Clone + Send + Sync + 'static {
    /// The device associated with this backend (e.g., a CPU thread pool, or a GPU device queue).
    type Device: Clone + Default + Send + Sync + std::fmt::Debug;

    /// Primitive array or buffer holding `f32` data.
    type FloatArray: Clone + Send + Sync + std::fmt::Debug;

    /// Primitive array or buffer holding `u32` data.
    type IntArray: Clone + Send + Sync + std::fmt::Debug;
}
