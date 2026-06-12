//! Forces dynamic linking of Brained.
//!
//! Dynamic linking causes Bevy to be built and linked as a dynamic library. This will make incremental builds compile much faster.

// Force linking of the main brained crate
#[expect(
    unused_imports,
    clippy::single_component_path_imports,
    reason = "This links the main brained crate when using dynamic linking, and as such cannot be removed or changed without affecting dynamic linking."
)]
use brained_internal;
