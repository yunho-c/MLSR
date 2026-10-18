//! Optional platform audio output for MLSR.
//!
//! Native output contains no synthesis/session policy or language bindings.
//! Apple graph objects are confined to their creating control thread. Completion
//! and configuration callbacks touch only atomics and never invoke user code.

#[cfg(any(test, all(feature = "apple", target_os = "macos")))]
mod counter;

#[cfg(all(feature = "apple", target_os = "macos"))]
pub mod apple;
