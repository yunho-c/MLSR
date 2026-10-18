//! Inference execution and target-specific integration.
//!
//! Core ML is the first backend. Its optional dependency is enabled by the
//! `coreml` feature on macOS and iOS only. Model execution is being implemented.

#[cfg(all(feature = "coreml", any(target_os = "macos", target_os = "ios")))]
mod coreml;
