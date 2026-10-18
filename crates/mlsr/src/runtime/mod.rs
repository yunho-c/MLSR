//! Inference execution and target-specific integration.
//!
//! Core ML is the first backend. Its optional dependency is enabled by the
//! `coreml` feature on macOS and iOS only. Model execution is being implemented.

mod error;
mod tensor;

pub use error::{Error, Result};
pub use tensor::{DType, HostTensor, TensorData};

#[cfg(all(feature = "coreml", any(target_os = "macos", target_os = "ios")))]
#[allow(unsafe_code)] // Audited scoped buffer copies live only in this module.
#[allow(dead_code)] // Adapter primitives are connected to the session in task 3.
mod coreml;
