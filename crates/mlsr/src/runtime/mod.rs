//! Inference execution and target-specific integration.
//!
//! Core ML is the first backend. Its optional dependency is enabled by the
//! `coreml` feature on macOS and iOS only. Model execution is being implemented.

mod error;
mod session;
mod tensor;

pub use error::{Error, Result};
pub use session::{ComputePolicy, Model, OutputBuffer, Session, SessionStats, Tensor};
pub use tensor::{DType, HostTensor, TensorData};

#[cfg(all(feature = "coreml", any(target_os = "macos", target_os = "ios")))]
#[allow(unsafe_code)] // Audited scoped buffer copies live only in this module.
mod coreml;
