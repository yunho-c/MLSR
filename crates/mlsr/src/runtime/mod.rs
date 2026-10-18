//! Inference execution and target-specific integration.
//!
//! Core ML is the first backend. Its optional dependency is enabled by the
//! `coreml` feature on macOS and iOS only. Sessions own native models/tensors on
//! a persistent worker, with explicit host transfers and optional output reuse.

mod error;
mod session;
mod tensor;

pub use error::{Error, Result};
pub use session::{ComputePolicy, Model, OutputBuffer, Session, SessionStats, Tensor};
pub use tensor::{DType, HostTensor, TensorData};

/// Native runtime required by a verified model bundle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// Apple Core ML compiled models.
    CoreMl,
    /// ExecuTorch programs delegated to Qualcomm HTP.
    QnnHtp,
}

#[cfg(all(feature = "coreml", any(target_os = "macos", target_os = "ios")))]
#[allow(unsafe_code)] // Audited scoped buffer copies live only in this module.
mod coreml;

#[cfg(all(feature = "qnn", target_os = "windows", target_arch = "aarch64"))]
#[allow(unsafe_code)] // Private, synchronous C boundary; storage stays on one worker.
mod qnn;
