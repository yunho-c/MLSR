//! MLSR's public Rust API and text-to-speech implementation.
//!
//! This initial scaffold exposes version information. Model loading and
//! synthesis will be added after the first model and runtime are selected.

#![deny(unsafe_code)]

pub mod audio;
pub mod config;
pub mod engine;
pub mod error;
pub mod models;
pub mod runtime;
pub mod text;

/// Version of the Rust implementation, shared by all language bindings.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
