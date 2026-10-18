//! MLSR's public Rust API and text-to-speech implementation.
//!
//! Supertonic synthesis runs from original text to owned Float32 PCM, using
//! verified local assets and an optional persistent Core ML runtime. Language
//! bindings currently expose version queries; synthesis FFI is subsequent work.

#![deny(unsafe_code)]

pub mod audio;
pub mod config;
pub mod engine;
pub mod error;
pub mod job;
pub mod models;
pub mod runtime;
pub mod text;

/// Version of the Rust implementation, shared by all language bindings.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
