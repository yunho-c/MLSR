//! MLSR's public Rust API and text-to-speech implementation.
//!
//! Version information and a persistent, optional Core ML tensor runtime are
//! available. The Supertonic text-to-PCM pipeline is the next integration step.

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
