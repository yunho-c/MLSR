//! C ABI adapter for the MLSR Rust library.
//!
//! Regenerate `include/mlsr.h` with `cargo run -p mlsr-codegen` after changing
//! exports. Version queries allocate nothing. Synthesis uses owned registry
//! handles, explicit release, and panic-contained C entry points.

mod api;
mod handles;
mod types;
pub use api::*;
pub use types::*;

use std::ffi::c_char;

// A static, NUL-terminated copy of the core crate's version. The workspace keeps
// both package versions aligned; the assertion catches accidental divergence.
const VERSION_C: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");

/// Returns the C ABI compatibility version. Bindings must check it before use.
///
/// Version 1 preserves the original queries and adds the synthesis job API.
// SAFETY: The exported symbol is unique and its signature uses the C ABI.
#[unsafe(no_mangle)]
pub extern "C" fn mlsr_abi_version() -> u32 {
    1
}

/// Returns the library version as a borrowed, NUL-terminated UTF-8 string.
///
/// The pointer remains valid while the library is loaded. Do not modify or free
/// it. This function is thread-safe and never returns a null pointer.
// SAFETY: The exported symbol is unique and its signature uses the C ABI.
#[unsafe(no_mangle)]
pub extern "C" fn mlsr_version() -> *const c_char {
    VERSION_C.as_ptr().cast()
}

#[cfg(test)]
mod tests {
    #[test]
    fn ffi_and_core_versions_match() {
        assert_eq!(super::VERSION_C.strip_suffix('\0'), Some(mlsr::VERSION));
    }
}
