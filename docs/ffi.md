# C ABI contract

`include/mlsr.h` is generated from `crates/mlsr-ffi` and is the shared contract
for Dart, Go, Python, and direct C consumers. The Rust crate builds a `cdylib`:
`libmlsr_ffi.dylib` on macOS, `libmlsr_ffi.so` on Linux, and `mlsr_ffi.dll` on
Windows. It also builds a static library (`.a` or `.lib`). Static native consumers
must link the required system libraries/frameworks; see the Core ML build tool.
The macOS build uses an `@rpath` install name and reserves Mach-O header space
so app packagers can relocate the library and rewrite its load commands.

## Current API

| Function | Contract |
| --- | --- |
| `uint32_t mlsr_abi_version(void)` | Returns compatibility version `1` |
| `const char *mlsr_version(void)` | Returns a borrowed, non-null, NUL-terminated UTF-8 version string |

Both functions are thread-safe, allocate nothing, and cannot unwind. The version
string remains valid while the native library is loaded. Callers must not modify
or free it. The bindings copy its contents into their own string types.

Bindings check the ABI version before interpreting the rest of the interface.
It is independent of the package's semantic version. Preserve existing symbols
and signatures for compatible additions; breaking ABI changes require a new ABI
version and binding updates. Version 1 currently promises only these queries.

## Rules for future engine APIs

- Represent engines, jobs, and owned audio with opaque handles and matching
  release functions. Define null handling and failure outputs for every call.
- Explicitly document lifetime, mutation, thread affinity, concurrent calls,
  cancellation, and whether destruction waits for running work.
- Keep allocation and deallocation in the same library. Never require callers
  to use their allocator to free Rust-owned memory.
- Use C-compatible layouts and explicit lengths. Do not export Rust strings,
  vectors, references, `Result`, futures, or trait objects.
- Translate failures into status codes with owned diagnostics. Keep panic
  unwinding within Rust; recovery from a caught panic must not reuse potentially
  inconsistent engine state. An abort cannot be recovered by a binding.
- Transfer audio in chunks. A pull-based job/chunk API is the initial design
  direction; add callback APIs only with defined thread and lifetime contracts.
- For operations retaining data after return, copy caller data into Rust-owned
  storage or define explicit ownership transfer. Respect cgo's pointer retention
  rules and each language's garbage collector.

These are constraints for future synthesis FFI. The synchronous Rust engine
already produces owned PCM, but no engine/job/audio symbols are exported in this
C ABI yet. Add ABI and installed-package tests as that interface is introduced.
