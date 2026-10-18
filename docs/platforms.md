# Platform status

Cross-platform support requires both a portable API and a working inference
runtime on each target. No inference runtime or TTS model is integrated yet.

| Target | Scaffold configuration | Validation |
| --- | --- | --- |
| macOS arm64 | Rust shared library; C, Dart, Go, Python | Local validation during initialization |
| macOS x64 | Host builds supported by Dart hook and development tooling | Not locally tested |
| Linux x64, glibc | Native host builds; CI job configured | Awaiting CI run |
| Linux arm64, glibc | Native host builds configured | Not locally tested |
| Windows x64, MSVC Rust | Native host builds; CI job configured | Awaiting CI run |
| Windows arm64, MSVC Rust | Dart target mapping configured | Not locally tested; C/Go toolchain needs qualification |
| Android / iOS | Future target-specific builds and packaging | Not implemented |
| Browser / WebAssembly | Future dedicated binding and compatible runtime | Not implemented |

The CI workflow checks Rust, the C header/ABI, all three bindings, and a Python
wrapper wheel on macOS, Linux, and Windows. Configuring a CI job does not establish
that it has passed. Current tests exercise version queries only.

## Toolchains

- Production Rust crates declare Rust 1.88+ and edition 2024. Development uses
  the stable channel; CI also checks the production crates and Core ML feature
  on 1.88.0.
- Python package: Python 3.10+. Development commands: Python 3.11+.
- Go package: Go 1.22+, cgo enabled, and a compatible native C toolchain.
- Dart package: Dart 3.11+, Cargo, and the Rust target for the host.
- Regenerating Dart bindings: libclang, recommended version 18 or newer.

The Dart hook rejects cross-compilation and static-only linking requests rather
than silently bundling a host library. Linux packaging currently assumes glibc;
musl builds and compatibility baselines require separate qualification. Windows
CI uses MSVC Rust output and GCC/MinGW for C/cgo consumers of the DLL.

GPU/NPU acceleration, runtime dependency bundling, supported minimum OS versions,
and release artifacts must be documented alongside each implemented backend.

## Core ML build boundary

`cargo build -p mlsr-ffi --features coreml` enables the pinned `cidre` bindings
on macOS/iOS. Default builds and non-Apple targets do not depend on Apple
frameworks. Both shared and static C libraries are built. The runtime adapter
is being implemented; this feature alone does not provide synthesis.

Use `MACOSX_DEPLOYMENT_TARGET=15.0` and `IPHONEOS_DEPLOYMENT_TARGET=18.0` for
Core ML builds, matching the pinned model manifest. iOS device and simulator
compile/link jobs are configured separately from actual device qualification.
Rust 1.88 is the minimum declared by cidre 0.29.0. Debug stripping is explicitly
disabled because affected Rust/LLVM toolchains produce dylibs rejected by the
macOS 27 loader ([upstream issue](https://github.com/rust-lang/rust/issues/157750)).
No Swift runtime adapter or separate MLSR system-binding crate is needed.
