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

- Production Rust crates declare Rust 1.85+ and edition 2024. Development uses
  the stable channel; CI also checks the two production crates on 1.85.0.
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
