# Platform status

Cross-platform support requires both a portable API and a working inference
runtime on each target. Supertonic text-to-PCM synthesis is implemented through
Core ML and qualified on macOS arm64, including the public C ABI and installed
Dart/Go/Python synthesis packages and a Flutter playback example. The
[POC qualification](internal/poc-qualification.md) records the exact host, artifacts,
timing boundaries, and remaining device/release gates.

The recommended model is now the [T=384 FP16 Core ML release](internal/supertonic3-fp16-promotion.md).
Its native replay and Flutter checks are recorded separately; the older POC's
INT4 benchmarks and ANE traces do not establish FP16 performance.

| Target | Scaffold configuration | Validation |
| --- | --- | --- |
| macOS arm64 | Rust engine/CLI; shared/static libraries; optional Core ML; Flutter player | Full text-to-PCM parity, installed C/Dart/Go/Python checks, real Flutter playback/cancel/export, release app build/signature checks, warm benchmark and process-attributed ANE traces on M4 Pro/macOS 27.0.1 |
| macOS x64 | Host builds; optional Core ML | Cross-compiled/linked locally; inference not executed |
| Linux x64, glibc | Native host builds; CI job configured | Awaiting CI run |
| Linux arm64, glibc | Native host builds configured | Not locally tested |
| Windows x64, MSVC Rust | Native host builds; CI job configured | All-target/all-feature Rust cross-check passed; native execution and CI run unverified |
| Windows arm64, MSVC Rust | Dart target mapping configured | Not locally tested; C/Go toolchain needs qualification |
| iOS arm64 / simulator | Core ML shared/static artifacts, minimum iOS 18 | Cross-compiled/linked locally; device execution and app packaging unqualified |
| Android | Future runtime and target-specific packaging | Not implemented |
| Browser / WebAssembly | Future dedicated binding and compatible runtime | Not implemented |

The CI workflow checks Rust, the C header/ABI, all three bindings, and a Python
wrapper wheel on macOS, Linux, and Windows. Configuring a CI job does not establish
that it has passed. Ordinary package tests exercise version queries and portable
ABI failures. Explicit model tests exercise synthesis, cancellation and cleanup;
`tools/check_synthesis_bindings.py` runs them outside the checkout on Apple Silicon.

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
frameworks. Both shared and static C libraries are built. The Rust runtime
executes supplied tensor graphs; the Rust `Engine` assembles the Supertonic
pipeline and exposes jobs through the shared C ABI.

Use `MACOSX_DEPLOYMENT_TARGET=15.0` for macOS or
`IPHONEOS_DEPLOYMENT_TARGET=18.0` for iOS Core ML builds, matching the pinned
model manifest. Avoid setting the iOS variable for host C compiler checks.
iOS device and simulator
compile/link jobs are configured separately from actual device qualification.
Rust 1.88 is the minimum declared by cidre 0.29.0. Debug stripping is explicitly
disabled because affected Rust/LLVM toolchains produce dylibs rejected by the
macOS 27 loader ([upstream issue](https://github.com/rust-lang/rust/issues/157750)).
No Swift runtime adapter or separate MLSR system-binding crate is needed.

`python3 tools/check_coreml.py` tests real CPU Core ML using the tiny checked-in
fixture, including C callers linked against static/shared libraries and ctypes
calls from foreign threads. `--cross` also builds Intel macOS and iOS device/
simulator artifacts; `--toolchain 1.88.0` selects an already installed toolchain.
It sets deployment targets explicitly and verifies their Mach-O metadata.
The exported probe is a development
example, separate from the production C ABI. Native static consumers must link
the Apple frameworks and compiler runtime shown in that tool; an archive alone
does not carry Rust's native linkage metadata to a C/Xcode linker.

The initial Rust runtime qualification ran on M4 Pro/macOS 27.0.1 with Rust 1.95
and Xcode SDKs macOS 27.0/iOS 27.1. Rust 1.88 host tests and foreign-caller checks
are also exercised locally. The final POC additionally records warm public-job
performance, public compute plans, and process-attributed ANE execution with a
CPU-only control on this host. These do not establish physical ANE residency or
execution placement for every model. Actual macOS 15/iOS 18 runtime execution,
physical iOS devices, and release distribution remain separate gates.
