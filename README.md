# MLSR

A cross-platform text-to-speech library with its main logic in Rust and a shared
C ABI for Dart, Go, and Python.

**Status:** initialized development scaffold. The Rust library and all three
bindings expose native version/ABI queries. Model loading, inference, speech
synthesis, streaming, and playback are not implemented yet. No model weights
are downloaded by the build.

## Layout

```text
crates/mlsr/          Public Rust API and TTS implementation modules
crates/mlsr-ffi/      Shared C ABI adapter
include/mlsr.h       Generated C header
bindings/dart/       Dart package, generated FFI, and native build hook
bindings/go/         Go module using cgo
bindings/python/     Python package using ctypes
model-manifests/     Model download metadata, checksums, and licenses
tests/fixtures/      Small, shared test inputs
tests/integration/   Tests of the actual C ABI
tools/codegen/       Pinned cbindgen header generator
tools/dev.py         Portable development commands
docs/                Architecture, FFI contract, and platform status
```

Rust applications depend directly on `mlsr`. Other languages call
`mlsr-ffi`, which delegates implementation to `mlsr`. See
[architecture](docs/architecture.md) for the module boundaries.

## Getting started

Install stable Rust, a native C toolchain, and Python 3.11+ for the development
commands. The repository selects stable Rust without changing the global
toolchain. Go checks need Go 1.22+ with cgo; Dart checks need Dart 3.11+.

```sh
# Compile the Rust library and its C ABI.
python3 tools/dev.py build

# Query the Rust API directly.
cargo run -p mlsr --example version

# Format/lint checks, Rust tests, C ABI smoke, and all three bindings.
python3 tools/dev.py check
```

On Windows, use `python` if `python3` is unavailable. The development C and Go
checks expect GCC/MinGW on PATH alongside the MSVC Rust toolchain. Build
dependencies are cached by Cargo and Dart; no Python virtual environment is
required. See [platforms](docs/platforms.md) for current validation and limitations.

Individual checks are available as `test-c`, `test-python`, `test-go`, and
`test-dart` subcommands. Each builds the native library and supplies its paths to
the selected language's test runner. Examples are colocated with their packages.
`python3 tools/dev.py test-standalone` additionally tests Dart and Go package
copies against relocated headers and native libraries outside the checkout.

## Generated bindings

Edit the exported Rust API, then regenerate both layers:

```sh
python3 tools/dev.py generate
python3 tools/dev.py check-generated
```

The generator uses pinned `cbindgen` and `ffigen` dependencies. Dart generation
also requires libclang; normal builds consume the generated files already in
the repository. The Dart generator uses ffigen's programmatic API.

Commit `Cargo.lock`, `bindings/dart/pubspec.lock`, `include/mlsr.h`, and the
generated Dart bindings along with relevant source changes. Normal Rust builds
do not regenerate or modify source files.

## Python package smoke test

With `pip`, `setuptools>=77`, and `wheel` available in the existing interpreter:

```sh
python3 tools/dev.py package-python
```

This produces a wheel in `target/wheels/` and tests an installed copy outside
the checkout. The initial wheel is a Python-only wrapper that requires a
separately available native library. Native binary bundling and release
publication are not configured. Rust and Dart packages are also marked private
while the API is being established.

## License

Apache-2.0. See [LICENSE](LICENSE). Each language package includes a copy so its
license is available when distributed independently.
