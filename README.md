# MLSR

A cross-platform text-to-speech library with its main logic in Rust and a shared
C ABI for Dart, Go, and Python.

**Status:** the first Supertonic 3 proof of concept is implemented and
[qualified on an M4 Pro](docs/poc-qualification.md): Rust synthesis and WAV CLI,
verified YAML bundles, a persistent cidre/Core ML runtime, installed
Dart/Go/Python bindings, and a [Flutter speak/stop player](examples/flutter_speak/).
It supports English/Korean, voices M1/F1, and
[bounded segment jobs with cancellation](docs/jobs.md). An experimental
[QNN backend](docs/qnn-runtime.md) now runs the pinned `superfast` preset on
Windows ARM64/X2 Elite through the same Rust jobs and C/Dart/Go/Python APIs.
Its adapter/lifecycle checks pass, but the preset fails the broader canonical
numerical screen; see the [QNN qualification report](docs/qnn-qualification.md).
Other devices, minimum OS versions, and distribution packages need qualification.
Builds never download model weights.

A pinned Supertonic 3 Core ML reference bundle and development-only
[capture/replay tools](tools/supertonic3/README.md) establish the baseline for
the Rust synthesis implementation. The model manifest uses YAML.
See [Supertonic synthesis](docs/supertonic3.md) for the Rust API, CLI, and evidence,
and the [Core ML runtime](docs/coreml-runtime.md) for its ownership contract.

## Layout

```text
crates/mlsr/          Public Rust API and TTS implementation modules
crates/mlsr-cli/      Local bundle verification and text-to-WAV CLI
crates/mlsr-ffi/      Shared C ABI adapter
include/mlsr.h       Generated C header
bindings/dart/       Dart package, generated FFI, and native build hook
bindings/go/         Go module using cgo
bindings/python/     Python package using ctypes
examples/flutter_speak/ macOS Flutter playback and WAV export example
model-manifests/     Model download metadata, checksums, and licenses
tests/fixtures/      Small, shared test inputs
tests/integration/   Tests of the actual C ABI
tools/codegen/       Pinned cbindgen header generator
tools/supertonic3/   Model verification and real reference-fixture capture/replay
tools/check_coreml.py Apple runtime, native callers, and build checks
tools/check_synthesis_bindings.py Installed real-synthesis package qualification
tools/dev.py         Portable checks and explicit real-model/benchmark commands
docs/                Architecture, FFI contract, and platform status
```

Rust applications depend directly on `mlsr`. Other languages call
`mlsr-ffi`, which delegates implementation to `mlsr`. See
[architecture](docs/architecture.md) for the module boundaries.

## Getting started

Install stable Rust, a native C toolchain, and Python 3.11+ for the development
commands. The repository selects stable Rust without changing the global
toolchain. Go checks need Go 1.22+ with cgo; Dart checks need Dart 3.11+.
The production Rust crates require Rust 1.88 or newer.

```sh
# Compile the Rust library and its C ABI.
python3 tools/dev.py build

# Query the Rust API directly.
cargo run -p mlsr --example version

# Synthesize on macOS using an already prepared local bundle (no Python needed).
MACOSX_DEPLOYMENT_TARGET=15.0 cargo run --release -p mlsr-cli --features coreml -- \
  synthesize --manifest model-manifests/supertonic3-coreml.yaml \
  --bundle /path/to/model-bundle --text "Hello from MLSR." \
  --language en --voice M1 --output hello.wav

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
