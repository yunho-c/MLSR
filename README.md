# MLSR

A cross-platform text-to-speech library with its main logic in Rust and a shared
C ABI for Dart, Go, and Python.

**Status:** the [developer SDK v0](docs/developer-sdk.md) is implemented on top of
the qualified Supertonic3 Core ML baseline. Rust, C, Dart, Go and Python expose
speech controllers, lazy synthesis, bounded rendering,
native Apple playback, retained seeking and original-text reader alignment. The
[Flutter reader](examples/flutter_speak/) uses these shared APIs. Start with the
[documentation](docs/README.md) for setup and API guides.

Real inference/output qualification targets macOS Apple Silicon, English/Korean
and voices M1/F1. The experimental Windows ARM64 [QNN Superfast backend](docs/qnn-runtime.md)
supports the headless SDK with the same voices, lazy jobs and WAV export;
its [subjective listening qualification is pending](docs/qnn-qualification.md).
Native Windows playback, physical iOS, other devices/minimum OS versions and
published native distribution packages need separate work and qualification.
See [platform support](docs/platforms.md) for the current matrix and requirements.
Builds never download model weights.

Flutter applications can [prepare a model asset package](docs/model-delivery.md)
and open `AppModels.supertonic3` without runtime model paths. Direct bundled
loading and verified installation support offline use on macOS arm64. Model
identity, payload hashes and licensing are recorded in YAML manifests.

## Layout

```text
crates/mlsr/          Public Rust API and TTS implementation modules
crates/mlsr-cli/      Local bundle verification and text-to-WAV CLI
crates/mlsr-ffi/      Shared C ABI adapter
crates/mlsr-playback/ Optional native output and playback clock adapter
include/mlsr.h       Generated C header
bindings/dart/       Dart package, generated FFI, and native build hook
bindings/flutter/    Flutter asset discovery and model installation adapter
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
docs/                Public API guides, model setup, and platform support
docs/internal/       Maintainer architecture, experiments, and qualification records
```

Rust applications depend directly on `mlsr`. Other languages call
`mlsr-ffi`, which delegates implementation to `mlsr`.

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

## Maintainer documentation

The [internal documentation index](docs/internal/README.md) covers architecture,
runtime design, implementation checklists and qualification evidence. The
[model tools](tools/supertonic3/README.md) capture and replay the pinned reference
used to validate synthesis changes.

## License

Apache-2.0. See [LICENSE](LICENSE). Each language package includes a copy so its
license is available when distributed independently.
