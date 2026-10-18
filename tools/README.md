# Development tools

Run `python3 tools/dev.py --help` for available commands. The runner uses only the
Python standard library and invokes the existing Cargo, Dart, Go, and C tools.
It does not create a Python virtual environment or change global toolchains.

`codegen/` pins cbindgen and generates the C header explicitly. The Dart generator
is in `bindings/dart/tool/generate_bindings.dart`. Both support checking whether
the committed output is current without overwriting it.

`test-standalone` copies the Dart and Go packages, header, and shared library to
a temporary directory and tests them there. This checks the prebuilt-library
path and catches dependencies on the checkout layout, including macOS library
relocation failures.

`package-python` additionally requires pip, setuptools 77+, and wheel in the
interpreter used to invoke the tool. It installs the built wrapper into a
temporary directory, runs its native integration tests there, and removes that
temporary installation when finished. Native libraries are still supplied
separately; nothing is published.

`supertonic3/` contains development-only YAML manifest validation, verified asset
preparation, and real Core ML reference capture/replay. See its
[README](supertonic3/README.md). `test-model-tools` runs the portable checks
without building Rust, loading weights, or requiring Core ML; it additionally
needs the packages in `supertonic3/requirements.txt`.

`check_coreml.py` tests the optional Apple runtime without downloading weights.
It compiles the tiny affine model with Xcode, runs Rust tests, builds static/shared
native artifacts, and executes real inference from C and Python ctypes callers.
`--cross` additionally links Intel macOS and iOS device/simulator artifacts.
It requires installed targets/toolchains and never installs them automatically.
The same command tests/builds the Core ML CLI. Full Supertonic synthesis replay
uses opt-in external assets via `python3 -m tools.supertonic3.synthesis_replay`;
see the [synthesis documentation](../docs/supertonic3.md).

`check_synthesis_bindings.py` explicitly qualifies real synthesis on macOS arm64:

```sh
python3 tools/check_synthesis_bindings.py --bundle /path/to/verified-bundle \
  --output-dir /path/to/new/qualification
```

It builds the release Core ML library, checks C callers linked both ways, installs
a Python wheel into a temporary directory (no venv), and relocates Go/Dart
packages, headers and native assets outside the checkout. All three run real
English/Korean M1/F1 synthesis, cancellation/backpressure/cleanup tests and their
cancel-then-WAV example. Go tests use the race detector. The verifier compares
PCM bytes across languages, validates WAVs independently with `afinfo`, checks
source/library/model hashes, and saves a YAML report and command log. The output
directory must not exist. `--no-build` uses an already built release library;
it is the caller's responsibility to build matching sources first. This is an
explicit integration test, not a latency benchmark or broad speech-quality study.

`benchmark_synthesis.py` runs the public job API in fresh sequential macOS arm64
processes. It preserves per-case warm first-audio/total samples, process peak RSS,
initialization with existing OS caches, and cancellation signal/quiescence times.
`--variant baseline` needs only the baseline bundle; `--variant all` needs the
optional graphs. See [execution variants](../docs/execution-variants.md) for
preparation, parity gates and the experimental fixed-vocoder limitation.
