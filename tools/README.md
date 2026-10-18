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
see the [synthesis documentation](../docs/internal/supertonic3.md).

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
optional graphs. See [execution variants](../docs/internal/execution-variants.md) for
preparation, parity gates and the experimental fixed-vocoder limitation.

## Explicit Apple Silicon qualification

The normal `check` command stays portable and does not load model weights. The
following commands consume already prepared local assets:

```sh
python3 tools/dev.py validate-reference --reference /path/to/reference
python3 tools/dev.py test-real --bundle /path/to/baseline-bundle \
  --reference /path/to/reference --output /path/to/new-suite \
  --variant-bundle /path/to/optional-variant-bundle --flutter
python3 tools/dev.py benchmark --bundle /path/to/baseline-bundle \
  --output /path/to/new-benchmark
```

Reference validation checks fixture integrity without inference. `test-real`
runs all enabled Rust tests (including opt-in model tests), saved-noise tensor
replay, standalone CLI WAV checks, and installed C/Dart/Go/Python tests. The
optional flags add optimized-model parity/lifecycle and the real Flutter UI
test; a full POC run includes both. It records command exit codes, source and
artifact hashes, and individual evidence receipts. A failed/partial run remains
failed in its report. Choose a new output directory for every run.

The benchmark command defaults to the baseline execution variant. It measures
fresh-process initialization with existing OS caches and warm public-job
latency separately from UI playback and instrumented trace timing. No command
clears system caches or installs toolchains. Reports and model weights stay
outside Git; compact qualification summaries are committed under
`tests/fixtures/supertonic3/`.

`profile_coreml.py --mobius /path/to/mobius --bundle /path/to/variant-bundle
--output /path/to/new-plans` calls the existing Möbius public compute-plan helper.
It verifies pinned models and saves preferred placement/cost estimates. These
are diagnostics, not proof of utilization or physical ANE weight residency.

For runtime evidence, build the `coreml_trace` Rust example with the `coreml`
feature, then run:

```sh
python3 tools/trace_synthesis.py --bundle /path/to/variant-bundle \
  --output /path/to/new-traces
python3 tools/analyze_synthesis_trace.py --capture /path/to/new-traces/capture.yaml \
  --output /path/to/new-traces/analysis.yaml
```

The tool loads models before attaching Instruments, then gates real speech on
Instruments' Darwin start notification. A small development-only helper compiles
the public SDK's `os_signpost` macros; it is not linked into the library or app.
Baseline, fused-eight-step, and CPU-only conditions use actual text and retained
model sessions. Raw traces, command logs, XML exports, PCM hashes, and process
IDs are retained. The analyzer requires every hardware event it attributes to
fit both a speech-request interval and an ANE evaluation interval from the
workload PID. It rejects missing intervals and unexpected counts. Attached traces
can lack model names; generic events establish process execution, not individual
model placement. These are diagnostic traces, not latency benchmarks.
