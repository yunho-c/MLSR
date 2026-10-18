# Supertonic 3 baseline tools

These development-only tools implement milestone 1: pin published model bytes,
prepare a verified local bundle, and capture/replay real reference fixtures.
The Rust runtime replay below additionally qualifies milestone 2. Public
Dart/Go/Python synthesis and Rust text-to-PCM orchestration are subsequent work.

The baseline is the published **v2** duration predictor and text encoder,
v2 bucketed INT4 VectorEstimators, and the unchanged FP16 vocoder. The v2 models
fix padding behavior; their names and hashes must not be substituted with v1
models or locally edited packages. New fusion/activation-quantization experiments
are outside this baseline.

## Dependencies

Use an existing Python 3.11+ interpreter. No command creates an environment or
installs packages automatically.

- Manifest, file-integrity, and portable tests: `requirements.txt` (PyYAML/NumPy).
- Real inference: macOS Apple Silicon and `requirements-coreml.txt`, including
  Core ML Tools 9.0. The first captured run used an existing Python 3.12 environment.
- Regenerating upstream frontend expectations: an existing interpreter with
  PyYAML, NumPy, and ONNX Runtime, because upstream `helper.py` imports it. That
  command does not execute ONNX models.

If dependencies are missing, install the appropriate requirements into the
development interpreter you have selected. These packages are not MLSR runtime
or language-binding dependencies.

Run commands from the MLSR repository root. Below, `python3` means the selected
existing interpreter; capture/replay specifically need the Core ML dependencies.

## Validate and prepare assets

```sh
python3 -m tools.supertonic3 validate-manifest
python3 tools/dev.py test-model-tools

python3 -m tools.supertonic3 prepare \
  --source "$HOME/.cache/fluidaudio/Models/supertonic-3" \
  --source ../supertonic-3-coreml \
  --output "$HOME/.cache/mlsr/models/supertonic3-coreml-int4-v1"

python3 -m tools.supertonic3 verify-bundle \
  --bundle "$HOME/.cache/mlsr/models/supertonic3-coreml-int4-v1"
```

Preparation searches the supplied roots for **matching bytes**, not just matching
filenames. It verifies every payload before use. On macOS/Linux, the default is
to symlink verified files, avoiding a second weight copy. Windows defaults to
copies; `--mode copy` is also available elsewhere. Keep linked sources in place.
Later source changes cause verification to fail before capture/replay.

Preparation never overwrites an existing output. For an existing bundle, use
`verify-bundle`. It rejects unlisted files inside compiled model directories,
apart from Finder metadata. Extra unrelated models in a supplied source root
are not imported. No sibling repository or shared model cache is modified.

Downloading is an explicit alternative:

```sh
python3 -m tools.supertonic3 prepare --download \
  --output /path/to/new/model-bundle
```

Only missing files are fetched from revision-pinned URLs, and every downloaded
file must match its size and checksum. Model loading and synthesis never fetch
assets. The complete locked payload is approximately 161.5 MiB.

## Capture real references

```sh
python3 -m tools.supertonic3 check-frontend \
  --bundle /path/to/model-bundle

python3 -m tools.supertonic3 capture \
  --bundle /path/to/model-bundle \
  --output /path/to/new/reference-capture

python3 -m tools.supertonic3 verify-reference \
  --reference /path/to/reference-capture
```

The default compute policy is `cpu_and_neural_engine`; `--compute-policy cpu_only`
is available for a separate control capture. A capture stores the policy and
Python, NumPy, Core ML Tools, Unicode, hardware, and OS versions. Reference tools
validate tensor interfaces before inference and save their own source hashes.

The corpus contains 14 real-audio cases and five rejected-input cases. Every
audio case saves actual noise, all model inputs/intermediates, and Float32 PCM.
See the [fixture contract](../../tests/fixtures/supertonic3/README.md). The full
initial capture is approximately 45.4 MiB. Keep large captures and weights outside
Git; only small metadata/expectations belong in `tests/fixtures/`.

A failed capture is marked with `capture-incomplete.yaml`; it cannot pass
verification. Its partial evidence is retained. Use a new output directory for
the next attempt. Capture duration includes loading, inference, and artifact I/O
and is not a synthesis benchmark.

## Replay and record evidence

```sh
python3 -m tools.supertonic3 replay \
  --bundle /path/to/model-bundle \
  --reference /path/to/reference-capture \
  --output /path/to/new/replay.yaml \
  --atol 0 --rtol 0

python3 -m tools.supertonic3 summary \
  --reference /path/to/reference-capture \
  --replay-report /path/to/replay.yaml \
  --output /path/to/new/reference-summary.yaml
```

Replay runs a fresh model pipeline using the saved initial-noise tensors and
recorded effective speaking rates. It checks every intermediate, output shape,
dtype, and frame count. Integer comparisons are exact; default float tolerances
are `atol=1e-5, rtol=1e-5`. The command above requests exact equality. Reports
always retain tolerances, maximum absolute error, RMSE, and exact-match status.
Non-passing comparisons return a nonzero exit status.

The summary can attach a successful replay only when its report identifies this
exact reference capture and covers every case. A summary without that argument
retains `replay: not_run`. Integrity checks, numerical replay, listening, and
performance qualification are reported separately. No perceptual or performance
qualification is implied by a numerical match.

## Replay through the Rust runtime

This section qualifies tensor execution with fixture-supplied model inputs.
For complete Rust synthesis from original text and saved noise, use
[`synthesis_replay`](../../docs/internal/supertonic3.md#persistent-model-pipeline).

Build the development transport and run either allocation mode against a
verified reference. This path needs NumPy/PyYAML, macOS and Xcode, but does not
import coremltools. Cargo never downloads model weights.

```sh
MACOSX_DEPLOYMENT_TARGET=15.0 cargo build --locked --release \
  -p mlsr --example coreml_replay --features coreml

python3 -m tools.supertonic3 replay-rust \
  --bundle /path/to/model-bundle --reference /path/to/reference-capture \
  --output /path/to/new/rust-ordinary.yaml

python3 -m tools.supertonic3 replay-rust \
  --bundle /path/to/model-bundle --reference /path/to/reference-capture \
  --output /path/to/new/rust-backings.yaml --output-backings
```

Each invocation defaults to two passes and byte-exact comparison (`--atol 0
--rtol 0`). The same Rust session loads six models once and serves all 28 cases.
The second mode retains two output buffers for each VE bucket. The driver
checks successful prediction counts and that per-case tensors are released;
the six persistent backing handles are accounted for separately. Backing reuse
is reported by actual native object identity and is never assumed.

The checked tensors are duration output, text embedding, every denoising state,
trimmed vocoder input, raw waveform, and trimmed PCM: 13 per case at eight steps.
The native TE/VE outputs drive subsequent predictions directly. Host reads in
this diagnostic harness preserve intermediates for comparison; it is not a
latency benchmark. Text preprocessing, initial noise, duration/bucket decisions,
and PCM trim length come from the saved fixtures. This runtime-only report
cannot be attached as evidence of a complete synthesis replay by `summary`.

The YAML report records the executable hash, source snapshot, fixture/manifest
hashes, compute policy, comparisons, buffer reuse, and resource counters. Rebuild
the binary before qualification; a source snapshot alone does not attest that an
arbitrary `--binary` was built from it. Bundles and references are verified before
and after inference. Interrupted/failed runs remain non-passing and preserve a
sibling `.native.log`; existing output paths are never overwritten. `--timeout`
sets the per-request limit (180 seconds by default). Temporary raw tensor files
are removed; model weights and saved references remain untouched.

## Rebuild frontend expectations independently

```sh
python3 -m tools.supertonic3.freeze_frontend \
  --upstream-root /path/to/supertonic \
  --bundle /path/to/model-bundle \
  --output /path/to/new/frontend.yaml
```

This verifies `py/helper.py` against the exact Git revision in the YAML manifest
before importing it. The produced positive expectations come from upstream,
not from `frontend.py`. Review any differences before replacing the tracked
fixture. MLSR adds explicit rejection for empty normalized text, unsupported
characters/languages, and overflow after normalization. It rejects inputs that
existing drivers might truncate to a fixed axis or wrap to uint16.

See [third-party notices](THIRD_PARTY_NOTICES.md) and the
[manifest provenance limitations](../../model-manifests/README.md).
