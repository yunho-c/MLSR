# Supertonic Core ML execution variants

The baseline remains the default. Fused VE is opt-in and retains exact numerical
agreement on the qualified corpus. Fixed vocoders are explicitly experimental:
their outputs differ from the flexible baseline even at their exact input sizes.

| Selection | Retained graphs | Routing | Qualification |
| --- | --- | --- | --- |
| `Baseline` / C value `0` | Six baseline models | Individual VE steps; flexible vocoder | Existing baseline parity and lifecycle checks |
| `Fused8` / C value `1` | Baseline plus one fused model | One VE prediction only for eight steps and bucket 128; baseline otherwise | Exact observable tensor/PCM comparisons against the baseline |
| `Fused8ExperimentalVocoder` / C value `2` | Baseline plus fused VE and two vocoders | Same VE policy; fixed vocoder only at true latent length 30 or 32 | Research only: PCM differs; perceptual quality unqualified |

Fixed vocoders never receive padded input as an approximation. Length 31 and
every other unsupported fixed length use the flexible graph. Non-eight-step
requests and buckets 256/512 still use individual VE steps. All graph choices
preserve the requested steps, duration, tokenization, noise and true latent shape.
`SynthesisOutput::route()` reports the choices actually used for a segment.

All selected models load once per engine. The set is bounded at six, seven or
nine models; it does not grow with requests. Baseline output-buffer reuse remains
available. Fused outputs use ordinary native allocation and remain native until
the final latent crop. Retention does not prove physical ANE residency.

## Asset preparation and selection

The checked-in `model-manifests/supertonic3-coreml-variants.yaml` pins every byte
of the archived FluidAudio graphs, transformation provenance and supported
interfaces. The weight blobs match the existing INT4 VE and FP16 vocoder pins.
The Rust loader embeds this description and requires the exact pinned baseline
manifest for optimized execution. The copied `execution-variants.yaml` in a
prepared bundle is an inspection aid; editing it does not override library pins.

Create a new bundle from existing local assets:

```sh
python3 -m tools.supertonic3.variants \
  --baseline-bundle /path/to/baseline-bundle \
  --experiment-dir /path/to/supertonic-fused-ve-experiment \
  --output /path/to/new/combined-bundle
```

The experiment directory must contain the pinned `noboundary/compiled/` and
`fixed-vocoders/` payloads. The tool verifies the source contents first, creates
file symlinks into a new directory, and verifies the result. It performs no model
download or conversion and does not modify the baseline. Keep source targets
immutable while a bundle is in use. An explicit optimized selection with missing
or corrupt assets fails before loading Core ML, rather than silently ignoring
the requested configuration.

In Rust, set `EngineOptions.execution_variant`. In the CLI, use
`--execution-variant fused8` or `--execution-variant fused8-experimental-vocoder`.
The shared C field and Python `execution_variant`, Go `ExecutionVariant`, and
Dart `executionVariant` use the numeric values above. All default to zero.

## Parity and performance boundaries

```sh
MACOSX_DEPLOYMENT_TARGET=15.0 cargo build --locked --release \
  -p mlsr --features coreml --example coreml_synthesis_replay
python3 -m tools.supertonic3.variant_replay \
  --bundle /path/to/combined-bundle --reference /path/to/reference \
  --output /path/to/new/replay

MACOSX_DEPLOYMENT_TARGET=15.0 cargo build --locked --release \
  -p mlsr-ffi --features coreml
python3 tools/benchmark_synthesis.py --bundle /path/to/combined-bundle \
  --variant-report /path/to/replay/report.yaml --output /path/to/new/benchmark
```

Replay covers the original 14 cases, 12 additional exact-length 30/31/32 probes
across English/Korean M1/F1, and three non-eight-step fallback probes. It saves
actual initial noise and paired WAVs. Two passes check model/tensor counts,
routing, all observable tensors and PCM; five invalid-input cases run per
variant. Fused graphs cannot expose their seven internal intermediate states;
the harness compares their inputs, final latent and waveform without inventing
those snapshots.

The report separately records each variant's bit-exact result. Its overall
passing gate requires baseline/fused parity and valid contracts/lifecycles for
all routes. Experimental fixed-vocoder PCM differences remain visible and do not
qualify those vocoders for production. The initial strict all-variant run is
retained as failed evidence. Its eight fixed-vocoder cases changed PCM with
maximum absolute errors ranging from 0.01050 to 0.09340; identical vocoder inputs
isolate the difference to the specialized vocoder execution. These numbers are
not perceptual scores or a quality approval.

The benchmark runs sequential fresh processes and measures the public job API,
including Python/C calls and owned PCM copies. Initialization includes verified
asset loading with existing OS compiler/file caches; it is not first-ever
uncached specialization. Warm first audio, complete generation, per-case real
time factors, cancellation signal/quiescence, and process maximum RSS are
reported separately. Each shape is warmed before timed repetitions. Memory
includes the host process and does not report accelerator allocation.

Cancellation still signals immediately, but an in-flight fused Core ML call
finishes before Rust can reach its next checkpoint. The benchmark records
prediction-triggered cancellation and stopped-consumer cancellation separately.
It does not assume a specific native operation was in flight from a timer alone.

## Further quantization candidates

W8A8 and dense INT8 remain evaluation candidates; they are not selectable
production paths. The FluidAudio archive's `w8a8-study`, `ve-w8a8-study` and
`ve-quantization-forensics` contain useful conversion and evaluation leads.
Their historical timings and quality observations are not MLSR qualification.
Any promotion needs matched inputs and complete synthesis, audio/perceptual
evaluation, memory/startup measurements and actual execution evidence.
Apple likewise describes activation-quantization benefits as dependent on
[execution placement](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html).
