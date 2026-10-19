# QNN Superfast integration qualification — 2026-10-18

The Windows ARM64 integration passes adapter, ownership, lifecycle, HTP execution
and native binding checks. **The Superfast preset remains experimental pending
subjective listening qualification.** Six canonical numerical comparisons exceed
the recorded thresholds; those results are retained as diagnostics and do not
determine perceptual quality. The [machine-readable snapshot](../tests/fixtures/supertonic3/qnn-summary.yaml)
records measurements, exact hashes and retained evidence locations.

## Target and configuration

Snapdragon X2 Elite X2E84100, SoC 88 / HTP V81, Windows 11 Home ARM64
10.0.28000, QAIRT 2.50.40.260831, ExecuTorch 1.5.0 at
`f7140a46ff38e919c557d45b102d3ff26097c8c9`, Rust 1.98.1. Four immutable PTE
programs total 212,190,568 bytes. The SDK/delegate hashes were verified against
[the manifest](../model-manifests/supertonic3-qnn-superfast.yaml).

Decoded compile specifications agree across all four programs: HTP backend,
FP16, burst performance mode, unsigned PD, requested VTCM 8 MB, offline prepare,
profiling off, shared buffers off, weight sharing off and multi-context mode off.
Requested VTCM is a compiler setting, not a measurement of physical memory.
The local target preparation adds SC8480XP/V81 to ExecuTorch's schemas;
Windows backend registration is already implemented by the pinned upstream revision.

Windows used the Balanced power scheme. A post-run power sample reported AC
connected, charging, 83%; earlier samples reported 81% and 85%. Power and
temperature were not controlled. Thermal WMI access was unavailable; energy,
temperature and sustained thermal behavior were not measured. No concurrent QNN
workload was launched by these checks during latency measurements.

## Numerical evidence

The shared English/Korean M1/F1 corpus covers normalization, punctuation and
decomposed Hangul. Added cases exercise true latent lengths 3/4/95/96/97 and
128/129 text positions. Invalid text/language/voice, zero speed and seven steps
are rejected. Boundaries use calibrated speaking rates, not naturalness tests.

All **23 cases × four passes** pass the adapter comparison: twelve successful
cases and eleven rejected cases, twice with each output-reuse mode. The 48
successful captures compare every tensor byte, including frontend IDs/masks,
actual capacity-128 noise, duration, TE, every one of eight VE outputs, full
vocoder input/output, PCM and exact frame counts. Four models remain loaded per
session. All raw captures were subsequently revalidated from their saved bytes.

Canonical comparisons use the pinned original unpadded ONNX models, the same
actual noise and QNN-derived duration. They separately record ONNX's independent
frame count. Each isolated VE comparison uses the QNN step's input; a second
ONNX trajectory measures accumulated drift. Vocoder comparisons use the true,
unpadded latent, explicitly testing the fixed-vocoder difference.

Thresholds were fixed before this corpus run: 5% relative L2 for model stages
and final latent, and 10% magnitude-STFT spectral convergence for final audio.
Waveform L2 is also recorded. No thresholds were widened. Six of twelve cases
fail the screen:

| Case | Final latent relative L2 | Spectral convergence | Failed gate |
| --- | ---: | ---: | --- |
| English M1 basic | 18.84% | 43.48% | Final latent, spectrum |
| English normalization | 4.80% | 12.44% | Spectrum |
| Korean M1 basic | 1.76% | 10.53% | Spectrum |
| Minimum L4 | 0.93% | 3.47% | Isolated vocoder: 5.73% |
| L96 English F1 | 5.03% | 9.98% | Final latent |
| 128 text positions | 11.35% | 27.89% | Final latent, spectrum |

All isolated VE comparisons remain below 1.16% relative L2; errors can accumulate
over the trajectory. Byte-exact adapter agreement locates these failures in the
existing preset/reference behavior. The earlier two-case feasibility screen did
not expose this broader failure set. Raw stages and paired Float32 WAVs,
including every failure, are retained. Listening qualification remains open.

## Device execution and lifecycle

The installed SDK's `qnn-net-run --profiling_level detailed` executed all four
**unchanged extracted contexts** on actual saved speech inputs. Supported
`qnn-profile-viewer` CSVs contain nonzero HTP operator cycles: DP 8.54 million,
TE 31.30 million, VE 68.33 million, vocoder 30.05 million. Every profiled output
matches the QNN reference byte for byte. These separate SDK runs prove device
execution of the pinned contexts; instrumentation timings are not production
synthesis latency. The Rust bridge executes those same PTEs through QNN.

The public Rust job probe checks repeated requests, one active job/Busy, bounded
queues, stalled consumers, late-output discard, exactly one terminal outcome,
cancel/restart, pending-work shutdown, idempotent shutdown, a newly loaded engine
after teardown and PCM surviving release. An atomic sequence changes strictly
around synchronous native `forward`; all twenty cancellation observations remain
inside the same native call across `cancel()`, covering TE, VE step 3 and vocoder.
Five additional cancellations release a producer blocked by a 16-frame queue.
There is no accelerator abort or preemption claim.

Final public-job measurements use a new process/context and warm filesystem
cache. Initialization, including hashes and four model loads, took **999 ms**.
One warmup per language/voice precedes five measured requests. English is one
segment; the Korean text is two. Times include Rust preparation, native calls,
transfers/copies and event consumption; they exclude playback and WAV writing.

| Request | Median first PCM | Median complete | P90 complete |
| --- | ---: | ---: | ---: |
| EN M1 | 90.17 ms | 90.27 ms | 105.74 ms |
| EN F1 | 85.89 ms | 85.96 ms | 95.04 ms |
| KO M1, two segments | 92.35 ms | 183.56 ms | 208.48 ms |
| KO F1, two segments | 86.07 ms | 163.54 ms | 212.92 ms |

Earlier runs measured 67–68 ms and 82–86 ms first PCM and are retained at
`ignore/qnn-jobs-20261018-c.json` and `ignore/qnn-jobs-20261018-final.json`.
The table uses the final binary after native failure cleanup was tightened;
these small uncontrolled samples are not a performance SLA. Cancellation
signaling was at most 0.002 ms at the host timer's resolution. Maximum observed
quiescence was 6.56 ms for TE, 12.52 ms for VE and 19.50 ms for vocoder; cancellation
waits for whichever call is in flight.

Warm host working set was 464,912,384 bytes, private allocation 231,124,992 bytes,
and peak working set 465,625,088 bytes. After shutdown these fell to 26,341,376
and 13,189,120 bytes respectively. These OS process counters exclude accelerator
and driver allocations. The bridge explicitly copies 1,753,092 output bytes per
segment across eleven predictions; Rust readback/PCM/queue copies and SDK staging
are additional. Output backings reuse Rust destinations, not HTP allocations.

## Public interfaces and portable checks

Real ARM64 C ABI, Python 3.14.7, Go 1.27.2/cgo and Dart 3.13.5 native-asset tests
pass. C tests exercise cancellation from another thread and retain an event after
engine destruction. Binding tests cover English/Korean M1/F1, Busy, stopped
consumers, cancellation, errors, reuse and retained audio. All four PCM files
match byte for byte across all three bindings. The Rust WAV CLI matches the
English M1 binding PCM. Go used LLVM-MinGW 20261006/Clang 23.1.3; the C smoke also
passed with MSVC. Dart's consuming-package test builds the QNN native asset using
the repository hook and explicit local bridge prerequisite.

Workspace tests, strict Clippy, generated C/Dart checks (Dart 3.11.1), portable
reference-tool tests, actual-bundle integrity checks and QNN runtime ownership/
recovery tests pass. Windows x64 all-target/all-feature checking passes without
QNN dependencies.
Apple runtime code was updated for explicit Int64 rejection; macOS/iOS inference
was not rerun from this Windows host. CI execution and distributable native
packages remain separate work.

## Model delivery integration — 2026-10-18

Merge `929d051` integrates local `main` at `82efec6` into `backend/qnn`.
The [model delivery snapshot](../tests/fixtures/supertonic3/qnn-model-delivery-summary.yaml)
records this build's hashes and results separately from the earlier measurements
above. Model storage retains feature bit 4; QNN moves to bit 8. The native C
smoke reports mask 13 (jobs, model store, QNN).

The explicit experimental catalog verifies QNN's backend and original manifest
hash. A real Rust store installation produces matching English/Korean M1/F1 PCM
through `Engine::from_model` and direct-bundle loading. Removing the caller's
model handle leaves the engine's lease intact; removal is blocked until engine
drop joins native shutdown. Retained PCM survives subsequent model removal.
Wrong-policy engine creation releases its temporary ownership correctly.

Dart's real C ABI test also passes when engine creation races caller model
disposal. Store removal stays blocked until engine disposal, and the native
handle count returns to its starting value. C, Python, Go/cgo and Dart native
bindings pass; all four PCM files match each other and the earlier qualification.
The qualification script now includes model-store fixtures and the real leased
QNN engine test in its consuming Dart package.

Validation passed 42 workspace tests, all five ignored QNN runtime tests, strict
Clippy, Windows x64 all-target/all-feature checking, C/Dart generation checks,
Dart analysis, three portable Dart tests, and 46 Python model-tool tests. Eight
workspace tests were initially ignored; the five QNN runtime tests were then run
explicitly. Two portable Dart real-model tests and one model-tool test were skipped;
the Dart real-model paths passed separately in the device suite. Apple inference
and the macOS Flutter adapter were not rerun on this Windows host.

The rebuilt replay passes all **92 adapter outcomes**, including byte-exact
tensors in all 48 successful captures. Canonical thresholds and the same six
failing cases are unchanged. The public job probe also passes, including twenty
in-flight native cancellations and five backpressure cancellations. The manifest,
four PTEs and native bridge are unchanged. The model remains experimental and
outside the builtin catalog. This integration does not refresh the earlier SDK
profiling or establish controlled performance measurements.

New raw evidence is retained under `ignore/qnn-replay-merge-20261018/`,
`ignore/qnn-bindings-merge-20261018/`, `ignore/qnn-jobs-merge-20261018.json`, and
`ignore/qnn-merge-*.log`; their key reports and logs are hashed in the snapshot.

## Developer SDK integration — 2026-10-18

Merge `c321ccb` integrates `main` at `09226ff` into `backend/qnn` and adapts
the new headless developer API. The [SDK snapshot](../tests/fixtures/supertonic3/qnn-sdk-summary.yaml)
records source/artifact hashes, exact checks, toolchains and retained evidence.
SDK bit 8 and Apple output bit 16 keep their main-branch meanings; QNN now uses
bit 32, and the real C smoke reports mask 45. Rebuild experimental consumers and
their native library together when updating from the earlier branch.

The real QNN SDK passes C, Python, Go and Dart checks. All four EN/KO M1/F1
cases remain byte-exact across SDK WAV payloads, legacy binding PCM and the
previous qualification. The three public SDK examples also export the same
70,546-frame English/M1 PCM. The consuming Dart project builds its native asset
through the QNN hook. Python and Go tests use their source packages; this pass
does not claim installed-wheel or relocated-Go-package qualification.

The Rust device test covers lazy cancellation, render/WAV parity, fixed-shape
failure context and recovery, resumed plans with original segment seeds, and
store lease release at facade close. Dart also checks facade creation racing
caller model disposal and successful store removal only after facade disposal.
Python and Go check verified-model readiness and independent facade ownership.
Native handles return to their starting counts, and owned PCM survives closure.

QNN voices and structured failures now identify `qnn`; qualified languages remain
empty. Unsupported step counts fail before admission. Requests for another
model cannot replace current speech. Portable tests exposed a shared seek race
that published a new position before its sentence highlight; both now update
together. The existing seek regression passed ten consecutive runs after the fix.

Validation passed **85 workspace tests**, strict all-target/all-feature Clippy,
Windows x64 cross-checking, generated C/Dart checks, Dart analysis/formatting and
46 model-tool tests. Nine workspace tests were initially ignored; the five QNN
runtime tests and the new QNN SDK test then passed explicitly on hardware. The
binding suites passed 12 Python, nine Go and nine Dart tests, with each suite's
Apple playback test skipped on Windows. One model-tool test was also skipped.

The full replay again passes **92 adapter outcomes**; all **48 successful raw
captures** were independently revalidated byte for byte. Public jobs pass,
including twenty in-flight cancellations and five backpressure cancellations.
The same **six of twelve canonical comparisons fail** at the unchanged 5% stage/
final-latent and 10% spectral thresholds. The manifest, four PTEs and bridge are
unchanged. QNN remains experimental and outside the builtin qualified catalog.

Windows native speech output returns `unsupported_capability`. This pass adds
no Windows output backend, new QAIRT profiles, controlled performance claims or
native package publication. macOS/iOS Core ML and Flutter execution were not
rerun on this Windows host. Raw evidence is retained under
`ignore/qnn-sdk-bindings-20261018-final/`, `ignore/qnn-sdk-replay-20261018/`,
`ignore/qnn-sdk-jobs-20261018.json` and `ignore/qnn-sdk-*.log`.

## Installed packages and CI — 2026-10-18

The [installed-package snapshot](../tests/fixtures/supertonic3/qnn-installed-summary.json)
qualifies a built Python wheel and a relocated Go package against the real QNN
backend. Consumers, headers, MLSR DLL, bridge and ExecuTorch delegate are copied
outside the checkout. The Python import probe verifies the installed package and
feature mask 45. The embedded ARM64 Python runtime has its own isolated search
path, including for child processes; no source-package path remains. QAIRT and
FastRPC remain explicitly installed runtime prerequisites.

Twelve Python and nine Go tests pass, including lazy jobs, cancellation, cleanup,
verified-model leases and SDK exports. Each suite skips its Apple-only output
test. All four EN/KO M1/F1 SDK WAV payloads and legacy PCM match the earlier
qualification byte for byte; both installed/relocated public examples match too.
The native DLL, bridge, manifest and model artifacts are unchanged. Evidence is
retained in `C:/Users/hocbu/.cache/mlsr/qnn-installed-20261018/`.

[CI run 39177132548](https://github.com/yunho-c/MLSR/actions/runs/39177132548)
passes all nine jobs, including native/package checks on Linux/macOS/Windows,
Flutter fixtures and Core ML/Apple compilation on stable Rust and Rust 1.88.
Apple counter concurrency tests run without opening an audio device; a physical
Apple playback check was not available on this Windows host.

The current quality decision is to retain the existing Superfast artifacts and
use subjective listening evaluation to determine perceptual qualification. The
six numerical-screen failures and unchanged thresholds remain in the historical
snapshots. No numerical pass or listening result is claimed by this package pass.

## Reproduction and retained evidence

Follow [QNN setup and commands](qnn-runtime.md). The YAML snapshot hashes the core
replay, job, profile, binding and environment reports. Retained evidence includes:

- `ignore/qnn-replay-20261018-final/`: complete reference/Rust/ONNX stages, per-case
  frame decisions, rejection outcomes, paired WAVs and failed numerical checks.
- `ignore/qnn-profile-20261018-final/`: extracted contexts, SDK logs, metadata and
  HTP profiling CSVs for all four graphs.
- `ignore/qnn-jobs-20261018-release.json` and its `.log`: public Rust measurements
  and lifecycle evidence. Earlier attempts and measurements remain alongside it.
- `ignore/qnn-bindings-20261018-final/`: C, Python, Go and Dart logs, build inputs,
  package lockfile, native-library hash and matching PCM.
- `ignore/qnn-cli-release.json` and its `.wav`/`.log`: CLI synthesis and byte
  comparison with binding PCM. `ignore/qnn-final-*.log` retains portable checks
  and the four real-device ownership/recovery tests.
- `ignore/qnn-environment-20261018-release.json` and
  `ignore/qnn-profile-20261018-final/compile-options.json`: environment snapshots
  and decoded PTE options. Earlier captures remain alongside them.

The next model-quality work is subjective listening evaluation of the existing
pinned preset; paired QNN/canonical WAVs are retained with the numerical evidence.
This integration provides reproducible cases for that evaluation. Unsupported
steps/shapes/variants fail explicitly; no fallback or replacement preset is used.
