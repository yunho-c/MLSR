# First Core ML proof of concept

The seven-task slice is complete on the tested macOS Apple Silicon host. Rust
owns Supertonic 3 preprocessing, model execution, segmentation, bounded jobs,
cancellation, and PCM delivery. The C ABI, installed Dart/Go/Python packages,
CLI, and Flutter player all exercised real synthesis. The default remains the
six-model baseline; fused eight-step VE is opt-in, and fixed vocoders remain
explicitly experimental.

The compact, hashed evidence index is
[`poc-summary.yaml`](../../tests/fixtures/supertonic3/poc-summary.yaml). Raw reports,
audio, screenshots, traces, and failed attempts are retained under
`~/.cache/mlsr/qualification/poc-v1/`. These are local qualification artifacts,
not files downloaded by builds or committed model weights.

## Host and scope

Qualification ran on Apple M4 Pro, macOS 27.0.1 (26A434), on AC power with
`powermode=0`. Toolchains were Rust 1.95.0, Xcode 27.1 (27A9275), macOS SDK 27.0,
Flutter 3.41.4 / Dart 3.11.1, Python 3.14.8, and Go 1.27.1. Production crates also
passed a Rust 1.88.0 all-target/all-feature check. The macOS deployment target is
15.0; iOS artifacts use 18.0 with SDK 27.1.

The explicit combined suite (`task7-suite-v1`) passed all 16 commands, with the
same 175 recorded source files before and after execution. This document and
its evidence index were added afterward; they do not change runtime behavior.

| Boundary | Evidence |
| --- | --- |
| Reference to Rust | 14 cases, two passes, 672 tensor comparisons at zero absolute/relative tolerance; five invalid-input cases and four seeded production/diagnostic checks |
| Optional graphs | 29 cases × three variants × two passes; baseline and fused VE exact; fixed-vocoder differences recorded and kept experimental; 15 invalid-input checks |
| Rust jobs | Real-model cancellation, backpressure, reuse, segmentation and lifecycle tests, including normally ignored model tests |
| Native/package use | C shared/static callers; relocated installed Python, Go and Dart packages; four English/Korean M1/F1 PCM comparisons exact across bindings; cancel/reuse/WAV examples; Go race detector |
| Flutter playback | Real synthesis and muted AVAudioEngine playback, long-text stop, engine reuse, English/Korean export, playback completion distinct from generation, zero remaining native handles |
| Export integrity | Independent RIFF/Float32 parsing and `afinfo`; Flutter English/Korean PCM exactly matches Python with identical text, voice, steps, speed and seed 0 |
| Build portability | Portable aggregate check, generated bindings, strict Clippy, Rust 1.88 check, Windows MSVC type-check, Intel macOS and iOS device/simulator cross-build/link and deployment metadata checks |

## Public-job timing and memory

`task7-benchmark-v1` measured the baseline through Python's public C ABI wrapper,
from `Engine.start` through copied first audio and terminal completion. Playback
and instrumented trace overhead are excluded. Twelve cases cover English,
Korean, M1/F1, normalization, punctuation, and four calibrated L30/L32 probes;
each has seven warm repetitions, in alternating case order. The raw report
preserves every request and per-case sample.

| Measurement | Result |
| --- | ---: |
| New-process engine initialization | 480.75 ms |
| First request, first PCM / complete | 177.22 / 221.15 ms |
| Warm first PCM, median / maximum over 84 requests | 40.20 / 43.70 ms |
| Warm complete synthesis, median / maximum | 41.93 / 81.88 ms |
| Maximum warm real-time factor | 0.027863 |
| Maximum cancellation-to-quiescence across 12 checks | 3.54 ms |
| Child-process peak RSS | 154,189,824 bytes (147.05 MiB) |

All 84 warm requests synthesized faster than their audio duration. First audio
is available after a segment is vocoded; small delivered PCM chunks do not imply
incremental inference within a segment. The first-request corpus entry contains
two sentences, so its complete time is longer than its first-audio time.

Initialization means a new process and engine using precompiled models and
existing OS specialization/file caches. It is **not** a pristine-install or
cache-cleared measurement: earlier task-6 baseline initialization took 14.34 s.
The single initialization sample cannot establish general startup latency.
RSS is the child process's `/usr/bin/time -l` maximum, including Python and Core
ML; it is not ANE memory. This is one machine/run without a guarantee of no
unrelated system activity. The earlier per-variant comparison remains in
[`variants-summary.yaml`](../../tests/fixtures/supertonic3/variants-summary.yaml).

## Flutter and release artifacts

The [example](../../examples/flutter_speak/README.md) keeps model work in the Dart
adapter isolate and playback in a separate Swift AVAudioEngine adapter. Native
playback schedules at most 22,050 frames plus one pending chunk. The final debug
integration run observed 20,480 scheduled and 4,096 pending frames, a 46.18 ms
maximum UI heartbeat gap, 12.04 ms from stop to playback flush, and 73.90 ms to
controller idle. Those UI timings have a different boundary from the Rust job
benchmark. Both English and Korean export paths passed; the English WAV contains
70,669 frames and the Korean WAV 126,533.

Screenshots were inspected for playing/completed layouts. Integration playback
was muted, so this establishes rendering and lifecycle behavior, not listening
quality. The release app built successfully; strict recursive code-signature
verification, arm64 architecture, and macOS 15.0 deployment metadata passed.
Manual release GUI interaction is unqualified because the available UI tool
could not attach a window. Real GUI integration coverage is from the debug app.

| Artifact | Logical bytes |
| --- | ---: |
| Pinned baseline model payload | 169,328,070 |
| Release C dynamic library | 1,729,280 |
| Release C static library | 21,111,824 |
| Rust CLI | 2,386,296 |
| Flutter release app, 29 real files, excluding symlink aliases | 22,835,652 |

These are logical file sizes, not private APFS allocation. Models remain external
to the app. The example disables the App Sandbox for local bundle access and is
not notarized or qualified for App Store distribution.

## Placement and actual execution

The existing Möbius public compute-plan helper inspected all six baseline models
and the fused VE graph. Plans prefer ANE for most estimated work, including the
vocoder. Preferred placement and estimated costs are not execution evidence.

`task7-traces-v4` attached Instruments after model initialization, then used a
Darwin notification to release the workload. A development-only helper emits
public SDK signposts around real Rust jobs. Attribution requires a hardware
interval to fit both a job interval and an ANE evaluation interval from the
same workload PID. The exact trace binary also appears in the combined suite's
artifact receipt.

| Condition | Requests | Native predictions | Attributed ANE hardware predictions |
| --- | ---: | ---: | ---: |
| Baseline, CPU + ANE | 13 | 143 | 130 |
| Baseline, CPU only | 4 | 44 | 0 |
| Fused8, CPU + ANE | 13 | 52 | 39 |

Attached traces omit model names and provide generic hardware labels. These
counts establish process-attributed ANE execution and a negative CPU control;
they do not establish individual model placement, vocoder ANE execution,
utilization, or physical weight residency. Traced timings are not benchmarks.
Earlier launch-under-profiler attempts timed out while loading fused models;
those failed captures are preserved. The successful attach workflow avoids
profiling model initialization and makes that exclusion explicit.

## Reproduce

Prepare local pinned bundles/reference fixtures using the
[model tools](../../tools/supertonic3/README.md), then choose fresh output directories:

```sh
python3 tools/dev.py check
python3 tools/dev.py test-model-tools
python3 tools/check_coreml.py --cross
python3 tools/dev.py validate-reference --reference /path/to/reference
python3 tools/dev.py test-real --bundle /path/to/baseline-bundle \
  --reference /path/to/reference --variant-bundle /path/to/variant-bundle \
  --flutter --output /path/to/new-suite
python3 tools/dev.py benchmark --bundle /path/to/baseline-bundle \
  --variant-report /path/to/new-suite/variants/report.yaml \
  --output /path/to/new-benchmark
```

See [development tools](../../tools/README.md) for placement/trace commands and the
example README for release builds. No command creates a venv, upgrades global
toolchains, clears OS caches, or downloads model weights during synthesis.

## Next milestone and remaining gates

Integrate the already-confirmed QNN execution path under the existing Rust model
and bounded job contracts. Pin its artifacts and tensor interfaces in YAML,
implement persistent sessions/native tensor ownership, and replay the shared
fixtures with explicit backend tolerances before exposing it through the same
C/Dart/Go/Python API. Then add Android packaging/playback and qualify cancellation,
memory, latency and cleanup on a named physical Qualcomm device. That second
backend should inform any necessary abstraction changes before LiteRT/XNNPACK
CPU coverage is added.

Physical iOS devices and actual minimum OS versions still need execution tests.
Linux/Windows native runtime tests, portable inference backends, binary package
distribution, notarization and remote CI results remain unqualified. Fixed
vocoders, W8A8 and dense INT8 need matched audio/perceptual evaluation before
promotion; the existing exact fused-VE result does not qualify those changes.
