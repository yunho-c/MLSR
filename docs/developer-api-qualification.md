# Developer API v0 qualification

All eight steps in the [acceptance checklist](developer-api-progress.md) are
complete for **macOS Apple Silicon, English/Korean and voices M1/F1**. Rust owns
the scheduler, synthesis, playback policy, retention, timeline and WAV writer.
C, Dart, Go and Python expose that behavior, and the Flutter example uses the
public SDK for both its speak and reader flows. See the [SDK guide](developer-sdk.md)
for entry points, ownership and examples.

The final installed and release runs used source commit
`b3a64ec3df1d08e2836f39e49516b92c608f2185`, including the paused frame-zero
highlight and model-lock lifetime fixes. Their recorded source hashes match the
qualified implementation. Subsequent documentation and timing-tool additions do
not change that implementation. The native library SHA-256 is
`cd3e58ff0c50a5472af25c4918a8ec5ac3f9731b879ea202be9c91014ed86206`.

Qualification ran on an Apple M4 Pro with macOS 27.0.1, Rust 1.95.0, Python 3.14.8,
Go 1.27.1, Dart 3.11.1 and Flutter 3.41.4. The Core ML baseline uses the pinned
Supertonic3 artifacts and `cpu_and_neural_engine` policy. Native output tests are
muted; they exercise the real output graph without claiming listening-quality
or acoustic-onset measurements.

## Installed public consumers

[`check_synthesis_bindings.py`](../tools/check_synthesis_bindings.py) builds or
accepts a prebuilt native library, then runs consumers outside the checkout.
The final run is `step8-installed-v2` under the external evidence root below.

| Consumer | Checks passed |
| --- | --- |
| C shared and static linkage | Existing ABI smoke tests and new owned developer handles, lazy state/completion, rendering, no-clobber WAV, native speech drain and zero registry handles |
| Python installed wheel | 13 tests, including asyncio cancellation/late-result cleanup, single PCM consumer, independent states, verified model ownership, native reader and fire-and-forget; old/new export examples |
| Go relocated module/header/library | Tests with the race detector, `go vet`, context ownership/cancellation, native reader, verified model lifetime and old/new examples |
| Dart relocated package/native asset | Analyzer and 9 tests covering old/new jobs, paused consumers, UTF-16 spans, native reader/replay and export; old/new examples |
| Supplemental legacy Dart model lease | The opt-in installed-copy test skipped by the main matrix was explicitly enabled and passed in `step8-dart-lease-v1`; engine creation races caller model disposal, holds the lease, then permits removal after shutdown |

The tests for the language-specific ownership rules live in
[Python](../bindings/python/tests/test_sdk.py),
[Go](../bindings/go/sdk_test.go), [Dart](../bindings/dart/test/sdk_test.dart)
and the [C consumer](../tests/integration/developer_smoke.c). The package runs
use actual Core ML inference and native output as well as portable error tests.
Python is installed into a local target directory, with no virtual environment.

Four cases match **every Float32 PCM bit** across the five new consumers
(C shared/static, Python, Go, Dart) and the three existing language job APIs.
Settings are eight steps, normalized speaking rate 1.0 / raw speed 1.05,
mono 44,100 Hz, and the seeds below. English text is `Hello from MLSR.`;
Korean text is `안녕하세요. 반갑습니다.`.

| Language / voice | Seed | Frames | PCM SHA-256 prefix |
| --- | ---: | ---: | --- |
| en / M1 | 1729 | 70,669 | `8b6c2c6384af` |
| en / F1 | 1730 | 70,792 | `2c4de2d4506c` |
| ko / M1 | 1731 | 126,287 | `6114543b5917` |
| ko / F1 | 1732 | 126,533 | `5c9eb0db13b23` |

The three new public export examples also produce identical shared-writer WAV
bytes. Their PCM matches the existing English/M1 result. Full hashes are in the
[tracked summary](../tests/fixtures/developer-api/qualification.json).

## Flutter and offline release

The [Flutter integration](../examples/flutter_speak/integration_test/speak_test.dart)
exercises rapid replacement/enqueue/stop, paused clocks, rate/volume, retained
seeking, completed replay with a new request ID, previous/next sentence, original
UTF-16 highlights, typed unsupported-character errors, exclusive WAV export and
zero handles. A separate native playback integration passes through the public
Flutter model loader. The shared AudioClip writer also has a no-clobber test.
The UI's `DemoController` presents and owns SDK handles; the former app-local
PCM channel, Swift player, scheduling state machine and WAV serializer are gone.

The debug UI evidence is recorded in
[`flutter.json`](../tests/fixtures/developer-api/flutter.json). Its observations
use a 20 ms UI pump interval and are separate from the release timing probe.

`step8-release-v4` builds a separate consumer, relocates the release app and runs
four phases in a new application container. All pass:

| Phase | Model copies | Additional result |
| --- | ---: | --- |
| Automatic first launch | 0 | Loads from relocated bundled assets through the ordinary UI controller |
| Automatic relaunch | 0 | Reuses bundled assets without an installed copy |
| Installed-copy first load | 29 files / 169,340,543 bytes | Verified publication; Tts retains the model lease after the caller releases its handle |
| Installed-copy relaunch | 0 | Reuses the installed version; removal succeeds after Tts shutdown |

The app is ad-hoc signed and sandboxed with no networking entitlement. Each
phase verifies permission denial for three existing developer files and a live
local network listener. English/M1 and Korean/F1 with seed 0 match the original
qualified POC's PCM exactly through both rendering and retained native speech.
Native playback reaches its final source frame, cancellation joins, retained
bytes return to zero and registry handles return from zero to zero in all phases.

The new common Rust WAV writer uses a different valid header from the former
demo writer. The expected PCM hashes were extracted from the original reference
WAVs after verifying their full-file hashes; the checker requires those exact
sample bytes. This changes the container comparison, not the numerical audio
gate. The historical delivery
runs and their limitations remain in the
[model delivery record](model-delivery-qualification.md).

## Timing and memory boundaries

[`benchmark_developer_api.py`](../tools/benchmark_developer_api.py) ran against
the installed Python wheel and the same release native library after the package
and release suites finished. It used battery power (89% at start), existing
system/model caches, one inference warmup and one native output warmup. These are
12 warm observations: the four cases above, repeated three times. They are not
peak-power results, cross-device estimates or a comparison against the earlier
low-level POC benchmark.

| Measurement | Median | Maximum | Boundary |
| --- | ---: | ---: | --- |
| First PCM | 77.52 ms | 99.30 ms | Public synthesis admission through the first copied Python PCM chunk |
| Synthesis complete | 101.74 ms | 164.56 ms | Admission through consumer completion and native quiescence |
| First retained speech PCM | 79.63 ms | 134.65 ms | Speak admission through a nonzero retained-buffer snapshot |
| First playback position | 197.34 ms | 242.26 ms | Speak admission through a positive latency-adjusted source position |
| Speech complete | 2,490.66 ms | 3,113.93 ms | Speak admission through actual native output drain and quiescent completion; includes the spoken duration |

Speech observations poll every 2 ms. The output graph reported about 94.13 ms
downstream latency; position is an adjusted native clock, not a microphone or
speaker timestamp. Initial Tts loading took 452.14 ms in this cached location.
First output warmup through the complete `Ready.` utterance took 1,638.84 ms;
that includes playback and must not be called graph-startup time.

Three running-speech cancellations took 0.028–0.042 ms to return the cancellation
request and 6.29–13.79 ms to reach quiescent completion. The independently run
native driver test recorded **9.35 ms for one native flush call**. These measure
different calls and states. Neither the request-return time nor the driver
flush time is an acoustic-silence measurement or a worst-case bound on an
in-flight model prediction.

| Memory/accounting observation | Result | Boundary |
| --- | ---: | --- |
| Retained PCM peak / final | 506,132 / 0 bytes | Owned disk PCM across the timing workload; default quota remains 128 MiB |
| Native registry handles | 0 before / 0 after | C-owned handle registry, including cleanup of all measured requests |
| Peak process RSS | 125,779,968 bytes (119.95 MiB) | Python process including Rust/Core ML mappings; excludes other processes and physical driver allocations |
| Native output buffer peak / final | 4 / 0 buffers, 44,100 / 0 frames | Separate driver test configured to exactly these limits; default Tts output permits 88,200 frames / 32 buffers |

The driver run additionally passes pause/rate changes, all five rates,
starvation/restart, full-queue rejection, stale epochs and synthetic native
configuration-change recovery. The earlier [offline pitch probe](playback-probe.md)
supplies pitch-preservation evidence. Driver counters, disk quotas, caller-owned
clips, runtime allocations and process RSS remain separate quantities.

## Regression and build checks

The final workspace all-feature tests pass (76 core, 7 CLI and 8 FFI tests).
Opt-in model/native tests are run separately, not counted as ordinary portable
test passes. Strict all-feature Clippy, Rust 1.88 all-target checks and Windows
core cross-compilation pass. The generated C header and Dart bindings are current.
Flutter analysis, installed Dart analysis, Go race/vet and Python lint/format
checks also pass. Windows compilation establishes a build boundary; it does not
qualify a Windows inference or playback backend.

Two lifecycle repairs were made during acceptance. The reader now reports the
first original text range when paused at frame zero, covered by shared and
Flutter regressions. Model-store lock ownership now explicitly unlocks on owner
drop: a duplicated descriptor previously kept an OS lock alive after its MLSR
owner closed. A deterministic descriptor-duplication regression fails before
the repair and passes after it. A separate intermittent store-test audit changed
from three failures in 12 runs to 12 passing runs after the repair; the original
intermittent trigger was not traced to a particular Core ML call or subprocess.
The final installed and relocated release checks include both repairs.

## Evidence and reproduction

The compact [qualification summary](../tests/fixtures/developer-api/qualification.json)
records source/library/manifest hashes, report hashes, PCM cases, copy counts and
measurements. Full reports, logs, copied packages, model assets, relocated apps
and WAVs remain outside Git under
`~/.cache/mlsr/qualification/developer-api-v0/`:

- `step8-installed-v2`: C static/shared and installed language consumers.
- `step8-dart-lease-v1`: supplemental old Engine model-store ownership test.
- `step8-release-v4`: four relocated sandboxed release phases.
- `step8-timing-v1`: public SDK timing, raw samples and retained WAVs.
- `step8-native-output-v1`: native flush, clock, rate and allocation probe.
- `step8-generated-v1`: generated C/Dart consistency check.
- `step7-flutter-v3`: debug reader/speak UI and native integration logs/screenshots.
- `store-lock-audit-v1` / `v2`: pre/post-repair repetitions, including failed logs.

Earlier failed artifacts are preserved. Initial step-8 release attempts found a
payload-cache manifest differing from the catalog pin and then a probe lint error.
Preparation now supplies the original pinned manifest explicitly. No model hash
or audio tolerance was relaxed. The previous Flutter failures and their frame-zero
repair are also retained in the step-7 record.

Build the release library with native output, then use fresh output directories:

```sh
MACOSX_DEPLOYMENT_TARGET=15.0 cargo build --locked --release \
  -p mlsr-ffi --features coreml,apple-output

python3 tools/check_synthesis_bindings.py --developer-api --no-build \
  --bundle /path/to/pinned-payloads \
  --verified-model-directory /path/to/catalog-verified-model-directory \
  --output-dir /path/to/new-installed-evidence

python3 tools/qualify_model_delivery.py \
  --bundle /path/to/pinned-payloads \
  --output /path/to/new-release-evidence --flutter /path/to/flutter

MLSR_LIBRARY_PATH=/path/to/new-installed-evidence/installed/native/lib/libmlsr_ffi.dylib \
PYTHONPATH=/path/to/new-installed-evidence/installed/python-installed \
python3 tools/benchmark_developer_api.py \
  --manifest model-manifests/supertonic3-coreml.yaml \
  --bundle /path/to/pinned-payloads --output /path/to/new-timing-evidence

MLSR_PLAYBACK_REPORT=/path/to/new-native-output-report.json \
MACOSX_DEPLOYMENT_TARGET=15.0 cargo test --locked --release \
  -p mlsr-playback --features apple \
  native_clock_lifecycle_bounds_and_configuration_recovery -- --ignored --nocapture
```

For the optional legacy Dart store test, set `MLSR_TEST_DELIVERY_BUNDLE` to the
catalog-verified model directory when running its copied package's tests.
Run timing probes after other model workloads finish. Tools use existing
toolchains, never download weights and never clear developer caches.

## Scope left for later milestones

This qualifies the essential developer API on this macOS host, not a published
distribution or universal backend support. QNN/Android, physical iOS, other
devices/minimum OS versions, real device-route changes, mobile audio-session
interruptions, notarization and remote CI remain separate work. Native macOS
interruption support is explicitly advertised as unavailable; controller policy
is covered with deterministic fake events. Word alignment, persistent synthesis
caching, background media controls, SSML/cloning and quantization promotion are
outside this milestone. These results do not establish physical ANE residency,
new speech quality, or numerical identity across Core ML versions.
