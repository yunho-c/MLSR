# Bundled model delivery qualification

The macOS arm64 slice packages the pinned Supertonic3 baseline through Flutter
assets and loads it without application users entering model paths. A separate
downstream consumer passed the four release cases below in two fresh-container
runs on M4 Pro / macOS 27.0.1,
using Flutter 3.41.4 and the `cpu_and_neural_engine` policy. This qualifies the
delivery path on that device and OS; mobile adapters and other backends follow.

## What the release test establishes

`tools/qualify_model_delivery.py` copies the Dart and Flutter packages into a
consumer outside this checkout. Its native hook receives a prebuilt library;
the consumer contains no Rust workspace. Preparation generates a local asset
package from verified model bytes. The test builds a release app, relocates it,
checks its signature and final asset hashes, and runs it with a unique bundle
identifier whose application data did not previously exist.

The release App Sandbox has no networking entitlement. Each run confirms that
opening three existing developer files fails with permission denial: the source
manifest, developer model bundle manifest, and consumer's prebuilt native
library. A connection to an actually listening local TCP socket is also denied.
This tests independence from those development inputs without renaming or
deleting a developer's cache or checkout. The app resolves its models inside the
relocated bundle.

| Case | Model files copied | Checks |
| --- | ---: | --- |
| First automatic launch | 0 | Fresh app data; direct bundled model; ordinary controller startup |
| Automatic relaunch | 0 | Direct loading again; no installed model copy |
| First installed-copy load | 29 | Native streaming, verified publication, engine lease blocks removal |
| Installed-copy relaunch | 0 | Existing version reused; removable after engine shutdown |

The 29 files contain 169,328,070 payload bytes and the 12,473-byte original YAML
manifest: **169,340,543 bytes total**. The direct path adds no model payload copy
to application support. Core ML maintains its own specialization caches; these
are separate from the model store. The measured release probe app is about
192.6 MB in logical file size, before distribution compression. This is neither
private APFS allocation nor a complete installed/cache storage measurement.

Each phase synthesizes English/M1 and Korean/F1 with steps 8, speed 1.05, seed 0.
The checks require exact WAV hashes against the previous qualified POC, finite
contiguous PCM, one completion event, bounded muted AVAudioEngine playback,
drained completion, stop/cancel quiescence, exclusive WAV export, ordinary
controller temporary-file cleanup, and zero native handles. Muting exercises the
real playback queue without claiming a new listening assessment. Forced
`installedCopy` exercises native materialization on macOS; automatic fallback
when assets are not directory-addressable is covered by small adapter tests.

| Output | Frames | WAV SHA-256 |
| --- | ---: | --- |
| Hello from MLSR. / M1 | 70,669 | `2e531ccb505d74a524338131a424f61fba853efcbffd2dacaee380b0112245d3` |
| Korean greeting / F1 | 126,533 | `cb75390773f1248a1eb21f9634e935d0c11aa5e075ba45e599f3a54f813c16a5` |

The ordinary controller's English WAV must match the same reference. No audio
tolerance or automatic retry is used to turn a mismatching run into a pass.

## Evidence and limits

Compact hashes and observations are recorded in
[`model-delivery/qualification-summary.yaml`](../tests/fixtures/model-delivery/qualification-summary.yaml).
Full reports, logs, consumers, relocated apps and WAVs are retained locally under
`~/.cache/mlsr/qualification/model-delivery-v1/`. Reports record the runtime source
commit and file hashes, qualification-script/probe hashes, native input hash,
final model inventory, bundle identity, denied file accesses, copy counts,
timings, and native handle counts. The runtime source is commit `df524b7`; the
qualification commit adds the checker, probe and evidence documentation.

The two passing runs measured ordinary controller loading at 14.55 and 14.83
seconds with fresh application data and about 0.42–0.43 seconds on relaunch.
Installing and verifying the model copy took 0.39–0.41 seconds, followed by
14.21–14.40-second first engine loads from the new location. These are individual
observations, not a performance
benchmark or a fully cold system measurement. The direct engine measurements
later in each phase follow the ordinary controller and already have warm model
caches. Apple describes device specialization and path/configuration-dependent
caching in its [Core ML lifecycle discussion](https://developer.apple.com/videos/play/wwdc2023/10049/).

Two earlier prototype runs produced different first-launch PCM, despite exact
asset hashes. One logged Core ML `E5RT` errors during loading; relaunching that
same artifact produced exact reference WAVs. Host controls using both native
builds and the relocated model directory also matched the reference. The cause
of the initial differences is unresolved; the errors alone do not prove a
cause or execution placement. These failed runs remain in `step6-release-v2`
and `step6-release-v3`. The checker retains the strict audio gate so recurrence
fails visibly. Its earlier playback-epoch errors were probe lifecycle mistakes,
fixed by running the ordinary controller before the direct playback probe and
continuing the native player's epoch sequence.

The artifact is locally ad-hoc signed, not notarized or an App Store submission.
The fresh application container does not clear shared OS caches. This test does
not establish per-model ANE placement, physical residency, startup reliability
across devices, or universal numerical identity across Core ML versions. Remote
CI has been configured but has not been run for these unpushed commits.

## Reproduce

Use the existing toolchains and a self-contained bundle with the original pinned
`mlsr-bundle.yaml`. The parent output directory must exist; the selected output
must be new and outside the checkout. This is an opt-in real-model check and
uses disk space for the consumer, generated assets, build, relocated app and
Core ML caches. It does not download weights or clear existing app/cache data.

```sh
python3 tools/qualify_model_delivery.py \
  --bundle /path/to/self-contained-model-bundle \
  --output /path/outside/checkout/new-qualification-directory \
  --flutter /path/to/flutter
```

Allow several minutes. PyYAML, Rust, Flutter, CocoaPods and Xcode must already be
available. The script preserves failed reports as well as successful ones. Keep
`report.json`, phase logs and `audio/` together when archiving a result. Raw
models and application artifacts stay outside Git.
