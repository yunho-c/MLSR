# Windows speech playback

The optional `windows-output` feature connects the shared speech controller to
WASAPI shared-mode output. It is independent of QNN: headless rendering can run
without it, and enabling it opens no audio device until speech starts. The first
implementation supports normal playback speed, pause/resume, per-stream volume,
replace/enqueue, retained seeking/replay, bounded buffering and device-clock drain.

## Enable output

Prepare the [QNN runtime](qnn-runtime.md), then build the Windows ARM64 library:

```powershell
. ./tools/qnn_env.ps1
cargo build --locked --release -p mlsr-ffi --features qnn,windows-output
```

Rust callers enable those features on `mlsr`; C/Python/Go callers load/link the
matching `mlsr-ffi` library. The C feature mask is 109: jobs 1, model store 4,
developer SDK 8, QNN 32 and Windows output 64. ABI 1 and all existing layouts are
unchanged. `TtsOptions` defaults to native output when compiled; explicit headless
mode still opens no endpoint and rejects speech with `unsupported_capability`.

For Dart, add `windows-output: true` beside `qnn: true` under
`hooks.user_defines.mlsr` in the consuming application's `pubspec.yaml`. Use
`ComputePolicy.qnnHtp`; Python uses `compute_policy="qnn_htp"`, Go uses
`EngineOptions{QNNHTP: true}`, and C uses compute policy 2. Existing speech APIs
apply without a Windows-specific controller or PCM loop.

For example, with the Python wrapper and prepared DLL environment:

```python
from mlsr import Tts

with Tts(manifest_path, bundle_path, compute_policy="qnn_htp") as tts:
    with tts.speak("Hello from MLSR.") as speech:
        speech.wait(30)  # Completion includes native playback drain.
```

`output_capabilities.rate_range` is absent (`null`); rate 1 works and other rates
fail before replacing active speech. Pitch-preserving variable playback speed
requires a future time-stretch stage. Public synthesis speaking rate remains a
separate model control and still has the preset's fixed-shape limits.

## Clock, ownership and routing

Each controller owns one reusable COM/audio stream on its output thread. Its
5 ms control tick feeds a finite endpoint buffer; inference stays on another
worker. Float32 mono 44.1 kHz PCM is converted by the Windows shared audio engine
to the endpoint's mix format. Per-stream volume preserves the endpoint volume
and other controllers' gain. Both audio policies leave system routing unchanged.

Source PCM remains retained until the device clock presents it, with limits on
both frames and chunk count. `IAudioClock` supplies the playback position; empty
engine padding alone does not imply completion. After a starvation gap, the
stream resets only once submitted audio has drained, so silence cannot advance
the source-text cursor. Pause freezes the clock. Flush invalidates the old epoch
and discards its native/source buffers before replacement or seeking.

Default-render-endpoint changes pause output, reopen the current default and
reschedule retained unplayed PCM from the last observed position. The application
resumes explicitly. Notification storage is bounded and overflow is an error.
Device/service failures that prevent recovery finish the current speech with a
playback failure; a later request opens a fresh stream. System interruption
events are not advertised. Clock positions describe stream presentation, not a
sample-exact acoustic measurement at the speaker.

## Qualification

The [qualification snapshot](../tests/fixtures/developer-api/windows-output.json)
records the exact source commit, native/model hashes, toolchains and retained logs.

The Windows ARM64/X2 Elite checks use the actual rendering endpoint at zero
volume. The native test covers pause/resume, starvation, rebasing, stale-epoch
rejection, true device-clock drain and a simulated default-route notification
followed by a real endpoint rebuild. It does not claim physical unplug/Bluetooth
or every Windows device/driver configuration was tested.

The real QNN SDK test covers English M1 and Korean F1, retained PCM parity,
pause/seek/replay, invalid-rate admission, queued cancellation, replacement and
stream reuse. The C/Python/Go/Dart suite includes native playback alongside the
existing EN/KO M1/F1 headless parity checks. Ordinary CI compiles and tests the
Windows adapter on stable Rust and Rust 1.88 without opening a device.

```powershell
. ./tools/qnn_env.ps1
$env:MLSR_QNN_TEST_BUNDLE = 'C:/models/mlsr-qnn-superfast-v1'
cargo test --locked -p mlsr-playback --features wasapi real_wasapi -- --ignored --nocapture
cargo test --locked -p mlsr --features qnn,windows-output --test windows_output -- --ignored --nocapture
# Add -NativeOutput to the QNN binding qualification command:
./tools/check_qnn_bindings.ps1 -Bundle $env:MLSR_QNN_TEST_BUNDLE -OutputDirectory ignore/windows-output-new -Python C:/tools/python-arm64/python.exe -Go C:/tools/go-arm64/bin/go.exe -Dart C:/tools/dart-arm64/bin/dart.exe -CCompiler C:/tools/llvm-mingw/bin/aarch64-w64-mingw32-clang.exe -NativeOutput
```

Superfast artifacts are unchanged. Numerical differences remain diagnostics;
subjective listening qualification of the preset is pending.
