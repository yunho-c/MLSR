# MLSR Speak

A local Flutter player for the macOS Apple Silicon proof of concept. Write text,
choose English or Korean and voice M1/F1, then speak, stop, or export completed
speech as a mono Float32 WAV. Replace or enqueue a new script while reading;
pause, change volume/rate, seek retained audio, and navigate sentences.
Command-Enter speaks; Escape stops all queued speech.

## Run

Requires macOS 15+, Apple Silicon, Xcode, CocoaPods, Rust 1.88+, and Flutter
3.41.4 / Dart 3.11.1 (the qualified SDK). Prepare a verified model bundle using
the repository's [model tools](../../tools/supertonic3/README.md) first. The
checked-in `mlsr-models.yaml` reads the usual local cache; edit its `source` if
your verified payloads live elsewhere. This path is preparation input only.

```sh
# From the repository root, before Pub resolves the generated asset package:
cargo run --locked --release -p mlsr-cli -- models prepare-flutter \
  --config examples/flutter_speak/mlsr-models.yaml
cd examples/flutter_speak
flutter pub get --enforce-lockfile
flutter run -d macos
```

The app opens `AppModels.supertonic3` automatically. Models, their original YAML
manifest, voices and notices ship in the generated Flutter asset package. No
model paths or setup dialog are required at runtime. The Dart native build
hook builds and bundles MLSR with Core ML and native Apple output enabled. It resolves the active Xcode
SDK explicitly, tracks embedded YAML manifests as build inputs, and uses the
same persistent Rust engine as the CLI and other language bindings. There are
no model downloads in the app or build hook.

`flutter build macos --release` produces `MLSR Speak.app` under
`build/macos/Build/Products/Release/`. The release enables App Sandbox and
user-selected file access for WAV export; model resources are inside the app.
Debug/profile builds keep the sandbox disabled for test evidence and developer
tooling. This example currently builds only for the Apple Silicon host.
Notarization, App Store submission, Intel/mobile inference, and additional device
qualification remain separate work. See [model delivery](../../docs/model-delivery.md).

## Ownership and playback

`DemoController` owns public `Tts` and `Speech` handles and bounded presentation
subscriptions. It does not consume PCM, advance queues, schedule native buffers,
or implement a playback clock. `FlutterMlsr.openTts` uses the same verified model
delivery as the lower-level engine API. The shared Rust SDK owns replacement,
queueing, cancellation, native Apple playback, bounded temporary retention,
source-frame positions and final playback completion. The app's only custom
native channel opens the macOS save panel.

The reader uses original UTF-16 segment spans from `SpeechState`; alignment is
segment-level, not word-level. Edits affect the next request, while the reader
shows the immutable accepted text. Seek commits once at the end of a slider drag.
Seeking a completed recording creates a new paused replay with a new request ID.
The UI retains only active/queued handles and the latest displayed recording.

Export calls `Tts.writeSpeech` for a completed recording, using the shared
incremental WAV writer. It does not synthesize again or collect a full PCM array
in Dart. Existing destinations are preserved; partial cancelled speech is not
exportable. Closing the app joins native work and releases owned retention.

## Validate

```sh
flutter analyze --no-pub
flutter test test/audio_clip_test.dart --no-pub
flutter test integration_test/speak_test.dart -d macos --no-pub \
  --dart-define=MLSR_TEST_OUTPUT=/absolute/path/to/new-evidence-directory
```

The integration test uses real synthesis and muted native playback. It checks
UI heartbeats while loading/speaking, bounded retention, replacement/enqueue/stop,
pause/rate/volume/seek/replay, original Unicode spans and sentence navigation,
English/Korean WAV export, and native handle cleanup. UI observations of first
retained PCM, first presented position and cancellation completion are reported
separately; they do not measure acoustic onset or native flush latency. Those
output-driver boundaries have separate [native tests](../../docs/apple-output.md).
It saves screenshots, paired audio, and a report into a **new** directory. These
checks establish behavior on the tested machine, not audible listening quality
or compatibility with every supported minimum OS. The root
`tools/dev.py test-real --flutter` command includes these checks in the explicit
Apple Silicon suite.

Add `--dart-define=MLSR_TEST_MATERIALIZE=true` to exercise the full-size native
streaming path into Application Support. This is an integration-test option;
the ordinary app loads immutable bundled resources directly. Small portable
delivery tests live in `bindings/flutter`. The separate downstream release check
tests a relocated app with fresh data and denied access to developer files and
the network; see [release qualification](../../docs/model-delivery-qualification.md)
for evidence and reproduction commands.

The example is Apache-2.0. Fraunces and Manrope are bundled under the SIL Open
Font License; exact upstream revisions and hashes are in
`assets/fonts/provenance.json`, with the corresponding license texts alongside.
Those license texts preserve upstream bytes, including their trailing whitespace.
Model and preprocessing notices are included in the generated package and the
app's license page; the library license does not relicense model weights.
