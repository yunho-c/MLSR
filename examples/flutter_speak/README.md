# MLSR Speak

A local Flutter player for the macOS Apple Silicon proof of concept. Write text,
choose English or Korean and voice M1/F1, then speak, stop, or export completed
speech as a mono Float32 WAV. Command-Enter speaks; Escape stops.

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
hook builds and bundles MLSR with Core ML enabled. It resolves the active Xcode
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

`SpeechController` owns one engine and job. Model loading, event polling, and
native cleanup run in the Dart binding's adapter isolate. The UI keeps a small
waveform history and writes audio incrementally to an owned temporary WAV.
Completed exports create a new file; an existing destination is preserved.
Cancelled partial files are removed, and app exit cleans up the engine and
temporary directory.

The separate Swift `PcmPlayer` uses AVAudioEngine on a serial queue. It bounds
scheduled PCM to 22,050 frames (half a second), plus one pending native chunk.
Dart awaits each enqueue, preserving backpressure through to the Rust job's
bounded queue. Stop flushes scheduled audio before waiting for native inference
to quiesce. Request epochs discard stale playback callbacks and synthesis
results. Generation completion and actual playback completion are distinct.

## Validate

```sh
flutter analyze --no-pub
flutter test test/wav_sink_test.dart --no-pub
flutter test integration_test/speak_test.dart -d macos --no-pub \
  --dart-define=MLSR_TEST_OUTPUT=/absolute/path/to/new-evidence-directory
```

The integration test uses real synthesis and muted native playback. It checks
UI heartbeats while loading/speaking, bounded playback, immediate stop/flush,
reuse, playback completion, English/Korean WAV export, and native handle cleanup.
It saves screenshots, paired audio, and a report into a **new** directory. These
checks establish behavior on the tested machine, not audible listening quality
or compatibility with every supported minimum OS. The root
`tools/dev.py test-real --flutter` command includes these checks in the explicit
Apple Silicon suite.

Add `--dart-define=MLSR_TEST_MATERIALIZE=true` to exercise the full-size native
streaming path into Application Support. This is an integration-test option;
the ordinary app loads immutable bundled resources directly. Small portable
delivery tests live in `bindings/flutter`. Relocated offline release qualification
is tracked separately in [delivery progress](../../docs/model-delivery-progress.md).

The example is Apache-2.0. Fraunces and Manrope are bundled under the SIL Open
Font License; exact upstream revisions and hashes are in
`assets/fonts/provenance.json`, with the corresponding license texts alongside.
Those license texts preserve upstream bytes, including their trailing whitespace.
Model and preprocessing notices are included in the generated package and the
app's license page; the library license does not relicense model weights.
