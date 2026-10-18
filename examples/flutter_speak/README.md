# MLSR Speak

A local Flutter player for the macOS Apple Silicon proof of concept. Write text,
choose English or Korean and voice M1/F1, then speak, stop, or export completed
speech as a mono Float32 WAV. Command-Enter speaks; Escape stops.

## Run

Requires macOS 15+, Apple Silicon, Xcode, CocoaPods, Rust 1.88+, and Flutter
3.41.4 / Dart 3.11.1 (the qualified SDK). Prepare a verified model bundle using
the repository's [model tools](../../tools/supertonic3/README.md) first.

```sh
cd examples/flutter_speak
flutter pub get --enforce-lockfile
flutter run -d macos \
  --dart-define=MLSR_MANIFEST=/absolute/path/to/MLSR/model-manifests/supertonic3-coreml.yaml \
  --dart-define=MLSR_BUNDLE=/absolute/path/to/prepared-bundle
```

Both paths can instead be entered in the settings dialog. The Dart native build
hook builds and bundles MLSR with Core ML enabled. It resolves the active Xcode
SDK explicitly, tracks embedded YAML manifests as build inputs, and uses the
same persistent Rust engine as the CLI and other language bindings. There are
no model downloads in the app or build hook.

`flutter build macos --release` produces `MLSR Speak.app` under
`build/macos/Build/Products/Release/`. Paths supplied with `--dart-define` become
build-time defaults; omit them for an app configured through its settings.
This example currently builds only for the Apple Silicon host. It disables the
macOS App Sandbox to read the external, potentially symlinked model bundle.
App Store sandboxing, notarization, Intel/mobile builds, and model distribution
are separate packaging work.

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
  --dart-define=MLSR_MANIFEST=/absolute/path/to/MLSR/model-manifests/supertonic3-coreml.yaml \
  --dart-define=MLSR_BUNDLE=/absolute/path/to/prepared-bundle \
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

The example is Apache-2.0. Fraunces and Manrope are bundled under the SIL Open
Font License; exact upstream revisions and hashes are in
`assets/fonts/provenance.json`, with the corresponding license texts alongside.
Those license texts preserve upstream bytes, including their trailing whitespace.
