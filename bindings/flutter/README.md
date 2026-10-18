# MLSR for Flutter

`mlsr_flutter` resolves model resources; the portable `mlsr` Dart package and Rust
core own verification, installation, engine lifetimes, and synthesis. The current
native resource adapter and Core ML hook target Apple Silicon with a macOS 15
minimum. Release delivery is [qualified on M4 Pro / macOS 27.0.1](../../docs/internal/model-delivery-qualification.md);
the minimum OS and other devices still need validation. Other native platforms
require their resource adapter and runtime qualification before use.

Prepare a model asset package before `flutter pub get`, following
[model delivery](../../docs/model-delivery.md). Add this package and the generated
asset package as path dependencies in the app. Enable Core ML on the app's
`mlsr` build hook:

```yaml
dependencies:
  mlsr_flutter:
    path: /path/to/MLSR/bindings/flutter
  mlsr_app_models:
    path: .mlsr/model_assets
hooks:
  user_defines:
    mlsr:
      coreml: true
      apple-output: true
```

```dart
import 'package:mlsr_flutter/mlsr_flutter.dart';
import 'package:mlsr_app_models/app_models.dart';

final tts = await FlutterMlsr.openTts(model: AppModels.supertonic3);
try {
  final speech = tts.speak('Hello from MLSR.');
  try {
    await speech.done; // Synthesis and native playback have both finished.
    await tts.writeSpeech(speech, '/chosen/new-file.wav');
  } finally {
    await speech.dispose();
  }
} finally {
  await tts.dispose();
}
```

`openTts` uses the same verified delivery path as the existing low-level `open`
engine API. Playback, bounded retention, scheduling, reader controls, and WAV
writing belong to the shared SDK; apps only observe state and present controls.
For a headless application, omit `apple-output` and use
`TtsOptions(output: OutputMode.headless)`.

The default uses an addressable app-bundled directory without a duplicate copy.
When resources have no directory, the adapter streams files into a Rust-owned
installation and publishes only after verification. `ModelDelivery.installedCopy`
explicitly exercises that route. On macOS the store lives in Application Support,
under `<bundle identifier>/mlsr`; Rust does not choose this platform path.

`FlutterModelLoader.loadModel` exposes a verified model without creating an engine.
Its returned `ModelHandle` must be disposed. `ModelResources` is the bounded
streaming/lookup interface for further platform adapters and test fixtures.
Progress reports stages and completed-file byte counts. Waiting for another
installer is asynchronous and bounded; corruption is surfaced to the caller.

The macOS plugin uses Flutter's registrar keys and app bundle APIs. It validates
relative paths, restricts writes to this app's staging store, creates files
exclusively, and copies in 64 KiB chunks on a background queue. Channel messages
contain metadata and paths, not model bytes. Rust remains the final authority on
manifest identity, every payload hash, publication and cleanup eligibility.

Run `flutter test` here for small integrity fixtures, with no model downloads or
inference. Tests share the Dart package's tiny fixture generator and templates.
Native app resource lookup, full-size streaming and Core ML execution are checked
by the example and downstream release qualification, separately from these tests.
