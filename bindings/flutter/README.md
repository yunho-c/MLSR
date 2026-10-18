# MLSR for Flutter

`mlsr_flutter` resolves model resources; the portable `mlsr` Dart package and Rust
core own verification, installation, engine lifetimes, and synthesis. The current
native resource adapter and Core ML hook support macOS, with the shipped model
catalog qualified for Apple Silicon and macOS 15 or later. Other native platforms
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
```

```dart
import 'package:mlsr_flutter/mlsr_flutter.dart';
import 'package:mlsr_app_models/app_models.dart';

final engine = await FlutterMlsr.open(model: AppModels.supertonic3);
try {
  final job = await engine.start('Hello from MLSR.');
  await for (final event in job.events) {
    // Feed audio events to an app-owned bounded player or WAV sink.
  }
} finally {
  await engine.dispose();
}
```

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
