# MLSR for Dart

`dart:ffi` bindings to MLSR's Rust library, generated with `ffigen`. Requires
Dart 3.11+. The shared `Tts` facade provides native speech, independent
controllers, reader controls, bounded rendering and lazy synthesis. Real
synthesis and native output are currently qualified on macOS arm64.

```dart
final tts = await Tts.open(manifest: manifestPath, bundle: bundlePath);
try {
  final speech = tts.speak('Hello from MLSR.'); // Immediate handle; native PCM path.
  try {
    speech.setPlaybackRate(1.25); // Independent of synthesis; preserves pitch.
    await speech.done;          // Actual output drain, not just synthesis EOF.
    await tts.writeSpeech(speech, 'spoken.wav'); // Retained PCM; no inference.
  } finally {
    await speech.dispose();
  }
  final clip = await tts.render('Export me.');
  await clip.writeWav('clip.wav');
  await tts.renderToFile('Incremental export.', 'stream.wav');
} finally {
  await tts.dispose();
}
```

`RequestOptions` freezes locale, stable voice ID (such as `supertonic3:F1`), public
speaking rate and typed `SupertonicOptions`. Normal speaking rate is 1.0;
`speech.request()` / `job.request()` report effective model settings separately.
`getVoices()` discovers catalog-only metadata; `model.getVoices()` reports verified
assets and `tts.getVoices()` loaded-engine readiness. Locale fallback is explicit.
Use `Tts.fromModel` with the existing verified `ModelHandle`, or
`FlutterMlsr.openTts(model: AppModels.supertonic3)` for automatic offline delivery.

The default `speak` mode replaces that controller's active and queued speech.
`mode: SpeakMode.enqueue` keeps FIFO order; `tts.createController()` creates a
separate scope sharing the inference lane and budgets. The controller exposes
native capabilities. Speech supports pause/resume, volume, pitch-preserving rate,
retained-frame `seek`, previous/next sentence and original-sentence navigation.
Navigation returns a new `Speech` when inference restarts. `replayFrom` creates a
new identity sharing a completed recording; the original `done` is immutable.

`speech.snapshot()` is an independent state read. Each `speech.states` access
creates a new single-subscription coalesced stream. Pausing a subscription stops
polling, so it cannot grow an event queue. States include source-frame position,
buffered end, seekable ranges, whole estimate/final duration and original text
alignment. `activeTextRange.start/end` use Dart UTF-16 indices; `utf8` is explicit.
No word timestamps are invented. `done` resolves cancellation with its first
reason; failure raises `SynthesisException` with category/stage/request/backend.

Advanced synthesis is lazy until audio is listened to or `start()` is called:

```dart
final job = tts.synthesize(text, options: const RequestOptions(locale: 'ko'));
try {
  // snapshot(), states and done do not start inference or consume audio.
  await for (final chunk in job.audio) {
    await consumePcm(chunk.audio.samples, chunk.audio.format.sampleRate);
  }
  await job.done;
} finally {
  await job.dispose();
}
```

Cancelling an audio subscription cancels unfinished generation as `consumerDropped`.
Cancelling a state subscription affects only that observer. An async `listen`
callback does not itself backpressure Dart; use `await for` or pause explicitly.
For independently cancellable exports, create a job and use `tts.collect(job)` /
`tts.writeJob(job, path)` while retaining its direct `cancel()` control. Dart
Futures themselves do not provide cancellation. Speak/control calls are short
native admission operations; polling uses a persistent adapter isolate, while
render/file/cleanup use separate workers so they cannot starve cancellation/state.

Always await disposal for deterministic cleanup. SDK finalizers are a fallback:
discarded Speech handles preserve fire-and-forget playback, while discarded
unconsumed synthesis jobs cancel. Owner disposal joins work and writers; live
child handles retain final outcomes until their own disposal. Clips own Dart
Float32 samples independently. Collection defaults to 32 MiB, file export to 30
minutes, with explicit queue/lookahead/storage budgets. Existing files are
preserved; only owned partial files are removed. `AudioClip.writeWav` copies PCM
to a worker/native buffer; `renderToFile` streams directly without full collection.

The original low-level `Engine` / `Job` API retains its raw speed, event offsets
and explicit-disposal behavior:

```dart
final engine = await Engine.load(manifest: manifestPath, bundle: bundlePath);
try {
  final job = await engine.start('Hello from MLSR.', language: 'en', voice: 'M1');
  try {
    await for (final event in job.events) {
      if (event.kind == EventKind.audio) {
        await consumePcm(event.samples, event.sampleRate); // Your bounded player.
      } else if (event.kind == EventKind.failed) {
        throw event.error!;
      }
    }
  } finally {
    await job.dispose();
  }
} finally {
  await engine.dispose();
}
```

Loading, polling, waits and destruction run on a dedicated adapter isolate.
`job.cancel()` signals Rust directly, so it does not queue behind a blocking poll;
`await job.wait()` observes quiescence. Reuse the engine between jobs; a competing
submission reports `engine_busy`. Always await disposal; native resources are
not reclaimed by a Dart finalizer.

Each job has one stream subscription. `await for`/paused subscriptions preserve
backpressure. An async `listen` callback alone does not pause the stream: pause
explicitly while awaiting a slow consumer. There is no unbounded adapter audio
queue. Returned `Float32List` samples are Dart-owned and survive disposal. Text
spans index original UTF-8 bytes, not Dart UTF-16 code units. Stream cancellation
also disposes the job; synthesis completion is separate from playback completion.

```sh
dart pub get --enforce-lockfile
dart run example/version.dart
dart test
```

In the repository checkout, the build hook compiles the Rust library in release
mode and bundles it as a native code asset. Rust and a native linker must be
installed. The hook currently supports host builds on macOS, glibc Linux, and
Windows with the MSVC Rust toolchain, for x64/arm64. See the root platform document
for the distinction between configured targets and validated targets.

To enable synthesis in a consuming application within the checkout:

```yaml
hooks:
  user_defines:
    mlsr:
      coreml: true
      apple-output: true
```

This opts into a macOS 15+ Core ML/native-output build. Omit `apple-output` for a
headless native build, or set `TtsOptions(output: OutputMode.headless)` per facade.
The default hook supplies the portable ABI without synthesis or native output.
Both feature options currently support macOS host builds; they do not configure
iOS packaging.

The [Flutter speak/stop example](../../examples/flutter_speak/README.md) uses this
hook with Flutter 3.41.4. Supported hook/code-assets ranges allow that SDK's pinned
dependencies as well as the newer versions in the standalone Dart lockfile. On
macOS the hook resolves the active Xcode SDK/compiler and tracks the embedded
YAML manifests, so native builds work from Xcode and rebuild when model pins change.

Outside the checkout, supply a matching prebuilt library through the consuming
application's `pubspec.yaml`:

```yaml
hooks:
  user_defines:
    mlsr:
      native-library: native/libmlsr_ffi.dylib
```

Use the corresponding `.so` or `.dll` on Linux or Windows. The path is relative
to the containing pubspec. It must match the target architecture and ABI; no
automatic download is configured. Mobile, browser, and cross-compilation support
remain future work. The package is marked `publish_to: none` during initialization.

With a Core ML library configured, run `dart run example/synthesize.dart
--manifest /path/to/manifest.yaml --bundle /path/to/bundle --output hello.wav
--cancel-first` from this package. The example cancels a job, reuses its engine,
and streams a new WAV file. From the repository root,
`tools/check_synthesis_bindings.py --bundle /path/to/bundle --output-dir /new/path`
qualifies a relocated package/native asset with real inference and this example.

`test/sdk_test.dart` covers catalog/verified readiness and owned WAV without model
execution. Set `MLSR_TEST_MANIFEST` and `MLSR_TEST_BUNDLE` for real shared-facade
tests; additionally opt into muted native playback with `MLSR_TEST_NATIVE_OUTPUT=1`.
The [installed SDK qualification](../../docs/internal/developer-api-qualification.md)
passes the old/new APIs, Korean/emoji/combining-text UTF-16 mapping, exact PCM/WAV,
bounded state observation, pending-done disposal, retained replay and model lease
ownership. The [developer API example](example/developer_api.dart) uses the shared
job/file writer; its optional `speak` argument plays and exports retained speech.

Regenerate bindings from the repository root with
`python3 tools/dev.py generate`. `tool/generate_bindings.dart` uses ffigen's
programmatic API; its older YAML configuration format is deprecated upstream.
Normal builds use the checked-in generated Dart file and do not need libclang.

Licensed under Apache-2.0; see `LICENSE`.
