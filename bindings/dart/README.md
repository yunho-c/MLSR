# MLSR for Dart

`dart:ffi` bindings to MLSR's Rust library, generated with `ffigen`. Requires
Dart 3.11+. Persistent engines expose bounded PCM streams, cancellation and
explicit cleanup. Real synthesis currently requires macOS arm64 and Core ML.

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
```

This opts into a macOS 15+ Core ML build. The default hook build supplies the
portable ABI without a synthesis backend. `coreml` is currently a macOS host
build option; it does not configure iOS packaging.

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

Regenerate bindings from the repository root with
`python3 tools/dev.py generate`. `tool/generate_bindings.dart` uses ffigen's
programmatic API; its older YAML configuration format is deprecated upstream.
Normal builds use the checked-in generated Dart file and do not need libclang.

Licensed under Apache-2.0; see `LICENSE`.
