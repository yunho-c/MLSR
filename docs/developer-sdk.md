# Developer SDK v0

The shared developer API is implemented in Rust and exposed through C, Dart,
Go and Python. It covers immediate speech handles, replace/enqueue controllers,
lazy single-consumer PCM, rendering, native playback, retained audio, reader
controls and original-text alignment. The [contract](developer-api.md) and
[acceptance checklist](internal/developer-api-progress.md) define its scope and evidence.

The qualified runtime/output target is macOS Apple Silicon. Core ML uses cidre;
optional Apple playback uses the separate `mlsr-playback` crate. Headless builds
need no output dependency. QNN/Android, physical iOS and other hardware/OS
qualification remain later milestones. The weights and preprocessing are still
the pinned Supertonic3 baseline; this SDK work does not promote a new model,
quantization mode or quality claim.

## Choose the public surface

| Need | Rust | Dart / Flutter | Python | Go |
| --- | --- | --- | --- | --- |
| Facade | `mlsr::tts::Tts` | `Tts.open`, `FlutterMlsr.openTts` | `Tts`, `Tts.open_async` | `OpenTTS` |
| Verified model without loading inference | `ModelHandle::open` | `ModelHandle.openDirectory`, Flutter loader | `ModelHandle`, `open_async` | `OpenModel` |
| Load from verified ownership | `Tts::open` | `Tts.fromModel` | `Tts.from_model`, `from_model_async` | `TTSFromModel` |
| Catalog / loaded voices | `VoiceCatalog` / `tts.voices()` | `getVoices()` / `tts.getVoices()` | `voices()` / `tts.voices` | `Voices()` / `tts.Voices()` |
| Speak | `tts.speak(request, options, mode)` | `tts.speak(text, ...)` | `tts.speak(text_or_request, ...)` | `tts.Say(ctx, text)`, `Speak` |
| Independent controller | `create_speech_controller` | `createController` | `create_controller` | `Controller(true)` |
| Advanced synthesis | `synthesize` | `synthesize` | `synthesize` | `Synthesize` |
| Collect / export | `render`, `render_to_file`, `write_speech` | `render`, `renderToFile`, `writeSpeech` | `render`, `render_to_file`, `write_speech` | `Render`, `RenderToFile`, `WriteSpeech` |
| Completion | `wait` / `completion` | `done` | `wait` / `wait_async` | `Wait(ctx)` / `Completion()` |
| Owned cleanup | `close` / `dispose` | `dispose` | `close` / `aclose`, context managers | `Close` |

Names vary by language, but the scheduler, outcome, source-frame timeline, limits
and errors come from Rust. The [additive C ABI](developer-c-api.md) uses owned
versioned metadata and PCM handles. ABI version 1 and the older `Engine`/job API
remain valid. Developer metadata schema 1 is advertised by feature bit 8;
macOS native output compilation is advertised by bit 16.

## Start with an example

The [Flutter example](../examples/flutter_speak/README.md) demonstrates the speak
button, rapid replace/enqueue/stop, pause/resume, rate/volume, retained seeking,
original-text highlighting, previous/next sentence and WAV export. Its
`DemoController` only owns and presents SDK handles. The app has no PCM method
channel, audio scheduling loop, private native player or WAV serializer.

The small public examples export through the shared writer, with an optional
native speak path:

- [Python](../bindings/python/examples/developer_api.py): bounded export and an
  independently cancellable job; `--speak --muted` exercises native completion.
- [Go](../bindings/go/examples/developer_api/main.go): context/SIGINT cancellation
  and bounded file rendering; `--speak` plays and exports retained audio.
- [Dart](../bindings/dart/example/developer_api.dart): async ownership and bounded
  job writing; the optional `speak` argument plays before retained export.

Use the [Dart](../bindings/dart/README.md), [Python](../bindings/python/README.md)
and [Go](../bindings/go/README.md) guides for state subscriptions, PCM consumption,
request options and native-library configuration. Packages remain development
packages with separately built native binaries; they are not published releases.

## Keep these boundaries explicit

Normal `speakingRate=1.0` maps to raw Supertonic speed 1.05. Playback rate changes
the native time-stretch node and leaves generated PCM and source-frame durations
unchanged. Request inspection exposes copied original text, resolved voice,
effective model settings and original navigation offset.

A synthesis job starts when its single audio consumer subscribes, or on explicit
`start`. Looking at state or waiting for completion does not start or drain it.
State observers are coalesced and bounded. Dart paused subscriptions stop polling;
Python iterators pull on demand; Go channels hold at most the latest pending state.
Closing a state observer does not cancel synthesis. Closing a PCM consumer early
cancels its job with `consumer_dropped`.

Speech returns its identity immediately and finishes only after native playback
has drained and inference is quiescent. Cancellation requests an independent
flush; completion joins any in-flight model call. Dropping a Speech wrapper can
leave fire-and-forget playback running; explicit cleanup is the deterministic
boundary. Go cancellation contexts retain control even if the wrapper is collected.

Default limits include 16 accepted requests, 256 KiB of queued source text,
44,100 queued PCM frames, 4,096-frame consumer chunks, 32 MiB render collection,
128 MiB retained audio and a 30-minute WAV frame limit. These are separate budgets.
Caller-owned clips and external runtime allocations are not included in the queue
or retention counters. The raw input limit remains 64 KiB per request.

Text ranges map the original request: UTF-8 bytes in C/Rust/Go, explicit UTF-8
and UTF-16 spans in metadata, and UTF-16 `SpeechTextRange.start/end` in Dart.
Python provides a safe original-text slice helper. Alignment is segment-level.
Unsupported tokenizer characters return a typed error; correct Unicode offsets
do not imply that every Unicode character has a model token.

Paused cursors also expose their segment at frame zero. Seeking within retention
keeps the current speech identity; replay after completion creates a new identity
and shares the existing recording. Navigation to ungenerated text starts a new
request using the frozen original plan and seed offsets. User WAV files survive
request/facade closure; temporary retained audio is owned and quota-limited.
