# MLSR for Go

A `cgo` binding to MLSR's shared C ABI. Requires Go 1.22+, a C compiler, the
generated `mlsr.h`, and the Rust shared library. The developer facade provides
shared speech controllers, native playback, reader controls, bounded rendering
and pull-based PCM. Real synthesis/native playback are currently qualified on
macOS arm64. Build with `--features coreml,apple-output` for speech, or
`--features coreml` for headless synthesis.

```go
tts, err := mlsr.OpenTTS(ctx, manifest, bundle,
    mlsr.EngineOptions{}, mlsr.DefaultTTSOptions())
if err != nil { return err }
defer tts.Close()

speech, err := tts.Say(ctx, "Hello from MLSR.")
if err != nil { return err }
defer speech.Close()
if err := speech.SetPlaybackRate(1.25); err != nil { return err }
if _, err := speech.Wait(ctx); err != nil { return err } // Actual output drain.
_, err = tts.WriteSpeech(speech, "spoken.wav", 0) // Retained PCM, no inference.
```

`Speak` accepts a `DefaultSpeechRequest(text)`, `DefaultSpeechOptions()` and
`Replace`/`Enqueue`; `Say` is the normal-rate replace helper. Options are copied on
admission. The new normalized `SpeakingRate=1.0` is separate from playback rate.
`speech.Request()` / `job.Request()` expose the accepted input and effective
model speed/steps/seed. Voice IDs are stable and model-scoped (`supertonic3:M1`).
`Voices()` gives catalog-only metadata, `tts.Voices()` loaded-engine readiness;
`ResolveVoice` requires explicit `SameLanguage` fallback for regional tags.
`OpenModel(ctx, directory, ModelDescriptor{CatalogYAML: catalog, ID: id,
Platform: platform})` verifies an immutable directory without loading inference;
`model.Voices()` reports `assets_verified`. `TTSFromModel` takes its own native
lease, so `model.Close()` is safe after loading. Model closure waits for concurrent
facade creation. The application supplies the trusted catalog and retains its
files for every derived engine.

`tts.Controller(true)` creates an independent replace/enqueue scope, while
`tts.Controller(false)` returns a caller-owned default-controller handle. Close
every returned controller. Closing the default closes it permanently; it does
not silently create another controller. `TTS.Close` cancels all work, but live
child handles remain inspectable until their own `Close`.

Speech supports pause/resume, volume, pitch-preserving rate, source-frame seek,
previous/next sentence and explicit original-sentence navigation. Navigation
returns `Navigation.Speech` when it starts a new request. Completed recordings use
`ReplayFrom` for a new identity sharing retained PCM; the old outcome is immutable.
Discarding a Speech preserves fire-and-forget playback. A cancellable context
continues to control it after Go finalization; its small completion watcher never
consumes audio and retires the native handle at terminal completion. Keep explicit
handles and `defer Close()` when deterministic resource cleanup is required.

State observation uses a one-slot channel and never drains audio:

```go
states, err := speech.States(observerContext)
if err != nil { return err }
defer states.Close()
for state := range states.C {
    showPosition(state.Position(), state.ActiveTextRange)
}
if err := states.Err(); err != nil { return err }
```

Slow readers skip intermediate revisions; terminal replaces the pending snapshot
before channel closure. `Close` or observer-context cancellation closes only the
subscription. Snapshot fields retain source-frame units through playback-rate
changes. UTF-8 ranges index Go strings; UTF-16 ranges are reported separately.
`speech.Wait(ctx)` cancels the request if its context expires and joins native
quiescence. Use `Completion()` to inspect pending/terminal state without cancelling.
Terminal failures return `*SDKError` with `Failure` category/stage/request/backend;
`errors.As(err, &oldError)` still works through the low-level `*Error` wrapper.

Headless rendering and advanced synthesis use the same Rust scheduler and writer:

```go
request := mlsr.DefaultSpeechRequest("Export me.")
clip, err := tts.Render(ctx, request, 0) // Zero selects the 32 MiB collection cap.
if err != nil { return err }
if _, err := clip.WriteWAV("clip.wav"); err != nil { return err }
if _, err := tts.RenderToFile(ctx, request, "stream.wav", 0); err != nil { return err }

job, err := tts.Synthesize(ctx, request) // Lazy: observing never starts generation.
if err != nil { return err }
defer job.Close()
audio, err := job.Audio() // Claims the only PCM consumer and starts it.
if err != nil { return err }
defer audio.Close()       // Early exit cancels consumer_dropped and joins.
for {
    chunk, err := audio.Next(ctx)
    if errors.Is(err, io.EOF) { break }
    if err != nil { return err }
    consume(chunk.Audio.Samples, chunk.Audio.Format)
}
```

There is no Go PCM producer/queue; a slow consumer backpressures Rust. `Start`
explicitly starts a lazy job without a consumer. Render/file contexts cancel
independently of blocking C calls. Files are finalized before publication,
existing paths are preserved, and only owned partial files are cleaned up. The
file cap defaults to 30 minutes; TTS options also bound queue/chunk/retained-store
budgets. `HeadlessOutput` disables device creation in a playback-capable build.

The original low-level API remains available:

Use `Open(manifest, bundle, EngineOptions{})`, then `Start(ctx,
DefaultRequest(text))`. Customize language, voice, seed and limits on that request
before starting it. `Next()` pulls one event at a time and returns `io.EOF` after
one completed/cancelled/failed terminal event. Check `event.Failure` on failure.
Call `Close()` on every job and engine, normally with `defer`; it cancels and waits
for native work. A competing job reports `*mlsr.Error` with code `engine_busy`.

The context watcher cancels even when the consumer stops calling `Next`; it does
not drain or buffer audio. `Cancel()` is immediate, while `Wait(timeoutMS)` waits
for quiescence. Returned samples are Go-owned and survive cleanup. Event text
spans use original UTF-8 byte offsets, matching Go string indices. Keep one pull
consumer; `Cancel`/`Close` may run concurrently with `Next`.

From the repository root, run `python3 tools/dev.py test-go`. The development
tool supplies the include, link, and runtime library paths. The Go module itself
does not depend on paths outside its package, so external consumers can install
the native header/library and configure `CGO_CFLAGS` and `CGO_LDFLAGS` themselves.

For manual builds, set `CGO_ENABLED=1`, add the header directory with `-I` to
`CGO_CFLAGS`, add the library directory with `-L` to `CGO_LDFLAGS`, and make the
shared library available to the OS loader. Linking `mlsr_ffi` is handled by the
binding. Cross-compilation also requires a target C toolchain and target Rust
library. Builds with `CGO_ENABLED=0` are not supported.

For example, after building the Core ML library from the repository root:

```sh
MACOSX_DEPLOYMENT_TARGET=15.0 cargo build --locked --release -p mlsr-ffi --features coreml
export CGO_CFLAGS="-I$PWD/include"
export CGO_LDFLAGS="-L$PWD/target/release -Wl,-rpath,$PWD/target/release"
cd bindings/go
go run ./examples/synthesize --manifest ../../model-manifests/supertonic3-coreml.yaml \
  --bundle /path/to/bundle --output hello.wav --cancel-first
```

The example cancels one job, reuses its engine and streams a new WAV file.
`tools/check_synthesis_bindings.py --bundle /path/to/bundle --output-dir /new/path`
qualifies a relocated module with real synthesis, `go test -race` and this example.

`sdk_test.go` covers portable catalog/file ownership. Set `MLSR_TEST_MANIFEST`
and `MLSR_TEST_BUNDLE` for shared-facade tests, and explicitly set
`MLSR_TEST_NATIVE_OUTPUT=1` for muted native reader/GC/context tests. The
[installed SDK qualification](../../docs/internal/developer-api-qualification.md) passes
with `go test -race`, including exact PCM/WAV, state coalescing, verified model
ownership and context-controlled fire-and-forget. The
[developer API example](examples/developer_api/main.go) uses bounded file rendering
and SIGINT cancellation; `--speak` additionally plays and exports retained speech.

The module path is `github.com/yunho-c/MLSR/bindings/go`. Future release tags for
this nested module must use the `bindings/go/` prefix, such as
`bindings/go/v0.1.0`.

Licensed under Apache-2.0; see `LICENSE`.
