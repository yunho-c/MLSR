# MLSR for Python

A `ctypes` binding to MLSR's shared Rust developer API. `Tts` provides native
speech, independent controllers, reader controls, bounded rendering, WAV export
and lazy advanced synthesis. Real synthesis and native playback are currently
qualified on macOS arm64. Build with `--features coreml,apple-output` for speech,
or `--features coreml` for headless synthesis.

Requires Python 3.10+ and a separately built MLSR shared library. Importing `mlsr`
does not load native code. The first API call loads `MLSR_LIBRARY_PATH` if set,
otherwise it looks for `mlsr_ffi` on the system library search path. Set the
environment variable before the first call; the loaded library is cached.

```python
import mlsr

with mlsr.Tts("model-manifests/supertonic3-coreml.yaml", "/path/to/bundle") as tts:
    print(tts.voices)  # Stable IDs and loaded-engine readiness.
    with tts.speak("Hello from MLSR.") as speech:
        speech.set_playback_rate(1.25)  # Pitch-preserving, independent of synthesis.
        speech.wait()                 # Actual playback completion, not just PCM EOF.
        tts.write_speech(speech, "spoken.wav")  # Retained audio; no new inference.

    clip = tts.render(mlsr.Request("Export me.", voice="supertonic3:F1"))
    clip.write_wav("clip.wav")
    tts.render_to_file("Stream directly to a file.", "streamed.wav")
```

`speak` returns immediately; native PCM stays in Rust. Default mode is `replace`
within `tts.controller`; use `mode="enqueue"` or `tts.create_controller()` for a
separate scope. Speech supports pause/resume, volume, playback rate, retained-frame
seek, previous/next sentence and explicit original-sentence navigation. A
navigation result contains a new `speech` when inference must restart. Completed
recordings use `speech.replay_from(frame)` to create a new identity; the original
completion remains immutable. Discarding a Speech object preserves fire-and-forget
playback. Explicit `close()` cancels/joins and releases its retained recording.
`speech.request` / `job.request` expose the copied accepted input, resolved voice,
effective model speed/steps/seed and original navigation start segment.

Read `speech.snapshot` or use `with speech.states() as states: ...` to observe
coalesced state without draining PCM. State includes source-frame position,
buffered end, seekable ranges, estimate/final duration and original text alignment.
`state.active_text_range.text(original)` slices Python strings correctly; `utf8`
and `utf16` are explicitly separate from Python character indices. Voice IDs are
model scoped (`supertonic3:M1`); locale fallback is explicit. `mlsr.voices()` gives
catalog metadata without loading models or output, and `tts.voices` gives loaded
engine readiness. `ModelHandle(directory, ModelDescriptor(catalog_yaml, id,
platform))` verifies an immutable directory without loading inference; its
`voices` report `assets_verified`. `Tts.from_model(model)` takes its own native
lease, so the model handle can be closed after loading. Both provide asynchronous
loading variants (`ModelHandle.open_async`, `Tts.from_model_async`). The catalog
must come from trusted application configuration and the application must retain
its model files for every derived engine.

Use `TtsOptions(output_mode="headless")` to disable output even
in a playback-capable build.

For asyncio, loading, blocking iteration, render, file writes and cleanup have
off-loop variants. No Python audio queue grows behind a slow consumer:

```python
async with await mlsr.Tts.open_async(manifest, bundle) as tts:
    async with tts.synthesize(mlsr.Request(text)) as job:
        assert job.snapshot.phase == "created"  # Observing does not start it.
        async with job.audio() as audio:        # Claims the only PCM consumer.
            async for chunk in audio:
                await consume(chunk.audio.samples, chunk.audio.format)
        await job.wait_async()

    clip = await tts.arender("Render off the event loop.")
    await tts.arender_to_file("Incremental export.", "async.wav")
```

Async task cancellation cancels the relevant render/job/speech and joins native
work before returning cancellation. State-observer cancellation closes only that
observer. Closing an unfinished PCM stream uses `consumer_dropped`. Use `with` /
`async with` for early iterator exits; an ordinary loop break does not close an
arbitrary Python iterator. Retained `awrite_speech` and `AudioClip.awrite_wav`
finish their bounded file operation before propagating task cancellation.

`Completion.outcome` distinguishes completed/cancelled/failed and records the first
cancellation cause. `wait` / `wait_async` raise `TtsError` on failure with category,
stage, request ID and backend. Snapshots retain that failure for late observers.
`wait(timeout=seconds)` returns `None` on timeout. Rendering defaults to a 32 MiB
collection budget; file rendering defaults to 30 minutes. Options expose queue,
chunk, lookahead and shared retained-storage caps. Writers preserve existing paths
and clean up only owned partial files. Returned samples remain Python-owned after
native cleanup. Always close caller-owned controllers/jobs/speeches/streams; owner
close ends work but preserves live child handles for their final observations.

The existing low-level `Engine` / `Job` API remains available with its original
raw speed and event meanings:

```python
import mlsr

with mlsr.Engine("model-manifests/supertonic3-coreml.yaml", "/path/to/bundle") as engine:
    with engine.start("Hello from MLSR.", language="en", voice="M1") as job:
        for event in job:
            if event.kind == "audio":
                consume_pcm(event.samples, event.sample_rate)  # Your bounded consumer.
            elif event.kind == "failed":
                raise event.error
```

Reuse an engine between jobs. `job.cancel()` wakes a blocked producer without
waiting for native inference; `job.wait(timeout_ms=100)` observes quiescence.
Context exit / `close()` cancels and waits. One active job is accepted per engine;
another raises `SynthesisError` with code `engine_busy`. Iteration has one consumer
and no hidden audio queue. Returned `array('f')` samples survive cleanup. Text
spans index the original input's UTF-8 bytes, not Python character positions.
Blocking calls release the GIL; run them on a worker thread in an async/UI app.

From the repository root, build the native library and run the WAV example:

```sh
MACOSX_DEPLOYMENT_TARGET=15.0 cargo build --locked --release -p mlsr-ffi --features coreml
MLSR_LIBRARY_PATH="$PWD/target/release/libmlsr_ffi.dylib" \
  PYTHONPATH=bindings/python/src python3 bindings/python/examples/synthesize.py \
  --manifest model-manifests/supertonic3-coreml.yaml --bundle /path/to/bundle \
  --output hello.wav --cancel-first
```

`--cancel-first` demonstrates cancellation and reuse before exporting speech.
The output must be a new path; model weights are never downloaded implicitly.

From a repository checkout, run `python3 tools/dev.py test-python` at the root.
`python3 tools/dev.py package-python` builds a wheel and tests an installed copy
outside the source package. The wheel currently contains Python code only and
requires the external native library; bundled native wheels are future work.
`tools/check_synthesis_bindings.py --bundle /path/to/bundle --output-dir /new/path`
qualifies an installed wheel with real speech, cancellation and a WAV example.

`tests/test_sdk.py` includes catalog/file/async ownership tests without model
assets. Set `MLSR_TEST_MANIFEST` and `MLSR_TEST_BUNDLE` for real shared-facade
stream/render/file equality and cancellation tests; additionally opt into muted
device tests with `MLSR_TEST_NATIVE_OUTPUT=1`. These SDK checks passed on the local
macOS Core ML/native-output host. Final multi-language/package qualification is
tracked in `docs/developer-api-progress.md`.

Licensed under Apache-2.0; see `LICENSE`.
