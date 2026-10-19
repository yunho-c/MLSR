# Shared headless developer API

The Rust `Tts` facade owns a loaded engine and one FIFO inference scheduler.
`Tts::open` accepts an existing verified `ModelHandle` and returns when ready;
`Tts::new` accepts an already loaded engine. Neither opens an audio device or
downloads assets. The original eager/busy-on-conflict `Engine` API is unchanged.

`Request` copies source text, resolved voice and options. `RequestOptions` keeps
`speaking_rate` separate from playback rate. Public rate 1.0 maps once to
Supertonic's raw speed 1.05; steps and seed use typed `BackendOptions`. Effective
low-level options are available for reproducing results.

QNN Superfast requires eight steps. Requests reject other step counts before
queue admission or speech replacement, and reject voices from another model
release. Fixed text/latent limits still produce typed failures; the SDK does not
silently split extra segments or select another preset. Voice metadata and
structured failures identify the actual backend (`qnn` or `coreml`). See
[QNN setup and qualification](qnn-runtime.md) for headless Windows ARM64 usage.

`tts.synthesize(request)` immediately returns a lazy `SynthesisJob`. Observing
`snapshot()`, `states()` or `wait()` does not start it. Calling `audio()` claims
the sole consumer and starts generation; `start()` can explicitly start without
a consumer. Generation then stops at bounded backpressure until consumed or
cancelled. Dropping the audio stream cancels unfinished work. A state subscription
stores only a revision and reads one shared latest snapshot, including a retained
terminal snapshot for late observers. There is no per-observer event queue.

Generation can reach `Generated` while queued PCM awaits consumption. Successful
job completion follows consumption of all PCM and native quiescence. Cancellation
is immediate as a request, but its terminal outcome waits for an in-flight native
call. The first cancellation reason is retained. Structured failures retain stage,
category, request identity, backend and diagnostic. During `Preparing`, a
cancellable duration-only pass plans every remaining frozen segment. It
publishes a whole-request estimate before PCM delivery, independently of a
stalled consumer, and synthesis reuses those decisions. A partial sum is never
presented as a total. Planning failures are reported at the planning stage;
final duration still comes from emitted frames. This moves duration prediction
ahead of first audio without increasing PCM budgets or duplicating model calls.

`tts.render` collects an `AudioClip` under a configured byte cap (default 32 MiB).
`AudioClip::write_wav`, `tts.render_to_file` and `tts.write_job` use the same Rust
Float32 WAV writer as the CLI. File rendering consumes bounded chunks, uses a
64 KiB file buffer and defaults to a 30-minute source-frame cap. It finalizes and
syncs an owned temporary before no-clobber publication. Failed writes, quotas,
cancellation or finalization leave no partial destination; existing/user-created
files are preserved. `collect`/`write_job` allow an advanced job to remain separately
cancellable and refuse another facade's job or an already claimed consumer.

Queue limits cover all accepted nonterminal requests, including lazy and
undrained ones: default 16 requests and 256 KiB of source text. The native PCM
queue, consumer PCM queue and one generated segment each have explicit limits.
The consumer queue also has a 32-chunk cap, so tiny chunks cannot accumulate
unbounded metadata. Caller-owned clips/chunks and retained handles are outside
these queue budgets. `close` rejects new work, cancels/join requests and waits for
render/file cleanup. File publication and closure are ordered: a publication that
has begun finishes before close returns; a close that wins prevents publication.

## Validation

Portable tests cover lazy start, independent/current/terminal observation,
single-consumer ownership, queue/count/text/chunk bounds, FIFO order, queued and
in-flight cancellation, first cancellation reason, paused/stalled consumers,
owner isolation, typed failures/reuse, render limits, incremental PCM parity,
no-clobber races, finalization I/O failures and close during a live writer.

`cargo test --locked --workspace`, strict all-feature/all-target clippy, Rust 1.88
workspace checks and Windows core target compilation passed. External native
tests remain explicitly opted into; portable success is not a device result.

`real_facade_render_stream_and_file_preserve_legacy_pcm` passed on the local Core
ML macOS host with the verified baseline, eight steps, raw speed 1.05 and seed
1729. Its two multi-sentence cases match every Float32 sample bit across the
existing engine, facade render, advanced stream and incremental WAV:

| Case | Frames |
| --- | ---: |
| English / M1 | 160,493 |
| Korean / F1 | 162,011 |

The retained native test log is under
`~/.cache/mlsr/qualification/developer-api-v0/step3-headless-v1/native-test.log`.
This establishes the shared headless layer. The [developer SDK guide](developer-sdk.md)
covers the implemented speech/output and language surfaces; the
[acceptance checklist](internal/developer-api-progress.md) tracks their combined qualification.
