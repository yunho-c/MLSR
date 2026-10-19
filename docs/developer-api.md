# Developer API v0 contract

This is the implementation contract for the developer API designed in
`ignore/chatgpt-conv1.md`. It extends the qualified Core ML synthesis slice;
completion of that earlier slice does not imply completion of this SDK.
[Progress and evidence](internal/developer-api-progress.md) track the eight accepted steps.

The v0 target is the full essential API on macOS Apple Silicon, with shared Rust
behavior and C/Dart/Go/Python interfaces. Device/backend availability is a separate
capability. QNN/Android, physical iOS qualification, optional word alignment,
persistent synthesis caching and background media controls follow this milestone.

## Ownership

`mlsr` owns synthesis, request/session scheduling, state, source mapping, retained
audio and timeline policy. `mlsr-playback` supplies optional platform output.
Language adapters translate these contracts; the Flutter example must not own a
second scheduler or speech state machine. The existing `bindings/flutter` package
continues to provide Flutter model delivery and host integration.

One `Tts` owns its engine and a default speech controller. It may be opened for
headless synthesis without an output device. All accepted synthesis work uses a
bounded FIFO scheduler over the engine's existing single inference lane. Playback
may continue after that lane finishes a request. A paused/backpressured generation
may hold the lane; later work waits until it progresses or is cancelled. There is
no promise of parallel inference. Separate `Tts` instances have separate engines.

The default output is opened lazily for speech. Optional output dependencies must
not burden headless builds. Custom outputs implement the same capacity, epoch,
clock and completion contract; unsupported operations return typed errors.

## Public operations

| Operation | Contract |
| --- | --- |
| `Tts.open` | Resolve/verify existing model delivery, load one engine; return only when ready. No implicit model download. |
| `getVoices` / `resolveVoice` | Catalog metadata and explicit locale/fallback selection, independent of loading inference graphs. |
| `speak` | Snapshot options, return a `Speech` immediately, schedule generation/playback. Default mode is replace. |
| `render` | Collect a completed `AudioClip` under a finite configured memory cap. |
| `renderToFile` | Incrementally write WAV; return path/format/frame metadata after successful finalization. |
| `synthesize` | Create an advanced single-consumer synthesis handle with separate state observation. |
| `close` / `dispose` | Cancel owned requests, flush output, join in-flight work and writers, release handles and owned temporary files. Idempotent. |

Existing `Engine.start` remains eager and busy-on-conflict. Existing C structs,
UTF-8 offsets and exported functions keep their meaning. New C functions/types are
additive and capability-checked; generated headers and bindings change together.

The new advanced `SynthesisJob` is lazy until its audio consumer subscribes or an
explicit `start()` is called. State observation does not consume PCM or start
generation. `done` alone does not start a lazy job. Once started, its producer
obeys native backpressure even without a consumer. Cancellation before start is
a valid terminal outcome. Facade render/speech methods attach consumers themselves.

## Requests, voices and text

Options and input strings are copied/frozen when accepted. Changing defaults
cannot mutate active or queued requests. Model-specific steps/seeds remain typed
backend options; execution policy/variant remain engine configuration. Public
`speakingRate=1.0` maps to the backend's declared normal setting (currently raw
Supertonic speed 1.05); the low-level raw speed API remains unchanged. The effective
speed is recorded in request metadata. `playbackRate` never changes generated PCM.

Voice metadata contains a stable model-scoped ID, display name, supported language
tags, model/revision, runtime capabilities, network requirement and availability.
`supertonic3:M1` identifies the same conditioning voice across Core ML and future
QNN; the raw M1/F1 aliases remain accepted by the low-level API. Do not invent an
accent/region from a language-wide model tag. Explicit fallback can select the
same supported language; strict resolution must not silently choose another.
Catalog existence, verified asset availability and engine readiness are distinct.

A frozen segment plan retains original text, sentence/segment IDs, model input,
and original source ranges. Canonical offsets are half-open UTF-8 bytes. The
public Dart `SpeechTextRange` uses half-open UTF-16 code units into the original string.
Both representations are derived from one checked mapping; normalized positions
must never be used to slice original text. Existing byte-offset fields remain
documented bytes. Navigation reuses the original plan and original segment seed
offsets. Unicode scalars and extended graphemes are not cut by input splitting.
The current 64 KiB raw-input and per-model segment limits remain explicit.

## Speech identity and terminal semantics

Every returned handle has a stable request ID and one immutable final outcome.
Queued requests replaced before inference also terminate. The handle's last state
and outcome remain inspectable until disposal; disposal does not invent success.

| Outcome | Meaning |
| --- | --- |
| completed synthesis | All generated PCM has reached the selected consumer and its final metadata is settled. File rendering also finalized the file. |
| completed speech | Input ended and the output reports that the final audio finished playback, including its processing/device drain. |
| cancelled | Requested cancellation, replacement, navigation or owner closure; reason is preserved. |
| failed | Typed category, stage, message, request ID and available backend information. Partial audio may already have played. |

`Speech.done` resolves completed/cancelled outcomes and reports failures as the
language's typed error. Rust/C retain a structured failed outcome; Dart/Python
futures raise the corresponding error, and Go returns it from completion wait.
`cancel()` immediately invalidates/flushes playback; its asynchronous completion
waits for native quiescence and cleanup. An in-flight model call is not preempted.

Replace cancels that controller's active and queued speech, then appends the new
request. It never cancels an unrelated render or another controller/engine.
Enqueue preserves FIFO order; pause does not skip to the next speech. Queue count
and total queued UTF-8 bytes have finite configurable limits. Overflow is an
explicit error, never silent loss. Cancelling one queued handle removes only it.

## State observation

State snapshots have a monotonic revision and independent synthesis/playback
states, e.g. queued/running/finished and buffering/playing/paused/ended. Terminal
outcome is separate and guarded exactly once. Empty output while synthesis is
active means buffering, not success. Paused playback can coexist with completed
synthesis.

Each state subscriber gets the latest snapshot first. Observation is independent
of the single PCM consumer. Position updates may coalesce; observers are not an
audit log. Per-subscriber buffering is bounded, including paused listeners, and
the terminal snapshot is retained. A slow observer must not block playback or
create unbounded native/language queues. No user callback runs on an audio thread.

## Output adapter and controls

Output commands and callbacks carry a playback epoch. Stop, replacement and seek
advance it; callbacks from older epochs cannot finish or restart current speech.
The output accepts bounded PCM and reports available capacity without blocking
the controller's cancellation path. Native scheduling, not Dart/Go/Python PCM
round-trips, is the default system playback path.

Pause freezes playback position. For retained speech, lookahead gates the start
of another model segment; an admitted segment finishes and its complete PCM is
retained and seekable even while paused or output is full. PCM queues and native
output remain independently bounded. Resume preserves the current media position.
Volume is independent of synthesis. Playback-rate changes preserve pitch and the
source-audio timeline; a 60-second clip remains a 60-second clip at 1.5x. Supported
rate ranges are advertised and validated. Position comes from the output's media
clock, with documented processing/device latency limits, never synthesis wall time.

The Apple implementation uses AVAudioEngine/player nodes. Played-back completion
also fires when a player is stopped, so epoch/cancellation state must guard it.
Native callback work is minimal; it must not call stop/reconfigure synchronously.
Processing-latency and time-stretch position mapping are explicit qualification
gates, not inferred from a successful build or buffer-accepted callback.

Host-managed audio policy is supported. Route/interruption events enter the same
ordered controller path. Losing the current route pauses safely; interruption end
does not override an explicit user pause/cancellation. Platforms advertise actual
policy capabilities. Mobile audio-session execution is not claimed from a macOS
test. Existing bundled/offline model delivery and release sandboxing are retained.

## Retention, timeline and navigation

Generation lookahead, PCM queues, render collection and disk retention have
separate finite limits. The initial reader mode retains complete generated audio in an
owned temporary store under a configurable disk quota. Quota failure is typed;
there is no implicit eviction or hidden loss of seekability. A no-retention mode
may expose only its actual available ranges. User output files are not temporary
store ownership and are never deleted by session cleanup.

| Timeline field | Meaning |
| --- | --- |
| position | Presented source-audio media position; frozen while paused. |
| bufferedUntil | End of contiguous audio ready to supply ahead of the position. |
| seekableRanges | Exactly the retained source-audio intervals available for replay. |
| estimatedDuration | Nullable prediction for the whole request; partial segment sums are not a whole-request estimate. |
| duration | Authoritative length from final emitted frames, once all output is known. |
| activeTextRange | Original text corresponding to the presented segment, or the selected paused cursor including frame zero; no fabricated word timestamps. |

Seek is valid only inside retained ranges. A seek outside them returns an explicit
result without moving playback. Valid seeks flush/invalidate old scheduled audio,
rebase the output clock and feed the retained store from the requested frame.
Seeking while paused remains paused. Seeking never re-synthesizes retained PCM.
The retained endpoint is a valid cursor as well. After playback has completed,
`replay_from` returns a new `Speech` sharing the recording; the original outcome
stays immutable and the new `done` represents the replay's actual drain. State
subscriptions end with their request and callers observe the new replay handle.

Previous/next sentence use sentence IDs from the plan, including sentences split
into several model segments. Navigation to ungenerated text is an explicit new
speech request from that plan position, with original source mappings and seeds;
the previous handle terminates as navigation-cancelled. It does not pretend that
ungenerated text has a known time offset. A cancellable duration-only pass
predicts the complete request before PCM generation and reuses those decisions
during synthesis; final PCM duration always remains authoritative. The pass
adds preparation latency proportional to the number of remaining segments,
without expanding audio queues or the segment admission horizon. UI timeline
scales use the final duration or whole-request estimate, with buffered/seekable regions
shown separately and an explicit unknown-total state while planning.
Navigation returns an explicit retained seek, new speech handle, or document
boundary. Pending requests retain their order behind a new navigation request;
an older handle cannot navigate over another current speech. See the
[reader API](reader-api.md) for budgets, ownership and qualification.

## Rendering and errors

WAV serialization and format/frame metadata are shared Rust functionality. File
rendering is incremental and bounded. Existing files are preserved by default;
only owned incomplete files may be cleaned up. Completion is published only after
header/data finalization succeeds. Disk full, quota exceeded, invalid seek,
unsupported capability, invalid input, unavailable voice, engine/output failure
and closed handles have stable distinct categories.

Tests cover fake deterministic output and real Apple output separately. The
accepted synthesis reference remains the numerical oracle. Qualification must
distinguish first PCM from first playback, cancellation-to-flush from inference
quiescence, and API coverage from device/backend/distribution qualification.
