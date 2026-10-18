# Shared speech controller

`Tts::with_output` installs an `OutputFactory` without opening a device. `speak`
uses a reusable default `SpeechController`; `create_speech_controller` creates an
independent queue over the same inference lane. All speech policy lives in Rust.
An adapter supplies bounded native enqueue, clock, pause/rate/volume, flush,
played-back completion and system events. It is created and used on one output
thread, so platform graph objects do not need to cross threads or implement Send.

Each accepted `speak` returns a `Speech` immediately. Replace cancels only that
controller's active and queued speech; enqueue preserves FIFO order. Admission
or capability failure leaves existing speech intact. Queue count/text limits come
from the shared scheduler; replay queues separately obey those pending count/text
limits plus one active request. A cancelled queued handle terminates without inference.
There can be at most 16 live controllers per owner. Renders and other controllers
remain independent of replacement. Dropping a Speech handle preserves convenient
fire-and-forget playback; explicit cancellation or controller/Tts closure stops it.

`Speech` exposes current/coalesced state, completion, pause/resume, volume and
pitch-preserving playback-rate commands. State reports generation and playback
separately, including buffering, playing, pause and final device drain. State
observers retain only a revision and one shared current snapshot. Commands are
accepted synchronously; state confirms their native application. Unsupported
controls/rates fail before modifying the current request.

Synthesis completion does not resolve Speech completion. The controller declares
end of input only after it has passed every PCM chunk to output, then waits for a
matching-epoch played-back drain. Cancel flushes output independently of inference
backpressure, then waits for in-flight synthesis to quiesce before its outcome.
Every terminal outcome is immutable; a cancellation that wins the final state
transition retains its reason even when a played-back callback races with it.

Replacement/cancellation advance playback epochs. Stale output snapshots cannot
finish current speech. Output failure retires that graph, cancels/joins its
synthesis and reports a typed failure; the next speech may open a fresh graph.
Source-frame positions, whole-request estimates, authoritative duration and
original UTF-8/UTF-16 segment highlighting remain separate values. The
[reader layer](reader-api.md) adds retention, seeking, completed-recording replay
and sentence navigation without another scheduler.

Route changes request pause. Interruption end does not override a user pause;
host-managed policy requires an explicit resume. These are shared policy rules,
tested with ordered simulated events. Native route/interruption support must be
advertised and qualified separately by each platform adapter.

## Validation and remaining integration

The original ten deterministic output tests cover lazy opening and graph reuse; playback drain
after generation; replace scope; FIFO/pause/rate/volume; prompt flush during a
gated in-flight model call; queued cancellation; stale callback rejection; a
controlled cancel-versus-final-completion race; admission failure preservation;
output failure/recovery; route/interruption policy and fire-and-forget lifetime.

`cargo test --locked --workspace` passed (60 core tests, seven CLI tests and four
FFI tests; two external-asset core tests skipped). Strict all-feature/all-target
clippy, Rust 1.88 workspace checks and Windows core target compilation passed.
Those step-4 counts are historical. Subsequent [Apple adapter](apple-output.md)
and [reader](reader-api.md) tests add native and retained-audio qualification.
New C/language facade surfaces, Flutter migration and full installed/native
acceptance remain tracked separately.
