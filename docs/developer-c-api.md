# Developer C API

The additive SDK uses ABI 1 with feature bit `8`. Existing engine/job functions,
layouts, raw model speed and UTF-8 event offsets retain their meanings. Bit `16`
advertises compiled macOS native output; model execution still needs bit `2` and
a verified model. See the generated [header](../include/mlsr.h) for every pointer,
layout and ownership contract. Initialize options with their default functions.

`mlsr_tts_open` loads paths; `mlsr_tts_from_model` retains an existing verified
model lease. Both block during loading. Output is lazy. Facade output mode `1`
forces headless operation; `2` selects host-managed output or returns an explicit
unsupported-capability error. Mode `0` uses the compiled platform default. Queue,
chunk, lookahead and shared retained-storage budgets are validated before use.

## Handles and metadata

All handles are checked registry IDs, not pointers. A returned handle belongs to
the caller. Release SDK handles with `mlsr_sdk_release`, old model/error/event
handles with their existing functions. Wrong-kind release preserves ownership.
Release of Tts/controller/speech/synthesis/stream cancels and joins its owned work;
run it off a UI/event-loop thread. In-flight calls keep safe Rust references.

`mlsr_sdk_forget` is restricted to speech and synthesis. Forgetting speech does
not cancel fire-and-forget playback. Forgetting synthesis drops that ownership;
a live PCM consumer still owns its job. Explicit disposal uses release. Closing a
facade does not release caller-owned child registry IDs or saved metadata. The
caller must still release those handles; `mlsr_live_handles` diagnoses registry
ownership, not physical memory.

Metadata is UTF-8 JSON in an owned SDK handle:

```json
{"schema_version":1,"type":"speech_state","data":{}}
```

`mlsr_json_read` borrows its bytes until release. Copy, decode, then release. The
v1 type names are `voice_catalog`, `resolved_voice`, `tts`, `controller`,
`speech_state`, `synthesis_state`, `audio`, `completion`, `text_plan`,
`navigation`, `wav_file`, `request`, and `error`. Field names match the serializable Rust
types, with snake-case enum tags. Ranges have `start` and exclusive `end` fields;
explicit `utf8` and `utf16` ranges retain original text coordinates. A duration
is unknown when null. Playback rate does not change source-frame units.

PCM is never JSON. `mlsr_audio_read` borrows an interleaved Float32 view from an
audio handle. Copy before release; releasing concurrently with a borrowed view
is invalid. Language adapters copy once into language-owned storage. The shared
`mlsr_pcm_write_wav` serializer borrows caller PCM only for its synchronous call.

## Synthesis and playback

`mlsr_sdk_request` exposes the copied input/resolved voice, public speaking rate,
typed backend options, effective model speed/steps/seed and original start segment.
This observation works before planning without starting a lazy request. Replays
report their own request identity and their shared original synthesis input.

`mlsr_tts_synthesize` creates a lazy job; observing state, plan or completion does
not start it. `mlsr_synthesis_start` starts explicitly; `mlsr_synthesis_audio`
claims its sole consumer and starts it. A second consumer fails without harming
the first. `mlsr_audio_stream_poll` returns exactly one audio/completion handle
on OK, no handle on TIMEOUT, and END after terminal delivery. Abandoning the
stream cancels with `consumer_dropped` and joins without draining PCM.

`mlsr_tts_controller` selects the facade's default controller or creates an
independent controller. `mlsr_speak` accepts either owner and returns immediately
after admission. Replace/enqueue and cancellation run in Rust. No playback PCM
crosses the language boundary. `mlsr_speech_command` applies cancel, pause,
resume, volume or independent pitch-preserving rate. Seeking uses retained source
frames. Completed-recording replay returns a new speech identity sharing stored
PCM. Navigation returns boundary/seeked/started metadata and a new speech handle
only for started. Original text/segment/sentence identities survive navigation.

`mlsr_sdk_snapshot`, `mlsr_sdk_wait` and `mlsr_sdk_states` are independent of PCM.
Each state watcher stores a revision and shared latest state, not an event log.
Polls return current/latest/terminal state, then END; concurrent polls on the
same consumer return `consumer_busy`. Blocking polls are capped at 30 seconds.
Language adapters must preserve this bounded/coalesced behavior and never create
an unbounded broadcast or paused-listener queue.

Failed requests still produce a completion with `outcome.kind=failed` and a typed
failure (`category`, `stage`, `request_id`, `backend`, `message`). High-level done
futures raise that failure; cancellation resolves normally with its first reason.
Synchronous errors retain the old code/message view and add the same structured
failure through `mlsr_error_details` when applicable.

## Rendering and export

`mlsr_tts_collect` / `mlsr_tts_write_job` take explicit lazy jobs so the caller can
cancel while a blocking render runs elsewhere. Convenience render/file calls
create a job internally. Collection defaults to 32 MiB; file writing defaults to
30 minutes. Existing files are preserved, partial temporary files are owned and
removed, and success is published only after finalization.

`mlsr_tts_write_speech` exports a completed retained recording belonging to that
facade with bounded reads, without inference or a full PCM copy. Export failure
does not change the immutable speech outcome. Owner close joins active writers;
user-published files survive close and speech disposal.

Portable ABI tests cover metadata ownership, explicit locale fallback, structured
errors, old error compatibility, option sizes, PCM validation, no-clobber writing
and partial cleanup. Shared reader tests verify retained export sample identity,
zero extra inference, foreign-owner rejection and file survival. Cross-language
and installed-package acceptance is tracked separately in
[the progress checklist](developer-api-progress.md).
