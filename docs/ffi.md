# C ABI contract

`include/mlsr.h` is generated from `crates/mlsr-ffi` and is the shared contract
for Dart, Go, Python, and direct C consumers. The Rust crate builds a `cdylib`:
`libmlsr_ffi.dylib` on macOS, `libmlsr_ffi.so` on Linux, and `mlsr_ffi.dll` on
Windows. It also builds a static library (`.a` or `.lib`). Static native consumers
must link the required system libraries/frameworks; see the Core ML build tool.
The macOS build uses an `@rpath` install name and reserves Mach-O header space
so app packagers can relocate the library and rewrite its load commands.

## Compatibility and capabilities

`mlsr_abi_version()` still returns `1`; `mlsr_version()` returns a borrowed,
NUL-terminated UTF-8 string valid while the library is loaded. These original
queries are unchanged. Bindings check the ABI before interpreting layouts.
`mlsr_features()` reports `MLSR_FEATURE_JOBS` (1), `MLSR_FEATURE_COREML` (2)
on Apple builds with Core ML enabled, `MLSR_FEATURE_MODEL_STORE` (4),
`MLSR_FEATURE_DEVELOPER_SDK` (8), `MLSR_FEATURE_APPLE_OUTPUT` (16) on macOS with
native output enabled, `MLSR_FEATURE_QNN_HTP` (32) on Windows ARM64 with QNN enabled,
and `MLSR_FEATURE_WINDOWS_OUTPUT` (64) with Windows WASAPI output enabled. Building the
portable API does not supply a TTS backend. `compute_policy` selects `0` Core ML
CPU, `1` Core ML CPU/Neural Engine, or `2` QNN HTP. ABI version and layouts remain
unchanged. QNN accepts only execution variant `0` with its separately pinned
Superfast manifest; Core ML's optimized variants cannot select QNN graphs.

The QNN headless build reports mask 45; adding Windows output reports 109.
Rebuild experimental QNN consumers and
their native library together: the earlier branch used bit 8 for QNN, before
integrating the main branch's SDK capability. SDK and Apple output bits retain
their main-branch values; no structure layout changes are required.

All options and event/error views have fixed C layouts. Initialize options with
`mlsr_engine_options_default` / `mlsr_job_options_default`; their `struct_size`
must equal the current layout's size. Reserved fields must be zero. Read views
with their exact `sizeof` value. Future incompatible layouts require new types
and entry points or an ABI change, rather than reading past an older struct.

`execution_variant` selects `0` baseline, `1` fused eight-step L128 VE with
fallback, or `2` fused VE plus **experimental** exact L30/L32 vocoders. Value 2
changes PCM and has no perceptual qualification. See [execution variants](internal/execution-variants.md).

## Handles and ownership

`MlsrHandle` is a non-reused 64-bit registry ID, not an address. Zero is null.
Stale IDs and wrong handle kinds fail with `invalid_handle`. Releasing zero is a
successful no-op; releasing a nonzero handle twice is an error. All successful
engine/job/event/error acquisitions require their matching release function.
Failure and timeout/end outputs are zeroed. A non-null error output owns a
diagnostic on failure; read and release it even when no other handle was returned.

`MlsrString` is a UTF-8 pointer plus byte length, without a required terminator.
Input strings are borrowed during the call and copied before asynchronous use.
Output strings and PCM are borrowed views owned by an event or error handle.
Copy them before releasing that owner; never pass them to the caller's allocator.
An event remains valid after job or engine destruction. Bindings copy PCM into
language-owned storage before releasing events.

Callers must supply aligned, valid readable/writable ranges, and disjoint output
pointers. Null checks and length validation cannot make arbitrary invalid C
pointers safe. Do not release an event/error concurrently with reading or using
its borrowed views. An in-flight engine/job operation retains internal ownership
when another thread releases the handle; subsequent lookups fail safely.

## Job lifecycle

1. `mlsr_engine_open` verifies the local YAML bundle and loads persistent models.
   It blocks; call it off UI/event-loop threads. No assets are downloaded.
2. `mlsr_job_start` copies text, language, voice and options. One active job is
   permitted per engine; another submission reports `engine_busy`. Validation
   after submission and inference errors arrive as failed terminal events.
3. `mlsr_job_poll` waits for at most the supplied milliseconds. `MLSR_OK` owns
   an event; `MLSR_TIMEOUT` owns nothing; `MLSR_END` means the terminal event was
   already consumed. Keep one consumer per job.
4. `mlsr_event_read` borrows prediction, audio, or terminal metadata. Release
   every event after use. Exactly one terminal outcome is delivered: completed,
   cancelled, or failed. Engine-local job IDs support stale-result rejection;
   combine them with engine identity when using multiple engines.
5. `mlsr_job_cancel` immediately sets cancellation, discards queued data and
   wakes blocked producers. It does not wait for an in-flight native prediction.
   `mlsr_job_wait` observes quiescence without consuming events; a stopped
   consumer must cancel or resume consuming for a blocked producer to finish.
6. `mlsr_job_release` cancels and waits. `mlsr_engine_shutdown` cancels and joins
   its workers; `mlsr_engine_release` also drops the registry handle. These calls
   can block and belong off UI threads. Existing job/event handles remain owned
   by their callers and must still be released.

Cancellation may run concurrently with polling. Polling, release, model loading,
and worker joins never hold the global registry lock while waiting for native
work. C entry points contain Rust unwinding; an abort or invalid foreign pointer
cannot be recovered this way. A panicked orchestration worker retires its engine.

## Audio, spans and bounds

Audio is mono, finite Float32 PCM at 44,100 Hz. Frame positions are relative to
the job's generated audio. Segment spans are half-open **UTF-8 byte ranges in the
original input**, not normalized text offsets, Python character indices, or Dart
UTF-16 indices. Prediction events precede their segment's audio. Terminal frame
counts record frames published, including frames that cancellation may discard.
Synthesis completion is independent of a host player's playback completion.

Default limits are 44,100 queued frames, 4,096 frames per chunk, and 1,572,864
frames per segment. Queue and segment allocation bounds are separate. Adapters
must preserve pull backpressure; retaining events or copying audio into an
unbounded host queue defeats the application's memory bound. See [jobs](jobs.md)
for segmentation, limits and cancellation semantics.

`mlsr_live_handles()` counts registry entries for leak diagnostics; it is not a
memory or ANE-residency measurement. The public C tests exercise shared/static
libraries, concurrent cancellation, errors, retained audio and complete cleanup.
`tools/check_synthesis_bindings.py` additionally qualifies installed/relocated
Python, Go and Dart packages with real models outside the checkout.
