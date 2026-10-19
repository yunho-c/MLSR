# Reader controls and retained recordings

Speech supports two storage policies. The default retains every generated frame
for recording/export. Opt into **on-demand seeking** for a reader that can seek
anywhere in the planned document, generating uncached passages directly and
evicting old cached segments when necessary. The Flutter example uses this mode.

| Binding | Enable on-demand seeking |
| --- | --- |
| Rust | `SpeechOptions::default().with_on_demand_seeking(true)` |
| Dart | `SpeechOptions(onDemandSeeking: true)` |
| Go | Set `DefaultSpeechOptions().OnDemandSeeking = true` on the options value. |
| Python | `SpeechOptions(on_demand_seeking=True)` |
| C | Set `MlsrSpeechOptions.retain_audio = 2` after initializing defaults. Values 0 and 1 retain their no-storage/full-recording meanings. |

On-demand mode requires retention and an output adapter that can flush/rebase.
Invalid options fail admission without replacing current speech. Disabling
retention omits seeking storage. Playback and rendering remain separate:
`Tts::render` returns caller-owned memory, and WAV exports are caller-owned files
that reader cleanup never deletes.

## Seeking across a document

Before PCM generation, a cancellable duration-only pass predicts every segment
in the frozen text plan. It retains small duration/shape decisions, releases
native tensors, and reuses each prediction during synthesis. There is no partial
sum masquerading as the whole duration. Planning adds preparation latency
proportional to segment count; seeking is unavailable until it finishes.

An on-demand seek resolves the selected time to an **original segment and local
frame offset**. The controller flushes old scheduled audio, generates only the
missing destination segment, then supplies PCM from that offset. Skipped segments
are not generated. Cached destinations reuse their exact PCM. The document,
original segment IDs/seeds, source mappings and active speech identity survive
seeks, as do pause, playback rate and volume. Sentence navigation uses the same
path, including for ungenerated sentences.

Seek admission returns without waiting for inference or file I/O. `seeking` and
`seek_target_frame` expose the pending destination until output has accepted its
audio (or the document endpoint). The latest seek wins; obsolete generation is
cancelled before a replacement is scheduled. Cancellation takes effect at native
checkpoints and must wait for an in-flight model call, so a cold destination can
show a loading state. This is not a zero-latency or hard real-time guarantee.

Measured segment lengths refine the timeline. Pending and playing positions are
anchored to segment/local offset, so an earlier correction cannot select different
text. An offset beyond a segment's actual end clamps to that end. Original segment
identity and synthesis settings are preserved when evicted audio is regenerated.
The whole scale can change when predictions are corrected; neither estimated nor
actual source duration changes with playback rate.

| State | Meaning |
| --- | --- |
| `position_frame` | Confirmed output cursor. A seek can move it while paused or buffering. |
| `seekable_ranges` | In on-demand mode, the full planned document `[0, total)`. In full-recording mode, actual retained audio. The endpoint is a valid cursor. |
| `buffered_ranges` | Actual ready cached audio; potentially disjoint. Partial/cancelled reader segments are never advertised as ready. |
| `buffered_until_frame` | End of the contiguous cached interval containing the cursor; the cursor itself if uncached. |
| `seeking`, `seek_target_frame` | Accepted seek still awaiting output rebasing or target audio. |
| `estimated_frames` | Whole-document estimate, refined by known actual lengths. Null before planning. |
| `duration_frames` | Actual document duration once every segment's length has been measured, even if some PCM was later evicted. |
| `recording_complete` | Every segment's PCM is currently retained and can supply a complete recording. |

A scrubber uses `duration_frames ?? estimated_frames` for its scale and
`seek_target_frame ?? position_frame` for its pending cursor. Buffered ranges
are a separate overlay, never a total-duration estimate or an on-demand seek
restriction. An invalid target leaves playback unchanged. No-retention, released
or closed storage exposes empty navigable ranges.

## Cache and generation limits

`Tts::configured` accepts `RetentionOptions` separately from inference and output
limits. Defaults are **128 MiB of PCM and 128 stores per Tts**, shared across
controllers and caller-retained completed handles. A full recording uses one
store; each cached reader segment uses one store. `retention_stats` reports
current/peak PCM bytes and store counts, excluding filesystem and RSS overhead.
Limits are configurable up to 16 GiB and 4,096 stores.

The byte quota holds about **12 minutes 41 seconds** at 44,100 Hz mono Float32
(176,400 PCM bytes/second); the store-count quota can bind earlier. On-demand mode
evicts the least recently used segments of its own document, protecting the
currently selected/presented segment. It does not evict other documents or full
recordings. Replays share the document cache and do not copy PCM. Actual duration
measurements survive eviction. Anonymous temporary files close/delete on final
release or owner closure.

A segment must fit in the cache. Automatic prefetch defers if the protected
current segment occupies the remaining quota. If the required destination cannot
fit, or other retained recordings hold the quota, the request fails with a typed
retention error. Quota and I/O failures never expose incomplete cached audio.
Full-recording mode does not evict history; exhausting its quota fails the request.

The **two-second source-frame lookahead** (88,200 frames by default) remains the
horizon for automatically starting the next segment during playback. Once started,
a whole bounded model segment drains into storage independently of pause/output
capacity; ready audio can extend beyond that horizon. An explicit on-demand seek
selects a new position anywhere in the document and generates there. It does not
increase the output queue or generate everything between the old and new cursor.

Native events, synthesis chunks, pending output and the audio driver have
separate finite bounds. Cache I/O is performed in bounded chunks on the controller
worker, never on the native audio callback. Only one segment job per reader is
in flight, through the shared inference queue. Paused readers stop automatic
generation beyond their horizon and release the inference lane between jobs.

A request is still limited to **64 KiB of raw UTF-8 text** before admission.
The cache allows a document longer than its PCM budget, but this is not a new
whole-book ingestion API: applications must split larger text into requests.

## Completion, replay and export

`Speech.done` is immutable and belongs to one accepted playback request. Seeking
within active speech preserves it. Reaching the end after skipping passages can
complete playback with an incomplete recording. In reader mode, `Completion.frames`
counts the measured frames of unique generated document segments, not an invented
full duration or the number of frames listened to.

`replay_from(frame, options, mode)` returns a **new Speech** and outcome after
successful completion. Reader replay shares the original document and cache and
can regenerate missing segments. Full-recording replay shares retained PCM without
inference. Both keep original source coordinates; observe the new handle's state.

Previous/next sentence respect original sentence IDs, including sentences split
into multiple model segments, and return `Boundary` at document ends. In reader
mode, active navigation returns `Seeked` even if destination audio must be generated.
In full-recording mode, navigation outside retained audio returns `Started(new_speech)`
for an original-plan suffix and ends the old handle with `Navigation` cancellation.
That legacy suffix starts its timeline at zero but preserves original IDs, seeds
and source ranges. Older handles cannot navigate over another current request.

`Tts::write_speech` requires completed playback with all PCM currently cached.
It rejects skipped/evicted coverage before creating an output file; it never
exports holes as silence. A complete export pins its segment files under the same
quota while the incremental writer runs. For an independent complete export, use
`render_to_file`, which explicitly generates the whole request.

`dispose()` cancels/joins this request and releases its storage reference; other
replay handles retain their shared references. Closing Tts invalidates all owned
retention, including completed handles, while preserving outcomes and all
caller-owned clips/export files.

## Evidence

The [on-demand seek qualification](../tests/fixtures/developer-api/on-demand-seek.json)
records source hashes and the tested commands. Portable reader regressions cover
direct uncached seeks, disjoint intervals, latest-seek cancellation during a blocked
native call, cache eviction/regeneration, duration corrections, and a seek accepted
during final drain. They also check close/quiescence, full export PCM, inference
availability while paused, and reading a document with room for only one segment.
The real prepared-segment test generates in order `2, 0, 1, 2`, compares every
Float32 bit with linear synthesis and checks that duration predictions are not
repeated. Real Python/Go and muted macOS Flutter tests exercise far seeks and cached
returns through the public APIs. That original qualification covers Core ML INT4
on the tested Mac.

```sh
MLSR_TEST_BUNDLE=/absolute/verified/model cargo test --locked -p mlsr --all-features \
  real_prepared_segments_match_linear_synthesis -- --ignored --nocapture
```

The [merge qualification](../tests/fixtures/developer-api/duration-main-merge.json)
adds the recommended T384 FP16 bundle and its complete-sentence segmentation.
English/M1 and Korean/F1 sentences exceeding the old 128-token window remain
whole; their prepared segments match every Float32 bit of linear synthesis in
order `2, 0, 1, 2`, without repeating duration predictions. The muted macOS Flutter
reader also passes unbuffered seeks, cached returns, paused controls and export
with this bundle. The legacy INT4 parity check still passes. Duration preparation
and generation share backend-specific token types and duration limits; QNN
execution of the merged reader has not been qualified on Windows hardware.

```sh
MLSR_TEST_BUNDLE=/absolute/verified/t384-fp16-model cargo test --locked -p mlsr --all-features \
  real_t384_prepared_sentences_match_linear_synthesis -- --ignored --nocapture
```

Earlier full-recording and duration-planning evidence follows; its no-eviction
and retained-only seek behavior applies to the default full-recording policy.

Portable tests cover partial/completed generation seeks, invalid-seek preservation,
paused segment admission, full retention with a full output queue, forward and
backward seeks beyond lookahead, actual-versus-predicted frame pacing, replacement
and closure during admission waits, long sentences, original Unicode ranges,
plan reuse, inference-free replay, shared storage, replay queue limits,
stale-handle navigation, quota failure, OS write failure, no-retention behavior
and cleanup. They also preserve the earlier cancellation/output/scheduler tests.

The [segment-retention qualification](../tests/fixtures/developer-api/segment-retention.json)
records the regression failing before the change and passing afterward. Its
muted macOS Flutter run retained 199,007 frames (4.51 seconds) while paused at
zero with the default 88,200-frame generation horizon, then sought to frame
92,296 without resuming. The native reader test preserves exact suffix PCM,
and portable tests cover quota failure while paused and the UTF-8 input limit.

Duration-planning regressions check cancellation during an in-flight predictor,
no partial total publication, estimates under PCM backpressure, one prediction
per segment, frame-limit failure before audio, and final-length correction. The
Flutter widget tests cover both timeline scales, buffer appends, unknown duration,
unavailable targets and replacement during a held drag. A paused seek within the
same text segment must also publish a new state revision; the controller keeps
its previous position until the output clock confirms the new cursor, while
publishing seek admission separately.

The real duration test compares every PCM sample against the incremental path
for English/M1 and Korean/F1 at raw synthesis speeds 0.84, 1.05 and 1.26. It also
checks exactly 11 predictions per segment at eight steps and unchanged retained
native tensor counts. The [duration-planning qualification](../tests/fixtures/developer-api/duration-planning.json)
records source hashes, Flutter integration, the repaired seek notification, and
a 59.94-second playback with its total estimated before the first retained PCM
and no backward progress steps across 2,471 sampled snapshots. These are muted
state observations, not acoustic timing or a throughput benchmark. Run the
exact-PCM test with:

```sh
MLSR_TEST_BUNDLE=/absolute/verified/model cargo test --locked -p mlsr --all-features \
  real_duration_planning_reuses_predictions_and_preserves_pcm -- --ignored --nocapture
```

The opt-in muted native reader test uses real Core ML and AVAudioEngine. It seeks
while paused and playing, navigates to a new sentence, and compares every retained
Float32 bit against a separately rendered original-plan suffix. The recording is
then replayed with a new handle while storage remains shared, and final disposal
returns disk accounting to zero. The [tracked result](../tests/fixtures/developer-api/reader.json)
contains source hashes and observations; external logs are under
`~/.cache/mlsr/qualification/developer-api-v0/step6-reader-v2/`.

```sh
MLSR_TEST_BUNDLE=/absolute/verified/model cargo test -p mlsr --all-features \
  real_reader_seek_navigation_exact_retention_and_completed_replay -- --ignored --nocapture
```

This test qualifies shared Rust reader behavior on macOS. The
[combined qualification](internal/developer-api-qualification.md) also covers language
wrappers, the Flutter reader UI and installed packages. A paused cursor at frame
zero exposes the first segment's original text range; the shared regression and
Flutter previous-sentence test cover that boundary.
