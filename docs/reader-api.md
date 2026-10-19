# Reader controls and retained recordings

The shared Rust controller retains generated PCM in owned temporary storage by
default. `SpeechOptions::with_retention(false)` omits that storage. Playback and
rendering remain separate: `Tts::render` returns caller-owned memory, and WAV
exports are caller-owned files that reader cleanup never deletes.

`Tts::configured` accepts `RetentionOptions` independently of inference queue and
output limits. The defaults are **128 MiB total PCM and 128 recordings per Tts**,
shared across its controllers and completed caller-retained handles. A replay
shares its recording; it does not charge a second copy. `retention_stats` reports
current/peak PCM bytes and store count, not physical filesystem/RSS overhead.
Quota failure is typed (`FailureStage::Retention`); no recording is silently
evicted. Anonymous temporary files are closed/deleted on last release or owner
closure. A failed write retires the owned file instead of exposing partial data.

New generation is retained only within `SpeechOptions::lookahead_frames` ahead
of the presented source position (default 88,200). Paused generation reaches that
cap and backpressures; retained history behind the cursor remains on disk. The
native event queue, advanced consumer queue, one pending output chunk and native
output have separate finite bounds. A backward seek does not permit unlimited
additional generation. File conversion uses a 16 KiB scratch block and bounded
PCM reads. Storage/control work is serialized on the controller thread; native
and filesystem calls do not carry a hard real-time deadline.

| Operation | Behavior |
| --- | --- |
| `seek(frame)` | Valid during running/paused speech, including after generation finishes. Flushes and rebases output, preserves pause/rate/volume, reads retained PCM without inference. State confirms application. |
| `replay_from(frame, options, mode)` | Valid for a completed recording. Returns a **new Speech** with its own immutable outcome and the same retained PCM/source plan. Observe the new handle's state. |
| `previous_sentence` / `next_sentence` | Navigate original spoken sentences, including a sentence split across several model segments. Return `Boundary` at document ends. |
| `navigate_to_sentence(id)` | Returns `Seeked` for available retained audio or `Started(new_speech)` for a new original-plan suffix/replay. An older handle cannot navigate over a different current request. |
| `dispose()` | Cancels/joins this request and releases its recording reference. Other replay handles keep their own references; final outcome/metadata remain inspectable. |

One `Speech.done` describes one accepted playback request and never changes after
completion. Seeking within an active request preserves its identity. Replaying a
completed recording gets a new identity, so awaiting its `done` waits for that
replay's actual output drain. Its timeline/duration still uses the complete
recording's source coordinates, even when replay starts partway through it.

Seekable ranges describe actual retained source frames, normally `[0, end)`.
The endpoint itself is also a valid cursor: it waits for more generated audio or
finishes when input has ended. Outside-range seeks return `invalid_seek` without
moving output. No-retention/released recordings expose empty ranges. Closing Tts
invalidates all its retained audio, including completed handles; it preserves
their outcomes and all caller-owned clips/export files.

For a sentence whose audio is not retained, navigation creates a new request
using the **same frozen text plan and original segment seed offsets**. It does
not slice/re-normalize a suffix or invent an unavailable timestamp. The previous
active handle ends with `Navigation` cancellation. The new suffix timeline starts
at zero; original sentence/segment IDs and UTF-8/UTF-16 source ranges remain
unchanged. Queued requests keep their order behind the navigation request.

`buffered_until_frame` is the end ready in the retained store; `seekable_ranges`
describe replayable data. Whole-request `estimated_frames` remains null until all
required predictions are known. The developer facade runs a cancellable
duration-only pass over the remaining frozen segments before generating PCM,
then reuses each prediction during synthesis. The complete estimate is available
even while playback is paused or PCM queues are full. Planning retains only
small duration/shape decisions, bounded by the validated text, and releases each
segment's temporary native tensors. It adds an upfront predictor pass to first
audio latency; it does not run the predictor twice or expand PCM lookahead.
Cancellation takes effect between predictions and waits for an in-flight call.

`duration_frames` comes from final generated frames and overrides the estimate.
Neither field changes with playback rate. Highlighting follows the
presented segment through seeks/rate changes; it does not claim word alignment.

A progress bar or scrubber uses `duration_frames ?? estimated_frames` as its
full scale. Retained ranges describe which parts of that scale can be selected;
their growing endpoint is not a total duration. The Flutter example shows the
available region separately, stays indeterminate while the total is unknown,
and leaves playback unchanged when a drag ends outside retained audio.

## Evidence

Portable tests cover partial/completed generation seeks, invalid-seek preservation,
paused lookahead, long sentences, original Unicode ranges, plan reuse, inference-
free replay, shared storage, replay queue limits, stale-handle navigation, quota
failure, OS write failure, no-retention behavior and cleanup. They also preserve
the earlier cancellation/output/scheduler tests.

Duration-planning regressions check cancellation during an in-flight predictor,
no partial total publication, estimates under PCM backpressure, one prediction
per segment, frame-limit failure before audio, and final-length correction. The
Flutter widget tests cover both timeline scales, buffer appends, unknown duration,
unavailable targets and replacement during a held drag. A paused seek within the
same text segment must also publish a new state revision; the controller keeps
its previous snapshot until the output clock confirms the new cursor.

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
