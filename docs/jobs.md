# Synthesis jobs and bounded output

`Engine::start(SynthesisRequest, JobOptions)` copies the request and returns a
`Job`. One orchestration worker owns the loaded pipeline. The existing native
runtime worker continues to own Core ML objects. One request may be active on
an engine; competing starts and synchronous calls return `Error::Busy`.
`Engine::synthesize` remains a synchronous, single-segment convenience API.

The job stream contains predicted segment duration, audio chunks, and exactly
one terminal report. Every event has an engine-local, monotonically increasing
job ID. Every segment/chunk includes its segment ID and a half-open **UTF-8 byte
range in the original input**. Normalized offsets are never substituted for
source offsets. Chunk positions are frame offsets in concatenated job audio;
samples are finite mono Float32 at 44,100 Hz. PCM has no implicit silence,
crossfade, clipping, or normalization between segments.

`Job::poll(timeout)` distinguishes an event, timeout, and exhausted stream.
Predicted durations are available after the duration model and before segment
denoising. Completed reports give final frames and segment count. Cancelled or
failed reports count frames already published, including any subsequently
discarded queue contents. Synthesis completion does not mean playback completion.

`Engine::start_plan` and the developer `Tts` facade additionally run a cancellable
duration-only pass over the remaining frozen segments before PCM generation.
`Job::estimated_frames()` becomes available after the complete pass, independently
of event queue capacity. Only small duration/shape decisions are retained and
reused; synthesis does not repeat the predictor calls. Segment events still use
actual emitted start offsets. The legacy `Engine::start` path keeps incremental
prediction and returns no whole-request estimate. Final report frames remain
authoritative on both paths.

## Text policy

The complete request is limited to 64 KiB of UTF-8. Text is split at newlines,
then sentences using pinned Unicode rules with common English titles (`Dr.`,
`Mr.`, `Mrs.`, `Ms.`, `Prof.`) kept with the following name. Synthesis and reader
navigation share those boundaries; separate sentences are never packed into one
model segment.

Supertonic3 segments target at most 70 source graphemes, or 57 for Korean and
Japanese, excluding surrounding whitespace. These quality targets are separate
from the hard limit of 128 normalized Unicode tokens, including language tags
and terminal punctuation. Normalization can expand text, so the token limit can
require smaller segments. All segments are checked against the real tokenizer
before the first model call. Unsupported input fails explicitly.

When a sentence exceeds a limit, the splitter prefers the last comma, semicolon
or colon (including CJK forms) that leaves a substantial prefix, then a word
boundary, then an extended grapheme boundary. Numeric separators in `1,234` and
`12:30` are not clause breaks. Word splits move earlier when possible to avoid
leaving a tail shorter than one-third of the character target. This is a bounded
heuristic, not a grammatical parser. Source punctuation stays with its clause;
forced word splits still use the model's existing terminal-punctuation rules.

The prefix scan is capped at 4 KiB to bound work on removable content. Empty
normalized spans are skipped; a single grapheme that cannot fit a segment is
rejected. These skips can leave gaps between source spans, but spoken ranges are
exact original substrings and are never truncated. Seed `s` produces segment `n`
with seed `s.wrapping_add(n)`; changing segmentation therefore changes noise and
prosody. A paragraph stream is not promised to match one unsegmented utterance,
and new plans may change across library versions. A frozen plan and its suffixes
retain their original segment IDs and seeds for their lifetime.

## Backpressure and cancellation

`JobOptions` independently bounds queued PCM frames, chunk frames, and predicted
segment frames. Defaults are 44,100 queued frames, 4,096 per chunk, and the model's
maximum 1,572,864 frames per segment. Metadata/data queues also cap at 32 events;
the terminal report has its own reserved slot and never waits for capacity.

Duration limits are checked before allocating latents or running the remaining
graphs. While delivering audio, the producer holds at most one complete segment
plus the bounded queue. It waits for queue capacity before copying another chunk
or starting another segment. The pipeline's temporary tensors are independently
bounded by supported model shapes. These are allocation bounds, not claims about
Core ML's physical memory or ANE residency. Audio retained by the caller is owned
by that caller and no longer counts toward the queue limit.

`cancel()` sets a shared flag, discards queued data, and wakes blocked producers.
It bypasses the inference command queue. Checkpoints run before/after model
predictions and between VE iterations. An in-flight native prediction is allowed
to return with all inputs/outputs alive; its late result is discarded. `wait()`
observes quiescence without draining data, so a non-cancelled, backpressured job
still needs a consumer. Cancellation after a terminal outcome discards remaining
queued data but does not rewrite the already recorded outcome.

Dropping a job cancels and waits. Shutting down/dropping an engine cancels its
active request and joins its workers. Neither operation waits for a consumer.
Returned chunks survive both. A caught inference panic retires the backend;
ordinary input/inference errors permit another request after termination. An
abort cannot be caught. Host adapters must execute blocking waits/destruction off
UI threads and check job IDs to reject events already pulled before a stop.

Portable tests use a controlled in-flight call, queue saturation, duration
failures and injected panics. The opt-in real-model test checks streamed PCM
against corresponding single-segment calls, cancellation/restart, model-load
counts and native tensor lifetime. Later FFI/binding tests must preserve this
contract rather than introduce an unbounded adapter queue. The
[job qualification record](../tests/fixtures/supertonic3/jobs-summary.yaml)
identifies the actual tests and saved-noise regression evidence.
