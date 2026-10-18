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

## Text policy

The complete request is limited to 64 KiB of UTF-8. Paragraphs are split at
newlines, then sentences with pinned Unicode segmentation. Oversized sentences
prefer whitespace boundaries and otherwise use extended grapheme boundaries.
Each segment is independently normalized, including language tags and terminal
punctuation, and must fit 128 tokens. All segments are checked against the real
tokenizer before the first model call. Unsupported input fails the job explicitly.

The prefix scan is capped at 4 KiB to bound work on removable content. Empty
normalized spans are skipped; a single grapheme that cannot fit a segment is
rejected. These skips can leave gaps between source spans, but spoken ranges are
exact original substrings and are never truncated. Seed `s` produces segment `n`
with seed `s.wrapping_add(n)`; changing segmentation therefore changes noise and
prosody. A paragraph stream is not promised to match one unsegmented utterance.

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
