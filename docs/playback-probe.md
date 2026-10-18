# Apple output feasibility

The explicit, muted `mlsr-playback` example qualifies the native primitives for
the developer API, not a finished speech controller. Run it with a new report:

```sh
cargo run --locked -p mlsr-playback --example apple_probe -- /absolute/new-report.json
```

The [recorded result](../tests/fixtures/developer-api/playback-probe.json) contains
source/lock hashes, host/toolchain and individual observations. It passed on the
local Apple Silicon macOS 27.0.1 host. MSRV 1.88, strict clippy and Windows target
compilation passed; no Windows playback or iOS device execution is implied.

Use `objc2-avf-audio` 0.3.2 for the optional output crate. Its generated player,
time conversion, completion-type and time-pitch APIs cover the required surface.
The existing `cidre` Core ML adapter remains unchanged. All native graph ownership
stays on an output control thread, with minimal completion callbacks and
ordered epoch validation on that control thread. Output dependencies are optional;
a headless MLSR build does not require them. The subsequent
[reusable adapter](apple-output.md) has its own native qualification.

The probe establishes these boundaries:

- A 2-second 440 Hz tone rendered offline at 1.5x lasts about 1.338 seconds at
  about 439.79 Hz. This checks duration/pitch, not perceptual speech quality.
- The live source-frame clock advances at approximately the selected 1.25x rate.
  It is a render clock, with reported downstream latency of about 85 ms on this
  graph/device. Adjust for downstream latency and clamp to the actual media range;
  do not expose it as a sample-exact speaker timestamp.
- While paused, native player time is unavailable. Latch the last valid public
  position. The first native clock after resume can regress within processing
  latency; monotonic-clamp within an epoch, with explicit rebasing on seek.
- `DataPlayedBack` waits for playback drain, but also fires on stop. Stop must
  invalidate the epoch before that callback can affect a speech outcome.
- Stop/reuse succeeds without accepting the cancelled completion. The measured
  stop-call duration is one native-call observation, not acoustic silence latency
  or inference cancellation-to-quiescence.

The first two local attempts are retained under the qualification directory as
`step1-playback-v1` and `step1-playback-v2`. They rejected an unavailable paused
clock and an initially assumed monotonic raw resume clock. Those observations
drive the explicit clock contract rather than being hidden by a passing build.
Versions v3/v4 pass with bounded resume checks and reported latency. Reader seek
and public SDK acceptance remain separate from these primitive checks.

Apple API references: [player nodes](https://developer.apple.com/documentation/avfaudio/avaudioplayernode),
[time pitch](https://developer.apple.com/documentation/avfaudio/avaudiounittimepitch),
and [generated bindings](https://docs.rs/objc2-avf-audio/0.3.2/objc2_avf_audio/).
