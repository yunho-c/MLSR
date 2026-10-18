# Reusable Apple output

`mlsr`'s optional `apple-output` feature installs a lazy native output factory on
macOS. `Tts::new`/`open` create no audio graph; the first active speech opens it on
its controller thread. Subsequent requests reuse that graph, including ordinary
cancellation and replacement. Headless builds omit all output dependencies. Core
ML continues to use cidre; output uses pinned objc2 AVFAudio bindings in the
separate `mlsr-playback` crate.

`output::apple::AppleOutputFactory` can be supplied through `Tts::with_output` to
select native queue limits and host-managed policy. The default queue holds at
most 88,200 mono Float32 frames and 32 native buffers. Stopped buffers awaiting a
callback count against the same limits. A full queue allocates nothing and never
blocks the independent flush path. The model/consumer queues have their own caps.

The output supports mono 44,100 Hz, volume 0–1, pause/resume, source-frame rebasing,
and independent pitch-preserving rate 0.5–2.0. It leaves the main mixer/output
hardware format connection implicit. Callbacks update atomics only; the owning
thread performs stop, reset and recovery. Graph objects never cross threads.

The public clock subtracts reported downstream latency, clamps to accepted source
frames and remains monotonic within an epoch. Pause latches it. Once scheduled
audio drains during a generation gap, the player timeline stops and is rebased
at the next chunk; time spent waiting does not become media time. End of input
alone cannot complete speech: the final `DataPlayedBack` callback must arrive.
Flush invalidates the epoch before old callbacks can affect another request.

A configuration notification pauses output and retains the unplayed native PCM.
Explicit resume reschedules that bounded remainder from the latched position;
it does not invoke synthesis. Both macOS policy modes leave audio-session
category/activation ownership to the host. Native configuration events are
advertised; AVAudioSession interruptions are **not** advertised on this macOS
adapter. iOS audio-session behavior and physical device-route changes remain
unqualified. Controller interruption policy is covered separately by fake-output
tests.

## Qualification

The muted tests below open the current macOS output device only when explicitly
requested. The tracked [result](../tests/fixtures/developer-api/apple-output.json)
records host, source hashes and actual observations. External logs are under
`~/.cache/mlsr/qualification/developer-api-v0/step5-native-v2/`.

```sh
cargo test -p mlsr-playback --features apple \
  native_clock_lifecycle_bounds_and_configuration_recovery -- --ignored --nocapture
MLSR_TEST_BUNDLE=/absolute/verified/model cargo test -p mlsr --all-features \
  real_speech_native_playback_pause_replace_reuse_and_drain -- --ignored --nocapture
```

The native driver test exercises all five rates (0.5, 1, 1.25, 1.5, 2), rate
changes during playback, frozen pause, starvation/restart, final drain, full-queue
backpressure, flush/stale callbacks, paused rebasing and a synthetic native
configuration notification. Peak owned native storage was 4 buffers/44,100
frames under those exact test limits; it returned to zero. These are owned buffer
counters, not physical driver memory or RSS. The earlier offline
[pitch probe](playback-probe.md) checks duration and pitch of a 440 Hz tone.

The real Core ML test renders and plays English M1 and Korean F1 with matching
final frame counts (160,493 and 162,011). It exercises pause/resume, live rate and
volume, queued cancellation, active replacement, final drain and closure through
the shared controller. All requests used one native graph. Output was muted;
this is lifecycle/clock qualification, not a listening-quality assessment.

Both tests reported about 84.84 ms downstream latency on this device/graph.
Source progress is an adjusted render-clock observation, not an acoustic speaker
timestamp. Recorded flush-call and cancellation-completion times have separate
boundaries. Short-clip start/drain includes engine/DSP/device overhead and must
not be interpreted as synthesis latency or a steady-state playback benchmark.

The first real-speech run (`step5-native-v1`) found that cancellation consumed
between enqueues incorrectly retired a healthy graph. The controller now takes
its cancellation/flush path and preserves that graph. A gated deterministic test
reproduces the ordering; the following native run passed with one graph open.

Workspace tests, strict clippy, Rust 1.88 compilation and Windows target
compilation cover the shared code. They do not imply native playback on Windows
or iOS. Public C/Dart/Go/Python translations and Flutter use of this adapter are
the following SDK step, not established by these Rust tests.
