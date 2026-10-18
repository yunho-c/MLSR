# Developer API v0 implementation and acceptance

The [contract](developer-api.md) is the target. The existing
[Core ML POC](poc-qualification.md) supplies the synthesis foundation. This table
must not be marked complete merely because a subset of tests passes.

| Step | Deliverable | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Developer contracts and Apple clock/completion/rate spike | Complete | [Contract](developer-api.md), [native probe and limitations](playback-probe.md), `tests/fixtures/developer-api/playback-probe.json` |
| 2 | Voice catalog/resolution and frozen Unicode segment plans | Complete | [Rust API and validation](voices-and-text-plans.md); SDK translations in step 7 |
| 3 | Shared render/stream/WAV APIs with state and completion | Pending | |
| 4 | Shared speech handles, bounded scheduler and state machine | Pending | |
| 5 | Reusable native Apple playback and host policy adapter | Pending | |
| 6 | Retention, seeking, reader timeline and navigation | Pending | |
| 7 | Public Rust/C/Dart/Go/Python SDKs and Flutter migration | Pending | |
| 8 | Three examples, installed-package and full acceptance qualification | Pending | |

## Requirement checklist

Each row needs concrete tests and current source evidence before final sign-off.
`Pending` means it has not been established for the new SDK, even if a low-level
primitive or the older demonstration already supports part of it.

| ID | Required behavior | Status / evidence |
| --- | --- | --- |
| API-01 | Installed public facade: open, speak, render, renderToFile, synthesize, dispose | Pending |
| API-02 | Same shared behavior through Rust, C, Dart, Go and Python; idiomatic async adapters | Pending |
| API-03 | Additive ABI; old functions/layouts/offset meanings preserved | Pending |
| VOICE-01 | Stable IDs, metadata, supported languages/capabilities | Rust complete: `voice::tests`; SDK translation pending |
| VOICE-02 | Catalog/verified-assets/loaded-engine readiness distinction | Rust complete: voice and engine tests; SDK translation pending |
| VOICE-03 | Explicit locale resolution/fallback, missing voice errors | Rust complete: `explicit_fallback_never_silently_changes_language_or_voice`; SDK translation pending |
| TEXT-01 | Frozen original sentence/segment plan and immutable request options | Frozen plan complete; facade option snapshot pending |
| TEXT-02 | Correct original UTF-8/UTF-16 mapping including Korean/emoji/combining text | Rust complete: `text::plan::tests`; Dart translation pending |
| TEXT-03 | Explicit input limits, segmentation without truncation, stable navigation seeds | Rust complete: segmentation/suffix tests plus real seeded PCM equality; navigation integration pending |
| SYN-01 | Lazy advanced audio consumption/explicit start; states do not drain audio | Pending |
| SYN-02 | Independent bounded state snapshots and synthesis completion | Pending |
| SYN-03 | Speaking-rate normalization and typed backend options | Pending |
| RENDER-01 | Bounded in-memory AudioClip with format/frame metadata | Pending |
| RENDER-02 | Shared incremental WAV output, no clobber, owned partial cleanup | Pending |
| RENDER-03 | Cancellation/disk/quota failures propagate; parity across bindings | Pending |
| SESSION-01 | Immediate per-request Speech handle, immutable outcome exactly once | Pending |
| SESSION-02 | Replace scope, enqueue order, bounded queue count/text bytes | Pending |
| SESSION-03 | Queued cancellation, old callback rejection, close/join races | Pending |
| STATE-01 | Independent synthesis/playback states; buffering is not completion | Pending |
| STATE-02 | Late/current snapshot, bounded paused/slow subscribers, retained terminal | Pending |
| OUTPUT-01 | Optional output dependency and native PCM path for all bindings | Pending |
| OUTPUT-02 | Bounded enqueue, independent flush/cancel path | Pending |
| OUTPUT-03 | Pause/resume clock and final played-back completion | Pending |
| OUTPUT-04 | Volume and independent pitch-preserving playback rate | Pending |
| OUTPUT-05 | Host-managed policy, route/interruption handling and explicit capabilities | Pending |
| STORE-01 | RAM/disk budgets; typed quota failure; cleanup only owned resources | Pending |
| TIME-01 | Position, bufferedUntil, seekableRanges, estimated/final duration | Pending |
| TIME-02 | Seek during generation/completion/pause; invalid seek preserves position | Pending |
| TIME-03 | Original-text highlight follows presented audio through rate/seek changes | Pending |
| TIME-04 | Previous/next sentence and explicit ungenerated-text navigation | Pending |
| SDK-01 | Flutter uses public SDK without a duplicate speech/session implementation | Pending |
| SDK-02 | Existing offline bundled model delivery and release sandbox preserved | Pending |
| EXAMPLE-01 | Speak button: rapid replace/enqueue/cancel | Pending |
| EXAMPLE-02 | Reader: pause/rate/volume/seek/highlight/navigation | Pending |
| EXAMPLE-03 | PCM/WAV export with limits and cancellation | Pending |
| QA-01 | Portable deterministic lifecycle/error/Unicode tests | Pending |
| QA-02 | Installed packages and actual Core ML/native playback tests | Pending |
| QA-03 | Existing exact PCM reference plus full new SDK qualification | Pending |
| QA-04 | First-PCM/first-playback/flush/quiescence/memory measurements kept separate | Pending |

## Deferred scope

QNN/Android and physical iOS qualification follow this API milestone. Word
alignment, persistent synthesis caching, background media controls, voice cloning,
full SSML and quantization promotion are not hidden acceptance dependencies. All
essential SDK rows above remain required; platform claims must name tested hosts.
