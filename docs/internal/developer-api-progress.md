# Developer API v0 implementation and acceptance

The [contract](../developer-api.md) is implemented and qualified for the agreed
macOS Apple Silicon scope. The [SDK guide](../developer-sdk.md) describes the public
surfaces; the [qualification record](developer-api-qualification.md) names the
tested artifacts, hosts, measurements and remaining platform limits. The existing
[Core ML POC](poc-qualification.md) remains the synthesis reference.

| Step | Deliverable | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Developer contracts and Apple clock/completion/rate spike | Complete | [Contract](../developer-api.md), [native probe and limitations](playback-probe.md), `tests/fixtures/developer-api/playback-probe.json` |
| 2 | Voice catalog/resolution and frozen Unicode segment plans | Complete | [Rust API and validation](../voices-and-text-plans.md); SDK translations in step 7 |
| 3 | Shared render/stream/WAV APIs with state and completion | Complete | [Shared Rust facade and qualification](../headless-sdk.md); SDK translations in step 7 |
| 4 | Shared speech handles, bounded scheduler and state machine | Complete | [Controller and deterministic output tests](../speech-controller.md); native output in step 5 |
| 5 | Reusable native Apple playback and host policy adapter | Complete | [Native adapter and qualification](apple-output.md); language wiring in step 7 |
| 6 | Retention, seeking, reader timeline and navigation | Complete | [Reader API, storage limits and native qualification](../reader-api.md); all SDKs expose the shared behavior |
| 7 | Public Rust/C/Dart/Go/Python SDKs and Flutter migration | Complete | [Additive C ABI](../developer-c-api.md), [public SDKs](../developer-sdk.md); Flutter uses shared Tts/Speech for scheduling, playback, reader controls and WAV export; verified model ownership/readiness in all bindings |
| 8 | Speak/reader/export examples, installed-package and full acceptance qualification | Complete | [Installed C/Dart/Go/Python, relocated sandboxed Flutter release, exact PCM and separate timing/memory evidence](developer-api-qualification.md) |

## Requirement checklist

Each row below has shared implementation tests and, where applicable, installed
SDK or real native integration evidence. "Complete" is scoped to the contract's
macOS target; it does not imply every backend, language version or device is
qualified. The source/test links are the durable checks; full external logs and
their hashes are indexed by the qualification record.

| ID | Required behavior | Status / evidence |
| --- | --- | --- |
| API-01 | Installed public facade: open, speak, render, renderToFile, synthesize, dispose | Complete: [installed consumers][installed] and [all language SDK tests][sdk-tests] |
| API-02 | Same shared behavior through Rust, C, Dart, Go and Python; idiomatic async adapters | Complete: [shared synthesis][synthesis], [C consumer][c-sdk], Python asyncio cancellation, Go contexts/race tests and Dart isolates/streams |
| API-03 | Additive ABI; old functions/layouts/offset meanings preserved | Complete: [C ABI/schema gates](../developer-c-api.md), generated header/bindings current, installed old/new jobs and exact PCM; supplemental legacy Dart engine lease check |
| VOICE-01 | Stable IDs, metadata, supported languages/capabilities | Complete: [voice tests][voices] and installed SDK catalog/resolution tests; metadata-supported languages are distinct from en/ko qualification |
| VOICE-02 | Catalog/verified-assets/loaded-engine readiness distinction | Complete: [voice tests][voices], Python/Go real verified-model ownership tests, Dart model readiness and Flutter verified loading |
| VOICE-03 | Explicit locale resolution/fallback, missing voice errors | Complete: `explicit_fallback_never_silently_changes_language_or_voice` and installed SDK resolution/error tests |
| TEXT-01 | Frozen original sentence/segment plan and immutable request options | Complete: [plans][plans], [request tests][request], copied request/effective-setting inspection in all SDKs |
| TEXT-02 | Correct original UTF-8/UTF-16 mapping including Korean/emoji/combining text | Complete: [plan boundary tests][plans], Dart UTF-16 SDK tests and [Flutter original-text highlights][flutter] |
| TEXT-03 | Explicit input limits, segmentation without truncation, stable navigation seeds | Complete: plan/suffix tests, [native exact retained-suffix comparison](../reader-api.md), SDK original segment/seed metadata |
| SYN-01 | Lazy advanced audio consumption/explicit start; states do not drain audio | Complete: [lazy/observer/single-consumer tests][synthesis] and installed C/Python/Go/Dart lazy job tests |
| SYN-02 | Independent bounded state snapshots and synthesis completion | Complete: [slow observer/quiescence tests][synthesis], language pull/coalesced adapters and installed cleanup tests |
| SYN-03 | Speaking-rate normalization and typed backend options | Complete: [request tests][request], effective-setting SDK assertions and four-case exact legacy PCM comparison |
| RENDER-01 | Bounded in-memory AudioClip with format/frame metadata | Complete: [render limit tests][synthesis], owned C PCM views and installed language clip/format tests |
| RENDER-02 | Shared incremental WAV output, no clobber, owned partial cleanup | Complete: Rust writer/consumer tests, installed SDK export/no-clobber tests and Flutter export integration |
| RENDER-03 | Cancellation/disk/quota failures propagate; parity across bindings | Complete: shared OS failure/quota/writer cancellation tests, typed SDK errors, five SDK consumers plus legacy PCM parity |
| SESSION-01 | Immediate per-request Speech handle, immutable outcome exactly once | Complete: [controlled final-completion race][speech], installed speech completion and independent completed-replay identities |
| SESSION-02 | Replace scope, enqueue order, bounded queue count/text bytes | Complete: [scoped replace/admission/FIFO tests][speech], [scheduler budgets][synthesis], Flutter replace/enqueue integration |
| SESSION-03 | Queued cancellation, old callback rejection, close/join races | Complete: [epoch/cancellation/close tests][speech], native driver and language ownership/context/disposal tests |
| STATE-01 | Independent synthesis/playback states; buffering is not completion | Complete: [fake output tests][speech], installed native speech waits and Flutter generation-finished-while-playing assertion |
| STATE-02 | Late/current snapshot, bounded paused/slow subscribers, retained terminal | Complete: revision-only Rust watches, Dart paused subscriptions, Python pull iterators and Go latest-state channel tests |
| OUTPUT-01 | Optional output dependency and native PCM path for all bindings | Complete: [optional native factory](apple-output.md), installed C/Dart/Go/Python native speech; headless SDK paths remain available |
| OUTPUT-02 | Bounded enqueue, independent flush/cancel path | Complete: [native driver][native-output] full-queue/stale callback tests and controlled cancellation-before-inference-quiescence regression |
| OUTPUT-03 | Pause/resume clock and final played-back completion | Complete: native driver, installed SDK and relocated release drain checks; acoustic timestamps not claimed |
| OUTPUT-04 | Volume and independent pitch-preserving playback rate | Complete: native five-rate/live-change tests, [offline pitch probe](playback-probe.md), SDK and Flutter controls |
| OUTPUT-05 | Host-managed policy, route/interruption handling and explicit capabilities | Complete for macOS: host policy, fake interruption policy and synthetic native configuration recovery; native interruption capability false, physical route/iOS unqualified |
| STORE-01 | RAM/disk budgets; typed quota failure; cleanup only owned resources | Complete: shared quota/lookahead/OS failure tests, SDK close/export ownership, release retention/handle counters and deterministic model-lock lifetime regression |
| TIME-01 | Position, bufferedUntil, seekableRanges, estimated/final duration | Complete: [shared timeline tests][speech], installed SDK snapshots and Flutter reader integration |
| TIME-02 | Seek during generation/completion/pause; invalid seek preserves position | Complete: Rust/native and language reader tests; completed recording replay creates a new Speech sharing PCM |
| TIME-03 | Original-text highlight follows presented audio through rate/seek changes | Complete: native source-range tests, Dart UTF-16 and Flutter seek/rate/highlight integration, paused frame-zero regression |
| TIME-04 | Previous/next sentence and explicit ungenerated-text navigation | Complete: original sentence grouping/frozen suffix/stale-handle tests, native exact PCM and Flutter previous/next navigation |
| SDK-01 | Flutter uses public SDK without a duplicate speech/session implementation | Complete: [DemoController][demo-controller] owns/observes SDK handles; native player, PCM channel and app WAV serializer removed |
| SDK-02 | Existing offline bundled model delivery and release sandbox preserved | Complete: all four relocated release phases, denied developer-file/network access, direct/installed model ownership and zero handles |
| EXAMPLE-01 | Speak button: rapid replace/enqueue/cancel | Complete: [Flutter UI and integration][flutter] |
| EXAMPLE-02 | Reader: pause/rate/volume/seek/highlight/navigation | Complete: [Flutter reader and integration][flutter] |
| EXAMPLE-03 | PCM/WAV export with limits and cancellation | Complete: [Python/Go/Dart examples](../developer-sdk.md#start-with-an-example), all run from installed/relocated packages |
| QA-01 | Portable deterministic lifecycle/error/Unicode tests | Complete: workspace all-feature tests, strict Clippy, Rust 1.88 checks, Windows core cross-compilation and SDK tests |
| QA-02 | Installed packages and actual Core ML/native playback tests | Complete: [installed matrix and relocated Flutter release](developer-api-qualification.md) |
| QA-03 | Existing exact PCM reference plus full new SDK qualification | Complete: four legacy/SDK cases, two original POC cases through all four release phases; exact sample bytes, no audio tolerance |
| QA-04 | First-PCM/first-playback/flush/quiescence/memory measurements kept separate | Complete: [public SDK timing and native-driver measurements](developer-api-qualification.md#timing-and-memory-boundaries), with source-clock and allocation boundaries stated |

[installed]: ../../tools/check_synthesis_bindings.py
[sdk-tests]: developer-api-qualification.md#installed-public-consumers
[c-sdk]: ../../tests/integration/developer_smoke.c
[synthesis]: ../../crates/mlsr/src/synthesis/tests.rs
[speech]: ../../crates/mlsr/src/speech/tests.rs
[voices]: ../../crates/mlsr/src/voice.rs
[plans]: ../../crates/mlsr/src/text/plan.rs
[request]: ../../crates/mlsr/src/request.rs
[native-output]: ../../crates/mlsr-playback/src/apple.rs
[flutter]: ../../examples/flutter_speak/integration_test/speak_test.dart
[demo-controller]: ../../examples/flutter_speak/lib/speech_controller.dart

## Deferred scope

QNN/Android and physical iOS qualification follow this API milestone. Word
alignment, persistent synthesis caching, background media controls, voice cloning,
full SSML and quantization promotion are not hidden acceptance dependencies. All
essential SDK rows above remain required; platform claims must name tested hosts.
