# Architecture

## Dependency direction

```text
Dart / Go / Python --> mlsr-ffi --> mlsr
Rust applications -------------> mlsr
```

The Cargo workspace shares version metadata, a lockfile, and build outputs.
`mlsr` and `mlsr-ffi` are the default members; `tools/codegen` is a development
tool and is not a runtime dependency. Portable model loading uses Serde, YAML/JSON
parsers, and SHA-256. JSON reads the upstream tokenizer/configuration/voice assets;
model manifests use YAML. The optional Apple-only
`coreml` feature adds pinned cidre bindings. The portable frontend and noise
preparation use pinned Unicode normalization and random-generation crates.

## Rust modules

| Module | Responsibility |
| --- | --- |
| `engine` | Model lifetime, synthesis orchestration, jobs, and cancellation |
| `config` | Model paths, runtime selection, and validated synthesis options |
| `error` | Errors independent of language-specific exceptions/status codes |
| `audio` | PCM buffers and chunks with explicit format and ownership |
| `text` | Text processing shared across supported models |
| `models` | Model-specific tokenization, tensor preparation, inference sequence, and voice formats |
| `runtime` | Executing networks and integrating optional platform backends |
| `tts`, `request`, `voice` | Developer facade, immutable requests, voice discovery and resolution |
| `synthesis`, `completion` | Shared inference scheduling, lazy PCM consumption and terminal outcomes |
| `speech`, `output`, `retention` | Speech controllers, output contracts, reader timelines and retained audio |

[Supertonic text-to-PCM synthesis](supertonic3.md) is implemented, including local
asset verification, frontend, duration/bucket planning, model sequencing, and
owned audio. `Engine` exposes synchronous synthesis and lower-level jobs;
`Tts` adds the shared developer facade and request scheduler. The
[Core ML session](coreml-runtime.md) owns native models/tensors on one worker;
native bindings stay private. The pinned reference tools qualify every input,
intermediate, shape decision, and PCM result from original text with saved noise.
[Jobs, cancellation and long-text segmentation](../jobs.md) run on a dedicated
orchestration worker with bounded pull output. The [synthesis C ABI](../ffi.md)
exposes that contract through checked handles, owned events and explicit release.

Keep model architecture separate from the inference backend: a model pipeline
specifies what to execute, while a runtime supplies execution. Split modules into
separate crates when dependencies, compilation cost, or reuse justify it.

## Language adapters

`mlsr-ffi` owns C-compatible layouts, handle conversions, and error translation.
Generate its header with [cbindgen](https://github.com/mozilla/cbindgen).

Dart uses [dart:ffi](https://dart.dev/interop/c-interop), generated declarations,
and a [build hook](https://dart.dev/tools/hooks) to bundle the native library.
Go uses [cgo](https://pkg.go.dev/cmd/cgo) and an externally available native
header/library. Python uses [ctypes](https://docs.python.org/3/library/ctypes.html)
with lazy loading of the native library. Dart runs blocking calls on an adapter
isolate and exposes a pull-driven stream; Go provides context cancellation and
`Close`; Python provides context-managed iteration. Each adapter copies returned
PCM into language-owned storage and preserves Rust's bounded output.

Text normalization, model policy, inference scheduling, and synthesis stay in
Rust. Host-language integrations should not reimplement them. Native playback
is optional through `mlsr-playback`; headless builds omit platform output
dependencies.

The [Flutter example](../../examples/flutter_speak/README.md) demonstrates that
boundary: its Dart `DemoController` owns and observes public `Tts`/`Speech`
handles. Rust owns speech scheduling, native Apple playback, retained audio and
WAV serialization. The app's custom native channel only opens the save panel.
Cancellation flushes playback independently of in-flight inference, and playback
epochs reject stale callbacks.

## Assets and distribution

Large model weights are external assets. YAML manifests in `model-manifests/`
track their revisions, URLs, checksums, licenses, tensor contracts, and runtime
compatibility. The first manifest pins the published Supertonic 3 v2 Core ML
models, with bucketed INT4 VectorEstimators and FP16 shared stages.

`tools/supertonic3/` verifies local assets and captures/replays real reference
tensors using Python/Core ML and replay saved runtime inputs through the Rust
session. Its dependencies are development-only and do not
enter the Rust crate, FFI library, or language packages. Small text/token fixtures
and capture summaries live in Git; model bundles, complete tensor captures, and
WAVs live in a user-selected external cache. These references establish porting
correctness, not broad speech-quality or performance qualification.

The [platform matrix](../platforms.md) separates build support from tested runtime
behavior. Installed C/Dart/Go/Python consumers and a relocated Flutter release
have [qualification evidence](developer-api-qualification.md) on macOS Apple
Silicon. Packages remain development packages with separately built native
libraries; published native distributions are future work.

`mlsr-cli` verifies local bundles and synthesizes complete Float32 WAV files
through the public Rust engine. It owns argument parsing and uses the shared
Rust WAV writer in `mlsr`. No Python code executes in the production synthesis
path.
