# Architecture

## Dependency direction

```text
Dart / Go / Python --> mlsr-ffi --> mlsr
Rust applications -------------> mlsr
```

The Cargo workspace shares version metadata, a lockfile, and build outputs.
`mlsr` and `mlsr-ffi` are the default members; `tools/codegen` is a development
tool and is not a runtime dependency. Default builds have no third-party runtime
dependencies. The optional Apple-only `coreml` feature adds pinned cidre bindings.
The JSON development transport dependency is confined to examples/tests; model
manifests and qualification reports use YAML.

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

Version queries and the [Core ML tensor runtime](coreml-runtime.md) are implemented.
Its sessions own native models/tensors on one worker and expose portable owned
data and opaque handles. Native bindings stay private; a separate system-binding
crate is unnecessary. Model/tokenization orchestration remains a subsequent
integration step. The pinned Supertonic bundle and saved-tensor replay tools
qualify runtime outputs independently of that future text-to-PCM implementation.

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
with lazy loading of the native library. Package APIs provide idiomatic cleanup,
exceptions, and concurrency integration as those capabilities are implemented.

Text normalization, model policy, inference scheduling, and synthesis stay in
Rust. Host-language integrations should not reimplement them. Audio playback is
a separate concern from producing PCM and is not a dependency of the core.

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

Native releases will need an explicit OS/architecture/backend matrix, compatible
library dependencies, and tests of installed packages. The present scaffold only
has local builds, a Python wrapper wheel, and configurable external native
libraries. It does not publish packages or ship prebuilt runtime bundles.

An optional `mlsr-cli` crate can be added when a synthesis command has useful
behavior. Until then, `cargo run -p mlsr --example version` exercises the Rust API.
