# Core ML runtime

The optional `coreml` feature uses cidre 0.29.0 on macOS 15+/iOS 18+. The
private adapter is in `crates/mlsr/src/runtime/coreml`; no Objective-C types or
borrowed native pointers form part of the public Rust API or C ABI.

## Ownership and validation

`HostTensor` owns contiguous row-major Float32 or Int32 values. Construction
rejects empty/zero dimensions, overflowing element/byte counts, and data-length
mismatches. Represent scalar model inputs as `[1]`.

Native arrays own retained `MLMultiArray` objects. Buffer access is scoped to
Core ML's callbacks, with checked byte spans and element strides. Reading an
output does not assume contiguous storage. Initialized allocations and typed
copies preserve integer and floating-point representations. Native wrappers
are explicitly neither Send nor Sync; the session owns them on one thread.

Predictions validate named input dtypes/shapes against Core ML's description,
reject duplicate/unknown names, and validate requested outputs before returning
retained tensors. Core ML reports missing required inputs. Each load/prediction
uses an autorelease pool; arrays retain their storage independently of temporary
feature providers. Native errors and exceptions at call boundaries become owned
Rust diagnostics containing operation, domain/name, code, and message.

Unsafe byte copies are confined to the private adapter; the rest of `mlsr`
denies unsafe code. Native paths must identify existing compiled `.mlmodelc`
directories. Asset acquisition and YAML-manifest integrity verification belong
to the model bundle layer; loading a graph does not download or convert it.

## Tests

```sh
MACOSX_DEPLOYMENT_TARGET=15.0 cargo test -p mlsr --features coreml
```

Apple tests compile the tiny [affine fixture](../tests/fixtures/coreml/README.md)
with Xcode and execute it on CPU. They exercise independently known arithmetic,
stride-aware copies, bit-preserving tensor transfer, output lifetime, malformed
inputs/models, error recovery, and a controlled Objective-C exception. Portable
host-tensor validation runs without the feature. Real Supertonic reference replay
is an additional opt-in qualification.

## Persistent sessions

`Session::coreml(ComputePolicy::CpuAndNeuralEngine)` starts one worker with a
bounded queue. `CpuOnly` supports diagnostic controls. `load` caches compiled
models by canonical path for that session's lifetime; callers verify asset
integrity and keep the bundle immutable. Separate sessions provide independent
models and compute policies. `warm_up` explicitly predicts with supplied inputs
and discards the outputs; it does not manufacture speech/model-specific inputs.

`upload`, `zeros`, `write`, `predict`, and `read` operate through opaque handles.
Prediction forwards native arrays directly. `read` is the explicit host-copy
boundary. Tensor drops enqueue release; model handles share the session cache.
Handles keep their owner alive, and mixing handles from different sessions is
rejected. Cloned sessions may be called concurrently; their native work remains
serialized without unsafe Send/Sync implementations.

`shutdown` closes every client/handle for that session and joins the worker after
queued work finishes. Dropping its final client/handle also shuts down and joins.
Native work is synchronous and cannot be interrupted mid-prediction. Worker
panics terminate the session and become errors rather than leaving waiting
callers blocked. Abandoned replies release otherwise unreachable tensors.

`stats` reports loaded models, retained tensor handles, and successful prediction
calls (including warm-up). Tests cover cache reuse, repeated calls, release,
concurrent clients, cross-session rejection, shutdown races, and worker failure.
