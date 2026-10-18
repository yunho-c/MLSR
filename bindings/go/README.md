# MLSR for Go

A `cgo` binding to MLSR's shared C ABI. Requires Go 1.22+, a C compiler, the
generated `mlsr.h`, and the Rust shared library. Persistent engines expose
bounded segment jobs, context cancellation and owned Float32 audio. Synthesis
supports macOS arm64/Core ML and experimental Windows ARM64/QNN.

Use `Open(manifest, bundle, EngineOptions{})`, then `Start(ctx,
DefaultRequest(text))`. Customize language, voice, seed and limits on that request
before starting it. `Next()` pulls one event at a time and returns `io.EOF` after
one completed/cancelled/failed terminal event. Check `event.Failure` on failure.
Call `Close()` on every job and engine, normally with `defer`; it cancels and waits
for native work. A competing job reports `*mlsr.Error` with code `engine_busy`.

The context watcher cancels even when the consumer stops calling `Next`; it does
not drain or buffer audio. `Cancel()` is immediate, while `Wait(timeoutMS)` waits
for quiescence. Returned samples are Go-owned and survive cleanup. Event text
spans use original UTF-8 byte offsets, matching Go string indices. Keep one pull
consumer; `Cancel`/`Close` may run concurrently with `Next`.

From the repository root, run `python3 tools/dev.py test-go`. The development
tool supplies the include, link, and runtime library paths. The Go module itself
does not depend on paths outside its package, so external consumers can install
the native header/library and configure `CGO_CFLAGS` and `CGO_LDFLAGS` themselves.

For manual builds, set `CGO_ENABLED=1`, add the header directory with `-I` to
`CGO_CFLAGS`, add the library directory with `-L` to `CGO_LDFLAGS`, and make the
shared library available to the OS loader. Linking `mlsr_ffi` is handled by the
binding. Cross-compilation also requires a target C toolchain and target Rust
library. Builds with `CGO_ENABLED=0` are not supported.

For Windows ARM64, follow [QNN setup](../../docs/qnn-runtime.md), build
`mlsr-ffi --features qnn`, and use native ARM64 Go with a compatible LLVM-MinGW
C compiler. The Windows binding explicitly links `mlsr_ffi.dll.lib` (the DLL
import library); `mlsr_ffi.lib` is Rust's separate static archive. Put the shared
library on PATH and dot-source `tools/qnn_env.ps1` for QNN dependencies. Pass
`EngineOptions{QNNHTP: true}` to `Open` with the QNN manifest and verified bundle.
`CPUOnly` and `QNNHTP` are mutually exclusive. The preset remains
[experimental because of model-quality failures](../../docs/qnn-qualification.md).

For example, after building the Core ML library from the repository root:

```sh
MACOSX_DEPLOYMENT_TARGET=15.0 cargo build --locked --release -p mlsr-ffi --features coreml
export CGO_CFLAGS="-I$PWD/include"
export CGO_LDFLAGS="-L$PWD/target/release -Wl,-rpath,$PWD/target/release"
cd bindings/go
go run ./examples/synthesize --manifest ../../model-manifests/supertonic3-coreml.yaml \
  --bundle /path/to/bundle --output hello.wav --cancel-first
```

The example cancels one job, reuses its engine and streams a new WAV file.
`tools/check_synthesis_bindings.py --bundle /path/to/bundle --output-dir /new/path`
qualifies a relocated module with real synthesis, `go test -race` and this example.

The module path is `github.com/yunho-c/MLSR/bindings/go`. Future release tags for
this nested module must use the `bindings/go/` prefix, such as
`bindings/go/v0.1.0`.

Licensed under Apache-2.0; see `LICENSE`.
