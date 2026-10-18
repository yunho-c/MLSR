# MLSR for Python

A `ctypes` binding to MLSR's Rust implementation. Persistent engines expose
bounded segment jobs, cancellation, owned Float32 audio, and structured errors.
Synthesis supports macOS arm64/Core ML and experimental Windows ARM64/QNN.

Requires Python 3.10+ and a separately built MLSR shared library. Importing `mlsr`
does not load native code. The first API call loads `MLSR_LIBRARY_PATH` if set,
otherwise it looks for `mlsr_ffi` on the system library search path. Set the
environment variable before the first call; the loaded library is cached.

```python
import mlsr

with mlsr.Engine("model-manifests/supertonic3-coreml.yaml", "/path/to/bundle") as engine:
    with engine.start("Hello from MLSR.", language="en", voice="M1") as job:
        for event in job:
            if event.kind == "audio":
                consume_pcm(event.samples, event.sample_rate)  # Your bounded consumer.
            elif event.kind == "failed":
                raise event.error
```

Reuse an engine between jobs. `job.cancel()` wakes a blocked producer without
waiting for native inference; `job.wait(timeout_ms=100)` observes quiescence.
Context exit / `close()` cancels and waits. One active job is accepted per engine;
another raises `SynthesisError` with code `engine_busy`. Iteration has one consumer
and no hidden audio queue. Returned `array('f')` samples survive cleanup. Text
spans index the original input's UTF-8 bytes, not Python character positions.
Blocking calls release the GIL; run them on a worker thread in an async/UI app.

From the repository root, build the native library and run the WAV example:

```sh
MACOSX_DEPLOYMENT_TARGET=15.0 cargo build --locked --release -p mlsr-ffi --features coreml
MLSR_LIBRARY_PATH="$PWD/target/release/libmlsr_ffi.dylib" \
  PYTHONPATH=bindings/python/src python3 bindings/python/examples/synthesize.py \
  --manifest model-manifests/supertonic3-coreml.yaml --bundle /path/to/bundle \
  --output hello.wav --cancel-first
```

`--cancel-first` demonstrates cancellation and reuse before exporting speech.
The output must be a new path; model weights are never downloaded implicitly.

For Windows ARM64, follow [QNN setup](../../docs/qnn-runtime.md), build
`mlsr-ffi --features qnn`, and run ARM64 Python with the prepared DLLs:

```powershell
. ./tools/qnn_env.ps1
$env:MLSR_LIBRARY_PATH = "$PWD/target/release/mlsr_ffi.dll"
$env:PYTHONPATH = "$PWD/bindings/python/src"
```

Select `compute_policy="qnn_htp"` when constructing `Engine` with
`model-manifests/supertonic3-qnn-superfast.yaml` and its verified bundle. The
helper sets `MLSR_DLL_DIRECTORIES`; the binding registers those semicolon-separated
directories with Python's Windows DLL loader and keeps them alive with the library.
The preset has [open model-quality failures](../../docs/qnn-qualification.md).

From a repository checkout, run `python3 tools/dev.py test-python` at the root.
`python3 tools/dev.py package-python` builds a wheel and tests an installed copy
outside the source package. The wheel currently contains Python code only and
requires the external native library; bundled native wheels are future work.
`tools/check_synthesis_bindings.py --bundle /path/to/bundle --output-dir /new/path`
qualifies an installed wheel with real speech, cancellation and a WAV example.

Licensed under Apache-2.0; see `LICENSE`.
