# Supertonic 3 QNN integration

The first QNN target is Windows ARM64 on Snapdragon X2 Elite X2E84100
(SoC 88, HTP V81), using QAIRT 2.50.40.260831 and the pinned ExecuTorch
1.5.0 build. Other Snapdragon generations and Android need their own exports
and qualification. The `superfast` programs are experimental; numerical
screening and ASR diagnostics do not establish perceptual quality.

## Local assets

[The YAML manifest](../model-manifests/supertonic3-qnn-superfast.yaml) pins the
four PTE programs, their original conversion metadata, tokenizer/configuration,
M1/F1 voices, and license texts. It also records the SDK/runtime identities,
conversion revisions, source paths and commands. The four programs total
212,190,568 bytes. Model weights and runtime dependencies stay outside Git.

Prepare a self-contained copy from the existing `supertonic-npu` checkout:

```powershell
python -m tools.supertonic3.qnn_bundle prepare --source ../supertonic-npu --output C:/models/mlsr-qnn-superfast-v1
python -m tools.supertonic3.qnn_bundle verify --bundle C:/models/mlsr-qnn-superfast-v1
$env:MLSR_QNN_TEST_BUNDLE = 'C:/models/mlsr-qnn-superfast-v1'
cargo test --locked -p mlsr loads_pinned_qnn_bundle -- --ignored
```

Preparation verifies all source files before creating a fresh destination,
copies the files instead of retaining links into research worktrees, then
verifies the copies. It never downloads or recompiles. The Rust bundle loader
verifies every payload and requires the exact pinned QNN contracts and hashes.
Keep the bundle immutable for every associated engine's lifetime.

Runtime DLLs and the system FastRPC driver are external dependencies. The
manifest records their origin separately from the model payload. The local
preparation command does not package or relicense the Qualcomm SDK.

## Tensor and synthesis contract

Each PTE exposes one `forward` method and one output, exposed by the adapter as
`output`. The YAML input ordinals preserve the ExecuTorch argument order. Input
names come from the pinned conversion metadata. All public floating tensors
are Float32, with FP16 HTP computation and no integer quantization. Text IDs
are Int64. All tensors are contiguous row-major.

The preset accepts batch one, 128 text positions, exactly eight flow evaluations,
and true latent lengths 4 through 96. Every VE and vocoder call uses the fixed
96-position shape. Its maximum PCM capacity is 294,912 mono frames at 44,100 Hz
(about 6.687 seconds). Requests outside these limits must fail explicitly.
Text segmentation alone cannot guarantee that the predicted duration fits.

The VectorEstimator output is the **updated latent**, including classifier-free
guidance and Euler integration in FP16. Rust must not integrate it a second
time. The source metadata identifies `flow_output: latent` and
`update_precision: htp-fp16`; velocity-only experiments are different programs.

MLSR preserves its Float32 duration policy:
`max(0.05, raw_duration / max(speed, 0.05))`, truncates requested PCM frames,
and rounds latent positions upward. The prototype's native runner lacks those
floors; replay must distinguish policy boundaries from adapter errors.

Rust keeps its pinned ChaCha8 Gaussian generator. QNN generates capacity-128
noise, crops **each channel** to the true length, then pads with zeros to 96.
Saved-noise replay accepts the actual `[1,144,128]` tensor, independent of RNG
seed conventions. Equal seeds do not imply equality with NumPy noise.

The QNN vocoder consumes the full fixed latent, followed by PCM trimming.
Core ML's crop-before-vocoder path remains a separate contract. Qualification
must compare this fixed vocoder with canonical unpadded inference, including
short sequences and the upper shape boundary.

## Native build and ownership

Use the ARM64 ExecuTorch installation prepared by the pinned `supertonic-npu`
checkout. The bridge calls that revision's Windows backend registration export
to register with its own ExecuTorch instance. Ordinary Cargo builds never build
ExecuTorch or download the SDK.

```powershell
./tools/build_qnn.ps1 -ExecuTorchPrefix ../supertonic-npu/build/executorch-arm64/install
. ./tools/qnn_env.ps1
python -m tools.supertonic3.qnn_bundle verify-runtime --sdk C:/Qualcomm/AIStack/QAIRT/2.50.40.260831 --bridge $env:MLSR_QNN_BRIDGE_DIR
cargo build --locked --release -p mlsr-cli -p mlsr-ffi --features qnn
cargo test --locked --release -p mlsr --features qnn runtime::qnn -- --ignored --test-threads=1
target/release/mlsr.exe synthesize --manifest model-manifests/supertonic3-qnn-superfast.yaml --bundle C:/models/mlsr-qnn-superfast-v1 --compute-policy qnn-htp --text "Hello from MLSR." --language en --voice M1 --output hello.wav
```

`qnn_env.ps1` selects the installed SDK, V81 skeleton and FastRPC driver, adds
runtime DLL directories to this shell's PATH, and sets `MLSR_QNN_BRIDGE_DIR` for
Cargo. The bridge and backend DLLs must remain loadable for every native host.
Python uses `MLSR_DLL_DIRECTORIES` from this helper to register the explicit
dependency directories with Windows' DLL loader.
Select `ComputePolicy::QnnHtp`, CLI `--compute-policy qnn-htp`, or C ABI compute
policy `2`. Feature bit `32` (`MLSR_FEATURE_QNN_HTP`) reports the compiled QNN capability. The ABI layout
and default Core ML selection are unchanged.

The model ownership/store API uses feature bit `4`, developer SDK bit `8`, and
Apple output bit `16`. A QNN headless build reports mask `45`. Rebuild native
libraries and bindings together when updating the experimental branch: QNN used
bit `4` before model delivery and bit `8` before this SDK integration.

For leased models, explicitly load
[`catalog-qnn-experimental.yaml`](../model-manifests/catalog-qnn-experimental.yaml)
with `ModelCatalog::parse` and resolve `supertonic3-qnn-superfast-v1` for
`windows-arm64`. Supply the self-contained bundle (including `mlsr-bundle.yaml`)
to `ModelHandle::open` or a `ModelStore` installation, then call
`Engine::from_model` with `ComputePolicy::QnnHtp`. C callers use
`mlsr_engine_open_model` with compute policy `2`; Dart callers use
`Engine.fromModel` with `ComputePolicy.qnnHtp`. Engine ownership keeps the store
lease until native shutdown completes. The builtin qualified catalog still
contains only Core ML; the Flutter resource plugin remains macOS-only.

The headless developer SDK uses the same explicit policy: Rust `Tts::open`, C
`mlsr_tts_from_model`, Dart `Tts.fromModel`, Python `Tts.from_model`, and Go
`TTSFromModel`. Manifest/path entry points (`Tts.open`, Python `Tts`, Go `OpenTTS`,
and C `mlsr_tts_open`) also accept it. Use headless output for lazy PCM jobs,
bounded rendering, WAV export and original-text plans. Loaded voices report
`backend: qnn` and an empty `qualified_languages` list. Eight steps are required
at admission; fixed latent/duration limits remain typed synthesis failures.
Closing the facade joins work and releases its model lease; caller-owned PCM
survives closure. Native Windows speech output returns `unsupported_capability`.

One Rust worker owns persistent ExecuTorch modules, their contexts, typed host
allocations and cached tensor handles. The narrow C++ bridge validates PTE
tensor metadata, wraps borrowed input allocations and copies each synchronous
result into an owned Rust destination. Optional VE backings reuse that
destination; this is not direct HTP output binding or zero-copy execution.
ExecuTorch owns accelerator transfers and memory registration. Program bytes
use 128-byte-aligned retained allocations. Public data needs natural Float32/
Int64 alignment; the backend performs its own transfer staging.

Input allocations and descriptors outlive each native call and remain retained
until replaced or the module is destroyed. Modules are destroyed before their
inputs and program bytes. If execution fails, the bridge destroys the module
before releasing either generation of borrowed views; the next prediction
reloads it from the retained program. Readback returns an independent owned snapshot.
`live_tensors` counts client handles, not the bounded last-input references
retained by modules. Native operations are serialized across sessions because
the pinned delegate shares backend resources. Cancellation bypasses this queue
but waits for an in-flight graph to return before discarding its output.

## Qualification status

The [2026-10-18 qualification report](qnn-qualification.md) records passing
real-device stage replay, model reuse, buffer ownership, native failure recovery,
Unicode paths, public jobs, C/Python/Go/Dart and HTP profiling. MLSR matches the
existing QNN runner byte for byte. **Six of twelve canonical ONNX comparisons
fail the fixed numerical screen.** These are diagnostics, not a perceptual
verdict. Keep this preset experimental pending subjective listening qualification.

The job contract remains [the current Rust contract](jobs.md): one active job,
bounded output, UTF-8 byte source offsets and cancellation independent of the
inference queue. The shared `Tts` facade adds a bounded FIFO and lazy requests
above that engine; speech output remains a separate capability.

## Reproduce device checks

Use fresh output paths. Raw tensors, audio and logs are intentionally retained
on failure. These are development tools; Python/ONNX are not runtime dependencies.
Replay needs NumPy, PyYAML, ONNX Runtime and SoundFile plus the pinned sibling
checkout's native runner and canonical ONNX assets. Profiling additionally uses
that checkout's prepared ExecuTorch Python package, host `flatc`, and the SDK's
x64 inspection bindings and ARM64 command-line tools.

```powershell
. ./tools/qnn_env.ps1
$bundle = 'C:/models/mlsr-qnn-superfast-v1'
$manifest = 'model-manifests/supertonic3-qnn-superfast.yaml'
cargo build --locked --release -p mlsr --features qnn --example qnn_synthesis_replay --example qnn_qualification
python -m tools.supertonic3.qnn_replay --source ../supertonic-npu --bundle $bundle --output ignore/qnn-replay-new
# Exit 1 currently records the canonical failures; inspect report.yaml.
target/release/examples/qnn_qualification.exe $manifest $bundle ignore/qnn-jobs-new.json
python -m tools.supertonic3.qnn_profile --source ../supertonic-npu --bundle $bundle --replay ignore/qnn-replay-new --output ignore/qnn-profile-new
```

Run latency checks without concurrent device work. `qnn_qualification` exercises
the public job API and records initialization, warm requests, cancellation during
TE/VE/vocoder, backpressure, restart, memory and shutdown. `qnn_profile` extracts
the unchanged PTE contexts and compile options, runs detailed SDK profiles on
saved replay inputs and checks outputs against the reference.

For all bindings, supply installed **ARM64** Python, Go, Dart 3.11+ and a compatible
ARM64 LLVM-MinGW compiler. The script builds the shared ABI, tests C and each
binding sequentially, exercises Dart's native-asset hook and leased SDK model
disposal, then compares EN/KO M1/F1 PCM bytes across the legacy and headless SDK
surfaces. It also runs the three public SDK export examples and checks that
Windows native speech reports `unsupported_capability`. It does not download
model weights, the SDK or toolchains.

The opt-in Rust SDK test checks PCM/WAV parity, lazy cancellation, fixed-shape
failure context and recovery, original segment seeds on resumed plans, and
model lease release at close:

```powershell
$env:MLSR_QNN_TEST_BUNDLE = $bundle
cargo test --locked --release -p mlsr --features qnn --test qnn_sdk -- --ignored --nocapture
```

```powershell
./tools/check_qnn_bindings.ps1 -Bundle $bundle -OutputDirectory ignore/qnn-bindings-new -Python C:/tools/python-arm64/python.exe -Go C:/tools/go-arm64/bin/go.exe -Dart C:/tools/dart-arm64/bin/dart.exe -CCompiler C:/tools/llvm-mingw/bin/aarch64-w64-mingw32-clang.exe
```

To qualify an installed Python wheel and relocated Go package, use a host Python
with pip/setuptools and a fresh output directory outside the checkout. The
consumer Python must be ARM64. Embedded Python is supported by copying its runtime
and replacing only that copy's search path with the installed package. The SDK
and FastRPC driver stay external runtime prerequisites; native MLSR/bridge DLLs
and headers are copied alongside the consumers.

```powershell
python tools/check_qnn_packages.py --bundle $bundle --python C:/tools/python-arm64/python.exe --go C:/tools/go-arm64/bin/go.exe --cc C:/tools/llvm-mingw/bin/aarch64-w64-mingw32-clang.exe --output C:/qualification/qnn-installed-new --reference-report ignore/qnn-bindings-new/sdk-pcm.json
```

Record device/OS/toolchain, power source and thermal measurement limitations in
an environment JSON alongside the reports. The checked-in qualification snapshot
retains the captured fields. Revalidate all saved Rust tensors and summarize:

```powershell
python -m tools.supertonic3.qnn_summary --replay ignore/qnn-replay-new --jobs ignore/qnn-jobs-new.json --profile ignore/qnn-profile-new --bindings ignore/qnn-bindings-new --environment ignore/qnn-environment-new.json --compile-options ignore/qnn-profile-new/compile-options.json --output ignore/qnn-summary-new.yaml
```
