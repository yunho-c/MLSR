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
checkout. Its Windows backend registration patch is required: a stock backend
DLL has a different registration boundary. Ordinary Cargo builds never build
ExecuTorch or download the SDK.

```powershell
./tools/build_qnn.ps1 -ExecuTorchPrefix ../supertonic-npu/build/executorch-arm64/install
. ./tools/qnn_env.ps1
cargo build --locked --release -p mlsr-cli -p mlsr-ffi --features qnn
cargo test --locked --release -p mlsr --features qnn runtime::qnn::tests -- --ignored --test-threads=1
```

`qnn_env.ps1` selects the installed SDK, V81 skeleton and FastRPC driver, adds
runtime DLL directories to this shell's PATH, and sets `MLSR_QNN_BRIDGE_DIR` for
Cargo. The bridge and backend DLLs must remain loadable for every native host.
Select `ComputePolicy::QnnHtp`, CLI `--compute-policy qnn-htp`, or C ABI compute
policy `2`. Feature bit `4` reports the compiled QNN capability. The ABI layout
and default Core ML selection are unchanged.

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
inputs and program bytes. Readback returns an independent owned snapshot.
`live_tensors` counts client handles, not the bounded last-input references
retained by modules. Native operations are serialized across sessions because
the pinned delegate shares backend resources. Cancellation bypasses this queue
but waits for an in-flight graph to return before discarding its output.

## Qualification status

Asset verification and portable contract tests are independent of execution.
The persistent adapter has passed real-device model reuse, buffer ownership,
alias rejection, metadata failure recovery and Unicode-path checks. Full stage
replay, public job/C ABI checks and device qualification remain separate checks;
the initial feasibility smoke does not establish their results.

The job contract remains [the current Rust contract](jobs.md): one active job,
bounded output, UTF-8 byte source offsets and cancellation independent of the
inference queue. Playback and the future `Tts`/`Speech` facade are separate.
