# T384 Superfast bundle

The experimental `supertonic3-qnn-superfast-t384-v1` bundle supports the
`feat/duration-planning` API with 384 normalized text tokens and three matched
flow/vocoder pairs. It keeps Superfast's FP16 computation, exact GELU, shared CFG
prefix and eight separate flow updates. The target remains Windows ARM64,
Snapdragon X2 Elite X2E84100, SoC 88 / HTP V81, QAIRT 2.50.40.260831.

| True latent length | Selected flow and vocoder | Maximum PCM capacity |
| --- | --- | --- |
| 4–128 | L128 | 393,216 frames / 8.92 seconds |
| 129–256 | L256 | 786,432 frames / 17.83 seconds |
| 257–512 | L512 | 1,572,864 frames / 35.67 seconds |

The eight PTE programs total 603,350,344 bytes. MLSR verifies all program and
metadata hashes, loads the eight graphs once, and selects the smallest fitting
bucket after duration prediction. Input over 384 normalized tokens or duration
over the largest bucket is rejected; text is never truncated. Segmentation
uses the verified text capacity, preserving complete sentences when they fit.
Long predicted durations can still exceed the audio capacity.

The pinned source noise has shape `[1,144,512]`. Each channel is cropped to the
true length and padded to the selected bucket. The vocoder consumes that whole
masked bucket; PCM is trimmed to the requested frame count. Prepared durations
use exactly the same routing as ordinary synthesis. Rendering a prepared segment
uses TE, eight flow calls and vocoding, without repeating duration inference.

The original T128/L96 bundle remains independently pinned. Its capacity-128
noise and program bytes are preserved. Seeded PCM between the two bundles is
expected to differ. The native ABI, QNN feature bit and compute policy are shared;
Python and Go select the new manifest/model ID without new API fields.

## Prepare and run

Use the matching `feat/superfast-t384` checkout in `supertonic-npu`. The pinned
manifest references `artifacts/superfast-t384-20261019`. With those verified local
artifacts present:

```powershell
$manifest = 'model-manifests/supertonic3-qnn-superfast-t384.yaml'
$bundle = 'C:/models/supertonic3-qnn-superfast-t384-v1'
python -m tools.supertonic3.qnn_bundle prepare --manifest $manifest --source ../supertonic-npu --output $bundle
python -m tools.supertonic3.qnn_bundle verify --manifest $manifest --bundle $bundle
. ./tools/qnn_env.ps1
cargo build --locked --release -p mlsr-cli -p mlsr-ffi --features qnn
target/release/mlsr.exe synthesize --manifest $manifest --bundle $bundle --compute-policy qnn-htp --text "A complete sentence can now stay together through planning and synthesis." --language en --voice M1 --output sentence.wav
```

For the model store, explicitly load `catalog-qnn-experimental.yaml`, resolve
`supertonic3-qnn-superfast-t384-v1` for `windows-arm64`, and use `QnnHtp` / `qnn_htp`.
The builtin catalog continues to contain qualified Core ML bundles. Runtime DLLs
and the Qualcomm SDK remain external dependencies, as described in
[QNN runtime setup](qnn-runtime.md).

To rebuild, use `python scripts/build_superfast_context.py --output <fresh-directory>`
in the sibling project. Export sources and original ONNX hashes are recorded in
each component's metadata. Every FP32 transformed graph is checked against its
original ONNX before compilation, and every compiled graph must be a single QNN
delegate with no CPU compute. Rebuilds may change program or metadata bytes;
they require a newly generated manifest and device qualification. The explicit
`tools.supertonic3.qnn_context_manifest` command locks a newly assembled preset;
it never overwrites a manifest.

## Device checks

```powershell
. ./tools/qnn_env.ps1
$env:MLSR_QNN_CONTEXT_BUNDLE = $bundle
$env:MLSR_QNN_CONTEXT_REPORT = 'ignore/qnn-context-new-run'
cargo test --locked --release -p mlsr --features qnn --test qnn_context -- --ignored --nocapture --test-threads=1
```

On the recorded X2 Elite, all eight programs loaded and native checks passed at
text lengths 383/384 (385 rejected) and latent lengths 4/127/128/129/255/256/257/512.
Oversized duration and unsupported steps fail at their expected boundaries.
English M1 and Korean F1 prepared plans preserve complete sentences, execute
one duration prediction per segment, and reproduce linear PCM bit for bit in
random order `[2,0,1,2]`, with stable live tensor counts.

The initial release run loaded the graphs in 1.41 seconds, excluding asset
verification. Four longer EN/KO × M1/F1 cases generated 13.6–16.0 seconds of
audio in 194.7–198.9 ms per warm call, all using L256. Three repetitions per
case reproduced identical PCM. These are measurements from this host and
workload, not a performance guarantee for every bucket.

The [qualification snapshot](../tests/fixtures/supertonic3/qnn-context-summary.json)
records program identities, measurements and evidence hashes. Installed Python
and relocated Go packages passed their SDK tests and produced identical PCM
through legacy synthesis, developer rendering and WAV examples. The original
T128/L96 profile still matches its prior EN/KO × M1/F1 PCM hashes exactly.

Combining this implementation with `backend/windows-output` also passed real,
muted WASAPI playback: pause/resume, seek/replay, retention, cancellation,
replacement and native drain. Python and Go passed their on-demand seek tests
against that combined build. Windows playback remains a separate branch/feature;
the headless build does not open an audio endpoint. The snapshot records the
exact combined tree used for these checks.

The device probe saves four Float32 WAVs and `device.json` when a fresh report
directory is supplied. Subjective listening remains pending; numerical export
equivalence and device checks do not establish perceptual quality. The bundle
stays explicitly experimental while the user evaluates the audio.
