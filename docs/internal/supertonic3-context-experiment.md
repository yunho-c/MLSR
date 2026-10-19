# Supertonic3 larger-context experiment

Recorded on 2026-10-18, against MLSR `2299689`, on Apple M4 Pro / macOS 27.0.1.
The experiment produces local comparison audio and T=384 Core ML exports. It
did not itself qualify a replacement shipping bundle or change Flutter's text policy.
The user subsequently judged FP16 near-identical to FP32 and approved promotion;
the separate [FP16 promotion record](supertonic3-fp16-promotion.md) tracks native
integration and its validation. The measurements below remain historical evidence.
The [compact evidence](../../tests/fixtures/supertonic3/context-summary.yaml)
records source hashes, measured ranges, model contracts and external reports.

## Reference and comparison controls

The target is the [official Supertonic3 demo](https://huggingface.co/spaces/Supertone/supertonic-3),
pinned to revision `4402acca094e2ff2b3dff446e0bf915f3fccc462`. Its live JavaScript
matched that revision when inspected. The demo packs sentences toward 300 source
characters for English, but keeps overlong sentences intact. It uses dynamic
text and latent dimensions, rather than the published Core ML bundle's fixed
T=128 text interface. See its [pinned source](https://huggingface.co/spaces/Supertone/supertonic-3/blob/4402acca094e2ff2b3dff446e0bf915f3fccc462/script.js).

For the reported English passage, this gives three chunks with normalized text
lengths **173, 294 and 295**, compared with MLSR's fourteen chunks. The flagged
word pairs are inside the official chunks. T=384 accommodates all three; it is
not a universal guarantee for the demo's overlong-sentence behavior.

`quality_probe.py` rendered both chunk plans through the **same official FP32
ONNX models** for M1 and F4/Olivia: 34 chunk captures in total. Each local model's
SHA-256 matched the demo's published LFS object. Voice styles and normalization
were also checked. The controls use eight steps and raw speed 1.05, corresponding
to the demo's UI speed 1.0. Both comparison layouts have zero added join silence;
an additional official-layout WAV retains the demo's 0.3-second gaps.

These are local ONNX Runtime CPU renders, not recordings of the browser demo.
Noise is reproducible PCG64, not the browser's unseeded Box-Muller generator.
Different chunk plans have different inputs and noise shapes. The actual noise
and every denoising state are saved; conversion comparisons replay identical
noise tensors rather than assuming equal RNG seeds imply equal tensors.

## What the export checks found

The new export starts from the Möbius PyTorch ports at
`26f5fa8059d5ab3d34449adce6ff6ed86734e522`, loading the official ONNX weights.
Two corrections were necessary in this experimental export path:

1. **LayerNorm epsilon:** the port defaulted to `1e-5`; all 71 corresponding
   LayerNorm nodes in the four official graphs specify approximately `1e-6`.
   The tool reads that value and checks layer counts before applying it.
   BatchNorm is left at its own graph-defined value.
2. **The true sequence endpoint:** adding masked zero padding changes the right
   boundary seen by temporal convolutions. The dynamic graph uses replicated
   edge values. Before each masked ConvNeXt convolution, the corrected port
   extends the last valid value through the padded tail, then retains the
   original output masks. This works with prefix masks: valid values followed
   by padding. The vocoder still receives the true-length latent.

The boundary effect was reproduced directly in the official ONNX models:
padding text to T=384 changed predicted duration by 1.19–1.66%; adding latent
bucket padding changed the final valid latent by 31.5–83.9% NRMSE relative to the
text-padded control. These observations concern the padding experiment and the
new exporter. They do **not** establish a defect in the separately published v2
Core ML graphs used by the shipping bundle.

After both corrections, fixed-shape FP32 PyTorch closely matches dynamic ONNX
across all six official-layout voice/chunk cases. Core ML FP32 on CPU also
closely matches, while FP16 has larger remaining numerical differences:

| Implementation versus dynamic official ONNX | Text embedding NRMSE | Final valid latent NRMSE | Waveform NRMSE |
| --- | --- | --- | --- |
| Corrected PyTorch FP32 | 0.000065–0.000079% | 0.00050–0.00456% | 0.0071–0.0781% |
| Corrected Core ML FP32, CPU | 0.000072–0.000080% | 0.00144–0.01240% | 0.0361–0.2361% |
| Corrected Core ML FP16, CPU+NE | 0.270–0.315% | 5.47–24.99% | 31.3–81.0% |

NRMSE is RMS difference divided by the reference tensor's RMS, excluding padded
coordinates. Waveform NRMSE is sensitive to phase and is **not** a word-error or
perceptual quality score. The FP32 control supports the corrected conversion;
it does not establish that the FP16 differences are audible or acceptable.
FP16 CPU-only replay also differs, so the discrepancy is not confined to ANE
execution. INT4 has not been introduced in these new exports.

## Artifacts and execution checks

The external root is
`~/.cache/mlsr/experiments/supertonic3-t384-20261018/`. Important directories:

| Directory | Contents |
| --- | --- |
| `onnx-controls/` | Official and MLSR chunk layouts, M1/F4 WAVs, exact noise and intermediate tensors |
| `padding-port-v3/` | Separate ONNX text-padding and latent-padding controls; epsilon-corrected port |
| `boundary-port-final/` | Port with true-endpoint correction, compared against dynamic ONNX |
| `coreml-boundary-fp16/` | Five FP16 packages and compiled models, YAML manifest, source snapshots |
| `coreml-boundary-fp32/` | Equivalent FP32 Core ML control |
| `coreml-boundary-ane-replay/` | Six complete FP16 CPU+NE replays and M1/F4 audio |
| `coreml-boundary-cpu-replay/` | Six complete FP16 CPU-only controls |
| `coreml-boundary-fp32-replay/` | Six complete FP32 CPU-only controls and M1/F4 audio |

The text encoder and duration predictor use T=384; VectorEstimators use
T=384 with L=256 and L=512. The vocoder accepts true lengths 4–512. For this
passage, M1 uses true lengths 154/252/257 and F4 uses 132/218/222. Export targets
iOS 18 / macOS 15. These experimental manifests are separate from the strict
shipping bundle schema.

Core ML Tools 9.0 and PyTorch 2.8.0 performed conversion under Python 3.12.15;
ONNX Runtime 1.20.1 and PyTorch 2.12.0 performed reference/port checks under
Python 3.13.16. Core ML Tools warns that Torch 2.8 is beyond its tested version;
the real-model comparisons above provide the recorded checks for this run.
Existing interpreters and cached dependencies were reused. ONNX 1.19.1 for
Python 3.12 was installed into the external experiment directory; no virtual
environment or global package update was required.

The `coreml-cli` compute-plan implementation reports all 800 counted operations
in each VectorEstimator and all 107 vocoder operations assigned to ANE. The
duration predictor and text encoder have mixed CPU/ANE plans. These are device
assignments and estimated costs, **not measured residency or utilization**.
The report is `coreml-boundary-placement.yaml`. Per-call timings in replay
reports are diagnostics, not a controlled performance benchmark; some runs
overlapped conversion or placement inspection.

Every replay here freezes duration, true latent length and noise to the official
reference. Core ML duration predictions are recorded and compared, but do not
control that replay's length. Native duration planning, bucket changes and
end-to-end synthesis still need qualification before integration.

Downloads contains these auditable comparison families, each for M1 and F4:

- `Supertonic3-VOICE-ONNX-official-chunks.wav`
- `Supertonic3-VOICE-ONNX-mlsr-chunks.wav`
- `Supertonic3-VOICE-CoreML-T384-FP32-reference.wav`
- `Supertonic3-VOICE-CoreML-T384-FP16-boundary-corrected.wav`

Download receipts include SHA-256 hashes. The complete private passage and
large model/audio/tensor payloads remain outside Git. Ordinary portable checks
passed: 54 model-tool tests, including eight new tests for noise preservation,
nontruncation, join behavior, reference integrity, bucket routing and replay
boundaries.

## Reproducing the experiment

Use an existing interpreter with NumPy/PyYAML, plus ONNX Runtime for references,
PyTorch/ONNX for port checks, and Core ML Tools for export/replay. No tool installs
dependencies or downloads weights automatically. `analysis` below is the saved
official-demo inspection directory, containing the pinned source/tree, styles,
indexer and exact chunk plan. The MLSR report supplies its original text spans.

```sh
python3 -m tools.supertonic3.quality_probe \
  --analysis /path/to/official-demo-analysis \
  --onnx /path/to/official/onnx \
  --mlsr-report /path/to/clause-fix-v1/report.json \
  --output /path/to/new/onnx-controls

python3 -m tools.supertonic3.context_probe \
  --captures /path/to/onnx-controls --onnx /path/to/official/onnx \
  --mobius-source /path/to/mobius/models/tts/supertonic-3 \
  --output /path/to/new/boundary-port --preserve-boundaries

python3 -m tools.supertonic3.context_coreml export \
  --onnx /path/to/official/onnx \
  --mobius-source /path/to/mobius/models/tts/supertonic-3 \
  --probe /path/to/boundary-port --output /path/to/new/coreml \
  --text-capacity 384 --precision float16

python3 -m tools.supertonic3.context_coreml replay \
  --captures /path/to/onnx-controls --probe /path/to/boundary-port \
  --models /path/to/coreml --output /path/to/new/replay \
  --policy cpu_and_neural_engine
```

Omit `--preserve-boundaries` for the ordinary zero-padding port control. Export
with `--precision float32` and replay with `--policy cpu_only` for the FP32
conversion control. Output directories must be new. Reports distinguish a
completed measurement from a shipping or listening qualification.

## Remaining gates

1. Listen to the official-layout versus MLSR-layout ONNX controls, then the
   FP32/FP16 Core ML versions. Check the reported pauses and missing words;
   numerical proximity alone cannot approve speech quality.
2. Qualify native duration planning and FP16 precision on a broader corpus,
   including Korean, short inputs, overlong sentences and bucket boundaries.
   Evaluate INT4 separately only after establishing an acceptable FP16 baseline.
3. Integrate model-declared text capacity and natural sentence packing into the
   Rust pipeline. Preserve original-text mappings across multi-sentence model
   chunks for the speech controller; increasing the numeric limit alone is
   insufficient. Then test Flutter playback/export and public API behavior.
