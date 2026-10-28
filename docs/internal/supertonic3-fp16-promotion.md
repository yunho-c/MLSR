# Supertonic3 FP16 promotion

On 2026-10-18 the user accepted the corrected FP16 comparison audio as sounding
near-identical to the FP32 reference and requested Core ML as the product target.
The recommended model is now `supertonic3-coreml-fp16-t384-v1`. ONNX remains
confined to optional development reference/conversion tools; there is no ONNX
runtime backend in MLSR. A separate four-bit candidate has also been built and
checked; it remains outside the default release pending listening assessment.

## Model and API changes

The new [manifest](../../model-manifests/supertonic3-coreml-fp16.yaml) pins all six
compiled FP16 graphs, tokenizer/configuration and M1/F1 voice styles: 34 files,
456,965,584 bytes (435.8 MiB). The previous INT4 bundle was 161.5 MiB. The new
manifest retains export-source snapshots and input hashes; it identifies local
compiled artifacts, with no published download. Its exact file SHA-256 is
`4e872b2627fc50f6920549525eb4aed7d3572c8dd458b090083e62a827627c49`.

The text window is 384 normalized tokens. Estimator buckets remain 128, 256 and
512 latent frames, selected by native predicted duration; the vocoder still
receives the true latent length. All graphs use the LayerNorm epsilon and true
sequence endpoint corrections described in the [experiment](supertonic3-context-experiment.md).
The added L128 estimator uses the same corrected exporter.

The Rust frontend, text encoder and segmenter read the verified manifest's
capacity. A sentence that fits stays whole. Oversized sentences fall back to
clauses/words under both the normalized-token limit and a 300-source-grapheme
target (120 for Korean/Japanese). No text is truncated. Complete sentences remain
separate, preserving exact reader navigation. Thus the reported passage changes
from fourteen fragments to six sentences, rather than the demo's three packed
multi-sentence invocations. Multi-sentence packing would need additional timing
alignment to preserve sentence navigation and remains separate work.

`BundleManifest::recommended()`, the unloaded voice catalog, public examples and
the Flutter app select FP16. `baseline()` and the legacy catalog entry retain
the original T=128 INT4 identity and segmentation for reproducibility. Archived
fused execution variants still require that exact old manifest; they cannot be
mixed into the new release. No C ABI layout or binding ownership rules changed.

## Qualification

Host: Apple M4 Pro, macOS 27.0.1. Compact hashes and receipts are in
[fp16-summary.yaml](../../tests/fixtures/supertonic3/fp16-summary.yaml).
Full evidence lives under
`~/.cache/mlsr/experiments/supertonic3-t384-20261018/`.

- `release-reference/` captures 16 real Core ML cases and five rejected-input
  cases. Native FP16 duration predictions determine frame counts and buckets;
  timings are not forced from the earlier ONNX experiment. The corpus covers
  both voices in English/Korean, normalization, long inputs beyond T=128, and
  calibrated true latent lengths 127/128/129/255/256/257. Calibrated-speed cases
  are routing checks, not naturalness evaluations. Positive frontend fixtures
  come from the pinned independent upstream implementation.
- `release-rust-ordinary.yaml` and `release-rust-backings.yaml` each pass 768
  exact tensor comparisons over two passes, five input rejections and four
  seeded ordinary/diagnostic comparisons. These run original text through Rust
  duration, mask, bucket, denoising and PCM trimming logic, sharing only saved
  initial noise with the independent Core ML reference. Six models load once
  per session; resource accounting is checked throughout and at unload.
- Portable checks pass: 86 core, seven CLI and eight FFI tests; 56 Python model
  tool tests; all-feature workspace Clippy; Flutter analysis. Tests cover strict
  model interfaces, normalized 383/384/385 token boundaries, overlong sentences,
  Korean expansion, original spans and legacy compatibility.
- The final Flutter integration test passes with real synthesis and muted
  output: replace/enqueue/cancel, pause/rate/volume/seek, retained replay,
  English/Korean export, Unicode reader navigation and zero leaked handles.
  The macOS release build and deep/strict code-signature verification pass.
- `release-app-check/release-validation.json` exercises the actual native
  library embedded in the release app. The private reported passage has six
  sentences and six model segments, 2,082,770 frames (47.2283 seconds). Repeated
  synthesis and muted playback's retained export match exactly. Original UTF-8
  and UTF-16 spans reconstruct the input; navigation to sentences 2, 3 and 0
  selects the expected frames. A longer Korean request renders as two complete
  sentences. Retention is released and native handles return from zero to zero.
- The L128 compute plan assigns its counted operations to ANE, consistent with
  the earlier L256/L512 plans. Placement plans do not establish utilization,
  physical residency, latency or energy improvements.

The first Flutter run exposed an existing state-observer bug: a paused seek
within a fully generated segment could update native position without emitting
a new revision. Flutter kept displaying the old position. A focused portable
test fails before the fix and passes afterward. The output worker now publishes
the applied seek after a matching native snapshot, even when other fields are
unchanged. The original failed run is retained in `release-flutter-test/`; the
successful final run is in `release-flutter-final/`.

The new native M1 WAV is `~/Downloads/MLSR-M1-CoreML-FP16-T384.wav`. PCM SHA-256:
`c5477b0f7531fb22e5090095a681a6703b7565bbb0e79830c43b00aeb3b52a82`.
This six-sentence rendering still needs the user's listening assessment. Their
approval concerned the earlier three-chunk FP16/FP32 comparison. Numerical
replay and lifecycle checks do not establish perceptual equivalence across a
corpus. Pronunciation of `disliked` remains the deferred investigation.

## Four-bit candidate

`tools.supertonic3.quantize_context` applies Core ML Tools 9.0 four-bit k-means
palettization to eligible estimator weights (threshold 2,048 elements), following
the method in FluidAudio's Supertonic3 quantization types and Möbius'
`coreml/convert_ve_fused.py`. It starts from the corrected FP16 packages, verifies
their provenance/hashes, leaves other stages and conditioning bytes unchanged,
and writes an external candidate bundle. It does not alter the shipping catalog.

The candidate is `coreml-int4-candidate/bundle/` under the external evidence root.
Its complete payload is 170,100,274 bytes (162.2 MiB), **62.8% smaller** than FP16.
Both allocation modes pass 768 exact Rust/reference tensor comparisons, five
input rejections and four seeded-path comparisons each. Across the matched
16-case FP16/INT4 captures, text conditioning, actual initial noise, native
duration predictions, frame counts and bucket choices are identical. Only the
estimator weights change; the final latent and waveform do differ.

The unpadded final-latent NRMSE ranges from 41.4% to 121.7%, and waveform NRMSE
from 106.4% to 136.5%. These are numerical differences, not evidence that speech
is perceptually equivalent or degraded. No quality approval is inferred from
the successful runtime checks. Conversion emitted NumPy/scikit-learn matrix
warnings; logs are retained, and all sixteen cases produced finite tensors and
PCM. Static compute plans assign the counted operations of all three compressed
estimators to ANE; this does not prove utilization or residency.

The actual release library renders the same six-sentence M1 passage at exactly
the same frame count. The comparison is
`~/Downloads/MLSR-M1-CoreML-INT4-T384-candidate.wav`, alongside the FP16 WAV.
Native handles return to zero. Full receipts, comparison ranges and the small
local performance observation are in
[int4-summary.yaml](../../tests/fixtures/supertonic3/int4-summary.yaml).

A small paired battery-powered check on the same M4 Pro measured median complete
`Tts.render` time of 445.1 ms for FP16 and 428.4 ms for INT4 across six warm runs
each, in FP16/INT4/INT4/FP16 process blocks. Both render the 47.23-second M1 passage.
The timing includes Python PCM transfer. Peak process RSS, including load and
warm-up, was 199.6–210.5 MiB for FP16 and 195.4–197.1 MiB for INT4. This one-passage
observation used existing OS caches and an unisolated host with another MLSR app
open; it does not establish a general speedup, steady memory use or ANE memory.

Remaining promotion steps:

1. Compare matched WAVs by listening, including the reported problem locations.
   Broaden the performance checks to representative inputs/devices, including
   startup, process memory, complete synthesis latency and native placement.
   A smaller model alone is not evidence of faster execution or persistent
   ANE residency.
2. Promote INT4 only after those quality and runtime checks, with a distinct
   manifest/catalog identity and the same native/Flutter lifecycle gates.

This promotion qualifies the local macOS slice. Physical iOS, minimum supported
OS execution, redistributed model/native packages and broader listening remain
separate gates; historical INT4 performance/trace results do not qualify FP16.
