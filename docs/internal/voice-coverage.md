# Supertonic3 preset voice coverage

The `feat/voices` implementation introduces the separately pinned
`supertonic3-coreml-fp16-t384-voices-v1` release. It provides the ten preset
voices and 31 language choices in the [official demo](https://huggingface.co/spaces/Supertone/supertonic-3),
whose inspected revision is `4402acca094e2ff2b3dff446e0bf915f3fccc462`.
The demo's names are retained as display metadata; stable IDs remain
`supertonic3:M1` through `supertonic3:F5`.

## Asset identity and compatibility

The new [manifest](../../model-manifests/supertonic3-coreml-fp16-voices.yaml)
contains 42 files totaling 459,297,332 bytes. Every original FP16 graph,
tokenizer/configuration file and M1/F1 style is unchanged. Eight additional
official style JSON files add 2,331,748 bytes. Their Float32 conditioning
contracts are `[1,50,256]` and `[1,8,16]`; no graph re-export was required.
The original FP16, INT4 and QNN manifests and identities remain intact.

Manifest voice declarations now control loading and discovery. Artifact sets,
paths, checksums, graph interfaces and conditioning shapes remain checked.
Display names cannot change stable IDs or imply a regional accent. The existing
JSON metadata bridge carries the expanded catalog through C, Dart, Go and
Python without changing the C ABI layout.

## Validation on 2026-10-19

Host: Apple M4 Pro, macOS 27.0.1, CPU + Neural Engine policy. The compact tracked
receipt is [voices-summary.yaml](../../tests/fixtures/supertonic3/voices-summary.yaml).
Full reports, PCM hashes, 49 WAVs, Flutter screenshots and integration exports
are under `~/.cache/mlsr/experiments/supertonic3-voices-20261019/`.

- All **310 voice/language combinations** generated finite, non-silent, bounded
  mono Float32 PCM at 44.1 kHz. Settings: eight steps, raw speed 1.05, seed 1729.
  Samples and native-library/manifest hashes are recorded. M1 WAVs cover every
  language; English/Korean WAVs cover every preset.
- M1/F1 in English/Korean match the original FP16 release's PCM byte-for-byte
  under matched input/settings. Native handles return from zero to zero.
- The real Flutter integration test passes with muted native playback. It
  selects Olivia/F4 and Arabic, requests two steps and displayed speed 1.3×,
  and verifies effective raw speed 1.35 in immutable request metadata. It also
  passes the existing replace/enqueue/cancel, reader navigation, seek/replay,
  retained WAV export and handle-cleanup regressions.
- Widget coverage verifies ten voice choices, 31 language choices, preservation
  of edited text when switching language, explicit example insertion and Arabic
  direction. English and Arabic integration screenshots were visually reviewed.
- The macOS release app builds and passes deep/strict code-signature verification.
  Its actual packaged native library loads all ten voices from its packaged model
  assets. An Olivia/Arabic rendering matches the Flutter integration export
  byte-for-byte, with zero handles remaining after shutdown. The release receipt
  is `release-validation.json` alongside the other evidence.
- Portable Rust workspace tests pass: 87 core, seven CLI, eight FFI and two
  playback tests, with three external-asset core tests skipped. All-feature,
  all-target Clippy passes. Python model-tool tests: 60 passed. Portable Python,
  Go and Dart binding checks pass; separate real-model suites are not implied
  by their skipped asset-dependent cases. Flutter analysis and two widget/audio
  tests pass.

The smoke matrix is **not** an intelligibility, speaker-similarity, perceptual
quality or performance benchmark. The eight new presets retain empty
`qualified_languages`; M1/F1 retain their prior English/Korean qualification.
Broader listening, long-form multilingual evaluation, other devices and QNN
voice expansion require their own evidence. The demo's external voice-cloning
service is outside this preset-coverage change.

See [preparation and reproduction](../../tools/supertonic3/README.md#recommended-fp16-bundle).
