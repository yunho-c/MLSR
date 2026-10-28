# QNN and FP16 consolidation

This merge combines local `7d9e4fe` (T=384 FP16 Core ML and complete-sentence
segmentation) with upstream `751b8db` (QNN PR #1). The recommended Flutter/Core ML
bundle remains FP16. QNN Superfast remains an explicit experimental selection;
its manifest and catalog retain the exact upstream bytes.

## Resolutions

- Manifest validation selects the contract by both backend and text capacity:
  Core ML T=128, Core ML T=384, or QNN T=128. Crossed contracts are rejected.
- Paused seeking publishes position and original-text highlighting together,
  after the output confirms the new epoch. The pending notification also
  preserves observer updates when a seek stays in the same segment or position.
  A gated regression checks the snapshot before the new position is published.
- Documentation preserves FP16 setup and QNN headless usage. Shared T=128
  segmentation uses the conservative clause-aware policy; T=384 keeps complete
  sentences that fit. QNN's L96 duration ceiling remains independent of text fit.
- Core ML synthesis and variant replay reports include the shared development
  transport source introduced by the QNN refactor.

## Validation

The [compact record](../../tests/fixtures/supertonic3/consolidation-summary.yaml)
pins external reports and logs. Checks on the merged macOS arm64 tree passed:

- `python3 tools/dev.py check`: Rust, C, Python, Go, Dart and generated C header.
- All-feature workspace tests: 93 core, seven CLI, eight FFI and two playback
  tests passed; explicit external-asset/device tests remained ignored.
- Strict all-feature Clippy, Rust 1.88.0 compilation, and Windows x64 compilation
  of all targets/features passed.
- 57 portable model-tool tests, ten Flutter binding tests and app analysis passed.
- Real FP16 Core ML replay passed 768 exact tensor comparisons in each allocation
  mode, with five rejected inputs and four seeded-path comparisons per mode.
- Muted Flutter integration passed synthesis, reader controls, export and
  cancellation, with zero native handles before and after the test. Flutter
  reported a foregrounding warning; the test completed and wrote its passing report.

QNN execution was not rerun on a Windows ARM64 device. The Windows x64 check
does not compile the architecture-specific QNN bridge. Existing QNN device
results remain evidence for their recorded revisions, not a device qualification
of this merge. No new listening or performance claim follows from these checks.
