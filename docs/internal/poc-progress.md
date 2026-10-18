# macOS Apple Silicon POC

This tracks the seven-task implementation plan in `ignore/scratch3.md`. Later
project decisions use YAML manifests and cidre; the original JSON manifest and
Objective-C++ bridge suggestions are superseded. QNN remains the next backend
after this POC; its feasibility has already been confirmed by the project owner.

| Task | Current evidence / remaining gate |
| --- | --- |
| 1. Model bundle and reference | Implemented; `model-manifests/supertonic3-coreml.yaml`, `tools/supertonic3/`, pinned frontend/reference fixtures and summaries |
| 2. Persistent Core ML runtime | Implemented with cidre; saved-stage parity, native ownership/reuse, C static/shared callers and Apple cross-build records in `runtime-summary.yaml` |
| 3. Rust single-segment synthesis and CLI | Implemented; full tensor parity and standalone WAV receipts in `synthesis-summary.yaml`; five commits `7e8b00f` through `7c4ca74` |
| 4. Segmentation/jobs/cancellation/bounds | Implemented in Rust; see `jobs.md`, portable lifecycle tests and explicit real-model job test |
| 5. C ABI and Dart/Go/Python synthesis | Implemented; installed synthesis/cancellation/WAV examples outside checkout, cross-language PCM equality and lifecycle checks in `bindings-summary.yaml` |
| 6. Selectable ANE variants | Implemented; baseline/fused VE exact across 29 cases; fixed vocoders explicitly experimental due to PCM differences; lifecycle and per-case timing in `variants-summary.yaml`. Baseline remains the default. |
| 7. Flutter example and qualification | Implemented; full Apple suite, responsive Flutter speak/stop/playback/export, release app build/signature checks, timing/memory/artifact receipts, placement plans and runtime traces in `poc-summary.yaml`. All 84 warm baseline requests were faster than playback on M4 Pro. See `poc-qualification.md` for scope and remaining platform/release gates. |

Existing qualification summaries are historical records tied to their recorded
source hashes. They do not automatically qualify later changes. Each new task
adds relevant validation. The [final POC qualification](poc-qualification.md)
exercises the combined implementation and reports its actual host, artifacts
and measured boundaries. QNN integration is the next implementation milestone.
