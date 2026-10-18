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
| 6. Selectable ANE variants | Pending: integrate available fusion/fixed-vocoder exports, shape gates, parity, lifecycle and measured performance |
| 7. Flutter example and qualification | Pending: responsive speak/stop/playback/WAV UI, real Apple suite, timing/memory/artifact records, real-time warm synthesis |

Existing qualification summaries are historical records tied to their recorded
source hashes. They do not automatically qualify later changes. Each new task
adds relevant validation; the final POC qualification must exercise the combined
implementation and report its actual host, artifacts and measured boundaries.
