# Bundled model delivery

This implements the accepted six-change plan. The first supported delivery target
is the qualified macOS arm64 Core ML baseline. Flutter packages model data; the
native build hook packages code; Rust verifies and owns model lifetimes.

| Change | Status and completion gate |
| --- | --- |
| 1. Bundle discovery and catalog | Implemented: directory discovery, exact-byte YAML identity, pinned catalog; 25 portable Rust tests, 46 model-tool tests, real prepared-bundle discovery, strict Clippy and Rust 1.88 check passed |
| 2. Rust model store and C/Dart ownership | Implemented: verified atomic publication, cross-process locks and model/engine leases; portable Rust/C/Dart tests, real Core ML engine lifetime/synthesis test, strict Clippy, Rust 1.88 and Windows cross-check passed |
| 3. Generated Flutter asset package | Implemented: offline CLI preparation, exact inventories, regular files, notices, platform filters and Dart constants; deterministic/negative fixture tests, real 29-file package hash audit, Flutter Pub/analyze, strict Clippy and Rust 1.88 check passed |
| 4. Flutter model resolution | Pending: native resource lookup, direct loading and streamed installation, open by model ID |
| 5. Example migration | Pending: bundled baseline, no required paths, existing playback/cancel/export behavior |
| 6. Downstream release qualification | Pending: separate consumer, relocated offline release app, fresh app data, no developer cache/checkout dependency |

Completion requires the built release app to synthesize after relocation with
fresh application data, offline, without manual paths or access to the developer
model cache. Qualification must inspect the final artifact, not only source tests.
Keep raw reports outside Git and commit small evidence summaries. Runtime downloads,
additional backends, and mobile runtime qualification follow this bundled slice.
