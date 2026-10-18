# Voice discovery and frozen text plans

Voice IDs use model identity (`supertonic3:M1`, `supertonic3:F1`), independent of
the Core ML backend and of a bundle revision. The old engine accepts raw M1/F1
identifiers unchanged. A catalog reports exact bundle/revision/manifest identity,
31 frontend-supported languages and capabilities. Only the pinned, qualified
baseline advertises English/Korean as qualified languages. Language-wide tags do
not imply a particular accent.

`BundleManifest::voice_catalog()` requires no assets or runtime and reports
`CatalogOnly`; `Bundle::voice_catalog()` reports `AssetsVerified`;
`Engine::voice_catalog()` reports `EngineReady` while the engine is open. These
are snapshots, not promises that assets or an engine remain available forever.
Locale selection is explicit: strict `en-US` fails when only `en` is declared;
`VoiceFallback::SameLanguage` can resolve it to `en`. Unknown voice IDs and
unsupported locales have distinct error categories. Discovery never downloads.

`Bundle::plan_text()` and `Engine::plan_text()` freeze token-validated text into
`TextPlan`. The engine method executes on its worker and uses its existing single
operation lane; the `Tts` scheduler serializes synthesis requests for callers.
Plans keep the complete original string, model-normalized segment input, spoken
sentence identity, and paired original UTF-8/UTF-16 spans. One long sentence can contain
multiple model segments. Removed-only text is not assigned fabricated audio.

`Engine::start_plan(plan, start_segment, ...)` accepts a frozen plan or suffix
without re-segmenting or renumbering it. The plan's manifest must match the
engine. Audio starts at frame zero for this job, while source spans, segment IDs
and `seed.wrapping_add(original_segment_id)` retain their original identities.
The shared [reader controls](reader-api.md) reuse these frozen plans and original
indices for sentence navigation.

`Utf16Map` rejects interior UTF-8 bytes, interior surrogate pairs, reversed spans
and out-of-range offsets. Source slicing never uses normalized-text offsets.
The existing 64 KiB raw input and bounded, grapheme-safe segmentation limits apply.

Validation for this change:

- `cargo test --locked -p mlsr --lib`: 35 passed, two external-asset tests skipped.
- `cargo clippy --locked --workspace --all-targets -- -D warnings`.
- `cargo +1.88.0 check --locked --workspace --all-targets` and Windows target
  compilation of the core and its tests/examples.
- `real_frozen_suffix_preserves_seeded_pcm`, explicitly run with the verified
  baseline bundle and Core ML on the local macOS host: the suffix PCM matches
  separately synthesized original segments bit for bit at their original seeds.

Portable tests cover readiness, explicit locale fallback, unavailable voices,
Korean/emoji/combining characters, UTF-16 slicing, long-sentence grouping, frozen
suffix identity, invalid cursors and foreign-model plan rejection. The public
[SDK translations](developer-sdk.md) and [installed-package qualification](internal/developer-api-qualification.md)
cover readiness, request inspection and original-text spans. Old layouts and
UTF-8 fields remain unchanged.
