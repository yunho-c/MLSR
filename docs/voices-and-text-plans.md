# Voice discovery and frozen text plans

Voice IDs use model identity (`supertonic3:M1`, `supertonic3:F1`), independent of
the execution backend and of a bundle revision. The low-level engine accepts raw
model IDs unchanged. The recommended `supertonic3-coreml-fp16-t384-voices-v1`
bundle provides all ten presets from the official demo:

| Model ID | Display name | Model ID | Display name |
| --- | --- | --- | --- |
| M1 | Alex | F1 | Sarah |
| M2 | James | F2 | Lily |
| M3 | Robert | F3 | Jessica |
| M4 | Sam | F4 | Olivia |
| M5 | Daniel | F5 | Emily |

The catalog follows the ordered `contract.voices` declarations. Optional
`voice_names` must name exactly those voices; legacy manifests retain their old
labels and two-voice inventory. Every declared voice must have one locked
`voice_<id>` artifact with Float32 `[1,50,256]` text/latent conditioning and
`[1,8,16]` duration conditioning. Extra/missing artifacts, duplicate/unsafe IDs,
invalid display names, incompatible tensors and nonfinite values are rejected.
All other graph/runtime contracts remain strict. QNN's pinned artifact set is
unchanged.

A catalog reports exact bundle/revision/manifest identity, 31 frontend-supported
languages and capabilities. Only M1/F1 in the exact accepted Core ML releases
advertise English/Korean as qualified languages. The expanded release preserves
their original graph and voice bytes; the eight added presets have empty
`qualified_languages`. All 310 preset/language combinations have short real
synthesis smoke checks, which do not establish intelligibility, speaker similarity
or listening quality. See [voice coverage evidence](internal/voice-coverage.md).
Language-wide tags do not imply a particular accent. QNN entries report
`backend: qnn` with empty qualified languages; its quality gate remains open.

Flutter reads the loaded `Tts.getVoices()` catalog, including display names and
languages. Unloaded `getVoices()` describes the recommended release only; it
does not promise those assets are installed. The example provides all 31 language
choices, local sample texts, Arabic direction, and generation controls. Choosing
a language preserves existing text; **Try an example** explicitly replaces it.
Its 0.8–1.3× speech-speed slider matches the official demo's `raw_speed = speed +
0.05`; the SDK's normalized `speakingRate` is therefore `(speed + 0.05) / 1.05`.
Quality selects 2–16 generation steps. Both affect the next request; retained
audio's playback-rate control remains independent.

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
Supertonic3's [text policy](jobs.md#text-policy) keeps complete sentences in the
recommended T=384 window and uses clause-aware fallbacks, while T=128 bundles
retain conservative character targets. Every model's token limit is enforced.
Chunk boundaries may change with library versions; sentence navigation
and seeded suffixes remain consistent within each frozen plan.

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
