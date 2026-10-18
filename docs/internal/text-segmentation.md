# Supertonic3 text segmentation

The current policy is documented in [synthesis jobs](../jobs.md#text-policy).
This record explains the quality change and the limits of its validation.

## Motivation

A user-provided 737-character English/M1 passage produced missing words and
discontinuous prosody. Its exported 2,163,283 Float32 samples were identical
across the pre-SDK revision `82efec6`, current SDK revision `09226ff`, and a
control with Core ML output-buffer reuse disabled. The reported recording's
defects therefore predated the developer API migration.

The old segmenter filled the 128-normalized-token window and cut at the last
space. One join ended with `I want to.` and continued with `know more about
Adler.` The words were present in the model input, but split across independent
invocations. This identified a poor boundary, not proof of every audible cause.

[FluidAudio issue #669](https://github.com/FluidInference/FluidAudio/issues/669)
also reports degradation inside long chunks that fit the text window. Its
70-character Latin target and 57-character Korean/Japanese target informed this
change. Its reported WER improvement is upstream evidence on a different corpus
and host, not an MLSR quality measurement.

## Implementation

- Keep original sentences separate for reader navigation. Synthesis and text
  plans share Unicode sentence boundaries, with common English titles joined
  to the following name.
- Cap trimmed source graphemes at 70, or 57 for Korean/Japanese, while still
  checking the complete normalized input against the hard 128-token limit.
- Prefer comma/semicolon/colon boundaries and their CJK equivalents when the
  prefix has at least one-third of the character target. Preserve punctuation,
  closing quotes and source spans. Do not split numeric separators as clauses.
- Fall back to whitespace, rebalancing a short final tail when both resulting
  chunks fit. Use grapheme boundaries only when a word itself cannot fit.
- Keep the existing strict tokenizer, global normalization, model bytes,
  duration/noise calculations, audio-bucket selection and playback graph.
  Forced word breaks can still receive the existing added terminal period;
  this heuristic does not promise grammatical phrases for unpunctuated text.

Unlike FluidAudio's chunker, this implementation does not pack separate
sentences together, reconstruct source strings, discard delimiting commas, or
truncate normalized input. New chunk identities change per-segment noise seeds,
so previous whole-paragraph PCM hashes are historical evidence. Frozen plans,
their sentence groupings, and suffix seed offsets remain internally consistent.

## Validation on 2026-10-18

Base revision: `09226ff`; modified source hashes, native-library hashes and full
logs are recorded in
`~/.cache/mlsr/comparisons/quality-20261018/clause-fix-validation/summary.json`.

- Workspace tests: 73 core, seven CLI and eight FFI tests passed. Two portable
  external-asset tests remained ignored. Added regression coverage checks the
  Adler clause, punctuation, numeric separators, tail balancing, quality/token
  limits, combining characters, Korean expansion, titles, spans and seed IDs.
- Workspace Clippy, native-feature Clippy and Rust 1.88 core checks passed.
- The explicitly enabled real Core ML frozen-suffix test now includes the long
  Adler sentence split into three model segments. Its PCM matches separate
  invocations at the original segment seeds exactly.
- The actual native library bundled in the rebuilt Flutter release reproduces
  the new headless render exactly. Muted native playback at 2x completes and
  exports the same retained PCM. Paused replay navigation to sentences 2, 3 and
  0 selects the expected segment IDs, source frames and text ranges.
- UTF-8/UTF-16 spans reconstruct the complete original passage. A longer
  Korean/F1 request also renders successfully within the hard token limit.
  Native handles return from zero to zero and retained PCM is released.
- The macOS release build and deep/strict code-signature verification passed.
  Xcode emitted existing simulator-version and Metal-toolchain search-path
  warnings; these did not fail the macOS build.

The new English rendering has 14 segments in six sentences and 2,244,287 frames
(50.8909 seconds). PCM SHA-256:
`fffd90690bafaff156d604819550a2081df840fc051d656234340d38addf411b`.
The Adler phrase is one complete segment, and its preceding clauses retain
their commas. The comparison WAV is `~/Downloads/MLSR-clause-aware-v1.wav`.

The user's initial listening pass confirmed that the Adler phrase and
transitions are better. A subsequent pass identified remaining long pauses at
`mind | boggled`, `more, | I want to understand`, `or | a workbook`,
`living a | happy life`, `not | seeing`, `rather | see them`, and
`live | freely and happy`, plus an apparently missing `a` before `workbook`.
Those findings remain unresolved at this revision; the initial improvement
does not qualify the whole passage as natural or intelligible throughout.
Broader quality still needs corpus-level listening. Muted lifecycle, mapping
and PCM checks are separate from that assessment.
The reported pronunciation of `disliked` remains a deferred investigation;
no precision/runtime cause or general quality improvement is claimed here.
