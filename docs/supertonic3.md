# Supertonic 3 synthesis

The first synthesis target is macOS arm64 with the pinned Core ML v2 export,
English/Korean qualification, voices M1/F1, and 44,100 Hz mono Float32 PCM.

## Verified local bundles

`mlsr::models::supertonic3::Bundle::load(manifest_path, bundle_directory)` parses
the YAML manifest, checks the supported model/tensor/audio contracts, verifies
the sizes and SHA-256 of every payload, and loads owned voice/indexer data.
It does not load Core ML graphs or use the network. Portable tests do not need
Apple frameworks or downloaded weights.

Manifests are limited to 1 MiB; auxiliary assets to 4 MiB each. YAML duplicate
keys, unsupported schemas/contracts, ambiguous paths, missing or changed files,
extra compiled-model files, and malformed voice/tokenizer data produce errors.
The YAML parser is pinned [serde_yaml_ng](https://github.com/acatton/serde-yaml-ng),
which preserves the existing Rust 1.88 minimum. Upstream JSON assets keep their
original format and checksums; Float32 styles are parsed with accurate decimal
conversion and validated for shape and finite values.

File symlinks from the preparation tool are supported and verified by target
contents. Nested directory symlinks inside graphs are rejected to avoid cycles;
regular `.DS_Store` metadata is ignored. Bundle contents and symlink targets must
remain immutable while the bundle or an associated engine is in use. Hashes
check consistency with a trusted manifest, not its authenticity or protection
against concurrent replacement of model files by another process.

To verify the real bundle without inference:

```sh
MLSR_TEST_BUNDLE=/path/to/model-bundle cargo test --locked -p mlsr \
  loads_pinned_external_bundle -- --ignored
```

## Text frontend

`Bundle::encode_text(text, language)` returns the original text, normalized text,
token count, Int32 IDs, and Float32 mask. Normalization follows the pinned
upstream ordering, including language tags, symbol replacements, and whitespace
handling. `unicode-normalization` is pinned to Unicode 15.0 to match the capture;
Python's whitespace rules are explicit and do not change with Rust's Unicode
tables. The adapted upstream code retains its MIT notice in the Rust crate.

The first slice accepts at most 64 KiB of raw UTF-8 and 128 normalized Unicode
scalars including tags. It rejects empty content, unknown languages, unsupported
scalars, and overlong text; it never truncates or wraps code points to 16 bits.
Model language tags are accepted independently of qualification: the current
speech corpus qualifies only English and Korean. Normalized offsets count
Unicode scalars and must not be treated as UTF-8 bytes or original-text offsets.

Portable tests compare all 19 positive/negative corpus entries against the
independent upstream frontend fixture. To repeat with the actual tokenizer:

```sh
MLSR_TEST_BUNDLE=/path/to/model-bundle cargo test --locked -p mlsr \
  matches_frontend_with_real_tokenizer -- --ignored
```

Planning and synthesis are the next implementation steps. The public C ABI and
language bindings currently expose version queries.
