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

## Duration planning and noise

`SynthesisOptions` validates 1..=64 steps and a finite positive speed; defaults
are eight steps, speed 1.05, and seed zero. `SynthesisPlan::from_duration` uses
Float32 for the speed clamp, duration floor, and frame multiplication, then
truncates frames and rounds latent length upward. Only true lengths 4..=512
are accepted; the selected VE bucket is 128, 256, or 512. Nonfinite output,
overflow, and unsupported vocoder lengths fail explicitly.

`prepare_noise(seed)` starts a new ChaCha8 generator for each request and draws
row-major standard normal samples via pinned `rand_chacha` 0.9.0 and `rand_distr`
0.5.1 (Float64 sampling converted to Float32). The seed is reproducible within
this implementation, not a promise of matching NumPy or arbitrary future
dependency versions. This is synthesis noise, not a cryptographic API.
`prepare_saved_noise` accepts the actual Float32 noise tensor for independent
reference comparisons, validates its true shape and finite values, and preserves
every bit while padding each latent channel with zeros. True and padded masks
are generated alongside it.

The portable planner tests use `tests/fixtures/supertonic3/planning.yaml`, which
records raw prediction/speed/duration Float32 bits and decisions extracted from
the existing capture, including the 127/128/129 and 255/256/257 boundaries.

## Persistent model pipeline

`Pipeline::load(bundle, compute_policy, reuse_output_buffers)` loads the six
graphs once and uploads M1/F1 voice styles. The pipeline owns four style tensors,
two writable scalar tensors, and optionally six output buffers (two per bucket).
`synthesize` takes original text, language, voice, and validated options. It
computes duration, prepares noise/masks, encodes text, runs VE for the requested
steps, crops the final latent to its true length, vocodes, and trims PCM.
The export's VE result already includes integration; no additional Euler step
is applied in Rust.

The ordinary path reads back the duration scalar, final latent for cropping,
and waveform. TE and intermediate VE tensors stay native. Synthesis requires
`&mut self`, so two requests cannot alias the reusable buffers concurrently;
the pipeline can move between caller threads. Errors drop per-request tensors.
`shutdown` joins the worker, and owned output PCM remains valid after shutdown
or destruction. In-flight native calls are synchronous; cancellation belongs
to the subsequent job API.

`synthesize_with_diagnostics` runs the same logic with optional captured noise
and records host snapshots of every input/intermediate. This is a qualification
path, not a performance benchmark. `stats` reports successful loads/predictions
and native tensor handles; it is not a measurement of physical memory or ANE
residency.

```sh
MACOSX_DEPLOYMENT_TARGET=15.0 cargo build --locked --release \
  -p mlsr --example coreml_synthesis_replay --features coreml

python3 -m tools.supertonic3.synthesis_replay \
  --bundle /path/to/model-bundle --reference /path/to/reference-capture \
  --output /path/to/new/synthesis-ordinary.yaml

python3 -m tools.supertonic3.synthesis_replay \
  --bundle /path/to/model-bundle --reference /path/to/reference-capture \
  --output /path/to/new/synthesis-backings.yaml --output-backings
```

Each mode runs two passes over 14 cases, comparing all 24 saved tensors per
case byte-for-byte, checking normalization/frame/bucket decisions, rejecting
five invalid inputs before inference, and comparing normal versus diagnostic
PCM for four requests with Rust-generated noise. Boundary cases reuse recorded
speaking rates; they do not supply precomputed durations or lengths to Rust.
Reports include source/binary/reference hashes, resource counters, and backing
reuse observations. Interrupted or failed runs remain non-passing. Existing
outputs are not overwritten; model bundles and reference captures are verified
before and after. Rebuild the binary before qualifying a source revision.

The native opt-in test `repeated_requests_error_recovery_thread_move_and_shutdown`
checks recovery after bad inputs/noise, repeated requests, moving the pipeline
between threads, option effects, stable tensor counts, and output lifetime.
It requires the `coreml` feature and `MLSR_TEST_BUNDLE`, and runs with `--ignored`.

The top-level engine API and WAV CLI are the next implementation step. The
public C ABI and language bindings currently expose version queries.
