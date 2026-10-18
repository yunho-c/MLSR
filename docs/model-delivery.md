# Model bundle identity and delivery

Model data and executable libraries have separate packaging paths. Flutter assets
carry manifests, serialized graphs, weights, tokenizer data and voices. The native
build hook carries the Rust library. A Flutter adapter resolves model resources;
Rust validates their identity and manages their lifetime. Implementation status
is tracked in [model-delivery-progress.md](model-delivery-progress.md).

## Self-contained bundle

The existing payload layout is preserved. `mlsr-bundle.yaml` lives at the root and
all paths inside it are relative to that root. `Bundle::open(root)` discovers this
file. `Bundle::load(manifest, root)` remains available for explicit/custom paths.
Both APIs verify all listed payloads and the Supertonic tensor/audio contract.

`BundleManifest` parses the contract without loading model payloads. It retains
the original bytes, exposes the checked file inventory, and can verify a root.
Its `sha256()` is the hash of the **original YAML file bytes**. This differs from
the historical model tools' canonical/semantic manifest digest. Packaging must
copy the original file, including comments, instead of reserializing it.

The Python preparation CLI now preserves the supplied manifest's bytes. Existing
prepared directories are never overwritten; create a new directory to produce
an exact-byte manifest for catalog-based discovery. Existing explicit-path
loading remains compatible with previously prepared bundles.

## Catalog

`model-manifests/catalog.yaml` pins model IDs, backend names, qualified platform
keys, original manifest file hashes and package-relative asset directories. The
first entry is `supertonic3-coreml-int4-v1`, qualified for `macos-arm64` only.
Adding a platform to the schema's vocabulary does not qualify model execution.

`ModelCatalog::parse` rejects unknown fields, duplicate keys/IDs, overlapping or
unsafe asset roots, malformed hashes and unsupported platform names. A lookup
selects an exact ID and platform; it does not scan user directories or choose
"latest". The catalog itself must come from the application's trusted resources.

`ModelDescriptor::open(root)` checks the manifest hash and model ID before
reading payloads. This prevents a substituted manifest from authorizing different
weights merely by carrying matching checksums of its own. Model files and any
development symlink targets must remain immutable while an engine uses them.

The current raw baseline manifest SHA-256 is
`958b204a6e0ef2c527a9258b7fa3a27ce54ec5d0857ca69a6aa25da72f294dc5`.
The existing semantic digest is
`4a206fe65fbd38b8d03d9d9b395ddbe03590c1095c431bd0c6eb7e253978c50c`.
Optional execution variants already pin the former. Neither digest is a version
selection policy or a signature for an independently downloaded catalog.
