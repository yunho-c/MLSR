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

## Verified model ownership and store

`ModelHandle::open(descriptor, directory)` verifies immutable app resources in
place. `Engine::from_model(model.clone(), options)` reuses that verification and
retains the model until the native session has shut down. Explicit-path loaders
remain available. Direct resources must stay immutable; they are never deleted
by the store. Optional execution variants still verify their additional graphs.

The host supplies an application-support directory to `ModelStore::new(root)`.
Rust does not select platform directories. The layout is:

```text
<app-support>/mlsr/
  models/<id>/<backend>/<manifest-sha256>/mlsr-bundle.yaml
  models/<id>/<backend>/<manifest-sha256>/<payload files>
  staging/install-<random>/...
  locks/<identity-hash>.install
  locks/<identity-hash>.lease
```

`begin(descriptor)` takes an exclusive per-version writer lock and creates a
private staging directory. The host streams/copies the exact manifest inventory
into it and closes all writers before `commit()`. Rust rejects missing/extra
entries, symlinks, changed manifests, hashes, and incompatible metadata, then
publishes the complete directory with a same-filesystem rename. It rebases the
verified graph paths without hashing the payload a second time. Existing
versions are never overwritten. Commit failure or transaction drop cleans only
that transaction's unpublished staging directory.

`open(descriptor)` acquires a shared lease before verifying a published version.
`remove(descriptor)` requires both an exclusive writer lock and an exclusive
lease. Models, their clones, and engines protect the version until their last
owner releases it. Pipeline destruction explicitly joins the native session
before releasing the lease. Contention returns `model_store_busy` or
`model_in_use`; absent versions return `model_not_installed`. No operation waits
indefinitely for a filesystem lock. Lock files remain in place to keep their
inodes stable across processes.

These are advisory locks for cooperating processes in an application-private
store, not protection against arbitrary filesystem writers. Process crashes
release locks and can leave unpublished staging directories; readers never
discover them. Automatic garbage collection and power-loss durability guarantees
are outside this slice. Removal is explicit and scoped to one exact version.

C ABI feature bit `4` advertises the additive model/store functions; ABI version
1 and existing structs/functions remain compatible. `MlsrModelSpec` supplies a
trusted catalog, exact ID, and platform. Model and installation handles have
explicit release functions and cannot be confused with engine/job handles.
Commit consumes a transaction on success or failure; release its installation
handle afterward. The returned model handle has independent ownership.

Dart exposes `ModelDescriptor`, `ModelHandle`, `ModelStore`,
`ModelInstallation`, and `Engine.fromModel`. Filesystem verification, publishing,
and cleanup run on adapter isolates. Always await disposal; creating an engine
and disposing its source model concurrently is supported. The engine retains its
own lease. Platform asset lookup and streaming belong in `mlsr_flutter`.

Portable tests use small synthetic graph files, including cross-process locks,
installer exit, corruption, abort, symlinks, ABI misuse, and Dart cleanup. The
Dart fixtures retain small manifest templates, but generate tokenizer/voice data
at test time and include no model weights. Opt into the real Core ML lease test
with `MLSR_TEST_DELIVERY_BUNDLE` and a Core ML-enabled native library. It copies
a self-contained baseline into a temporary store, creates an engine concurrently
with model disposal, synthesizes, checks removal is blocked, then closes and
removes the version.
