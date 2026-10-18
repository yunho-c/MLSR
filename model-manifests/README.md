# Model manifests

`supertonic3-coreml-variants.yaml` pins optional archived fused-eight-step VE
and exact-length vocoder graphs. It is separate from the baseline/reference
identity. The Rust library embeds these pins; see
[execution variants](../docs/internal/execution-variants.md) for preparation and qualification.
Fixed vocoders are explicitly experimental because their PCM differs.

Use YAML for model manifests. [supertonic3-coreml.yaml](supertonic3-coreml.yaml)
pins the first development reference: Supertonic 3 v2 Core ML shared stages,
INT4 VectorEstimators at latent buckets 128/256/512, and M1/F1 preset voices.
There are 28 payload files, totaling 169,328,070 bytes. Weights are external;
ordinary builds never download them.

The experimental [QNN manifest](supertonic3-qnn-superfast.yaml) is pinned by its
original serialized bytes, including CRLF line endings. Its `.gitattributes`
exception preserves those bytes in Git and on every checkout; keep the existing
SHA-256 and bundle identity intact when editing unrelated files. Manifest identity
verification does not normalize YAML or line endings.

The manifest records:

- Publisher repository and full immutable revision. Each file URL is
  `<repository>/resolve/<revision>/<file.path>`.
- Exact SHA-256 and byte count of every required file, including compiled model
  descriptors and weights. Finder `.DS_Store` files are excluded.
- Input/output names, dtypes, shapes, internal precision, and audio/text limits.
- Model deployment target and the separately limited first-slice test target.
- Upstream model, preprocessing, conversion-tooling, and integration provenance.
- Model/voice licensing separately from the Apache-2.0 library and tools.

`schema_version: 1` describes this manifest format; it is independent of the
public C ABI version. `python3 -m tools.supertonic3 validate-manifest` performs
strict validation for this particular Supertonic export, including duplicate
YAML keys, unsafe paths, missing assets, unpinned revisions, and tensor contracts.
The tool is not a general model registry implementation.

The publisher does not record an exact exporter source revision. The manifest
explicitly distinguishes the pinned Möbius reference-tooling snapshot from an
export attestation. Published artifact bytes are pinned and locally verified;
bit-identical reconstruction of the exports from source is not established.

See [asset preparation and reference capture](../tools/supertonic3/README.md).
