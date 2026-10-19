# Model manifests

The recommended release is [supertonic3-coreml-fp16.yaml](supertonic3-coreml-fp16.yaml):
boundary-corrected FP16 Core ML, T=384 text, L=128/256/512 estimators and M1/F1.
It pins locally compiled payloads and their export provenance. The source URL
identifies the original model inputs; these compiled files are **not published
there**. Preparation requires verified local assets. See the
[promotion record](../docs/internal/supertonic3-fp16-promotion.md).
ONNX is used only by optional conversion/reference tools, not by the library.

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

The manifest records:

- Source repository and full immutable revision. For published bundles, each
  file URL is `<repository>/resolve/<revision>/<file.path>`. A manifest marked
  `provenance.local_export` must use verified local files instead.
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

For the legacy published INT4 bundle, the publisher does not record an exact exporter source revision. Its manifest
explicitly distinguishes the pinned Möbius reference-tooling snapshot from an
export attestation. Published artifact bytes are pinned and locally verified;
bit-identical reconstruction of the exports from source is not established.

See [asset preparation and reference capture](../tools/supertonic3/README.md).
