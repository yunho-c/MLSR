"""Prepare a new verified bundle with the pinned optional execution graphs.

No downloads, in-place bundle edits, or model conversion. File symlinks avoid
duplicating model weights; source contents must remain immutable while in use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .bundle import (
    DEFAULT_MANIFEST,
    ROOT,
    ReferenceError,
    digest,
    load_manifest,
    load_yaml,
    relative_path,
    verify_bundle,
    write_yaml,
)

VARIANT_MANIFEST = ROOT / "model-manifests/supertonic3-coreml-variants.yaml"
SOURCES = {
    "ve_fused8_128": "noboundary/compiled/VectorEstimator_v2_L128_int4_S8.mlmodelc",
    "vocoder_30": "fixed-vocoders/Vocoder_L30.mlmodelc",
    "vocoder_32": "fixed-vocoders/Vocoder_L32.mlmodelc",
}


def variant_manifest():
    manifest = load_yaml(VARIANT_MANIFEST)
    if manifest["schema_version"] != 1 or set(manifest["artifacts"]) != set(SOURCES):
        raise ReferenceError("Unsupported execution variant manifest")
    if digest(DEFAULT_MANIFEST) != manifest["base_manifest_sha256"]:
        raise ReferenceError("Variant description pins a different baseline")
    for artifact in manifest["artifacts"].values():
        root = relative_path(artifact["path"])
        for entry in artifact["files"]:
            name = relative_path(entry["path"])
            if not name.startswith(root + "/"):
                raise ReferenceError("Variant file outside its graph")
    return manifest


def verify_variants(bundle):
    manifest = variant_manifest()
    for artifact in manifest["artifacts"].values():
        expected = {entry["path"] for entry in artifact["files"]}
        actual = set()
        for file in (bundle / artifact["path"]).rglob("*"):
            if file.is_symlink() and file.is_dir():
                raise ReferenceError("Nested directory symlinks are not supported")
            if file.is_file() and not (
                file.name == ".DS_Store" and not file.is_symlink()
            ):
                actual.add(file.relative_to(bundle).as_posix())
        if actual != expected:
            raise ReferenceError(f"Variant inventory differs: {artifact['path']}")
        for entry in artifact["files"]:
            path = bundle / entry["path"]
            if (
                path.stat().st_size != entry["size_bytes"]
                or digest(path) != entry["sha256"]
            ):
                raise ReferenceError(f"Variant bytes differ: {path}")
    return manifest


def prepare(baseline, experiment, output):
    baseline = baseline.resolve(strict=True)
    experiment = experiment.resolve(strict=True)
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise ReferenceError("Output bundle already exists")
    manifest = load_manifest()
    verify_bundle(manifest, baseline)
    variants = variant_manifest()
    files = [
        (baseline / f["path"], f)
        for a in manifest["artifacts"].values()
        for f in a["files"]
    ]
    for name, artifact in variants["artifacts"].items():
        for entry in artifact["files"]:
            relative = Path(entry["path"]).relative_to(artifact["path"])
            files.append((experiment / SOURCES[name] / relative, entry))
    # Verify every source before creating the destination. No archive report is
    # trusted as a substitute for reading the actual payloads.
    for source, entry in files:
        if (
            source.stat().st_size != entry["size_bytes"]
            or digest(source) != entry["sha256"]
        ):
            raise ReferenceError(f"Source does not match pinned bytes: {source}")
    output.mkdir(parents=True)
    for source, entry in files:
        target = output / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source.resolve(strict=True))
    write_yaml(output / "execution-variants.yaml", variants)
    verify_bundle(manifest, output)
    verify_variants(output)
    print(f"Verified baseline plus three execution graphs: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-bundle", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.baseline_bundle, args.experiment_dir, args.output)


if __name__ == "__main__":
    main()
