"""Pin and assemble the local T=384 FP16 release from verified Core ML exports.

This writes a new manifest and a symlinked development bundle. Flutter packaging
copies its verified payloads into application assets. No model is published or
downloaded by this command.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from .bundle import DEFAULT_MANIFEST, ReferenceError, digest, load_yaml, write_yaml


def package(args) -> None:
    if args.manifest.exists() or args.manifest.is_symlink():
        raise ReferenceError("Manifest output already exists")
    legacy = load_yaml(DEFAULT_MANIFEST)
    manifest = copy.deepcopy(legacy)
    manifest["id"] = "supertonic3-coreml-fp16-t384-v1"
    manifest["source"] = {
        "repository": "https://huggingface.co/spaces/Supertone/supertonic-3",
        "revision": "4402acca094e2ff2b3dff446e0bf915f3fccc462",
    }
    manifest["provenance"] = {
        "local_export": True,
        "distribution_note": "Locally built Core ML payloads; source identifies the original model inputs, not downloadable compiled artifacts.",
        "upstream_model": legacy["provenance"]["upstream_model"],
        "upstream_preprocessing": legacy["provenance"]["upstream_preprocessing"],
        "conversion_tooling_reference": legacy["provenance"][
            "conversion_tooling_reference"
        ],
        "conditioning_manifest_sha256": digest(DEFAULT_MANIFEST),
        "export_corrections": [
            "LayerNorm epsilon read from official ONNX graphs",
            "Replicate the true sequence endpoint before masked ConvNeXt convolutions",
        ],
        "exports": {},
    }
    manifest["contract"]["text_tokens"] = 384
    manifest["runtime"][
        "placement_note"
    ] = "CPU+Neural Engine policy. VE/vocoder placement was inspected; no utilization or physical residency claim."
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for label, directory in [
        ("main", args.models),
        ("short_latents", args.short_models),
    ]:
        record = load_yaml(directory / "manifest.yaml")
        if (
            record["status"] != "complete"
            or record["text_capacity"] != 384
            or record["precision"] != "float16"
            or not record["preserve_boundaries"]
        ):
            raise ReferenceError(
                "Expected completed boundary-corrected T=384 FP16 export"
            )
        manifest["provenance"]["exports"][label] = {
            "manifest_sha256": digest(directory / "manifest.yaml"),
            "tool_sources_sha256": record["tool_sources_sha256"],
            "onnx_sha256": record["onnx_sha256"],
            "coremltools": record["coremltools"],
            "torch": record["torch"],
        }
        for key, artifact in record["artifacts"].items():
            if key in sources:
                raise ReferenceError(f"Duplicate export: {key}")
            sources[key] = (
                directory / artifact["compiled"],
                artifact["compiled_sha256"],
            )
    for name, artifact in manifest["artifacts"].items():
        if artifact["kind"] == "coreml_compiled":
            key = (
                name.replace("ve_", "vector_estimator_L")
                if name.startswith("ve_")
                else name
            )
            directory, hashes = sources[key]
            actual = {
                str(p.relative_to(directory)): digest(p)
                for p in directory.rglob("*")
                if p.is_file()
            }
            if actual != hashes:
                raise ReferenceError(f"Modified compiled model: {directory}")
            artifact["path"] = f"{key}_T384_fp16.mlmodelc"
            artifact["precision"] = {
                "weights": "float16",
                "activations": "float16",
                "interface": (
                    "int32/float32"
                    if name in ("duration_predictor", "text_encoder")
                    else "float32"
                ),
            }
            for tensors in [artifact["inputs"], artifact["outputs"]]:
                for input_name, tensor in tensors.items():
                    if input_name in ("text_ids", "text_mask", "text_emb"):
                        tensor["shape"][-1] = 384
            artifact["files"] = []
            for relative, sha in sorted(hashes.items()):
                source = directory / relative
                path = f"{artifact['path']}/{relative}"
                artifact["files"].append(
                    {"path": path, "sha256": sha, "size_bytes": source.stat().st_size}
                )
                destination = args.output / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(source.resolve())
        else:
            for entry in artifact["files"]:
                source = args.legacy_bundle / entry["path"]
                if (
                    source.stat().st_size != entry["size_bytes"]
                    or digest(source) != entry["sha256"]
                ):
                    raise ReferenceError(f"Modified conditioning asset: {source}")
                destination = args.output / entry["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(source.resolve())
    write_yaml(args.manifest, manifest)
    (args.output / "mlsr-bundle.yaml").write_bytes(args.manifest.read_bytes())
    print(f"{manifest['id']}: {digest(args.manifest)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "short-models", "legacy-bundle", "output", "manifest"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    package(parser.parse_args())


if __name__ == "__main__":
    main()
