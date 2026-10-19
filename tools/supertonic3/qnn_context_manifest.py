"""Lock a freshly assembled T384 Superfast preset into a new MLSR manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

import yaml

from .bundle import ReferenceError, load_yaml
from .qnn_bundle import MANIFEST, verify_file


def identity(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": digest, "size_bytes": path.stat().st_size}


def build(source: Path, compiled: Path, revision: str) -> dict:
    if not re.fullmatch("[0-9a-f]{40}", revision):
        raise ReferenceError("Conversion revision must be a full commit ID")
    source, compiled = source.resolve(), compiled.resolve()
    relative = compiled.relative_to(source).as_posix()
    preset = json.loads((compiled / "bundle.json").read_text())
    if (
        preset["preset"] != "superfast-t384"
        or preset["noise_length"] != 512
        or preset["frontend_text_capacity"] != 384
        or preset["flow_total_steps"] != 8
        or [b["latent_length"] for b in preset["buckets"]] != [128, 256, 512]
    ):
        raise ReferenceError("Expected the assembled T384 Superfast preset")
    provenance = json.loads((compiled / "provenance.json").read_text())
    manifest = copy.deepcopy(load_yaml(MANIFEST))
    manifest["id"] = "supertonic3-qnn-superfast-t384-v1"
    manifest["provenance"]["conversion"] = {
        "project": "supertonic-npu",
        "revision": revision,
        "assembly_command": "python scripts/build_superfast_context.py --output <fresh-directory>",
        "preset_provenance_sha256": identity(compiled / "provenance.json")["sha256"],
        "rebuild_note": (
            "Vendored Superfast transformations; graph metadata records original ONNX, "
            "export source and program hashes. Rebuilding is explicit; changed bytes "
            "require a new manifest and qualification."
        ),
    }
    contract = manifest["contract"]
    contract.update(
        text_tokens=384,
        latent_buckets=[128, 256, 512],
        vocoder_latent_range=[4, 512],
        noise_capacity=512,
        latent_length_policy=(
            "ceil(requested_frames / 3072); reject outside 4..512; select the smallest "
            "fitting bucket; pass the full masked state to its fixed vocoder and trim PCM"
        ),
        noise_policy=(
            "ChaCha8 as in MLSR, generate [1,144,512], crop each channel to true length "
            "and zero-pad to the selected bucket; replay accepts saved [1,144,512] Float32 noise"
        ),
    )
    original_artifacts = manifest["artifacts"]
    original_sources = manifest["provenance"]["local_sources"]
    artifacts, sources = {}, {}
    components = [
        ("duration_predictor", "duration_predictor", "frontend", 128),
        ("text_encoder", "text_encoder", "frontend", 128),
    ]
    components += [
        (artifact, component, f"l{length}", length)
        for length in (128, 256, 512)
        for artifact, component in (
            (f"ve_{length}", "vector_estimator"),
            (f"vocoder_{length}", "vocoder"),
        )
    ]
    for name, component, folder, length in components:
        metadata_path = compiled / folder / f"{component}.json"
        program_path = metadata_path.with_suffix(".pte")
        record = json.loads(metadata_path.read_text())
        pinned = provenance["components"][f"{folder}/{component}"]
        verify_file(
            program_path,
            {"sha256": pinned["pte_sha256"], "size_bytes": pinned["pte_bytes"]},
        )
        if identity(metadata_path)["sha256"] != pinned["metadata_sha256"]:
            raise ReferenceError(f"Changed export metadata: {metadata_path}")
        for key, expected in {
            "component": component,
            "precision": "fp16",
            "gelu": "exact",
            "text_length": 384,
            "latent_length": length,
            "delegates": 1,
            "cpu_compute": [],
            "qnn_sdk": "2.50.40.260831",
            "target_soc": 88,
            "htp_arch": 81,
        }.items():
            if record.get(key) != expected:
                raise ReferenceError(f"Unexpected {component}.{key}")
        path = f"graphs/{name}.pte"
        artifact = {
            "kind": "executorch_qnn",
            "path": path,
            "license": "OpenRAIL-M",
            "precision": copy.deepcopy(
                original_artifacts["duration_predictor"]["precision"]
            ),
            "inputs": {
                value["name"]: {
                    "ordinal": i,
                    "dtype": value["dtype"],
                    "shape": value["shape"],
                    "layout": "contiguous row-major",
                }
                for i, value in enumerate(record["inputs"])
            },
            "outputs": {
                "output": {
                    "ordinal": 0,
                    **record["output"],
                    "layout": "contiguous row-major",
                }
            },
            "files": [{"path": path, **identity(program_path)}],
        }
        if component in ("vector_estimator", "vocoder"):
            artifact["latent_bucket"] = length
        artifacts[name] = artifact
        metadata_name, metadata_target = name + "_metadata", f"graphs/{name}.json"
        artifacts[metadata_name] = {
            "kind": "conversion_metadata",
            "path": metadata_target,
            "license": "OpenRAIL-M",
            "files": [{"path": metadata_target, **identity(metadata_path)}],
        }
        sources[name] = f"{relative}/{folder}/{component}.pte"
        sources[metadata_name] = f"{relative}/{folder}/{component}.json"
    for name in (
        "tts_config",
        "unicode_indexer",
        "voice_M1",
        "voice_F1",
        "model_license",
        "executorch_license",
    ):
        artifacts[name] = original_artifacts[name]
        sources[name] = original_sources[name]
        verify_file(source / sources[name], artifacts[name]["files"][0])
    manifest["artifacts"] = artifacts
    manifest["provenance"]["local_sources"] = sources
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--conversion-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.source, args.compiled, args.conversion_revision)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(
            "# Experimental T384 Superfast; payloads and conversion metadata are immutable.\n"
        )
        yaml.safe_dump(manifest, stream, sort_keys=False, allow_unicode=True)
    print(f"{args.output}: {identity(args.output)['sha256']}")


if __name__ == "__main__":
    main()
