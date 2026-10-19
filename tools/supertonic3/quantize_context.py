"""Build an opt-in T=384 candidate with four-bit palettized estimator weights.

Uses verified FP16 mlpackages, never the old T=128 compressed models. Other
stages and conditioning assets remain byte-identical to the recommended bundle.
The candidate is external to the shipping catalog pending quality qualification.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import platform
import subprocess
import time

from .bundle import (
    ReferenceError,
    digest,
    load_manifest,
    load_yaml,
    verify_bundle,
    write_yaml,
)
from .context_coreml import tree_hashes
from .quality_probe import record_tool_sources


def quantize(args) -> None:
    import coremltools as ct
    import coremltools.optimize.coreml as cto

    base = load_manifest(args.manifest)
    if base["id"] != "supertonic3-coreml-fp16-t384-v1":
        raise ReferenceError("Expected the corrected FP16 release")
    verify_bundle(base, args.bundle)
    exports = {}
    for label, directory in [
        ("main", args.models),
        ("short_latents", args.short_models),
    ]:
        if (
            digest(directory / "manifest.yaml")
            != base["provenance"]["exports"][label]["manifest_sha256"]
        ):
            raise ReferenceError("Export differs from the FP16 release provenance")
        for name, artifact in load_yaml(directory / "manifest.yaml")[
            "artifacts"
        ].items():
            exports[name] = (directory, artifact)
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = args.output / "bundle"
    bundle.mkdir()
    manifest = copy.deepcopy(base)
    manifest["id"] = "supertonic3-coreml-int4-t384-candidate-v1"
    manifest["provenance"]["quantization"] = {
        "base_manifest_sha256": digest(args.manifest),
        "method": "Core ML Tools weight-only k-means palettization",
        "bits": 4,
        "stages": ["ve_128", "ve_256", "ve_512"],
        "qualification": "candidate; not promoted",
    }
    report = {
        "status": "incomplete",
        "kind": "int4_context_candidate",
        "coremltools": ct.__version__,
        "python": platform.python_version(),
        "base_manifest_sha256": digest(args.manifest),
        "artifacts": {},
        "tool_sources_sha256": record_tool_sources(args.output),
        "method": {"nbits": 4, "mode": "kmeans", "weight_threshold": 2048},
        "scope": "Estimator weights only; activations and other stages remain FP16",
    }
    write_yaml(args.output / "quantization.yaml", report)
    config = cto.OptimizationConfig(
        global_config=cto.OpPalettizerConfig(
            nbits=4,
            mode="kmeans",
            weight_threshold=2048,
        )
    )
    for name, artifact in manifest["artifacts"].items():
        if not name.startswith("ve_"):
            for entry in artifact["files"]:
                target = bundle / entry["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to((args.bundle / entry["path"]).resolve())
            continue
        directory, source = exports[name.replace("ve_", "vector_estimator_L")]
        package = directory / source["package"]
        if tree_hashes(package) != source["package_sha256"]:
            raise ReferenceError(f"Changed source mlpackage: {name}")
        started = time.monotonic()
        print(f"Palettizing {name}", flush=True)
        model = ct.models.MLModel(str(package), skip_model_load=True)
        compressed = cto.palettize_weights(model, config)
        stem = f"{name}_T384_int4"
        output_package = args.output / f"{stem}.mlpackage"
        compressed.save(str(output_package))
        subprocess.run(
            ["xcrun", "coremlcompiler", "compile", str(output_package), str(bundle)],
            check=True,
        )
        artifact["path"] = f"{stem}.mlmodelc"
        artifact["precision"] = {
            "weights": "4-bit k-means palettes for eligible weights; small weights retain FP16",
            "activations": "float16",
            "interface": "float32",
        }
        artifact["files"] = []
        compiled = bundle / artifact["path"]
        for relative, sha in sorted(tree_hashes(compiled).items()):
            artifact["files"].append(
                {
                    "path": f"{artifact['path']}/{relative}",
                    "sha256": sha,
                    "size_bytes": (compiled / relative).stat().st_size,
                }
            )
        report["artifacts"][name] = {
            "source_package_sha256": source["package_sha256"],
            "package_sha256": tree_hashes(output_package),
            "compiled_sha256": tree_hashes(compiled),
            "seconds": time.monotonic() - started,
        }
        write_yaml(args.output / "quantization.yaml", report)
        print(f"Finished {name}: {time.monotonic()-started:.2f}s", flush=True)
    write_yaml(bundle / "mlsr-bundle.yaml", manifest)
    verify_bundle(load_manifest(bundle / "mlsr-bundle.yaml"), bundle)
    report.update(
        status="complete", manifest_sha256=digest(bundle / "mlsr-bundle.yaml")
    )
    write_yaml(args.output / "quantization.yaml", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "bundle", "models", "short-models", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    quantize(parser.parse_args())


if __name__ == "__main__":
    main()
