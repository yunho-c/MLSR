"""Export and replay experimental larger-context Core ML models.

Requires macOS, coremltools, torch, ONNX and a local Möbius checkout for export.
Models and evidence stay outside the repository. This experimental manifest is
not accepted by the shipping MLSR bundle loader.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from .bundle import ReferenceError, digest, load_yaml, write_yaml
from .context_probe import (
    INPUTS,
    LATENT_BUCKETS,
    TorchPort,
    compare_capture,
    official_cases,
    replay,
)
from .quality_probe import joined_pcm, record_tool_sources
from .reference import write_float_wav


def tree_hashes(path: Path) -> dict:
    return {
        str(p.relative_to(path)): digest(p)
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }


def export(args) -> None:
    import coremltools as ct
    import torch

    qualification = load_yaml(args.probe / "report.yaml")
    if (
        qualification["status"] != "complete"
        or qualification["text_capacity"] != args.text_capacity
    ):
        raise ReferenceError(
            "A completed probe at the requested text capacity is required"
        )
    for name, sha in qualification["source_sha256"].items():
        if digest(args.mobius_source / "coreml" / name) != sha:
            raise ReferenceError(f"Port source changed since qualification: {name}")
    for name, sha in qualification.get("onnx_sha256", {}).items():
        if digest(args.onnx / name) != sha:
            raise ReferenceError(f"ONNX model changed since qualification: {name}")
    args.output.mkdir(parents=True, exist_ok=False)
    preserve_boundaries = qualification.get("preserve_boundaries", False)
    port = TorchPort(args.mobius_source, args.onnx, preserve_boundaries)
    # Reuse Möbius' compatibility patch for NumPy 2 scalar casts during conversion.
    from coreml.convert_coreml import _patch_int_cast  # noqa: F401

    class TextWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, text_ids, style, text_mask):
            return self.model(text_ids.long(), style, text_mask)

    report = {
        "status": "incomplete",
        "experimental": True,
        "text_capacity": args.text_capacity,
        "precision": args.precision,
        "minimum_deployment": "iOS18 / macOS15",
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "coremltools": ct.__version__,
        "numpy": np.__version__,
        "driver_sha256": digest(Path(__file__)),
        "probe_sha256": digest(args.probe / "report.yaml"),
        "onnx_sha256": {p.name: digest(p) for p in args.onnx.glob("*.onnx")},
        "normalization_adjustment": port.normalization,
        "artifacts": {},
    }
    report["preserve_boundaries"] = preserve_boundaries
    report["tool_sources_sha256"] = record_tool_sources(args.output)
    write_yaml(args.output / "manifest.yaml", report)
    torch.manual_seed(0)
    T = args.text_capacity
    samples = {
        "duration_predictor": [
            torch.zeros(1, T, dtype=torch.int32),
            torch.zeros(1, 8, 16),
            torch.ones(1, 1, T),
        ],
        "text_encoder": [
            torch.zeros(1, T, dtype=torch.int32),
            torch.zeros(1, 50, 256),
            torch.ones(1, 1, T),
        ],
        "vocoder": [torch.randn(1, 144, 4)],
    }
    outputs = {
        "duration_predictor": "duration",
        "text_encoder": "text_emb",
        "vector_estimator": "denoised_latent",
        "vocoder": "wav",
    }
    stages = [("duration_predictor", None), ("text_encoder", None)]
    stages += [("vector_estimator", L) for L in args.latent_buckets]
    stages += [("vocoder", None)]
    if args.stage:
        stages = [(name, length) for name, length in stages if name in args.stage]
    for name, L in stages:
        key = name if L is None else f"{name}_L{L}"
        started = time.monotonic()
        model = port.models[name]
        if name in ("duration_predictor", "text_encoder"):
            model = TextWrapper(model).eval()
        sample = samples.get(name)
        if name == "vector_estimator":
            sample = [
                torch.randn(1, 144, L),
                torch.randn(1, 256, T),
                torch.zeros(1, 50, 256),
                torch.ones(1, 1, L),
                torch.ones(1, 1, T),
                torch.tensor([0.0]),
                torch.tensor([8.0]),
            ]
        with torch.no_grad():
            traced = torch.jit.trace(model, tuple(sample))
        shapes = [tuple(v.shape) for v in sample]
        if name == "vocoder":
            shapes = [(1, 144, ct.RangeDim(lower_bound=4, upper_bound=512, default=4))]
        inputs = [
            ct.TensorType(
                name=k, shape=shape, dtype=np.int32 if k == "text_ids" else np.float32
            )
            for k, shape in zip(INPUTS[name], shapes)
        ]
        converted = ct.convert(
            traced,
            inputs=inputs,
            outputs=[ct.TensorType(name=outputs[name], dtype=np.float32)],
            convert_to="mlprogram",
            compute_precision=(
                ct.precision.FLOAT16
                if args.precision == "float16"
                else ct.precision.FLOAT32
            ),
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            skip_model_load=True,
        )
        package = args.output / f"{key}.mlpackage"
        converted.save(str(package))
        subprocess.run(
            ["xcrun", "coremlcompiler", "compile", str(package), str(args.output)],
            check=True,
        )
        compiled = args.output / f"{key}.mlmodelc"
        report["artifacts"][key] = {
            "package": package.name,
            "compiled": compiled.name,
            "package_sha256": tree_hashes(package),
            "compiled_sha256": tree_hashes(compiled),
            "seconds": time.monotonic() - started,
        }
        write_yaml(args.output / "manifest.yaml", report)
        print(f"Exported {key}: {time.monotonic()-started:.2f}s", flush=True)
    report["status"] = "complete"
    write_yaml(args.output / "manifest.yaml", report)


class CoreMLContext:
    def __init__(self, directory: Path, policy: str):
        import coremltools as ct

        self.version = ct.__version__
        self.models = {}
        self.timings = {}
        manifest = load_yaml(directory / "manifest.yaml")
        if manifest["status"] != "complete":
            raise ReferenceError("Incomplete model export")
        units = {
            "cpu_only": ct.ComputeUnit.CPU_ONLY,
            "cpu_and_neural_engine": ct.ComputeUnit.CPU_AND_NE,
        }[policy]
        for key, artifact in manifest["artifacts"].items():
            path = directory / artifact["compiled"]
            if tree_hashes(path) != artifact["compiled_sha256"]:
                raise ReferenceError(f"Compiled artifact changed: {key}")
            self.models[key] = ct.models.CompiledMLModel(str(path), compute_units=units)

    def predict(self, name: str, feeds: dict) -> np.ndarray:
        key = name
        if name == "vector_estimator":
            key = f"{name}_L{feeds['noisy_latent'].shape[-1]}"
        feeds = {
            k: np.ascontiguousarray(
                v, dtype=np.int32 if k == "text_ids" else np.float32
            )
            for k, v in feeds.items()
        }
        started = time.monotonic()
        result = next(iter(self.models[key].predict(feeds).values()))
        self.timings.setdefault(key, []).append(time.monotonic() - started)
        if not np.isfinite(result).all():
            raise ReferenceError(f"Nonfinite Core ML output: {key}")
        return np.asarray(result, dtype=np.float32).copy()


def qualify(args) -> None:
    reference = load_yaml(args.captures / "report.yaml")
    probe = load_yaml(args.probe / "report.yaml")
    manifest = load_yaml(args.models / "manifest.yaml")
    if reference["status"] != "complete" or probe["status"] != "complete":
        raise ReferenceError("Incomplete reference/probe")
    if (
        digest(args.captures / "report.yaml") != probe["reference_sha256"]
        or digest(args.probe / "report.yaml") != manifest["probe_sha256"]
    ):
        raise ReferenceError("Model/probe/reference provenance mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    engine = CoreMLContext(args.models, args.policy)
    report = {
        "status": "incomplete",
        "policy": args.policy,
        "precision": manifest["precision"],
        "coremltools": engine.version,
        "model_manifest_sha256": digest(args.models / "manifest.yaml"),
        "driver_sha256": digest(Path(__file__)),
        "load_seconds": time.monotonic() - started,
        "scope": "Reference duration/length/noise frozen; 8-step rollout; true-length vocoder",
        "cases": [],
    }
    report["tool_sources_sha256"] = record_tool_sources(args.output)
    write_yaml(args.output / "report.yaml", report)
    for case in official_cases(reference):
        audio = []
        for chunk in case["chunks"]:
            case_id = f"{case['id']}-{chunk['id']:02d}"
            row = next(r for r in probe["cases"] if r["id"] == case_id)
            source = args.captures / chunk["capture"]
            if digest(source) != chunk["capture_sha256"]:
                raise ReferenceError("Modified original capture")
            with np.load(source, allow_pickle=False) as loaded:
                saved = dict(loaded)
            started = time.monotonic()
            result = replay(
                engine,
                saved,
                manifest["text_capacity"],
                row["bucket"],
                reference["steps"],
            )
            elapsed = time.monotonic() - started
            path = args.output / f"{case_id}.npz"
            np.savez_compressed(path, **result)
            comparisons = {"dynamic_onnx": compare_capture(result, saved)}
            for name in ("bucket-padding", "torch-port"):
                variant = row["variants"][name]
                other = args.probe / variant["capture"]
                if digest(other) != variant["sha256"]:
                    raise ReferenceError("Modified probe capture")
                with np.load(other, allow_pickle=False) as loaded:
                    comparisons[name] = compare_capture(result, dict(loaded))
            report["cases"].append(
                {
                    "id": case_id,
                    "seconds": elapsed,
                    "capture": path.name,
                    "sha256": digest(path),
                    "comparisons": comparisons,
                }
            )
            audio.append(result["pcm"])
            print(
                f"{case_id}: {elapsed:.3f}s; port PCM nrmse "
                f"{comparisons['torch-port']['pcm']['nrmse']:.6g}",
                flush=True,
            )
            write_yaml(args.output / "report.yaml", report)
        label = manifest["precision"].replace("float", "fp")
        write_float_wav(
            args.output / f"{case['voice']}-coreml-{label}.wav",
            joined_pcm(audio),
            44100,
        )
    report.update(status="complete", prediction_seconds=engine.timings)
    write_yaml(args.output / "report.yaml", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export")
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--mobius-source", type=Path, required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--text-capacity", type=int, default=384)
    p.add_argument("--precision", choices=["float16", "float32"], default="float16")
    p.add_argument(
        "--latent-buckets",
        nargs="+",
        type=int,
        choices=[128, 256, 512],
        default=list(LATENT_BUCKETS),
    )
    p.add_argument("--stage", action="append", choices=list(INPUTS))
    p = sub.add_parser("replay")
    for name in ("captures", "probe", "models", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument(
        "--policy",
        choices=["cpu_only", "cpu_and_neural_engine"],
        default="cpu_and_neural_engine",
    )
    args = parser.parse_args()
    (export if args.command == "export" else qualify)(args)


if __name__ == "__main__":
    main()
