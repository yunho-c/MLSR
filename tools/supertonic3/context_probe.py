"""Experimental padding and PyTorch-port checks using saved official captures.

Duration, true latent length and actual noise are frozen from the dynamic ONNX
reference. This isolates text padding, latent padding, and port arithmetic; it
is not native end-to-end duration qualification or a shipping model manifest.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
import time

import numpy as np

from .bundle import ReferenceError, digest, load_yaml, write_yaml
from .quality_probe import OnnxReference, joined_pcm, metrics, record_tool_sources
from .reference import write_float_wav


INPUTS = {
    "duration_predictor": ("text_ids", "style_dp", "text_mask"),
    "text_encoder": ("text_ids", "style_ttl", "text_mask"),
    "vector_estimator": (
        "noisy_latent",
        "text_emb",
        "style_ttl",
        "latent_mask",
        "text_mask",
        "current_step",
        "total_step",
    ),
    "vocoder": ("latent",),
}

LATENT_BUCKETS = (256, 512)


def context_bucket(length: int) -> int:
    if not 4 <= length <= LATENT_BUCKETS[-1]:
        raise ReferenceError("Latent length is outside the experimental contract")
    return next(bucket for bucket in LATENT_BUCKETS if bucket >= length)


def pad_last(value: np.ndarray, capacity: int) -> np.ndarray:
    if capacity < value.shape[-1]:
        raise ReferenceError("Padding must not truncate a tensor")
    widths = [(0, 0)] * value.ndim
    widths[-1] = (0, capacity - value.shape[-1])
    return np.pad(value, widths)


def official_cases(reference: dict) -> list[dict]:
    cases = [c for c in reference["cases"] if c["policy"] == "official"]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ReferenceError("Expected nonempty unique official reference cases")
    for case in cases:
        chunks = case["chunks"]
        if not chunks or [c["id"] for c in chunks] != list(range(len(chunks))):
            raise ReferenceError("Expected ordered contiguous reference chunks")
    return cases


def replay(
    engine, saved: dict, text_capacity: int, latent_capacity: int, steps: int = 8
) -> dict:
    result = {key: saved[key].copy() for key in ("style_dp", "style_ttl", "frames")}
    result.update(
        text_ids=pad_last(saved["text_ids"], text_capacity),
        text_mask=pad_last(saved["text_mask"], text_capacity),
        initial_noise=pad_last(saved["initial_noise"], latent_capacity),
        latent_mask=pad_last(saved["latent_mask"], latent_capacity),
    )
    for model, output in [
        ("duration_predictor", "raw_duration"),
        ("text_encoder", "text_emb"),
    ]:
        result[output] = engine.predict(model, {k: result[k] for k in INPUTS[model]})
    latent = result["initial_noise"].copy()
    for step in range(steps):
        feeds = {
            k: result[k] for k in ("text_emb", "style_ttl", "latent_mask", "text_mask")
        }
        feeds.update(
            noisy_latent=latent,
            current_step=np.array([step], np.float32),
            total_step=np.array([steps], np.float32),
        )
        latent = engine.predict("vector_estimator", feeds)
        result[f"ve_step_{step:02d}"] = latent
    true_length = saved["initial_noise"].shape[-1]
    result["vocoder_input"] = np.ascontiguousarray(latent[..., :true_length])
    wav = engine.predict("vocoder", {"latent": result["vocoder_input"]}).reshape(-1)
    frames = int(saved["frames"])
    if wav.size < frames:
        raise ReferenceError("Vocoder output shorter than reference duration")
    result["pcm"] = wav[:frames].copy()
    return result


def compare_capture(actual: dict, expected: dict, steps: int = 8) -> dict:
    """Compare the valid prefix, excluding padding from the error denominator."""
    text_length = int(np.sum(expected["text_mask"]))
    length = int(np.sum(expected["latent_mask"]))
    result = {
        "raw_duration": metrics(actual["raw_duration"], expected["raw_duration"]),
        "text_emb": metrics(
            actual["text_emb"][..., :text_length],
            expected["text_emb"][..., :text_length],
        ),
        "pcm": metrics(actual["pcm"], expected["pcm"]),
    }
    for step in range(steps):
        key = f"ve_step_{step:02d}"
        result[key] = metrics(actual[key][..., :length], expected[key][..., :length])
    return result


class TorchPort:
    def __init__(self, source: Path, onnx: Path, preserve_boundaries: bool = False):
        import torch

        self.torch = torch
        self.version = str(torch.__version__)
        torch.set_num_threads(4)
        sys.path.insert(0, str(source))
        self.models = {}
        self.normalization = {}
        for name in INPUTS:
            module = importlib.import_module(f"coreml.{name}")
            builder = getattr(module, f"build_{name}_from_onnx")
            kwargs = {"tts_json_path": onnx / "tts.json"} if name == "vocoder" else {}
            self.models[name] = builder(onnx / f"{name}.onnx", **kwargs).eval()
            # The upstream port defaults to 1e-5, while these ONNX graphs use
            # 1e-6. Read the graph rather than carrying that discrepancy into
            # a new export. BatchNorm epsilon is intentionally left alone.
            self.normalization[name] = align_layer_norm_eps(
                self.models[name], onnx / f"{name}.onnx"
            )
            if preserve_boundaries and name != "vocoder":
                preserve_conv_boundaries(self.models[name])

    def predict(self, name: str, feeds: dict) -> np.ndarray:
        with self.torch.no_grad():
            values = [self.torch.from_numpy(feeds[k]) for k in INPUTS[name]]
            result = self.models[name](*values).numpy()
        if not np.isfinite(result).all():
            raise ReferenceError(f"Nonfinite {name} result")
        return result


def align_layer_norm_eps(model, path: Path) -> dict:
    import onnx
    import torch

    graph = onnx.load(path)
    eps = [
        next(float(a.f) for a in n.attribute if a.name == "epsilon")
        for n in graph.graph.node
        if n.op_type == "LayerNormalization"
    ]
    layers = [m for m in model.modules() if isinstance(m, torch.nn.LayerNorm)]
    if not eps or len(set(eps)) != 1 or len(eps) != len(layers):
        raise ReferenceError(f"Cannot map LayerNorm epsilon unambiguously: {path}")
    before = sorted({m.eps for m in layers})
    for layer in layers:
        layer.eps = eps[0]
    return {
        "count": len(layers),
        "original_port_epsilon": before,
        "onnx_epsilon": eps[0],
    }


def preserve_conv_boundaries(model) -> None:
    """Make masked fixed-width ConvNeXt match unpadded edge-replicated inputs.

    A zero tail is not equivalent to ONNX Pad(mode=edge) at the true sequence
    endpoint. Extend that endpoint before each temporal convolution, then keep
    the original output masks. Only prefix masks (ones followed by zeros) are
    supported by this experimental, batch-one speech pipeline.
    """
    import types
    import torch.nn.functional as functional
    from coreml.common import ConvNeXt1DBlock

    def forward(self, x, mask=None):
        convolution_input = x
        if mask is not None:
            endpoint = mask - functional.pad(mask[..., 1:], (0, 1))
            last_valid = (x * endpoint).sum(dim=-1, keepdim=True)
            convolution_input = x + (1.0 - mask) * last_valid
        h = self.dwconv(convolution_input)
        if mask is not None:
            h = h * mask
        h = self.norm(h)
        h = self.pwconv2(functional.gelu(self.pwconv1(h)))
        out = x + self.gamma * h
        return out if mask is None else out * mask

    for block in model.modules():
        if isinstance(block, ConvNeXt1DBlock):
            block.forward = types.MethodType(forward, block)


def probe(args) -> None:
    reference = load_yaml(args.captures / "report.yaml")
    if reference["status"] != "complete":
        raise ReferenceError("Reference capture is incomplete")
    for name, sha in reference["onnx_sha256"].items():
        if digest(args.onnx / name) != sha:
            raise ReferenceError(f"Reference model hash mismatch: {name}")
    args.output.mkdir(parents=True, exist_ok=False)
    onnx = OnnxReference(args.onnx)
    port = TorchPort(args.mobius_source, args.onnx, args.preserve_boundaries)
    report = {
        "status": "incomplete",
        "text_capacity": args.text_capacity,
        "reference_sha256": digest(args.captures / "report.yaml"),
        "onnx_sha256": reference["onnx_sha256"],
        "driver_sha256": digest(Path(__file__)),
        "torch": port.version,
        "onnxruntime": onnx.version,
        "normalization_adjustment": port.normalization,
        "preserve_boundaries": args.preserve_boundaries,
        "source_sha256": {
            p.name: digest(p)
            for p in sorted((args.mobius_source / "coreml").glob("*.py"))
        },
        "scope": "Reference duration/length/noise frozen; vocoder uses true length",
        "cases": [],
    }
    report["tool_sources_sha256"] = record_tool_sources(args.output)
    write_yaml(args.output / "report.yaml", report)
    for case in official_cases(reference):
        audio = {name: [] for name in ("text-padding", "bucket-padding", "torch-port")}
        for chunk in case["chunks"]:
            capture = args.captures / chunk["capture"]
            if digest(capture) != chunk["capture_sha256"]:
                raise ReferenceError(f"Modified capture: {capture}")
            with np.load(capture, allow_pickle=False) as loaded:
                saved = dict(loaded)
            length = saved["initial_noise"].shape[-1]
            bucket = context_bucket(length)
            tick = time.monotonic()
            previous = saved
            row = {
                "id": f"{case['id']}-{chunk['id']:02d}",
                "voice": case["voice"],
                "capture": chunk["capture"],
                "length": length,
                "bucket": bucket,
                "variants": {},
            }
            for name, engine, capacity in [
                ("text-padding", onnx, length),
                ("bucket-padding", onnx, bucket),
                ("torch-port", port, bucket),
            ]:
                result = replay(
                    engine, saved, args.text_capacity, capacity, reference["steps"]
                )
                path = args.output / f"{row['id']}-{name}.npz"
                np.savez_compressed(path, **result)
                row["variants"][name] = {
                    "capture": path.name,
                    "sha256": digest(path),
                    "vs_dynamic": compare_capture(result, saved),
                    "vs_previous": compare_capture(result, previous),
                }
                audio[name].append(result["pcm"])
                previous = result
            report["cases"].append(row)
            write_yaml(args.output / "report.yaml", report)
            target = "vs_dynamic" if args.preserve_boundaries else "vs_previous"
            comparisons = row["variants"]["torch-port"][target]
            print(
                f"{row['id']}: {time.monotonic()-tick:.2f}s; {target} "
                f"TE max_abs {comparisons['text_emb']['max_abs']:.6g}; "
                f"PCM nrmse {comparisons['pcm']['nrmse']:.6g}",
                flush=True,
            )
        for name, chunks in audio.items():
            write_float_wav(
                args.output / f"{case['voice']}-{name}.wav", joined_pcm(chunks), 44100
            )
    report["status"] = "complete"
    write_yaml(args.output / "report.yaml", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument(
        "--mobius-source",
        type=Path,
        required=True,
        help="Directory containing the Möbius coreml Python package",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-capacity", type=int, default=384)
    parser.add_argument("--preserve-boundaries", action="store_true")
    probe(parser.parse_args())


if __name__ == "__main__":
    main()
