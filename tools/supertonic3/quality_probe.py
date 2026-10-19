"""Development-only official ONNX controls for text segmentation and conversion.

The same models, voices, rate and steps render two saved chunk plans. Actual
noise and every stage output are retained for subsequent conversion replay.
This does not change the shipping model bundle or the Rust text policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import time

import numpy as np

from .bundle import ReferenceError, digest, write_yaml
from .frontend import normalize_text
from .reference import write_float_wav


def record_tool_sources(output: Path) -> dict:
    destination = output / "tool-sources"
    destination.mkdir()
    hashes = {}
    for name in (
        "bundle",
        "frontend",
        "reference",
        "quality_probe",
        "context_probe",
        "context_coreml",
    ):
        source = Path(__file__).with_name(f"{name}.py")
        shutil.copy2(source, destination / source.name)
        hashes[source.name] = digest(source)
    return hashes


def encode_normalized(text: str, indexer: list[int], capacity: int | None = None):
    """Encode every normalized character; padding must never truncate text."""
    capacity = len(text) if capacity is None else capacity
    if not text or capacity < len(text):
        raise ReferenceError("Text is empty or exceeds the requested capacity")
    tokens = []
    for char in text:
        if ord(char) >= len(indexer) or indexer[ord(char)] < 0:
            raise ReferenceError(f"Unsupported character U+{ord(char):04X}")
        tokens.append(indexer[ord(char)])
    ids = np.zeros((1, capacity), dtype=np.int64)
    mask = np.zeros((1, 1, capacity), dtype=np.float32)
    ids[0, : len(tokens)] = tokens
    mask[0, 0, : len(tokens)] = 1
    return ids, mask


def noise_for(text: str, voice: str, length: int, seed: int) -> np.ndarray:
    """Stable per-input noise, saved for replay rather than cross-RNG guessing.

    Different segmentation creates different inputs/shapes and therefore does
    not have sample-identical noise. Unchanged inputs retain the same noise.
    """
    key = f"{seed}\0{voice}\0{text}".encode()
    value = int.from_bytes(hashlib.sha256(key).digest()[:16], "little")
    return np.random.Generator(np.random.PCG64(value)).standard_normal(
        (1, 144, length), dtype=np.float32
    )


def joined_pcm(chunks: list[np.ndarray], gap_frames: int = 0) -> np.ndarray:
    if not chunks or gap_frames < 0:
        raise ReferenceError("Expected audio chunks and nonnegative join padding")
    pieces = []
    for i, chunk in enumerate(chunks):
        if i and gap_frames:
            pieces.append(np.zeros(gap_frames, dtype=np.float32))
        pieces.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
    return np.concatenate(pieces)


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    if actual.shape != expected.shape:
        raise ReferenceError(f"Shape mismatch: {actual.shape} != {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ReferenceError("Nonfinite tensor")
    diff = actual.astype(np.float64) - expected.astype(np.float64)
    rmse = float(np.sqrt(np.mean(diff * diff)))
    rms = float(np.sqrt(np.mean(expected.astype(np.float64) ** 2)))
    return {
        "shape": list(actual.shape),
        "max_abs": float(np.max(np.abs(diff))),
        "rmse": rmse,
        "nrmse": rmse / max(rms, 1e-12),
    }


class OnnxReference:
    def __init__(self, directory: Path, threads: int = 4):
        import onnxruntime as ort

        self.version = ort.__version__
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.models = {
            name: ort.InferenceSession(
                str(directory / f"{name}.onnx"),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            for name in (
                "duration_predictor",
                "text_encoder",
                "vector_estimator",
                "vocoder",
            )
        }

    def predict(self, name: str, feeds: dict) -> np.ndarray:
        result = self.models[name].run(None, feeds)[0]
        if not np.isfinite(result).all():
            raise ReferenceError(f"Nonfinite {name} result")
        return result

    def capture(
        self,
        normalized: str,
        indexer: list[int],
        style: dict,
        voice: str,
        seed: int,
        steps: int,
        speed: float,
    ) -> dict:
        ids, mask = encode_normalized(normalized, indexer)
        ttl = np.asarray(style["style_ttl"]["data"], dtype=np.float32).reshape(
            style["style_ttl"]["dims"]
        )
        dp = np.asarray(style["style_dp"]["data"], dtype=np.float32).reshape(
            style["style_dp"]["dims"]
        )
        raw = self.predict(
            "duration_predictor", {"text_ids": ids, "text_mask": mask, "style_dp": dp}
        )
        # Browser arithmetic: Number(float32 model output) * 1 / (UI speed+.05).
        duration = float(raw.reshape(-1)[0]) / speed
        frames = math.floor(duration * 44100)
        length = math.floor((duration * 44100 + 3072 - 1) / 3072)
        if frames <= 0 or not 4 <= length <= 512:
            raise ReferenceError(f"Unsupported reference duration {duration}")
        embedding = self.predict(
            "text_encoder", {"text_ids": ids, "text_mask": mask, "style_ttl": ttl}
        )
        noise = noise_for(normalized, voice, length, seed)
        latent_mask = np.ones((1, 1, length), dtype=np.float32)
        capture = {
            "text_ids": ids,
            "text_mask": mask,
            "style_dp": dp,
            "style_ttl": ttl,
            "raw_duration": raw,
            "text_emb": embedding,
            "initial_noise": noise,
            "latent_mask": latent_mask,
            "frames": np.array(frames, dtype=np.int64),
        }
        latent = noise.copy()
        total = np.array([steps], dtype=np.float32)
        for step in range(steps):
            latent = self.predict(
                "vector_estimator",
                {
                    "noisy_latent": latent,
                    "text_emb": embedding,
                    "style_ttl": ttl,
                    "latent_mask": latent_mask,
                    "text_mask": mask,
                    "current_step": np.array([step], dtype=np.float32),
                    "total_step": total,
                },
            )
            capture[f"ve_step_{step:02d}"] = latent
        wav = self.predict("vocoder", {"latent": latent}).reshape(-1)
        if len(wav) < frames:
            raise ReferenceError("Vocoder produced fewer frames than requested")
        capture["pcm"] = wav[:frames].copy()
        return capture


def capture_comparison(args) -> None:
    analysis, onnx = args.analysis, args.onnx
    snapshot = json.loads((analysis / "source-snapshot.json").read_text())
    for entry in snapshot["files"]:
        if digest(analysis / entry["path"]) != entry["sha256"]:
            raise ReferenceError(f"Modified demo source: {entry['path']}")
    tree = json.loads((analysis / "space-tree.json").read_text())
    sources = {}
    for row in tree:
        if row["path"].endswith(".onnx"):
            path = onnx / Path(row["path"]).name
            if digest(path) != row["lfs"]["oid"]:
                raise ReferenceError(f"ONNX bytes differ from demo: {path}")
            sources[path.name] = digest(path)
    official = json.loads((analysis / "official-chunks.json").read_text())
    mlsr = json.loads(args.mlsr_report.read_text())
    text = mlsr["text"]
    policies = {
        "official": [
            {"text": row["text"], "normalized": row["normalized"]} for row in official
        ],
        "mlsr": [
            {"text": row["original"], "normalized": row["normalized"]}
            for row in mlsr["plan"]["segments"]
        ],
    }
    if " ".join(row["text"] for row in policies["official"]) != text.strip():
        raise ReferenceError("Official chunk list does not reconstruct the passage")
    if "".join(row["text"] for row in policies["mlsr"]) != text:
        raise ReferenceError("MLSR spans do not reconstruct the passage")
    for chunks in policies.values():
        for row in chunks:
            if normalize_text(row["text"], "en") != row["normalized"]:
                raise ReferenceError("Saved normalization differs from reference")
    indexer = json.loads((analysis / "assets/onnx/unicode_indexer.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "incomplete",
        "text": text,
        "official_space_revision": snapshot["revision"],
        "source_snapshot_sha256": digest(analysis / "source-snapshot.json"),
        "chunk_plan_sha256": digest(analysis / "official-chunks.json"),
        "mlsr_report_sha256": digest(args.mlsr_report),
        "onnx_sha256": sources,
        "driver_sha256": digest(Path(__file__)),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "steps": 8,
        "speed": 1.05,
        "seed": 0,
        "noise": "PCG64 keyed by normalized text/voice/seed; actual tensors saved",
        "comparison": "Same FP32 ONNX models; dynamic axes; zero added gaps for both layouts",
        "cases": [],
    }
    report["tool_sources_sha256"] = record_tool_sources(args.output)
    write_yaml(args.output / "report.yaml", report)
    engine = OnnxReference(onnx)
    report["onnxruntime"] = engine.version
    started = time.monotonic()
    for voice in args.voices:
        style_path = analysis / f"assets/voice_styles/{voice}.json"
        style = json.loads(style_path.read_text())
        for policy, chunks in policies.items():
            name = f"{voice}-{policy}"
            directory = args.output / name
            directory.mkdir()
            pcm = []
            records = []
            for i, row in enumerate(chunks):
                tick = time.monotonic()
                tensors = engine.capture(
                    row["normalized"], indexer, style, voice, 0, 8, 1.05
                )
                path = directory / f"chunk-{i:02d}.npz"
                np.savez_compressed(path, **tensors)
                pcm.append(tensors["pcm"])
                records.append(
                    {
                        "id": i,
                        **row,
                        "tokens": len(row["normalized"]),
                        "frames": len(tensors["pcm"]),
                        "latent_length": tensors["initial_noise"].shape[-1],
                        "capture": str(path.relative_to(args.output)),
                        "capture_sha256": digest(path),
                        "seconds": time.monotonic() - tick,
                    }
                )
                print(
                    f"{name} {i+1}/{len(chunks)}: {records[-1]['seconds']:.2f}s",
                    flush=True,
                )
            merged = joined_pcm(pcm)
            write_float_wav(directory / "speech-zero-gap.wav", merged, 44100)
            if policy == "official":
                write_float_wav(
                    directory / "speech-demo-gaps.wav", joined_pcm(pcm, 13230), 44100
                )
            report["cases"].append(
                {
                    "id": name,
                    "voice": voice,
                    "voice_sha256": digest(style_path),
                    "policy": policy,
                    "chunks": records,
                    "frames_zero_gap": len(merged),
                    "pcm_sha256": hashlib.sha256(merged.tobytes()).hexdigest(),
                }
            )
            write_yaml(args.output / "report.yaml", report)
    report.update(status="complete", elapsed_seconds=time.monotonic() - started)
    write_yaml(args.output / "report.yaml", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--mlsr-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--voices", nargs="+", choices=["M1", "F4"], default=["M1", "F4"]
    )
    capture_comparison(parser.parse_args())


if __name__ == "__main__":
    main()
