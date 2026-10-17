"""Capture and replay real Core ML reference tensors; never used by MLSR runtime."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys
import time

import numpy as np

from .bundle import (
    DEFAULT_CORPUS,
    ROOT,
    ReferenceError,
    canonical_digest,
    digest,
    load_yaml,
    relative_path,
    verify_bundle,
    write_yaml,
)
from .frontend import InputError, duration_frames, encode_text, select_bucket


def load_corpus(path: Path = DEFAULT_CORPUS) -> dict:
    corpus = load_yaml(path)
    if (
        type(corpus.get("schema_version")) is not int
        or corpus["schema_version"] != 1
        or not isinstance(corpus.get("cases"), list)
    ):
        raise ReferenceError("Invalid corpus schema")
    seen = set()
    for case in corpus["cases"]:
        name = case.get("id", "")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in seen:
            raise ReferenceError(f"Invalid or duplicate case ID: {name}")
        seen.add(name)
        if not isinstance(case.get("text"), str) or case.get("voice") not in (
            "M1",
            "F1",
        ):
            raise ReferenceError(f"Invalid text/voice for {name}")
        if type(case.get("seed")) is not int or not 0 <= case["seed"] < 2**64:
            raise ReferenceError(f"Invalid noise seed for {name}")
        steps = case.get("steps", corpus["defaults"]["steps"])
        speed = case.get("speed", corpus["defaults"]["speed"])
        if type(steps) is not int or not 1 <= steps <= 64:
            raise ReferenceError(f"Invalid step count for {name}")
        if (
            not isinstance(speed, (int, float))
            or not math.isfinite(speed)
            or speed <= 0
        ):
            raise ReferenceError(f"Invalid speaking rate for {name}")
        target = case.get("target_latent_length")
        if target is not None and (type(target) is not int or not 4 <= target <= 512):
            raise ReferenceError(f"Invalid target latent length for {name}")
    if not seen:
        raise ReferenceError("Corpus is empty")
    return corpus


def read_indexer(manifest: dict, root: Path) -> list[int]:
    path = root / manifest["artifacts"]["unicode_indexer"]["path"]
    return json.loads(path.read_text(encoding="utf-8"))


def check_frontend(corpus: dict, indexer: list[int], expected_path: Path) -> list[dict]:
    expected = load_yaml(expected_path)
    if expected.get("corpus_sha256") != canonical_digest(corpus):
        raise ReferenceError("Frontend fixture belongs to a different corpus")
    if set(expected["cases"]) != {case["id"] for case in corpus["cases"]}:
        raise ReferenceError("Frontend fixture case set differs from corpus")
    results = []
    for case in corpus["cases"]:
        target = expected["cases"][case["id"]]
        try:
            normalized, ids, mask = encode_text(case["text"], case["language"], indexer)
        except InputError as error:
            if error.code != case.get("expected_error") or error.code != target.get(
                "error"
            ):
                raise ReferenceError(
                    f"{case['id']}: unexpected frontend error {error.code}"
                ) from error
            results.append({"id": case["id"], "error": error.code})
            continue
        if case.get("expected_error") or target.get("error"):
            raise ReferenceError(
                f"{case['id']}: expected rejection, input was accepted"
            )
        tokens = ids[0, : len(normalized)].tolist()
        if normalized != target["normalized_text"] or tokens != target["tokens"]:
            raise ReferenceError(
                f"{case['id']}: preprocessing differs from upstream-derived fixture"
            )
        if int(mask.sum()) != len(normalized):
            raise ReferenceError(f"{case['id']}: invalid text mask")
        results.append({"id": case["id"], "tokens": len(normalized)})
    return results


def tensor_record(value: np.ndarray) -> dict:
    if (
        value.dtype not in (np.dtype("float32"), np.dtype("int32"))
        or not np.isfinite(value).all()
    ):
        raise ReferenceError(
            f"Unexpected tensor dtype or nonfinite values: {value.dtype}"
        )
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def write_float_wav(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    """IEEE Float32 mono WAV; NPZ remains the canonical tensor container."""
    data = np.asarray(pcm, dtype="<f4").tobytes(order="C")
    fmt = struct.pack("<HHIIHH", 3, 1, sample_rate, sample_rate * 4, 4, 32)
    fact = struct.pack("<I", len(data) // 4)
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"fact" + struct.pack("<I", len(fact)) + fact
    chunks += b"data" + struct.pack("<I", len(data)) + data
    path.write_bytes(b"RIFF" + struct.pack("<I", len(chunks) + 4) + b"WAVE" + chunks)


def environment() -> dict:
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "unicode_version": __import__("unicodedata").unidata_version,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "PyYAML", "coremltools")
        },
    }
    if sys.platform == "darwin":
        result["hardware"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        result["os_build"] = subprocess.check_output(
            ["sw_vers", "-buildVersion"], text=True
        ).strip()
    return result


class CoreMLReference:
    def __init__(self, manifest: dict, root: Path, compute_policy: str):
        if sys.platform != "darwin":
            raise ReferenceError("Core ML capture/replay requires macOS")
        import coremltools as ct

        policies = {
            "cpu_and_neural_engine": ct.ComputeUnit.CPU_AND_NE,
            "cpu_only": ct.ComputeUnit.CPU_ONLY,
        }
        if compute_policy not in policies:
            raise ReferenceError(f"Unknown compute policy: {compute_policy}")
        self.ct = ct
        self.units = policies[compute_policy]
        self.manifest = manifest
        self.root = root
        self.indexer = read_indexer(manifest, root)
        self.models = {}

    def model(self, name: str):
        if name not in self.models:
            print(f"Loading {name}", flush=True)
            path = self.root / self.manifest["artifacts"][name]["path"]
            self.models[name] = self.ct.models.CompiledMLModel(
                str(path), compute_units=self.units
            )
        return self.models[name]

    def predict(self, name: str, feeds: dict, output: str) -> np.ndarray:
        expected = self.manifest["artifacts"][name]["inputs"]
        if set(feeds) != set(expected):
            raise ReferenceError(f"Wrong input names for {name}")
        for key, value in feeds.items():
            spec = expected[key]
            if value.dtype != np.dtype(spec["dtype"]) or value.ndim != len(
                spec["shape"]
            ):
                raise ReferenceError(f"Wrong input dtype/rank: {name}.{key}")
            if any(
                isinstance(want, int) and got != want
                for got, want in zip(value.shape, spec["shape"])
            ):
                raise ReferenceError(f"Wrong input shape: {name}.{key} = {value.shape}")
            if not np.isfinite(value).all():
                raise ReferenceError(f"Nonfinite model input: {name}.{key}")
        outputs = self.model(name).predict(feeds)
        if output not in outputs:
            raise ReferenceError(f"Missing model output: {name}.{output}")
        value = np.asarray(outputs[output])
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ReferenceError(
                f"Unexpected model output dtype/values: {name}.{output}"
            )
        return np.ascontiguousarray(value).copy()

    def synthesize(
        self, case: dict, defaults: dict, saved_noise=None, replay_speed=None
    ):
        normalized, ids, mask = encode_text(
            case["text"], case["language"], self.indexer
        )
        voice_path = (
            self.root / self.manifest["artifacts"][f"voice_{case['voice']}"]["path"]
        )
        voice = json.loads(voice_path.read_text(encoding="utf-8"))
        ttl = np.asarray(voice["style_ttl"]["data"], dtype=np.float32).reshape(
            (1, 50, 256)
        )
        dp = np.asarray(voice["style_dp"]["data"], dtype=np.float32).reshape((1, 8, 16))
        raw = self.predict(
            "duration_predictor",
            {"text_ids": ids, "text_mask": mask, "style_dp": dp},
            "duration",
        )
        if raw.shape != (1,):
            raise ReferenceError(f"Unexpected duration shape: {raw.shape}")
        speed = np.float32(case.get("speed", defaults["speed"]))
        target = case.get("target_latent_length")
        if replay_speed is not None:
            speed = np.float32(replay_speed)
        elif target is not None:
            # Aim at the middle of a true-latent-length interval, avoiding
            # Float32 rounding on the sample-count boundary itself.
            target_seconds = np.float32((target - 0.5) * 3072 / 44100)
            speed = np.float32(raw[0] / target_seconds)
        if not np.isfinite(speed) or speed <= 0:
            raise ReferenceError("Invalid effective speaking rate")
        duration = np.maximum(
            np.float32(0.05), raw / np.maximum(speed, np.float32(0.05))
        ).astype(np.float32)
        frames = duration_frames(duration[0])
        length = (frames + 3071) // 3072
        if not 4 <= length <= 512:
            raise ReferenceError(
                f"Vocoder cannot accept true latent length {length}; no padding approximation"
            )
        if target is not None and length != target:
            raise ReferenceError(f"Boundary fixture requested L{target}, got L{length}")
        bucket = select_bucket(length)
        emb = self.predict(
            "text_encoder",
            {"text_ids": ids, "text_mask": mask, "style_ttl": ttl},
            "text_emb",
        )
        if saved_noise is None:
            noise = (
                np.random.Generator(np.random.PCG64(case["seed"]))
                .standard_normal((1, 144, length))
                .astype(np.float32)
            )
        else:
            noise = np.asarray(saved_noise).copy()
            if noise.shape != (1, 144, length) or noise.dtype != np.float32:
                raise ReferenceError(
                    "Saved noise does not match predicted latent shape/dtype"
                )
        latent_mask = np.ones((1, 1, length), dtype=np.float32)
        xt = np.pad(noise, ((0, 0), (0, 0), (0, bucket - length)))
        ve_mask = np.pad(latent_mask, ((0, 0), (0, 0), (0, bucket - length)))
        tensors = {
            "text_ids": ids,
            "text_mask": mask,
            "style_ttl": ttl,
            "style_dp": dp,
            "duration_raw": raw,
            "duration": duration,
            "text_emb": emb,
            "initial_noise": noise,
            "latent_mask": latent_mask,
            "ve_initial_latent": xt.copy(),
            "ve_latent_mask": ve_mask,
        }
        steps = case.get("steps", defaults["steps"])
        current_steps = np.arange(steps, dtype=np.float32)
        total_step = np.array([steps], dtype=np.float32)
        tensors.update(current_steps=current_steps, total_step=total_step)
        for step in range(steps):
            xt = self.predict(
                f"ve_{bucket}",
                {
                    "noisy_latent": xt,
                    "text_emb": emb,
                    "style_ttl": ttl,
                    "latent_mask": ve_mask,
                    "text_mask": mask,
                    "current_step": current_steps[step : step + 1],
                    "total_step": total_step,
                },
                "denoised_latent",
            )
            if xt.shape != (1, 144, bucket):
                raise ReferenceError(f"Unexpected latent output shape: {xt.shape}")
            tensors[f"ve_step_{step:02d}"] = xt.copy()
        vocoder_input = np.ascontiguousarray(xt[:, :, :length])
        wav = self.predict("vocoder", {"latent": vocoder_input}, "wav")
        if wav.shape != (1, length * 3072):
            raise ReferenceError(f"Unexpected waveform shape: {wav.shape}")
        pcm = wav[0, : min(frames, wav.shape[1])].copy()
        tensors.update(vocoder_input=vocoder_input, wav_raw=wav, pcm=pcm)
        metadata = {
            "id": case["id"],
            "text": case["text"],
            "language": case["language"],
            "voice": case["voice"],
            "normalized_text": normalized,
            "tokens": len(normalized),
            "seed": case["seed"],
            "steps": steps,
            "effective_speed": float(speed),
            "predicted_duration_seconds": float(duration[0]),
            "requested_frames": frames,
            "latent_length": length,
            "latent_bucket": bucket,
            "pcm_frames": len(pcm),
            "pcm_duration_seconds": len(pcm) / 44100,
            "purpose": "shape_boundary" if target else "speech_reference",
        }
        return metadata, tensors


def capture(
    manifest: dict,
    root: Path,
    corpus: dict,
    frontend_path: Path,
    output: Path,
    compute_policy: str,
) -> dict:
    verify_bundle(manifest, root)
    frontend_results = check_frontend(
        corpus, read_indexer(manifest, root), frontend_path
    )
    if output.exists() or output.is_symlink():
        raise ReferenceError(f"Reference output already exists: {output}")
    runner = CoreMLReference(manifest, root, compute_policy)
    output.mkdir(parents=True)
    started = time.perf_counter()
    report = {
        "schema_version": 1,
        "bundle_id": manifest["id"],
        "manifest_sha256": canonical_digest(manifest),
        "corpus_sha256": canonical_digest(corpus),
        "compute_policy": compute_policy,
        "environment": environment(),
        "noise_generator": "NumPy PCG64 standard_normal float64 converted to float32; replay uses saved tensor",
        "tool_files": {
            p.relative_to(ROOT).as_posix(): digest(p)
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "frontend": frontend_results,
        "cases": [],
    }
    try:
        for case in corpus["cases"]:
            if "expected_error" in case:
                continue
            print(f"Capturing {case['id']}", flush=True)
            metadata, tensors = runner.synthesize(case, corpus["defaults"])
            npz_path = output / f"{case['id']}.npz"
            wav_path = output / f"{case['id']}.wav"
            np.savez_compressed(npz_path, **tensors)
            write_float_wav(wav_path, tensors["pcm"], 44100)
            metadata["tensors"] = {
                name: tensor_record(value) for name, value in tensors.items()
            }
            metadata["files"] = [
                {"path": p.name, "size_bytes": p.stat().st_size, "sha256": digest(p)}
                for p in (npz_path, wav_path)
            ]
            report["cases"].append(metadata)
            print(
                f"  {metadata['tokens']} tokens; L{metadata['latent_length']} / bucket {metadata['latent_bucket']}; "
                f"{metadata['pcm_frames']} frames",
                flush=True,
            )
        verify_bundle(manifest, root)
        report["capture_wall_seconds"] = time.perf_counter() - started
        report["validation"] = {
            "fixture_integrity": "pending",
            "replay": "not_run",
            "listening": "not_performed",
            "performance": "not_benchmarked",
        }
        write_yaml(output / "reference.yaml", report)
        verify_reference(manifest, corpus, output)
        report["validation"]["fixture_integrity"] = "passed"
        write_yaml(output / "reference.yaml", report)
    except BaseException as error:
        write_yaml(
            output / "capture-incomplete.yaml",
            {"error": str(error), "partial_report": report},
        )
        raise
    return report


def verify_reference(manifest: dict, corpus: dict, root: Path) -> dict:
    if (root / "capture-incomplete.yaml").exists():
        raise ReferenceError("Reference capture is marked incomplete")
    report = load_yaml(root / "reference.yaml")
    if report.get("schema_version") != 1 or report.get(
        "manifest_sha256"
    ) != canonical_digest(manifest):
        raise ReferenceError("Reference uses a different model manifest")
    if report.get("corpus_sha256") != canonical_digest(corpus):
        raise ReferenceError("Reference uses a different corpus")
    expected = {c["id"] for c in corpus["cases"] if "expected_error" not in c}
    requests = {c["id"]: c for c in corpus["cases"]}
    ids = [c["id"] for c in report["cases"]]
    if len(set(ids)) != len(ids) or set(ids) != expected:
        raise ReferenceError("Reference case set is incomplete or duplicated")
    for case in report["cases"]:
        request = requests[case["id"]]
        for key in ("text", "language", "voice", "seed"):
            if case[key] != request[key]:
                raise ReferenceError(
                    f"Case metadata differs from corpus: {case['id']}.{key}"
                )
        if case["steps"] != request.get("steps", corpus["defaults"]["steps"]):
            raise ReferenceError("Reference step count differs from corpus")
        if {f["path"] for f in case["files"]} != {
            f"{case['id']}.npz",
            f"{case['id']}.wav",
        }:
            raise ReferenceError("Expected exactly one NPZ and WAV per reference case")
        for entry in case["files"]:
            path = root / relative_path(entry["path"])
            if (
                path.stat().st_size != entry["size_bytes"]
                or digest(path) != entry["sha256"]
            ):
                raise ReferenceError(f"Changed reference file: {path.name}")
        path = root / f"{case['id']}.npz"
        with np.load(path, allow_pickle=False) as tensors:
            if set(tensors.files) != set(case["tensors"]):
                raise ReferenceError(f"Tensor inventory differs: {case['id']}")
            for name, entry in case["tensors"].items():
                if tensor_record(tensors[name]) != entry:
                    raise ReferenceError(f"Changed tensor: {case['id']}.{name}")
            if tensors["initial_noise"].shape != (1, 144, case["latent_length"]):
                raise ReferenceError("Noise shape does not match latent length")
            if tensors["pcm"].shape != (case["pcm_frames"],):
                raise ReferenceError("PCM frame count mismatch")
            if case["latent_bucket"] != select_bucket(case["latent_length"]):
                raise ReferenceError("Incorrect latent bucket")
            length, bucket = case["latent_length"], case["latent_bucket"]
            if not 4 <= length <= 512:
                raise ReferenceError("Invalid true latent length")
            shapes = {
                "text_ids": (1, 128),
                "text_mask": (1, 1, 128),
                "style_ttl": (1, 50, 256),
                "style_dp": (1, 8, 16),
                "duration_raw": (1,),
                "duration": (1,),
                "text_emb": (1, 256, 128),
                "initial_noise": (1, 144, length),
                "latent_mask": (1, 1, length),
                "ve_initial_latent": (1, 144, bucket),
                "ve_latent_mask": (1, 1, bucket),
                "current_steps": (case["steps"],),
                "total_step": (1,),
                "vocoder_input": (1, 144, length),
                "wav_raw": (1, length * 3072),
                "pcm": (case["pcm_frames"],),
            }
            shapes.update(
                {
                    f"ve_step_{step:02d}": (1, 144, bucket)
                    for step in range(case["steps"])
                }
            )
            if set(tensors.files) != set(shapes):
                raise ReferenceError("Reference must retain every model intermediate")
            for name, shape in shapes.items():
                dtype = np.int32 if name == "text_ids" else np.float32
                if tensors[name].shape != shape or tensors[name].dtype != dtype:
                    raise ReferenceError(f"Wrong reference tensor shape/dtype: {name}")
            frames = duration_frames(tensors["duration"][0])
            if frames != case["requested_frames"] or case["pcm_frames"] != frames:
                raise ReferenceError("Duration and PCM frame counts disagree")
            if (frames + 3071) // 3072 != length:
                raise ReferenceError("Duration and latent length disagree")
            if not np.array_equal(tensors["pcm"], tensors["wav_raw"][0, :frames]):
                raise ReferenceError("PCM does not match the trimmed vocoder output")
            final_latent = tensors[f"ve_step_{case['steps'] - 1:02d}"][:, :, :length]
            if not np.array_equal(tensors["vocoder_input"], final_latent):
                raise ReferenceError(
                    "Vocoder input differs from the true-length final latent"
                )
            if not np.array_equal(
                tensors["ve_initial_latent"][:, :, :length], tensors["initial_noise"]
            ):
                raise ReferenceError("Initial VE state differs from the saved noise")
            if np.any(tensors["ve_initial_latent"][:, :, length:]):
                raise ReferenceError("Initial latent bucket padding is not zero")
            if not np.array_equal(
                tensors["current_steps"], np.arange(case["steps"], dtype=np.float32)
            ):
                raise ReferenceError("Saved step schedule is incorrect")
            if tensors["total_step"][0] != case["steps"]:
                raise ReferenceError("Saved total_step disagrees with the request")
    return report


def compare_tensors(
    expected: dict, actual: dict, atol: float, rtol: float
) -> list[dict]:
    if set(expected) != set(actual):
        raise ReferenceError("Replay tensor inventory changed")
    results = []
    for name in expected:
        left, right = expected[name], actual[name]
        if left.dtype != right.dtype or left.shape != right.shape:
            results.append(
                {"tensor": name, "passed": False, "reason": "shape/dtype mismatch"}
            )
            continue
        # Numerical equality treats +0.0 and -0.0 as equal. Preserve the
        # distinction when reporting bit-exact replay of a captured tensor.
        exact = left.tobytes(order="C") == right.tobytes(order="C")
        passed = (
            exact
            if left.dtype.kind in "iu" or (atol == 0 and rtol == 0)
            else bool(np.allclose(left, right, atol=atol, rtol=rtol))
        )
        error = left.astype(np.float64) - right.astype(np.float64)
        results.append(
            {
                "tensor": name,
                "passed": passed,
                "exact": exact,
                "max_abs": float(np.max(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error * error))),
            }
        )
    return results


def replay(
    manifest: dict,
    bundle: Path,
    corpus: dict,
    reference: Path,
    output: Path,
    atol: float,
    rtol: float,
) -> dict:
    if not math.isfinite(atol) or not math.isfinite(rtol) or min(atol, rtol) < 0:
        raise ReferenceError("Comparison tolerances must be finite and nonnegative")
    if output.exists():
        raise ReferenceError(f"Replay report already exists: {output}")
    verify_bundle(manifest, bundle)
    saved = verify_reference(manifest, corpus, reference)
    runner = CoreMLReference(manifest, bundle, saved["compute_policy"])
    lookup = {c["id"]: c for c in corpus["cases"]}
    result = {
        "schema_version": 1,
        "reference_sha256": digest(reference / "reference.yaml"),
        "environment": environment(),
        "tool_files": {
            p.relative_to(ROOT).as_posix(): digest(p)
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "atol": atol,
        "rtol": rtol,
        "cases": [],
    }
    for case in saved["cases"]:
        print(f"Replaying {case['id']} with saved noise", flush=True)
        with np.load(reference / f"{case['id']}.npz", allow_pickle=False) as arrays:
            expected = {name: arrays[name] for name in arrays.files}
        metadata, actual = runner.synthesize(
            lookup[case["id"]],
            corpus["defaults"],
            expected["initial_noise"],
            case["effective_speed"],
        )
        checks = compare_tensors(expected, actual, atol, rtol)
        metadata_match = all(
            metadata[key] == case[key]
            for key in (
                "normalized_text",
                "tokens",
                "steps",
                "effective_speed",
                "requested_frames",
                "latent_length",
                "latent_bucket",
                "pcm_frames",
            )
        )
        result["cases"].append(
            {
                "id": case["id"],
                "metadata_match": metadata_match,
                "passed": metadata_match and all(c["passed"] for c in checks),
                "tensors": checks,
            }
        )
    verify_bundle(manifest, bundle)
    result["passed"] = all(c["passed"] for c in result["cases"])
    output.parent.mkdir(parents=True, exist_ok=True)
    write_yaml(output, result)
    if not result["passed"]:
        raise ReferenceError(f"Reference replay failed; see {output}")
    return result


def compact_summary(report: dict) -> dict:
    """A reviewable record without machine paths or large tensor payloads."""
    fields = (
        "id",
        "language",
        "voice",
        "tokens",
        "steps",
        "effective_speed",
        "latent_length",
        "latent_bucket",
        "pcm_frames",
        "pcm_duration_seconds",
        "purpose",
    )
    return {
        "schema_version": 1,
        "bundle_id": report["bundle_id"],
        "manifest_sha256": report["manifest_sha256"],
        "corpus_sha256": report["corpus_sha256"],
        "compute_policy": report["compute_policy"],
        "environment": report["environment"],
        "tool_files": report["tool_files"],
        "cases": [
            {
                **{key: c[key] for key in fields},
                "noise_sha256": c["tensors"]["initial_noise"]["sha256"],
                "pcm_sha256": c["tensors"]["pcm"]["sha256"],
            }
            for c in report["cases"]
        ],
        "frontend": report["frontend"],
        "validation": report["validation"],
    }
