"""Replay verified saved inputs through MLSR's persistent Rust runtime.

YAML remains the manifest/report format. JSON lines and raw little-endian blobs
are only a development subprocess transport. No coremltools import is needed.
"""

from __future__ import annotations

import math
from pathlib import Path
import platform
import queue
import subprocess
import tempfile
import threading

import numpy as np

from .bundle import (
    ROOT,
    ReferenceError,
    canonical_digest,
    digest,
    verify_bundle,
    write_yaml,
)
from .reference import compare_tensors, verify_reference

INPUTS = (
    "text_ids",
    "text_mask",
    "style_ttl",
    "style_dp",
    "ve_initial_latent",
    "ve_latent_mask",
    "current_steps",
    "total_step",
)


def output_names(steps):
    return {"duration_raw", "text_emb", "vocoder_input", "wav_raw", "pcm"} | {
        f"ve_step_{step:02d}" for step in range(steps)
    }


def decode_outputs(descriptors, raw, steps):
    if set(descriptors) != output_names(steps):
        raise ReferenceError("Rust runtime output inventory changed")
    actual = {}
    for name, descriptor in descriptors.items():
        shape = descriptor["shape"]
        start, length = descriptor["offset"], descriptor["length"]
        if (
            descriptor["dtype"] != "float32"
            or not isinstance(shape, list)
            or not shape
            or any(type(dim) is not int or dim < 1 for dim in shape)
            or type(start) is not int
            or start < 0
            or type(length) is not int
            or length != math.prod(shape) * 4
            or start + length > len(raw)
        ):
            raise ReferenceError(f"Invalid runtime tensor descriptor: {name}")
        actual[name] = np.frombuffer(
            raw, dtype="<f4", count=length // 4, offset=start
        ).reshape(shape)
    return actual


class Worker:
    """One process for all cases; bounded waits and captured native diagnostics."""

    def __init__(self, binary, log, timeout):
        self.timeout = timeout
        self.process = subprocess.Popen(
            [str(binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            bufsize=1,
        )
        self.lines = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                self.lines.put(line)
        finally:
            self.lines.put(None)

    def call(self, job):
        import json

        self.process.stdin.write(json.dumps(job) + "\n")
        self.process.stdin.flush()
        try:
            line = self.lines.get(timeout=self.timeout)
        except queue.Empty as error:
            raise ReferenceError(
                "Rust runtime request timed out; see native log"
            ) from error
        if line is None:
            raise ReferenceError(
                f"Rust runtime exited ({self.process.poll()}); see native log"
            )
        reply = json.loads(line)
        if "error" in reply:
            raise ReferenceError(f"Rust runtime: {reply['error']}")
        return reply["ok"]

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.reader.join(timeout=5)
        self.process.stdin.close()
        self.process.stdout.close()


def replay_rust(
    manifest,
    bundle,
    corpus,
    reference,
    output,
    binary,
    passes=2,
    output_backings=False,
    atol=0.0,
    rtol=0.0,
    timeout=180.0,
):
    if platform.system() != "Darwin":
        raise ReferenceError("Rust Core ML replay requires macOS")
    if type(passes) is not int or passes < 1:
        raise ReferenceError("Pass count must be positive")
    if not all(math.isfinite(v) and v >= 0 for v in (atol, rtol)):
        raise ReferenceError("Tolerances must be finite and nonnegative")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ReferenceError("Request timeout must be positive and finite")
    binary = binary.resolve(strict=True)
    output = output.resolve()
    native_log = output.with_suffix(".native.log")
    if output.exists() or native_log.exists():
        raise ReferenceError("Replay output or native log already exists")
    verify_bundle(manifest, bundle)
    saved = verify_reference(manifest, corpus, reference)
    reference_hash = digest(reference / "reference.yaml")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "kind": "rust_runtime_replay",
        "passed": False,
        "scope": "saved runtime tensors; frontend and duration/bucket planning are supplied by fixtures",
        "manifest_sha256": canonical_digest(manifest),
        "reference_sha256": reference_hash,
        "corpus_sha256": canonical_digest(corpus),
        "compute_policy": saved["compute_policy"],
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "binary_sha256": digest(binary),
        "atol": atol,
        "rtol": rtol,
        "passes": passes,
        "output_backings": output_backings,
        "source_files": {
            p.relative_to(ROOT).as_posix(): digest(p)
            for p in sorted(
                {
                    ROOT / "Cargo.lock",
                    ROOT / "Cargo.toml",
                    ROOT / "crates/mlsr/Cargo.toml",
                    ROOT / "crates/mlsr/examples/coreml_replay.rs",
                    Path(__file__),
                    *list((ROOT / "crates/mlsr/src/runtime").rglob("*.rs")),
                }
            )
        },
        "cases": [],
        "performance": "not_benchmarked",
        "ane_residency": "not_qualified",
    }
    # Reserve the report exclusively before any later rewrite, including failures.
    # Concurrent runs choosing the same destination cannot overwrite each other.
    with output.open("x"):
        pass
    write_yaml(output, result)
    worker = None
    try:
        with native_log.open("x") as log, tempfile.TemporaryDirectory(
            prefix="mlsr-rust-replay-"
        ) as temporary:
            temporary = Path(temporary)
            worker = Worker(binary, log, timeout)
            print("Loading six models in one persistent Rust session", flush=True)
            loaded = worker.call(
                {
                    "command": "load",
                    "root": str(bundle.resolve()),
                    "models": {
                        name: item["path"]
                        for name, item in manifest["artifacts"].items()
                        if item["kind"] == "coreml_compiled"
                    },
                    "compute_policy": saved["compute_policy"],
                    "output_backings": output_backings,
                }
            )
            expected_live = 6 if output_backings else 0
            if loaded["stats"] != {
                "model_loads": 6,
                "live_tensors": expected_live,
                "predictions": 0,
            }:
                raise ReferenceError("Unexpected initial runtime resource counts")
            expected_predictions = 0
            for repetition in range(passes):
                for case in saved["cases"]:
                    with np.load(
                        reference / f"{case['id']}.npz", allow_pickle=False
                    ) as arrays:
                        data = bytearray()
                        descriptors = {}
                        for name in INPUTS:
                            arr = np.ascontiguousarray(arrays[name])
                            raw = arr.tobytes()
                            descriptors[name] = {
                                "shape": list(arr.shape),
                                "dtype": str(arr.dtype),
                                "offset": len(data),
                                "length": len(raw),
                            }
                            data.extend(raw)
                        input_file, output_file = (
                            temporary / "input.bin",
                            temporary / "output.bin",
                        )
                        input_file.write_bytes(data)
                        reply = worker.call(
                            {
                                "command": "run",
                                "input": str(input_file),
                                "output": str(output_file),
                                "tensors": descriptors,
                                "bucket": case["latent_bucket"],
                                "latent_length": case["latent_length"],
                                "steps": case["steps"],
                                "pcm_frames": case["pcm_frames"],
                            }
                        )
                        actual = decode_outputs(
                            reply["tensors"], output_file.read_bytes(), case["steps"]
                        )
                        expected = {
                            name: arrays[name]
                            for name in sorted(output_names(case["steps"]))
                        }
                        checks = compare_tensors(expected, actual, atol, rtol)
                        expected_predictions += case["steps"] + 3
                        stats_ok = reply["stats"] == {
                            "model_loads": 6,
                            "live_tensors": expected_live,
                            "predictions": expected_predictions,
                        }
                        reuse = reply["ve_backing_reused"]
                        reuse_valid = len(reuse) == case["steps"] and all(
                            type(v) is bool for v in reuse
                        )
                        reuse_valid = reuse_valid and (
                            output_backings or not any(reuse)
                        )
                        entry = {
                            "id": case["id"],
                            "pass": repetition + 1,
                            "bucket": case["latent_bucket"],
                            "latent_length": case["latent_length"],
                            "passed": stats_ok
                            and reuse_valid
                            and all(c["passed"] for c in checks),
                            "stats": reply["stats"],
                            "ve_backing_reused": reuse,
                            "tensors": checks,
                        }
                        result["cases"].append(entry)
                        write_yaml(
                            output, result
                        )  # An interrupted run stays non-passing.
                        print(
                            f"{case['id']} pass {repetition + 1}: {'passed' if entry['passed'] else 'FAILED'}",
                            flush=True,
                        )
            if worker.call({"command": "close"}) != {"closed": True}:
                raise ReferenceError("Rust runtime did not acknowledge shutdown")
            worker.process.stdin.close()
            result["worker_exit"] = worker.process.wait(timeout=10)
            verify_bundle(manifest, bundle)
            verify_reference(manifest, corpus, reference)
            if digest(reference / "reference.yaml") != reference_hash:
                raise ReferenceError("Reference metadata changed during replay")
            result["passed"] = result["worker_exit"] == 0 and all(
                c["passed"] for c in result["cases"]
            )
            result["tensor_comparisons"] = sum(
                len(c["tensors"]) for c in result["cases"]
            )
            result["all_tensors_exact"] = all(
                t["exact"] for c in result["cases"] for t in c["tensors"]
            )
            result["backings_reused"] = sum(
                sum(c["ve_backing_reused"]) for c in result["cases"]
            )
    except BaseException as error:
        result["passed"] = False
        result["error"] = str(error)
        raise
    finally:
        if worker is not None:
            worker.stop()
        write_yaml(output, result)
    if not result["passed"]:
        raise ReferenceError(f"Rust runtime replay failed; see {output}")
    return result
