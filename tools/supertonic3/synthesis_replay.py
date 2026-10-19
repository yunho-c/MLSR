"""Qualify Rust synthesis from original text/options and actual saved noise."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import platform
import tempfile

import numpy as np

from .bundle import (
    DEFAULT_CORPUS,
    DEFAULT_MANIFEST,
    ROOT,
    ReferenceError,
    canonical_digest,
    digest,
    load_manifest,
    verify_bundle,
    write_yaml,
)
from .reference import compare_tensors, load_corpus, verify_reference
from .rust_runtime import Worker, decode_tensor_map

METADATA = (
    "normalized_text",
    "tokens",
    "requested_frames",
    "latent_length",
    "latent_bucket",
    "pcm_frames",
)


def check_metadata(actual, case):
    return actual == {key: case[key] for key in METADATA}


def evidence_complete(report, saved, corpus, seeded_ids):
    """An empty, partial, duplicated, or failed run cannot qualify synthesis."""
    positive = {c["id"] for c in saved["cases"]}
    negative = {c["id"] for c in corpus["cases"] if "expected_error" in c}
    wanted = {
        (name, repetition + 1)
        for name in positive
        for repetition in range(report["passes"])
    }
    actual = [(c["id"], c["pass"]) for c in report["cases"]]
    rejected = [c["id"] for c in report["rejections"]]
    seeded = [c["id"] for c in report["seeded_checks"]]
    return (
        bool(positive)
        and bool(seeded_ids)
        and len(actual) == len(wanted)
        and set(actual) == wanted
        and len(rejected) == len(negative)
        and set(rejected) == negative
        and len(seeded) == len(seeded_ids)
        and set(seeded) == seeded_ids
        and report["worker_exit"] == 0
        and all(
            c["tensors"] and all(t["passed"] and t["exact"] for t in c["tensors"])
            for c in report["cases"]
        )
        and all(
            item["passed"]
            for group in ("cases", "rejections", "seeded_checks")
            for item in report[group]
        )
    )


def replay(
    manifest_path,
    bundle,
    corpus_path,
    reference,
    output,
    binary,
    passes=2,
    output_backings=False,
    timeout=180.0,
):
    if platform.system() != "Darwin":
        raise ReferenceError("Core ML synthesis qualification requires macOS")
    if type(passes) is not int or passes < 1:
        raise ReferenceError("Pass count must be positive")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ReferenceError("Timeout must be positive and finite")
    manifest = load_manifest(manifest_path)
    corpus = load_corpus(corpus_path)
    binary = binary.resolve(strict=True)
    output = output.resolve()
    native_log = output.with_suffix(".native.log")
    if output.exists() or native_log.exists():
        raise ReferenceError("Output report or native log already exists")
    verify_bundle(manifest, bundle)
    saved = verify_reference(manifest, corpus, reference)
    seeded_ids = {"en_m1_basic", "en_f1_basic", "ko_m1_basic", "ko_f1_basic"}
    if not seeded_ids.issubset({case["id"] for case in saved["cases"]}):
        raise ReferenceError(
            "The qualification corpus must include basic English/Korean cases for both voices"
        )
    reference_hash = digest(reference / "reference.yaml")
    sources = {
        ROOT / "Cargo.toml",
        ROOT / "Cargo.lock",
        ROOT / "crates/mlsr/Cargo.toml",
        ROOT / "crates/mlsr/examples/coreml_synthesis_replay.rs",
        ROOT / "crates/mlsr/examples/support/synthesis_replay.rs",
        *list((ROOT / "crates/mlsr/src").rglob("*.rs")),
        *list(Path(__file__).parent.glob("*.py")),
    }
    report = {
        "schema_version": 1,
        "kind": "rust_synthesis_replay",
        "passed": False,
        "scope": "Original text and recorded request settings through Rust; only initial noise is supplied by the reference",
        "manifest_sha256": canonical_digest(manifest),
        "manifest_file_sha256": digest(manifest_path),
        "corpus_sha256": canonical_digest(corpus),
        "reference_sha256": reference_hash,
        "compute_policy": saved["compute_policy"],
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "binary_sha256": digest(binary),
        "source_files": {
            p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(sources)
        },
        "passes": passes,
        "output_backings": output_backings,
        "atol": 0.0,
        "rtol": 0.0,
        "cases": [],
        "rejections": [],
        "seeded_checks": [],
        "performance": "not_benchmarked",
        "ane_residency": "not_qualified",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x"):
        pass
    write_yaml(output, report)
    worker = None
    try:
        with native_log.open("x") as log, tempfile.TemporaryDirectory(
            prefix="mlsr-synthesis-replay-"
        ) as temporary:
            temporary = Path(temporary)
            worker = Worker(binary, log, timeout)
            loaded = worker.call(
                {
                    "command": "load",
                    "manifest": str(manifest_path.resolve()),
                    "bundle": str(bundle.resolve()),
                    "compute_policy": saved["compute_policy"],
                    "output_backings": output_backings,
                }
            )
            predictions = 0
            live = (
                12 if output_backings else 6
            )  # Four styles, two scalars, optional six backings.

            def stats_ok(reply):
                return reply["stats"] == {
                    "model_loads": 6,
                    "live_tensors": live,
                    "predictions": predictions,
                }

            if not stats_ok(loaded):
                raise ReferenceError("Unexpected initial model/tensor counts")

            def job(case, diagnostic=True, noise=None):
                return {
                    "command": "run",
                    "text": case["text"],
                    "language": case["language"],
                    "voice": case["voice"],
                    "steps": case["steps"],
                    "speed": case["effective_speed"],
                    "seed": case["seed"],
                    "noise": noise,
                    "diagnostic": diagnostic,
                    "output": str(temporary / "output.bin"),
                }

            def decode(reply, dtypes):
                file = temporary / "output.bin"
                if file.stat().st_size > 64 * 1024 * 1024:
                    raise ReferenceError("Unexpected tensor output size")
                return decode_tensor_map(reply["tensors"], file.read_bytes(), dtypes)

            for repetition in range(passes):
                for case in saved["cases"]:
                    with np.load(
                        reference / f"{case['id']}.npz", allow_pickle=False
                    ) as arrays:
                        noise = arrays["initial_noise"]
                        noise_file = temporary / "noise.bin"
                        noise_file.write_bytes(noise.tobytes(order="C"))
                        reply = worker.call(
                            job(
                                case,
                                noise={
                                    "path": str(noise_file),
                                    "shape": list(noise.shape),
                                },
                            )
                        )
                        predictions += case["steps"] + 3
                        actual = decode(
                            reply,
                            {name: str(arrays[name].dtype) for name in arrays.files},
                        )
                        checks = compare_tensors(
                            {name: arrays[name] for name in arrays.files},
                            actual,
                            0.0,
                            0.0,
                        )
                    reuse = reply["ve_backing_reused"]
                    reuse_ok = len(reuse) == case["steps"] and all(
                        type(v) is bool for v in reuse
                    )
                    reuse_ok = reuse_ok and (output_backings or not any(reuse))
                    entry = {
                        "id": case["id"],
                        "pass": repetition + 1,
                        "passed": check_metadata(reply["metadata"], case)
                        and stats_ok(reply)
                        and reuse_ok
                        and all(c["passed"] for c in checks),
                        "metadata": reply["metadata"],
                        "stats": reply["stats"],
                        "ve_backing_reused": reuse,
                        "tensors": checks,
                    }
                    report["cases"].append(entry)
                    write_yaml(output, report)
                    print(
                        f"{case['id']} pass {repetition + 1}: {'passed' if entry['passed'] else 'FAILED'}",
                        flush=True,
                    )

            for case in corpus["cases"]:
                if "expected_error" not in case:
                    continue
                request = {
                    **case,
                    **{
                        "steps": corpus["defaults"]["steps"],
                        "effective_speed": corpus["defaults"]["speed"],
                    },
                }
                rejected = False
                try:
                    worker.call(job(request))
                except ReferenceError as error:
                    rejected = str(error).startswith(
                        f"Rust runtime: {case['expected_error']}:"
                    )
                passed = rejected and stats_ok(worker.call({"command": "stats"}))
                report["rejections"].append(
                    {
                        "id": case["id"],
                        "expected_error": case["expected_error"],
                        "passed": passed,
                    }
                )

            # Exercise the ordinary no-readback path against diagnostic execution
            # with Rust-generated noise, covering both voices and both languages.
            for case in [c for c in saved["cases"] if c["id"] in seeded_ids]:
                normal = worker.call(job(case, diagnostic=False))
                predictions += case["steps"] + 3
                ordinary = decode(normal, {"pcm": "float32"})["pcm"].copy()
                normal_ok = stats_ok(normal) and check_metadata(
                    normal["metadata"], case
                )
                diagnostic = worker.call(job(case))
                predictions += case["steps"] + 3
                with np.load(
                    reference / f"{case['id']}.npz", allow_pickle=False
                ) as arrays:
                    traced = decode(
                        diagnostic,
                        {name: str(arrays[name].dtype) for name in arrays.files},
                    )
                exact = ordinary.tobytes() == traced["pcm"].tobytes()
                report["seeded_checks"].append(
                    {
                        "id": case["id"],
                        "seed": case["seed"],
                        "passed": normal_ok
                        and stats_ok(diagnostic)
                        and check_metadata(diagnostic["metadata"], case)
                        and exact,
                        "normal_and_diagnostic_pcm_exact": exact,
                        "noise_sha256": hashlib.sha256(
                            traced["initial_noise"].tobytes()
                        ).hexdigest(),
                        "pcm_sha256": hashlib.sha256(ordinary.tobytes()).hexdigest(),
                    }
                )
            report["final_stats"] = worker.call({"command": "stats"})["stats"]
            if worker.call({"command": "close"}) != {"closed": True}:
                raise ReferenceError("Rust worker did not acknowledge shutdown")
            worker.process.stdin.close()
            report["worker_exit"] = worker.process.wait(timeout=10)
            verify_bundle(manifest, bundle)
            verify_reference(manifest, corpus, reference)
            if (
                digest(reference / "reference.yaml") != reference_hash
                or digest(manifest_path) != report["manifest_file_sha256"]
            ):
                raise ReferenceError("Manifest/reference changed during qualification")
            report["tensor_comparisons"] = sum(
                len(c["tensors"]) for c in report["cases"]
            )
            report["all_tensors_exact"] = all(
                t["exact"] for c in report["cases"] for t in c["tensors"]
            )
            report["backings_reused"] = sum(
                sum(c["ve_backing_reused"]) for c in report["cases"]
            )
            report["passed"] = evidence_complete(report, saved, corpus, seeded_ids)
    except BaseException as error:
        report["passed"] = False
        report["error"] = str(error)
        raise
    finally:
        try:
            if worker is not None:
                worker.stop()
        except BaseException as error:
            report["passed"] = False
            report["cleanup_error"] = str(error)
            raise
        finally:
            write_yaml(output, report)
    if not report["passed"]:
        raise ReferenceError(f"Synthesis qualification failed; see {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--binary",
        type=Path,
        default=ROOT / "target/release/examples/coreml_synthesis_replay",
    )
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--output-backings", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    report = replay(
        args.manifest,
        args.bundle,
        args.corpus,
        args.reference,
        args.output,
        args.binary,
        args.passes,
        args.output_backings,
        args.timeout,
    )
    print(
        f"Passed {report['tensor_comparisons']} exact tensor comparisons, {len(report['rejections'])} rejections, {len(report['seeded_checks'])} seeded-path comparisons"
    )


if __name__ == "__main__":
    main()
