"""Compare selectable Rust graph variants with saved inputs and real baselines."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path
import tempfile

import numpy as np

from .bundle import (
    DEFAULT_MANIFEST,
    ROOT,
    ReferenceError,
    digest,
    load_manifest,
    verify_bundle,
    write_yaml,
)
from .reference import compare_tensors, load_corpus, verify_reference, write_float_wav
from .rust_runtime import Worker, decode_tensor_map
from .synthesis_replay import METADATA
from .variants import VARIANT_MANIFEST, verify_variants

VARIANTS = ("baseline", "fused8", "fused8_fixed_vocoder")


def route(variant, case):
    length = case["latent_length"]
    return {
        "fused_ve": variant != "baseline"
        and case["steps"] == 8
        and case["latent_bucket"] == 128,
        "fixed_vocoder": (
            length if variant == "fused8_fixed_vocoder" and length in (30, 32) else None
        ),
    }


def replay(bundle, reference, output, binary):
    output.mkdir(parents=True, exist_ok=False)
    manifest = load_manifest()
    verify_bundle(manifest, bundle)
    verify_variants(bundle)
    saved = verify_reference(manifest, load_corpus(), reference)
    sources = [
        ROOT / "Cargo.toml",
        ROOT / "Cargo.lock",
        DEFAULT_MANIFEST,
        VARIANT_MANIFEST,
        ROOT / "crates/mlsr/examples/coreml_synthesis_replay.rs",
        *sorted((ROOT / "crates/mlsr/src").rglob("*.rs")),
        *sorted(Path(__file__).parent.glob("*.py")),
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    report = {
        "schema_version": 1,
        "kind": "rust_execution_variant_replay",
        "passed": False,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "binary_sha256": digest(binary),
        "source_files": hashes,
        "reference_sha256": digest(reference / "reference.yaml"),
        "compute_policy": "cpu_and_neural_engine",
        "passes": 2,
        "initialization": [],
        "cases": [],
        "rejections": [],
        "workers": [],
        "timing_scope": "Diagnostic replay timings are not performance results",
        "default": "baseline; this report qualifies numerical behavior only",
    }
    path = output / "report.yaml"
    write_yaml(path, report)
    cases = [
        dict(c, source="reference", tensors_path=reference / f"{c['id']}.npz")
        for c in saved["cases"]
    ]
    inputs = output / "inputs"
    inputs.mkdir()
    # New exact-vocoder probes use original corpus text/voices and calibrated
    # rates, with actual saved noise. They are routing probes, not prosody claims.
    basics = [
        c
        for c in saved["cases"]
        if c["id"] in ("en_m1_basic", "en_f1_basic", "ko_m1_basic", "ko_f1_basic")
    ]
    for case in basics:
        with np.load(reference / f"{case['id']}.npz", allow_pickle=False) as arrays:
            raw_duration = arrays["duration_raw"][0]
        for length in (30, 31, 32):
            new = dict(case)
            new.update(
                id=f"{case['id']}_L{length}",
                latent_length=length,
                latent_bucket=128,
                source="calibrated",
            )
            new["effective_speed"] = float(
                np.float32(raw_duration / np.float32((length - 0.5) * 3072 / 44100))
            )
            new["tensors_path"] = inputs / f"{new['id']}.npz"
            noise = (
                np.random.default_rng(case["seed"])
                .standard_normal((1, 144, length))
                .astype(np.float32)
            )
            np.savez(new["tensors_path"], initial_noise=noise)
            cases.append(new)
    for steps in (1, 7, 9):
        new = dict(
            cases[0], id=f"fallback_steps_{steps}", steps=steps, source="step_fallback"
        )
        new["tensors_path"] = inputs / f"{new['id']}.npz"
        with np.load(cases[0]["tensors_path"], allow_pickle=False) as arrays:
            np.savez(new["tensors_path"], initial_noise=arrays["initial_noise"])
        cases.append(new)
    worker = None
    try:
        with tempfile.TemporaryDirectory(prefix="mlsr-variant-replay-") as temp:
            temp = Path(temp)
            for variant in VARIANTS:
                with (output / f"{variant}.native.log").open("x") as log:
                    worker = Worker(binary, log, timeout=300)
                    loaded = worker.call(
                        {
                            "command": "load",
                            "manifest": str(DEFAULT_MANIFEST),
                            "bundle": str(bundle),
                            "compute_policy": "cpu_and_neural_engine",
                            "output_backings": True,
                            "execution_variant": variant,
                        }
                    )
                    expected_loads = {
                        "baseline": 6,
                        "fused8": 7,
                        "fused8_fixed_vocoder": 9,
                    }[variant]
                    report["initialization"].append({"variant": variant, **loaded})
                    predictions = 0
                    for repetition in range(2):
                        for case in cases:
                            with np.load(
                                case["tensors_path"], allow_pickle=False
                            ) as arrays:
                                expected = {
                                    name: arrays[name].copy() for name in arrays.files
                                }
                            noise = expected["initial_noise"]
                            noise_path = temp / "noise.f32"
                            noise_path.write_bytes(noise.tobytes())
                            reply = worker.call(
                                {
                                    "command": "run",
                                    "text": case["text"],
                                    "language": case["language"],
                                    "voice": case["voice"],
                                    "steps": case["steps"],
                                    "speed": case["effective_speed"],
                                    "seed": case["seed"],
                                    "diagnostic": True,
                                    "noise": {
                                        "path": str(noise_path),
                                        "shape": list(noise.shape),
                                    },
                                    "output": str(temp / "tensors.bin"),
                                }
                            )
                            expected_route = route(variant, case)
                            if (
                                variant == "baseline"
                                and case["source"] != "reference"
                                and repetition == 0
                            ):
                                dtypes = {
                                    name: desc["dtype"]
                                    for name, desc in reply["tensors"].items()
                                }
                                # Independently constrain the stage inventory before accepting
                                # a fresh reference for exact-length/step-routing probes.
                                names = set(expected) | {
                                    "text_ids",
                                    "text_mask",
                                    "style_ttl",
                                    "style_dp",
                                    "duration_raw",
                                    "duration",
                                    "latent_mask",
                                    "ve_initial_latent",
                                    "ve_latent_mask",
                                    "text_emb",
                                    "current_steps",
                                    "total_step",
                                    "vocoder_input",
                                    "wav_raw",
                                    "pcm",
                                }
                                names |= {
                                    f"ve_step_{i:02}" for i in range(case["steps"])
                                }
                                if set(dtypes) != names:
                                    raise ReferenceError(
                                        "Baseline probe inventory changed"
                                    )
                                dtypes = {
                                    name: "int32" if name == "text_ids" else "float32"
                                    for name in names
                                }
                                actual = decode_tensor_map(
                                    reply["tensors"],
                                    (temp / "tensors.bin").read_bytes(),
                                    dtypes,
                                )
                                expected = actual
                                if (
                                    actual["initial_noise"].tobytes() != noise.tobytes()
                                    or reply["metadata"]["latent_length"]
                                    != case["latent_length"]
                                ):
                                    raise ReferenceError(
                                        "Calibrated length or saved noise changed"
                                    )
                                np.savez(case["tensors_path"], **actual)
                                case.update(reply["metadata"])
                            else:
                                if expected_route["fused_ve"]:
                                    expected = {
                                        name: value
                                        for name, value in expected.items()
                                        if not name.startswith("ve_step_")
                                        or name == "ve_step_07"
                                    }
                                actual = decode_tensor_map(
                                    reply["tensors"],
                                    (temp / "tensors.bin").read_bytes(),
                                    {
                                        name: str(value.dtype)
                                        for name, value in expected.items()
                                    },
                                )
                            checks = compare_tensors(expected, actual, 0.0, 0.0)
                            predictions += 3 + (
                                1 if expected_route["fused_ve"] else case["steps"]
                            )
                            expected_stats = {
                                "model_loads": expected_loads,
                                "live_tensors": 12,
                                "predictions": predictions,
                            }
                            contract_passed = (
                                reply["route"] == expected_route
                                and reply["metadata"] == {k: case[k] for k in METADATA}
                                and reply["stats"] == expected_stats
                            )
                            pre_vocoder_exact = all(
                                c["passed"]
                                for c in checks
                                if c["tensor"] not in ("pcm", "wav_raw")
                            )
                            passed = contract_passed and all(
                                c["passed"] for c in checks
                            )
                            pcm = actual["pcm"]
                            wav = output / f"{variant}-{case['id']}.wav"
                            if repetition == 0:
                                write_float_wav(wav, pcm, 44100)
                            item = {
                                "variant": variant,
                                "id": case["id"],
                                "pass": repetition + 1,
                                "passed": passed,
                                "contract_passed": contract_passed,
                                "pre_vocoder_exact": pre_vocoder_exact,
                                "source": case["source"],
                                "request": {
                                    k: case[k]
                                    for k in (
                                        "text",
                                        "language",
                                        "voice",
                                        "steps",
                                        "seed",
                                        "effective_speed",
                                    )
                                },
                                "route": reply["route"],
                                "metadata": reply["metadata"],
                                "stats": reply["stats"],
                                "tensors": checks,
                                "wav_sha256": digest(wav),
                            }
                            report["cases"].append(item)
                            write_yaml(path, report)
                            print(
                                f"{variant} {case['id']} pass {repetition+1}: {'passed' if passed else 'FAILED'}",
                                flush=True,
                            )
                    for case in load_corpus()["cases"]:
                        if "expected_error" not in case:
                            continue
                        rejected = False
                        try:
                            worker.call(
                                {
                                    "command": "run",
                                    "text": case["text"],
                                    "language": case["language"],
                                    "voice": case["voice"],
                                    "steps": 8,
                                    "speed": 1.05,
                                    "seed": case["seed"],
                                    "diagnostic": False,
                                    "noise": None,
                                    "output": str(temp / "reject.bin"),
                                }
                            )
                        except ReferenceError as error:
                            rejected = str(error).startswith(
                                f"Rust runtime: {case['expected_error']}:"
                            )
                        stats = worker.call({"command": "stats"})["stats"]
                        report["rejections"].append(
                            {
                                "variant": variant,
                                "id": case["id"],
                                "passed": rejected and stats == expected_stats,
                            }
                        )
                    worker.call({"command": "close"})
                    worker.process.stdin.close()
                    code = worker.process.wait(timeout=30)
                    report["workers"].append({"variant": variant, "exit": code})
                    worker.stop()
                    worker = None
        verify_bundle(manifest, bundle)
        verify_variants(bundle)
        verify_reference(manifest, load_corpus(), reference)
        if (
            hashes != {str(p.relative_to(ROOT)): digest(p) for p in sources}
            or digest(binary) != report["binary_sha256"]
            or digest(reference / "reference.yaml") != report["reference_sha256"]
        ):
            raise ReferenceError("Sources or binary changed during qualification")
        report["probe_inputs"] = {
            p.name: digest(p) for p in sorted(inputs.glob("*.npz"))
        }
        report["variant_results"] = {
            variant: {
                "bit_exact": all(
                    c["passed"] for c in report["cases"] if c["variant"] == variant
                ),
                "status": (
                    "experimental; PCM differs; perceptual quality unqualified"
                    if variant == "fused8_fixed_vocoder"
                    else "required exact parity"
                ),
            }
            for variant in VARIANTS
        }
        report["complete"] = (
            len(report["cases"]) == len(cases) * 2 * 3
            and all(
                c["contract_passed"] and c["pre_vocoder_exact"] for c in report["cases"]
            )
            and len(report["rejections"]) == 15
            and all(c["passed"] for c in report["rejections"])
            and len(report["workers"]) == 3
            and all(w["exit"] == 0 for w in report["workers"])
        )
        report["passed"] = report["complete"] and all(
            report["variant_results"][v]["bit_exact"] for v in ("baseline", "fused8")
        )
    finally:
        if worker is not None:
            worker.stop()
        write_yaml(path, report)
    if not report["passed"]:
        raise ReferenceError(f"Variant qualification failed; inspect {path}")
    print(
        f"Baseline/fused VE exact parity passed across 29 cases and two passes; fixed-vocoder differences remain experimental: {path}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--binary",
        type=Path,
        default=ROOT / "target/release/examples/coreml_synthesis_replay",
    )
    args = parser.parse_args()
    replay(
        args.bundle.resolve(),
        args.reference.resolve(),
        args.output.resolve(),
        args.binary.resolve(),
    )


if __name__ == "__main__":
    main()
