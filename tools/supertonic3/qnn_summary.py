"""Revalidate retained QNN replay bytes and summarize external device evidence."""

import argparse
import json
from pathlib import Path

import numpy as np

from .bundle import load_yaml, write_yaml
from .qnn_replay import decode, metrics, sha, spectral


def distribution(values):
    return {
        "count": len(values),
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.percentile(values, 90)),
        "max_ms": float(max(values)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "replay",
        "jobs",
        "profile",
        "bindings",
        "environment",
        "compile-options",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    replay = load_yaml(args.replay / "report.yaml")
    profile = load_yaml(args.profile / "report.yaml")
    jobs = json.loads(args.jobs.read_text(encoding="utf-8-sig"))
    bindings = json.loads(
        (args.bindings / "report.json").read_text(encoding="utf-8-sig")
    )
    environment = json.loads(args.environment.read_text(encoding="utf-8-sig"))
    compile_options = json.loads(args.compile_options.read_text(encoding="utf-8-sig"))
    assert (
        replay["manifest_sha256"]
        == profile["manifest_sha256"]
        == bindings["manifest_sha256"]
        == jobs["provenance"]["manifest_sha256"]
    )
    assert (
        replay["bridge_sha256"]
        == bindings["bridge_sha256"]
        == jobs["provenance"]["bridge_sha256"]
    )
    assert profile["replay_report_sha256"] == sha(args.replay / "report.yaml")
    assert profile["compile_options_sha256"] == sha(args.compile_options)
    comparisons = []
    raw_runs = 0
    for case in replay["cases"]:
        if "canonical" not in case:
            assert len(case["runs"]) == 4 and all(run["passed"] for run in case["runs"])
            continue
        expected = {}
        for name, desc in case["tensors"].items():
            path = Path(desc["path"])
            assert sha(path) == desc["sha256"], f"Changed retained tensor: {path}"
            expected[name] = np.fromfile(path, desc["dtype"]).reshape(desc["shape"])
        directory = args.replay / case["request"]["id"]
        for run in case["runs"]:
            raw = (directory / f"rust-{run['reuse']}-{run['repeat']}.raw").read_bytes()
            actual = decode(run["reply"], raw, expected)
            assert all(
                metrics(actual[name], value)["byte_exact"]
                for name, value in expected.items()
            )
            assert run["passed"] and run["metadata_equal"]
            raw_runs += 1
        length = case["plan"]["latent_length"]
        latent = np.fromfile(
            directory / "canonical/ve_step_07.raw", np.float32
        ).reshape(1, 144, length)
        pcm = np.fromfile(directory / "canonical/pcm.raw", np.float32)
        latent_error = metrics(expected["ve_step_07"][..., :length], latent)[
            "relative_l2"
        ]
        convergence = spectral(expected["pcm"], pcm)
        canonical = case["canonical"]
        assert (
            abs(latent_error - canonical["trajectory"]["ve_step_07"]["relative_l2"])
            < 1e-12
        )
        assert abs(convergence - canonical["spectral_convergence"]) < 1e-12
        comparisons.append(
            {
                "case": case["request"]["id"],
                "true_latent_length": length,
                "frames": case["plan"]["frames"],
                "passed": canonical["passed"],
                "failed_checks": [
                    name for name, passed in canonical["checks"].items() if not passed
                ],
                "final_latent_relative_l2": latent_error,
                "spectral_convergence": convergence,
                "vocoder_relative_l2": canonical["stages"]["vocoder"]["relative_l2"],
            }
        )
    assert raw_runs == 48 and replay["adapter_passed"]
    assert profile["passed"] and jobs["passed"] and bindings["passed"]
    assert len(profile["components"]) == 4
    for component in profile["components"]:
        directory = args.profile / component["name"]
        assert sha(directory / "profile.csv") == component["profile_sha256"]
        assert sha(directory / "context.bin") == component["context_sha256"]
    warm = []
    for language in ("en", "ko"):
        for voice in ("M1", "F1"):
            runs = [
                r
                for r in jobs["runs"]
                if r["repeat"] > 0 and r["language"] == language and r["voice"] == voice
            ]
            warm.append(
                {
                    "language": language,
                    "voice": voice,
                    "segments": runs[0]["segments"],
                    "frames": runs[0]["frames"],
                    "first_pcm": distribution([r["first_pcm_ms"] for r in runs]),
                    "complete": distribution([r["complete_ms"] for r in runs]),
                }
            )
    evidence = {
        "replay": args.replay / "report.yaml",
        "public_rust_jobs": args.jobs,
        "htp_profile": args.profile / "report.yaml",
        "bindings": args.bindings / "report.json",
        "environment": args.environment,
        "serialized_compile_options": args.compile_options,
    }
    result = {
        "schema_version": 1,
        "status": (
            "experimental_integration_passed_numerical_screen_passed"
            if replay["canonical_screen_passed"]
            else "experimental_integration_passed_model_quality_gate_failed"
        ),
        "adapter_passed": replay["adapter_passed"],
        "canonical_screen_passed": replay["canonical_screen_passed"],
        "production_qualified": False,
        "manifest_sha256": replay["manifest_sha256"],
        "device": environment,
        "runtime": {
            "sdk": "2.50.40.260831",
            "executorch": "1.5.0",
            "soc": 88,
            "htp_arch": 81,
            "compile_options": compile_options["duration_predictor"][0]["options"],
            "same_options_all_four_graphs": all(
                v == compile_options["duration_predictor"]
                for v in compile_options.values()
            ),
        },
        "replay": {
            "cases": len(replay["cases"]),
            "positive_cases": len(comparisons),
            "negative_cases": len(replay["cases"]) - len(comparisons),
            "passes_per_backing_mode": 2,
            "backing_modes": [False, True],
            "full_raw_runs_revalidated": raw_runs,
            "acceptance": replay["acceptance"],
            "canonical_comparisons": comparisons,
            "rust_binary_sha256": replay["binary_sha256"],
            "bridge_sha256": replay["bridge_sha256"],
            "reference_binary_sha256": replay["reference_binary_sha256"],
        },
        "jobs": {
            "passed": True,
            "provenance": jobs["provenance"],
            "initialization_ms": jobs["initialization_ms"],
            "measurement": jobs["measurement"],
            "warm": warm,
            "memory": jobs["memory"],
            "model_loads": jobs["model_loads"],
            "live_client_tensors": jobs["live_client_tensors"],
            "native_cancellations": {
                "count": len(jobs["native_cancellations"]),
                "same_call_observed": sum(
                    v["same_native_call_across_cancel"]
                    for v in jobs["native_cancellations"]
                ),
                "signal": distribution(
                    [v["signal_ms"] for v in jobs["native_cancellations"]]
                ),
                "quiescence": distribution(
                    [v["quiescence_ms"] for v in jobs["native_cancellations"]]
                ),
                "stages": {
                    stage: distribution(
                        [
                            v["quiescence_ms"]
                            for v in jobs["native_cancellations"]
                            if v["stage"] == stage
                        ]
                    )
                    for stage in ("text_encoder", "ve_step_03", "vocoder")
                },
            },
            "backpressure_cancellations": {
                "signal": distribution(
                    [v["signal_ms"] for v in jobs["backpressure_cancellations"]]
                ),
                "quiescence": distribution(
                    [v["quiescence_ms"] for v in jobs["backpressure_cancellations"]]
                ),
            },
        },
        "htp": {
            "passed": True,
            "scope": profile["scope"],
            "components": [
                {
                    "name": c["name"],
                    "pte_sha256": c["pte_sha256"],
                    "total_node_cycles": c["profile"]["total_node_cycles"],
                    "reference_byte_exact": c["qnn_reference"]["byte_exact"],
                }
                for c in profile["components"]
            ],
        },
        "bindings": bindings,
        "limitations": [
            (
                "Canonical numerical screen failed; thresholds unchanged. No perceptual listening qualification."
                if not replay["canonical_screen_passed"]
                else "Numerical screen passed. No perceptual listening qualification."
            ),
            "Only Windows ARM64 SoC 88 / HTP V81; English/Korean M1/F1 exercised. Other devices/languages unqualified.",
            "Exactly eight steps, text capacity 128, true latent length 4..96. No automatic fallback.",
            "Power conditions observed; no energy/temperature telemetry, sustained thermal benchmark or cold-disk benchmark.",
            "Host process memory excludes accelerator/driver allocations. No zero-copy or hardware-preemption claim.",
            "macOS/iOS inference not rerun on this Windows host. SDK/runtime distribution packaging remains external.",
        ],
        "evidence": {
            name: {"path": str(path), "sha256": sha(path)}
            for name, path in evidence.items()
        },
    }
    write_yaml(args.output, result)
    print(f"Revalidated {raw_runs} Rust captures; wrote {args.output}")


if __name__ == "__main__":
    main()
