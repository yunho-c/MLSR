"""Explicit Apple Silicon job benchmark through the public C/Python adapter.

Reports process-cold initialization with existing OS compiler caches, warm first
audio, complete generation, cancellation-to-quiescence and process peak RSS.
No playback or model download. Each variant runs in a fresh, sequential process.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {"baseline": 0, "fused8": 1, "fused8_fixed_vocoder": 2}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stats(values):
    ordered = sorted(values)
    return {
        "median": statistics.median(values),
        "p95": ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)],
        "max": max(values),
    }


def worker(args):
    # The native library is separately built and pinned by the parent. Import is
    # local wrapper code; actual synthesis and scheduling remain in Rust.
    sys.path.insert(0, str(ROOT / "bindings/python/src"))
    from mlsr import Engine
    from mlsr._synthesis import library

    lib = library()
    baseline_handles = lib.mlsr_live_handles()
    corpus = yaml.safe_load(
        (ROOT / "tests/fixtures/supertonic3/corpus.yaml").read_text()
    )
    cases = [
        dict(c, steps=8, effective_speed=1.05)
        for c in corpus["cases"]
        if "expected_error" not in c and "target_latent_length" not in c
    ]
    if args.variant_report:
        qualification = yaml.safe_load(args.variant_report.read_text())
        if qualification.get("passed") is not True:
            raise ValueError("The supplied variant replay must pass")
        for item in qualification["cases"]:
            # These F1 inputs are single sentences, so job segmentation preserves
            # the calibrated exact L30/L32 vocoder shape used by the replay.
            if (
                item["variant"] == "baseline"
                and item["pass"] == 1
                and item["id"]
                in (
                    "en_f1_basic_L30",
                    "en_f1_basic_L32",
                    "ko_f1_basic_L30",
                    "ko_f1_basic_L32",
                )
            ):
                cases.append(dict(item["request"], id=item["id"]))
    report = {
        "variant": args.variant,
        "compute_policy": args.compute_policy,
        "passed": False,
        "timing_boundary": "Python Engine.start through copied audio / terminal event over public C ABI; no playback",
        "cold_definition": "new process and Engine; precompiled models and existing OS specialization/file caches",
        "samples": [],
        "cancellations": [],
    }
    started = time.perf_counter_ns()
    engine = Engine(
        ROOT / "model-manifests/supertonic3-coreml.yaml",
        args.bundle,
        execution_variant=VARIANTS[args.variant],
        compute_policy=args.compute_policy,
    )
    report["initialization_ms"] = (time.perf_counter_ns() - started) / 1e6

    def run(case):
        started = time.perf_counter_ns()
        first = None
        frames = 0
        pcm_hash = hashlib.sha256()
        with engine.start(
            case["text"],
            language=case["language"],
            voice=case["voice"],
            steps=case["steps"],
            speed=case["effective_speed"],
            seed=case["seed"],
        ) as job:
            for event in job:
                if event.kind == "audio":
                    if first is None:
                        first = (time.perf_counter_ns() - started) / 1e6
                    if (
                        event.start_frame != frames
                        or event.sample_rate != 44100
                        or event.channels != 1
                    ):
                        raise RuntimeError("Audio order or format changed")
                    frames += event.frames
                    pcm_hash.update(event.samples.tobytes())
                elif event.terminal and (
                    event.kind != "completed" or event.frames != frames
                ):
                    raise RuntimeError(
                        f"Unexpected terminal: {event.kind} {event.error}"
                    )
        elapsed = (time.perf_counter_ns() - started) / 1e6
        if first is None or frames == 0:
            raise RuntimeError("No playable audio")
        return {
            "id": case["id"],
            "first_audio_ms": first,
            "total_ms": elapsed,
            "frames": frames,
            "audio_seconds": frames / 44100,
            "real_time_factor": elapsed * 44.1 / frames,
            "pcm_sha256": pcm_hash.hexdigest(),
        }

    try:
        report["first_request"] = run(cases[0])
        # Warm every case/shape, explicitly excluded from measured warm samples.
        expected = {case["id"]: run(case) for case in cases}
        for repetition in range(args.repeats):
            order = cases if repetition % 2 == 0 else list(reversed(cases))
            for case in order:
                sample = dict(run(case), repetition=repetition + 1)
                if sample["pcm_sha256"] != expected[case["id"]]["pcm_sha256"]:
                    raise RuntimeError("Repeated PCM changed")
                report["samples"].append(sample)
        # Prediction event is published after DP. Delays sample subsequent work;
        # they are not assertions that a particular native call is in flight.
        for trigger, delay_ms in [
            ("after_prediction", 0),
            ("after_prediction", 2),
            ("after_prediction", 5),
            ("stalled_consumer", 100),
        ]:
            for repetition in range(3):
                with engine.start(
                    "Hello from MLSR.\nAnother sentence follows.",
                    queue_frames=1,
                    chunk_frames=1,
                ) as job:
                    event = job.poll(timeout_ms=30000)
                    if event is None or event.kind != "predicted":
                        raise RuntimeError("Missing cancellation trigger")
                    if trigger == "stalled_consumer":
                        while True:
                            event = job.poll(timeout_ms=30000)
                            if event is None or event.terminal:
                                raise RuntimeError("Missing stalled-consumer audio")
                            if event.kind == "audio":
                                break
                    time.sleep(delay_ms / 1000)
                    started = time.perf_counter_ns()
                    job.cancel()
                    signal_ms = (time.perf_counter_ns() - started) / 1e6
                    if not job.wait(timeout_ms=30000):
                        raise RuntimeError("Cancellation did not quiesce")
                    quiescence_ms = (time.perf_counter_ns() - started) / 1e6
                    terminal = job.poll(timeout_ms=0)
                    if terminal is None or terminal.kind != "cancelled":
                        raise RuntimeError("Cancellation leaked a late event")
                    report["cancellations"].append(
                        {
                            "trigger": trigger,
                            "delay_ms": delay_ms,
                            "repetition": repetition + 1,
                            "signal_ms": signal_ms,
                            "quiescence_ms": quiescence_ms,
                        }
                    )
        # Reuse after cancellation is part of the measured engine lifecycle.
        report["after_cancel"] = run(cases[-1])
    finally:
        engine.close()
    report["handles_before"] = baseline_handles
    report["handles_after"] = lib.mlsr_live_handles()
    report["by_case"] = {}
    for case in cases:
        values = [s for s in report["samples"] if s["id"] == case["id"]]
        report["by_case"][case["id"]] = {
            "request": case,
            **{
                key: stats([s[key] for s in values])
                for key in ("first_audio_ms", "total_ms", "real_time_factor")
            },
        }
    report["warm_all_samples_faster_than_playback"] = all(
        s["real_time_factor"] < 1 for s in report["samples"]
    )
    report["passed"] = (
        len(report["samples"]) == len(cases) * args.repeats
        and report["handles_after"] == baseline_handles
    )
    args.output.write_text(yaml.safe_dump(report, sort_keys=False, allow_unicode=True))
    if not report["passed"]:
        raise RuntimeError("Incomplete samples or native handles leaked")


def benchmark(args):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("This benchmark currently qualifies macOS arm64")
    if args.repeats < 3:
        raise ValueError("At least three warm repetitions are required")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT))
    from tools.supertonic3.bundle import load_manifest, verify_bundle
    from tools.supertonic3.variants import verify_variants

    manifest = load_manifest()
    verify_bundle(manifest, args.bundle)
    variants = verify_variants(args.bundle) if args.variant != "baseline" else None
    env = os.environ.copy()
    env["MLSR_LIBRARY_PATH"] = str(args.library.resolve(strict=True))
    source_paths = [
        ROOT / "Cargo.toml",
        ROOT / "Cargo.lock",
        Path(__file__),
        *sorted((ROOT / "crates/mlsr/src").rglob("*.rs")),
        *sorted((ROOT / "crates/mlsr-ffi/src").rglob("*.rs")),
        *sorted((ROOT / "bindings/python/src").rglob("*.py")),
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    report = {
        "schema_version": 1,
        "kind": "public_job_benchmark",
        "passed": False,
        "environment": {
            "platform": platform.platform(),
            "hardware": subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip(),
            "power": subprocess.check_output(
                ["pmset", "-g", "batt"], text=True
            ).strip(),
            "power_settings": subprocess.check_output(
                ["pmset", "-g", "custom"], text=True
            ).strip(),
        },
        "library_sha256": digest(args.library),
        "baseline_manifest_sha256": digest(
            ROOT / "model-manifests/supertonic3-coreml.yaml"
        ),
        "variant_manifest_sha256": digest(
            ROOT / "model-manifests/supertonic3-coreml-variants.yaml"
        ),
        "logical_artifact_bytes": {
            "native_library": args.library.stat().st_size,
            "baseline_bundle": sum(
                f["size_bytes"]
                for a in manifest["artifacts"].values()
                for f in a["files"]
            ),
            "optional_graphs": (
                sum(
                    f["size_bytes"]
                    for a in variants["artifacts"].values()
                    for f in a["files"]
                )
                if variants
                else 0
            ),
            "note": "Logical model sizes count shared weight blobs per graph; file symlinks avoid physical duplication in the prepared cache",
        },
        "source_files": hashes,
        "scope": "Sequential fresh processes; per-case alternating order; no unrelated-load exclusion guarantee",
        "variants": [],
    }
    try:
        for variant in VARIANTS if args.variant == "all" else [args.variant]:
            output = args.output / f"{variant}.yaml"
            command = [
                "/usr/bin/time",
                "-l",
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--bundle",
                str(args.bundle.resolve()),
                "--output",
                str(output.resolve()),
                "--variant",
                variant,
                "--repeats",
                str(args.repeats),
                "--compute-policy",
                args.compute_policy,
            ]
            if args.variant_report:
                command += ["--variant-report", str(args.variant_report.resolve())]
            print(
                f"Benchmarking {variant}: {args.repeats} warm passes plus lifecycle checks",
                flush=True,
            )
            log = args.output / f"{variant}.log"
            with log.open("x") as stream:
                subprocess.run(
                    command,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            result = yaml.safe_load(output.read_text())
            import re

            match = re.search(r"(\d+)\s+maximum resident set size", log.read_text())
            if not match:
                raise RuntimeError("Process peak RSS was not reported by macOS time")
            result["peak_process_rss_bytes"] = int(match[1])
            result["peak_memory_scope"] = (
                "macOS time -l child process maximum RSS, including Python and loaded Core ML; not ANE memory"
            )
            result["report_sha256"] = digest(output)
            if args.variant_report:
                qualification = yaml.safe_load(args.variant_report.read_text())
                result["numerical_qualification"] = qualification["variant_results"][
                    variant
                ]
            report["variants"].append(result)
        verify_bundle(manifest, args.bundle)
        if variants:
            verify_variants(args.bundle)
        if (
            hashes != {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
            or digest(args.library) != report["library_sha256"]
        ):
            raise RuntimeError("Sources or native library changed")
        report["passed"] = all(v["passed"] for v in report["variants"])
    finally:
        (args.output / "report.yaml").write_text(
            yaml.safe_dump(report, sort_keys=False, allow_unicode=True)
        )
    print(f"Benchmark complete: {args.output / 'report.yaml'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--library", type=Path, default=ROOT / "target/release/libmlsr_ffi.dylib"
    )
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    parser.add_argument("--variant-report", type=Path)
    parser.add_argument(
        "--compute-policy",
        choices=["cpu_only", "cpu_and_neural_engine"],
        default="cpu_and_neural_engine",
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        benchmark(args)


if __name__ == "__main__":
    main()
