#!/usr/bin/env python3
"""Capture real MLSR jobs with Instruments, including a CPU-only control.

The ANE table is system-wide. Analyze model labels only inside request intervals
emitted by the workload PID; do not equate a compute plan with actual execution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.supertonic3.bundle import load_manifest, load_yaml, verify_bundle, write_yaml
from tools.supertonic3.variants import verify_variants


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(args):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Requires Apple Silicon macOS and Xcode Instruments")
    output, bundle, binary = (
        args.output.resolve(),
        args.bundle.resolve(),
        args.binary.resolve(strict=True),
    )
    output.mkdir()
    manifest = ROOT / "model-manifests/supertonic3-coreml.yaml"
    report = {
        "schema_version": 1,
        "kind": "instruments_synthesis_capture",
        "passed": False,
        "scope": "Raw instrumented execution; analyze process intervals separately; not a latency benchmark",
        "platform": platform.platform(),
        "binary_sha256": sha(binary),
        "manifest_sha256": sha(manifest),
        "marker_source_sha256": sha(ROOT / "tests/integration/trace_signposts.m"),
        "capture_tool_sha256": sha(Path(__file__)),
        "capture_mode": "attach after model initialization; Darwin notification gates requests until recording starts",
        "conditions": [],
        "commands": [],
    }
    env = dict(os.environ, MACOSX_DEPLOYMENT_TARGET="15.0")
    env.pop("IPHONEOS_DEPLOYMENT_TARGET", None)
    with (output / "commands.log").open("x") as log:

        def run(command, timeout=240):
            command = list(map(str, command))
            print("+", " ".join(command), flush=True)
            print(json.dumps(command), file=log, flush=True)
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
            report["commands"].append(
                {"command": command, "exit_code": result.returncode}
            )
            result.check_returncode()

        try:
            verify_bundle(load_manifest(manifest), bundle)
            verify_variants(bundle)
            helper = output / "libmlsr_trace_signposts.dylib"
            run(
                [
                    "xcrun",
                    "clang",
                    "-dynamiclib",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-mmacosx-version-min=15.0",
                    ROOT / "tests/integration/trace_signposts.m",
                    "-o",
                    helper,
                ]
            )
            report["marker_library_sha256"] = sha(helper)
            for variant, policy, repetitions in [
                ("baseline", "ane", 12),
                ("baseline", "cpu", 3),
                ("fused8", "ane", 12),
            ]:
                name = f"{variant}-{policy}"
                trace, workload = output / f"{name}.trace", output / f"{name}.yaml"
                ready = output / f"{name}.ready"
                notification = "dev.mlsr.trace." + uuid.uuid4().hex
                child_env = dict(
                    env,
                    MLSR_TRACE_READY_PATH=str(ready),
                    MLSR_TRACE_NOTIFICATION=notification,
                )
                command = list(
                    map(
                        str,
                        [
                            binary,
                            manifest,
                            bundle,
                            workload,
                            variant,
                            policy,
                            repetitions,
                            helper,
                        ],
                    )
                )
                print(f"Loading {name} before attaching Instruments", flush=True)
                print(
                    json.dumps(
                        {
                            "workload_command": command,
                            "ready": str(ready),
                            "notification": notification,
                        }
                    ),
                    file=log,
                    flush=True,
                )
                with (output / f"{name}-workload.log").open("x") as child_log:
                    child = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=child_env,
                        stdout=child_log,
                        stderr=subprocess.STDOUT,
                    )
                    try:
                        deadline = time.monotonic() + 240
                        while not ready.exists() or not ready.read_text().strip():
                            if child.poll() is not None:
                                raise RuntimeError(
                                    f"Workload exited before trace attachment: {child.returncode}"
                                )
                            if time.monotonic() > deadline:
                                raise TimeoutError(
                                    f"Model initialization timed out: {name}"
                                )
                            time.sleep(0.05)
                        if int(ready.read_text()) != child.pid:
                            raise RuntimeError(
                                "Unexpected workload PID in trace handshake"
                            )
                        run(
                            [
                                "xcrun",
                                "xctrace",
                                "record",
                                "--template",
                                "Time Profiler",
                                "--instrument",
                                "Neural Engine",
                                "--instrument",
                                "Core ML",
                                "--instrument",
                                "os_signpost",
                                "--time-limit",
                                "60s",
                                "--output",
                                trace,
                                "--notify-tracing-started",
                                notification,
                                "--attach",
                                child.pid,
                            ],
                            timeout=120,
                        )
                        code = child.wait(timeout=10)
                        report["commands"].append(
                            {"command": command, "exit_code": code}
                        )
                        if code:
                            raise RuntimeError(
                                f"Speech workload failed: {name}, exit {code}"
                            )
                    finally:
                        if child.poll() is None:
                            child.terminate()
                            try:
                                child.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                child.kill()
                                child.wait()
                value = load_yaml(workload)
                expected = (repetitions + 1) * (11 if variant == "baseline" else 4)
                if (
                    value.get("passed") is not True
                    or value["stats"]["predictions"] != expected
                    or len(value["requests"]) != repetitions + 1
                    or len({(x["frames"], x["pcm_sha256"]) for x in value["requests"]})
                    != 1
                ):
                    raise RuntimeError(
                        f"Incomplete or unstable speech workload: {name}"
                    )
                for schema in (
                    "toc",
                    "ane-hw-intervals",
                    "os-signpost",
                    "coreml-os-signpost",
                ):
                    command = [
                        "xcrun",
                        "xctrace",
                        "export",
                        "--input",
                        trace,
                        "--output",
                        output / f"{name}-{schema}.xml",
                    ]
                    if schema == "toc":
                        command += ["--toc"]
                    else:
                        selector = f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]'
                        if schema == "os-signpost":
                            selector = '/trace-toc/run[@number="1"]/data/table[@schema="os-signpost" and @target-pid="SINGLE" and not(@category) and not(@subsystem)]'
                        command += ["--xpath", selector]
                    run(command)
                report["conditions"].append(
                    {
                        "name": name,
                        "workload": value,
                        "workload_sha256": sha(workload),
                        "xml_files": {
                            p.name: sha(p) for p in sorted(output.glob(f"{name}-*.xml"))
                        },
                    }
                )
                write_yaml(output / "capture.yaml", report)
            if sha(binary) != report["binary_sha256"]:
                raise RuntimeError("Trace workload binary changed during capture")
            if (
                sha(Path(__file__)) != report["capture_tool_sha256"]
                or sha(ROOT / "tests/integration/trace_signposts.m")
                != report["marker_source_sha256"]
            ):
                raise RuntimeError("Capture tooling changed during recording")
            verify_bundle(load_manifest(manifest), bundle)
            verify_variants(bundle)
            report["passed"] = True
        except Exception as error:
            report["error"] = str(error)
            raise
        finally:
            write_yaml(output / "capture.yaml", report)
    print(f"Captured real speech and CPU control: {output / 'capture.yaml'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Verified combined baseline/fused bundle",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "target/release/examples/coreml_trace"
    )
    capture(parser.parse_args())


if __name__ == "__main__":
    main()
