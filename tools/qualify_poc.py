#!/usr/bin/env python3
"""Explicit real Apple Silicon suite; local assets only, no downloads or venv."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.supertonic3.bundle import load_yaml, write_yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    paths = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
        )
        .decode()
        .split("\0")
    )
    prefixes = (
        "crates/",
        "bindings/",
        "tools/",
        "tests/",
        "model-manifests/",
        "examples/",
    )
    return {
        name: sha(ROOT / name)
        for name in sorted(set(paths))
        if name
        and (ROOT / name).is_file()
        and (
            name.startswith(prefixes)
            or name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml")
        )
        and not name.endswith(".md")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant-bundle", type=Path)
    parser.add_argument(
        "--flutter",
        action="store_true",
        help="Also build/run the macOS UI integration test",
    )
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error(
            "Requires macOS Apple Silicon, Xcode, and existing language toolchains"
        )
    args.bundle, args.reference = args.bundle.resolve(
        strict=True
    ), args.reference.resolve(strict=True)
    if args.variant_bundle:
        args.variant_bundle = args.variant_bundle.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir()  # A previous qualification must never be overwritten.
    env = dict(
        os.environ, MACOSX_DEPLOYMENT_TARGET="15.0", MLSR_TEST_BUNDLE=str(args.bundle)
    )
    env.pop("IPHONEOS_DEPLOYMENT_TARGET", None)
    if args.variant_bundle:
        env["MLSR_TEST_VARIANT_BUNDLE"] = str(args.variant_bundle)
    report = {
        "schema_version": 1,
        "kind": "apple_silicon_poc_suite",
        "passed": False,
        "full_poc": False,
        "platform": platform.platform(),
        "bundle": str(args.bundle),
        "reference": str(args.reference),
        "variant_bundle": str(args.variant_bundle) if args.variant_bundle else None,
        "flutter_requested": args.flutter,
        "commands": [],
        "evidence": {},
        "scope": "Correctness, installed packages and optional UI; timing is measured separately",
    }
    report_path = output / "report.yaml"
    write_yaml(report_path, report)
    with (output / "commands.log").open("x") as log:

        def run(*command, cwd=ROOT):
            command = list(map(str, command))
            print("+", " ".join(command), flush=True)
            print(
                json.dumps({"command": command, "cwd": str(cwd)}), file=log, flush=True
            )
            began = time.monotonic()
            result = subprocess.run(
                command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            report["commands"].append(
                {
                    "command": command,
                    "cwd": str(cwd),
                    "exit_code": result.returncode,
                    "elapsed_s": time.monotonic() - began,
                }
            )
            write_yaml(report_path, report)
            result.check_returncode()

        def evidence(name, path):
            value = load_yaml(path)
            if value.get("passed") is not True:
                raise RuntimeError(f"Incomplete or failed evidence: {path}")
            report["evidence"][name] = {
                "path": str(path),
                "sha256": sha(path),
                "passed": True,
            }

        try:
            if args.flutter:
                # Resolve platform tooling before taking the immutable source
                # snapshot, including the generated CocoaPods lockfile.
                run(
                    "flutter",
                    "pub",
                    "get",
                    "--enforce-lockfile",
                    cwd=ROOT / "examples/flutter_speak",
                )
                run(
                    "pod",
                    "install",
                    "--deployment",
                    cwd=ROOT / "examples/flutter_speak/macos",
                )
            report["source_files"] = sources()
            run(
                sys.executable,
                "-m",
                "tools.supertonic3",
                "verify-bundle",
                "--bundle",
                args.bundle,
            )
            run(
                sys.executable,
                "-m",
                "tools.supertonic3",
                "verify-reference",
                "--reference",
                args.reference,
            )
            rust_tests = [
                "cargo",
                "test",
                "--locked",
                "-p",
                "mlsr",
                "-p",
                "mlsr-ffi",
                "-p",
                "mlsr-cli",
                "--all-features",
                "--",
                "--include-ignored",
                "--test-threads=1",
            ]
            if not args.variant_bundle:
                rust_tests += [
                    "--skip",
                    "optimized_engines_cancel_recover_and_retain_a_bounded_model_set",
                ]
            run(*rust_tests)
            run(
                "cargo",
                "build",
                "--locked",
                "--release",
                "-p",
                "mlsr-ffi",
                "-p",
                "mlsr-cli",
                "--features",
                "coreml",
            )
            run(
                "cargo",
                "build",
                "--locked",
                "--release",
                "-p",
                "mlsr",
                "--features",
                "coreml",
                "--example",
                "coreml_synthesis_replay",
                "--example",
                "coreml_trace",
            )
            replay = output / "synthesis.yaml"
            run(
                sys.executable,
                "-m",
                "tools.supertonic3.synthesis_replay",
                "--bundle",
                args.bundle,
                "--reference",
                args.reference,
                "--output",
                replay,
                "--output-backings",
            )
            evidence("synthesis", replay)
            run(
                sys.executable,
                "-m",
                "tools.supertonic3.cli_smoke",
                "--bundle",
                args.bundle,
                "--reference",
                args.reference,
                "--replay-report",
                replay,
                "--output-dir",
                output / "cli",
            )
            evidence("cli", output / "cli/report.yaml")
            run(
                sys.executable,
                ROOT / "tools/check_synthesis_bindings.py",
                "--bundle",
                args.bundle,
                "--output-dir",
                output / "bindings",
                "--no-build",
            )
            evidence("bindings", output / "bindings/report.yaml")
            if args.variant_bundle:
                run(
                    sys.executable,
                    "-m",
                    "tools.supertonic3.variant_replay",
                    "--bundle",
                    args.variant_bundle,
                    "--reference",
                    args.reference,
                    "--output",
                    output / "variants",
                )
                evidence("variants", output / "variants/report.yaml")
            if args.flutter:
                app = ROOT / "examples/flutter_speak"
                model_config = output / "flutter-models.yaml"
                write_yaml(
                    model_config,
                    {
                        "schema_version": 1,
                        "package": "mlsr_app_models",
                        "output": str(app / ".mlsr/model_assets"),
                        "mlsr_package": str(ROOT / "bindings/dart"),
                        "models": [
                            {
                                "id": "supertonic3-coreml-int4-v1",
                                "dart_name": "supertonic3",
                                "source": str(args.bundle),
                                "manifest": str(
                                    ROOT / "model-manifests/supertonic3-coreml.yaml"
                                ),
                                "platforms": ["macos-arm64"],
                            }
                        ],
                    },
                )
                run(
                    "cargo",
                    "run",
                    "--locked",
                    "--release",
                    "-p",
                    "mlsr-cli",
                    "--",
                    "models",
                    "prepare-flutter",
                    "--config",
                    model_config,
                )
                run("flutter", "pub", "get", "--enforce-lockfile", cwd=app)
                run("flutter", "analyze", "--no-pub", cwd=app)
                run("flutter", "test", "test/audio_clip_test.dart", "--no-pub", cwd=app)
                run(
                    "flutter",
                    "test",
                    "integration_test/playback_test.dart",
                    "-d",
                    "macos",
                    "--no-pub",
                    cwd=app,
                )
                run(
                    "flutter",
                    "test",
                    "integration_test/speak_test.dart",
                    "-d",
                    "macos",
                    "--no-pub",
                    "--dart-define=MLSR_TEST_OUTPUT=" + str(output / "flutter"),
                    cwd=app,
                )
                evidence(
                    "flutter", output / "flutter/report.json"
                )  # JSON is a YAML subset.
            if sources() != report["source_files"]:
                raise RuntimeError(
                    "Sources changed during qualification; preserve this run and use a new output"
                )
            run(
                sys.executable,
                "-m",
                "tools.supertonic3",
                "verify-bundle",
                "--bundle",
                args.bundle,
            )
            report["artifacts"] = {
                name: {
                    "sha256": sha(ROOT / "target/release" / name),
                    "bytes": (ROOT / "target/release" / name).stat().st_size,
                }
                for name in (
                    "mlsr",
                    "libmlsr_ffi.dylib",
                    "libmlsr_ffi.a",
                    "examples/coreml_trace",
                )
            }
            report["passed"] = True
            report["full_poc"] = args.flutter and args.variant_bundle is not None
        except Exception as error:
            report["error"] = str(error)
            raise
        finally:
            write_yaml(report_path, report)
    print(f"Passed requested Apple suite: {report_path}", flush=True)


if __name__ == "__main__":
    main()
