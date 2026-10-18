#!/usr/bin/env python3
"""Opt-in macOS release delivery qualification; local model bytes, no downloads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import plistlib
import shutil
import socket
import subprocess
import time
import uuid

import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_sources(source, destination):
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(
            ".dart_tool",
            "build",
            ".mlsr",
            "Pods",
            "ephemeral",
            ".symlinks",
            "__pycache__",
            ".flutter-plugins-dependencies",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flutter", default="flutter")
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("requires macOS Apple Silicon")
    bundle = args.bundle.resolve(strict=True)
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        parser.error("output must be outside the developer checkout")
    output.mkdir()  # Never overwrite earlier evidence.
    env = dict(os.environ, MACOSX_DEPLOYMENT_TARGET="15.0")
    env.pop("IPHONEOS_DEPLOYMENT_TARGET", None)
    report = {
        "schema_version": 1,
        "passed": False,
        "platform": platform.platform(),
        "phases": {},
    }
    report["hardware"] = subprocess.check_output(
        ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
    ).strip()
    report_path = output / "report.json"

    def run(name, command, cwd=ROOT):
        print(f"{name}: {' '.join(map(str, command))}", flush=True)
        with (output / f"{name}.log").open("wb") as log:
            subprocess.run(
                list(map(str, command)),
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=600,
            )

    try:
        report["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        report["qualification_sources"] = {
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                Path(__file__).resolve(),
                ROOT / "tests/fixtures/model-delivery/release_probe.dart",
            ]
        }
        tracked = (
            subprocess.check_output(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                cwd=ROOT,
            )
            .decode()
            .split("\0")
        )
        report["source_files"] = {
            name: digest(ROOT / name)
            for name in sorted(set(tracked))
            if name.startswith(
                ("crates/", "bindings/", "examples/", "model-manifests/", "include/")
            )
            and not name.endswith(".md")
            and (ROOT / name).is_file()
        }
        run(
            "native-build",
            [
                "cargo",
                "build",
                "--locked",
                "--release",
                "-p",
                "mlsr-ffi",
                "--features",
                "coreml,apple-output",
            ],
        )
        run("cli-build", ["cargo", "build", "--locked", "--release", "-p", "mlsr-cli"])
        consumer = output / "consumer"
        copy_sources(ROOT / "examples/flutter_speak", consumer)
        for package in ["dart", "flutter"]:
            copy_sources(ROOT / "bindings" / package, consumer / "packages" / package)
        (consumer / "native").mkdir()
        library = consumer / "native/libmlsr_ffi.dylib"
        shutil.copyfile(ROOT / "target/release/libmlsr_ffi.dylib", library)
        report["native_input_sha256"] = digest(library)
        pubspec_path = consumer / "pubspec.yaml"
        pubspec = yaml.safe_load(pubspec_path.read_text())
        pubspec["dependencies"].update(
            {
                "mlsr": {"path": "packages/dart"},
                "mlsr_flutter": {"path": "packages/flutter"},
                "crypto": "^3.0.7",
            }
        )
        pubspec["hooks"]["user_defines"]["mlsr"] = {
            "coreml": True,
            "apple-output": True,
            "native-library": str(library),
        }
        pubspec_path.write_text(yaml.safe_dump(pubspec, sort_keys=False))
        model_config = {
            "schema_version": 1,
            "package": "mlsr_app_models",
            "output": ".mlsr/model_assets",
            "mlsr_package": "packages/dart",
            "models": [
                {
                    "id": "supertonic3-coreml-int4-v1",
                    "dart_name": "supertonic3",
                    "source": str(bundle),
                    "manifest": str(ROOT / "model-manifests/supertonic3-coreml.yaml"),
                    "platforms": ["macos-arm64"],
                }
            ],
        }
        config = consumer / "mlsr-models.yaml"
        config.write_text(yaml.safe_dump(model_config, sort_keys=False))
        run(
            "prepare",
            [
                ROOT / "target/release/mlsr",
                "models",
                "prepare-flutter",
                "--config",
                config,
            ],
            consumer,
        )
        shutil.copyfile(
            ROOT / "tests/fixtures/model-delivery/release_probe.dart",
            consumer / "lib/release_probe.dart",
        )
        bundle_id = "dev.mlsr.delivery.run" + uuid.uuid4().hex[:12]
        report["bundle_identifier"] = bundle_id
        container = Path.home() / "Library/Containers" / bundle_id
        if container.exists():
            raise RuntimeError("qualification app data is not fresh")
        report["container_absent_before_launch"] = True
        for file in [
            consumer / "macos/Runner/Configs/AppInfo.xcconfig",
            consumer / "macos/Runner.xcodeproj/project.pbxproj",
        ]:
            file.write_text(file.read_text().replace("dev.mlsr.mlsrSpeak", bundle_id))
        run("pub-get", [args.flutter, "pub", "get"], consumer)
        # Analyze the consumer. Copied packages' dev-only tests/tools have their
        # own dependency graphs and were checked in their respective packages.
        run("analyze", [args.flutter, "analyze", "--no-pub", "lib"], consumer)
        run(
            "release-build",
            [
                args.flutter,
                "build",
                "macos",
                "--release",
                "--no-pub",
                "--target",
                "lib/release_probe.dart",
            ],
            consumer,
        )
        built = consumer / "build/macos/Build/Products/Release/MLSR Speak.app"
        relocated = output / "relocated/MLSR Delivery.app"
        relocated.parent.mkdir()
        shutil.copytree(built, relocated, symlinks=True)  # Preserve framework links.
        run("signature", ["codesign", "--verify", "--deep", "--strict", relocated])
        entitlements = plistlib.loads(
            subprocess.check_output(
                ["codesign", "-d", "--entitlements", ":-", relocated],
                stderr=subprocess.DEVNULL,
            )
        )
        if (
            entitlements.get("com.apple.security.app-sandbox") is not True
            or entitlements.get("com.apple.security.network.client")
            or entitlements.get("com.apple.security.network.server")
        ):
            raise RuntimeError(
                "release must be sandboxed without networking entitlements"
            )
        report["entitlements"] = entitlements
        index = yaml.safe_load(
            (consumer / ".mlsr/model_assets/package-index.yaml").read_text()
        )
        model = index["models"][0]
        assets = (
            relocated
            / "Contents/Frameworks/App.framework/Resources/flutter_assets/packages/mlsr_app_models"
        )
        audited = []
        for entry in model["files"]:
            file = assets / model["asset_path"] / entry["path"]
            if (
                file.is_symlink()
                or file.stat().st_size != entry["size_bytes"]
                or digest(file) != entry["sha256"]
            ):
                raise RuntimeError(f"final app asset mismatch: {entry['path']}")
            audited.append(entry)
        for notice in [
            "supertonic3-OpenRAIL-M.txt",
            "supertonic-MIT.txt",
            "SOURCES.md",
            "MLSR-THIRD-PARTY.md",
        ]:
            if not (assets / "notices" / notice).is_file():
                raise RuntimeError(f"final app missing notice: {notice}")
        report["final_model_inventory"] = audited
        report["final_model_bytes"] = sum(e["size_bytes"] for e in audited)
        report["release_app_logical_bytes"] = sum(
            p.stat().st_size
            for p in relocated.rglob("*")
            if p.is_file() and not p.is_symlink()
        )
        executable = relocated / "Contents/MacOS/MLSR Speak"
        report["relocated_executable_sha256"] = digest(executable)
        forbidden = [
            ROOT / "model-manifests/supertonic3-coreml.yaml",
            bundle / "mlsr-bundle.yaml",
            consumer / "native/libmlsr_ffi.dylib",
        ]
        if not all(p.is_file() for p in forbidden):
            raise RuntimeError("denial probes require existing developer files")
        # Original POC PCM payloads, extracted from the previously qualified
        # WAVs after verifying their full-file hashes (documented in the report).
        # WAV serialization now belongs to Rust and its header differs.
        expected_pcm = {
            "hello": "1a4e3d62292e843bed979c680e7227b7d35f677770c9aea8ea138d3aee83ae3d",
            "korean": "e73fe4ec4fccc0aaefdbbc0d32a0a10cf29f14b02f20fceb964e844fdbfcd8fb",
        }
        report["expected_pcm_sha256"] = expected_pcm
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            for phase in [
                "automatic_first",
                "automatic_repeat",
                "installed_first",
                "installed_repeat",
            ]:
                print(f"Running relocated sandboxed app: {phase}", flush=True)
                phase_env = dict(
                    env,
                    MLSR_DELIVERY_PHASE=phase,
                    MLSR_FORBIDDEN_FILES=json.dumps(list(map(str, forbidden))),
                    MLSR_DENIED_NETWORK_PORT=str(listener.getsockname()[1]),
                )
                started = time.monotonic()
                with (output / f"{phase}.log").open("wb") as log:
                    result = subprocess.run(
                        [executable],
                        cwd=relocated.parent,
                        env=phase_env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=180,
                    )
                log = (output / f"{phase}.log").read_text(errors="replace")
                lines = [
                    line.split("MLSR_DELIVERY_RESULT=", 1)[1]
                    for line in log.splitlines()
                    if "MLSR_DELIVERY_RESULT=" in line
                ]
                if not lines:
                    raise RuntimeError(
                        f"{phase}: no app receipt, exit {result.returncode}"
                    )
                receipt = json.loads(lines[-1])
                receipt["process_wall_seconds"] = time.monotonic() - started
                report["phases"][phase] = receipt
                # Retain raw audio beside the report, including partial failed
                # runs. Do not make the evidence depend on keeping an app's data
                # container forever. The probe never removes these exports.
                audio = receipt.get("audio_evidence_directory")
                if audio:
                    destination = output / "audio" / phase
                    destination.mkdir(parents=True)
                    audio_hashes = {
                        f"{name}-export.wav": result["wav_sha256"]
                        for name, result in receipt.get("outputs", {}).items()
                    }
                    if "controller_wav_sha256" in receipt:
                        audio_hashes["controller-export.wav"] = receipt[
                            "controller_wav_sha256"
                        ]
                    for name, expected in audio_hashes.items():
                        source = Path(audio) / name
                        if digest(source) != expected:
                            raise RuntimeError(f"{phase}: exported audio changed")
                        shutil.copyfile(source, destination / name)
                if result.returncode != 0 or receipt.get("passed") is not True:
                    raise RuntimeError(f"{phase} failed: {receipt.get('failure')}")
                if (
                    not Path(receipt["bundled_directory"])
                    .resolve()
                    .is_relative_to(relocated.resolve())
                ):
                    raise RuntimeError("model lookup escaped relocated app")
                if receipt["copied_files"] != (29 if phase == "installed_first" else 0):
                    raise RuntimeError("unexpected first/repeat materialization count")
                if receipt["copied_bytes"] != (
                    report["final_model_bytes"] if phase == "installed_first" else 0
                ):
                    raise RuntimeError("unexpected model copy size")
                if (
                    phase != "installed_repeat"
                    and receipt["store_before"]["files"] != 0
                ):
                    raise RuntimeError("model store was not fresh before installation")
                if (
                    phase == "installed_repeat"
                    and receipt["store_before"]["files"] != 29
                ):
                    raise RuntimeError(
                        "repeat launch did not reuse the installed version"
                    )
                for name, expected in expected_pcm.items():
                    if receipt["outputs"][name]["pcm_sha256"] != expected:
                        raise RuntimeError(
                            f"{phase}: {name} differs from the qualified POC output"
                        )
                if receipt["controller_pcm_sha256"] != expected_pcm["hello"]:
                    raise RuntimeError("ordinary release controller PCM changed")
                expected_files = 29 if phase == "installed_first" else 0
                if receipt["store_after"]["files"] != expected_files:
                    raise RuntimeError("unexpected final installed model inventory")
                report_path.write_text(json.dumps(report, indent=2) + "\n")
        report["passed"] = True
    except Exception as error:
        report["failure"] = str(error)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Evidence: {report_path}", flush=True)


if __name__ == "__main__":
    main()
