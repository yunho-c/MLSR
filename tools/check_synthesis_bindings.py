#!/usr/bin/env python3
"""Explicit Apple real-model suite: C ABI and installed language packages.

Uses the existing Python interpreter, never a virtual environment. Ordinary CI
does not need model weights. Reports/audio are preserved in a new external path.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.dev import native_environment, library_name, VERSION
from tools.supertonic3.bundle import (
    load_manifest,
    verify_bundle,
    canonical_digest,
    write_yaml,
)
from tools.supertonic3.cli_smoke import wav_pcm


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "model-manifests/supertonic3-coreml.yaml",
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    if sys.platform != "darwin" or platform.machine() != "arm64":
        parser.error("This explicit suite requires macOS Apple Silicon")
    output = args.output_dir.resolve()
    output.mkdir()  # Preserve earlier evidence; caller must choose a fresh path.
    manifest, bundle = args.manifest.resolve(), args.bundle.resolve()
    model = load_manifest(manifest)
    report = {
        "schema_version": 1,
        "kind": "installed_synthesis_bindings",
        "passed": False,
        "manifest_sha256": canonical_digest(model),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "packages": {},
        "pcm": {},
        "examples": {},
    }
    report_path = output / "report.yaml"
    write_yaml(report_path, report)
    with (output / "commands.log").open("x") as log:

        def run(command, cwd=ROOT, env=None):
            print("+", " ".join(map(str, command)), flush=True)
            print(
                "+",
                json.dumps(list(map(str, command))),
                "cwd=",
                cwd,
                file=log,
                flush=True,
            )
            subprocess.run(
                list(map(str, command)),
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

        try:
            verify_bundle(model, bundle)
            if not args.no_build:
                env = dict(os.environ, MACOSX_DEPLOYMENT_TARGET="15.0")
                run(
                    [
                        "cargo",
                        "build",
                        "--release",
                        "--locked",
                        "-p",
                        "mlsr-ffi",
                        "--features",
                        "coreml",
                    ],
                    env=env,
                )
            library = ROOT / "target/release" / library_name()
            report["library_sha256"] = sha(library)
            sources = [
                ROOT / "Cargo.lock",
                ROOT / "Cargo.toml",
                Path(__file__),
                ROOT / "tests/integration/synthesis_smoke.c",
            ]
            for folder in (
                "crates/mlsr",
                "crates/mlsr-ffi",
                "bindings/python",
                "bindings/go",
                "bindings/dart",
            ):
                for path in (ROOT / folder).rglob("*"):
                    if (
                        path.is_file()
                        and not any(
                            part
                            in (".dart_tool", "build", "__pycache__", "mlsr.egg-info")
                            for part in path.parts
                        )
                        and path.suffix
                        in (".rs", ".toml", ".py", ".go", ".dart", ".yaml", ".lock")
                    ):
                        sources.append(path)
            report["source_files"] = {
                p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(sources))
            }
            with tempfile.TemporaryDirectory(
                prefix="mlsr-installed-synthesis-"
            ) as directory:
                root = Path(directory)
                native, include = root / "native/lib", root / "native/include"
                native.mkdir(parents=True)
                include.mkdir(parents=True)
                shutil.copy2(library, native)
                shutil.copy2(ROOT / "include/mlsr.h", include)
                env = native_environment(native, include)
                env.update(
                    MLSR_TEST_BUNDLE=str(bundle), MLSR_TEST_MANIFEST=str(manifest)
                )
                common = ["--manifest", manifest, "--bundle", bundle, "--cancel-first"]
                # Real public C API through both shared and static native artifacts.
                base = [
                    "xcrun",
                    "clang",
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-mmacosx-version-min=15.0",
                    ROOT / "tests/integration/synthesis_smoke.c",
                    "-I",
                    include,
                ]
                run(
                    [
                        *base,
                        "-L",
                        native,
                        "-lmlsr_ffi",
                        "-Wl,-rpath," + str(native),
                        "-o",
                        root / "shared",
                    ]
                )
                run([root / "shared", manifest, bundle], cwd=root, env=env)
                resource = Path(
                    subprocess.check_output(
                        ["xcrun", "clang", "-print-resource-dir"], text=True
                    ).strip()
                )
                frameworks = [
                    argument
                    for name in (
                        "Foundation",
                        "CoreFoundation",
                        "CoreGraphics",
                        "ImageIO",
                        "CoreVideo",
                        "CoreML",
                    )
                    for argument in ("-framework", name)
                ]
                run(
                    [
                        *base,
                        ROOT / "target/release/libmlsr_ffi.a",
                        "-L" + str(resource / "lib/darwin"),
                        "-lclang_rt.osx",
                        "-lobjc",
                        *frameworks,
                        "-o",
                        root / "static",
                    ]
                )
                run([root / "static", manifest, bundle], cwd=root, env=env)
                # Install a real wheel into a plain directory, without a venv.
                wheels = root / "wheels"
                run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "wheel",
                        "--no-deps",
                        "--no-build-isolation",
                        "--wheel-dir",
                        wheels,
                        ROOT / "bindings/python",
                    ]
                )
                installed = root / "python-installed"
                run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--no-deps",
                        "--no-compile",
                        "--target",
                        installed,
                        wheels / f"mlsr-{VERSION}-py3-none-any.whl",
                    ]
                )
                pytests = root / "python-tests"
                shutil.copytree(ROOT / "bindings/python/tests", pytests)
                pyexample = root / "synthesize.py"
                shutil.copy2(ROOT / "bindings/python/examples/synthesize.py", pyexample)
                pyenv = dict(
                    env,
                    PYTHONPATH=str(installed),
                    MLSR_TEST_OUTPUT=str(output / "python"),
                )
                run(
                    [sys.executable, "-m", "unittest", "discover", "-s", pytests, "-v"],
                    cwd=root,
                    env=pyenv,
                )
                run(
                    [
                        sys.executable,
                        pyexample,
                        *common,
                        "--output",
                        output / "python-example.wav",
                    ],
                    cwd=root,
                    env=pyenv,
                )
                report["packages"][
                    "python"
                ] = "installed wheel; tests and cancel/reuse/WAV example passed"
                # Relocated Go module/header/library, with the Go race detector.
                go = root / "go"
                shutil.copytree(ROOT / "bindings/go", go)
                goenv = dict(env, MLSR_TEST_OUTPUT=str(output / "go"))
                run(["go", "test", "-race", "-count=1", "./..."], cwd=go, env=goenv)
                run(["go", "vet", "./..."], cwd=go, env=goenv)
                run(
                    [
                        "go",
                        "run",
                        "./examples/synthesize",
                        *common,
                        "--output",
                        output / "go-example.wav",
                    ],
                    cwd=go,
                    env=goenv,
                )
                report["packages"][
                    "go"
                ] = "relocated module; race tests and cancel/reuse/WAV example passed"
                # Dart's declared prebuilt asset avoids rebuilding or discovering
                # Rust sources relative to the copied package.
                dart = root / "dart"
                shutil.copytree(
                    ROOT / "bindings/dart",
                    dart,
                    ignore=shutil.ignore_patterns(".dart_tool", "build"),
                )
                with (dart / "pubspec.yaml").open("a") as pubspec:
                    pubspec.write(
                        "\nhooks:\n  user_defines:\n    mlsr:\n      native-library: "
                        + json.dumps(str(native / library_name()))
                        + "\n"
                    )
                dartenv = dict(env, MLSR_TEST_OUTPUT=str(output / "dart"))
                run(["dart", "pub", "get", "--enforce-lockfile"], cwd=dart, env=dartenv)
                run(["dart", "analyze", "--fatal-infos"], cwd=dart, env=dartenv)
                run(["dart", "test", "--concurrency=1"], cwd=dart, env=dartenv)
                run(
                    [
                        "dart",
                        "run",
                        "example/synthesize.dart",
                        *common,
                        "--output",
                        output / "dart-example.wav",
                    ],
                    cwd=dart,
                    env=dartenv,
                )
                report["packages"][
                    "dart"
                ] = "relocated package/native asset; isolate stream/cancel tests and WAV example passed"
            for name in ("en_M1", "en_F1", "ko_M1", "ko_F1"):
                paths = {
                    language: output / language / f"{name}.f32"
                    for language in ("python", "go", "dart")
                }
                checksums = {language: sha(path) for language, path in paths.items()}
                if len(set(checksums.values())) != 1:
                    raise RuntimeError(f"Binding PCM differs: {name}: {checksums}")
                report["pcm"][name] = {
                    "frames": paths["python"].stat().st_size // 4,
                    "sha256": checksums["python"],
                    "three_bindings_byte_exact": True,
                }
            for language in ("python", "go", "dart"):
                path = output / f"{language}-example.wav"
                pcm = wav_pcm(path)
                if hashlib.sha256(pcm).hexdigest() != report["pcm"]["en_M1"]["sha256"]:
                    raise RuntimeError(f"Installed {language} example PCM differs")
                run(["afinfo", path])
                report["examples"][language] = {
                    "wav_sha256": sha(path),
                    "frames": len(pcm) // 4,
                    "pcm_byte_exact": True,
                }
            verify_bundle(model, bundle)
            if sha(library) != report["library_sha256"]:
                raise RuntimeError("Native library changed during qualification")
            for name, expected in report["source_files"].items():
                if sha(ROOT / name) != expected:
                    raise RuntimeError(f"Source changed during qualification: {name}")
            report["passed"] = True
        except BaseException as error:
            report["error"] = str(error)
            raise
        finally:
            write_yaml(report_path, report)
    print(
        "C static/shared and installed Python/Go/Dart synthesis passed; four PCM cases and three WAV examples are exact"
    )


if __name__ == "__main__":
    main()
