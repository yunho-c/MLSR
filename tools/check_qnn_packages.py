"""Qualify installed Python and relocated Go against real Windows ARM64 QNN.

Run from a shell prepared by qnn_env.ps1, using a Python with pip/setuptools.
The native ARM64 consumer Python may be the official embedded distribution.
Preserve every package, log and output in a fresh directory outside the checkout.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from check_qnn_sdk_pcm import pcm

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--go", required=True, type=Path)
    parser.add_argument("--cc", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-report", required=True, type=Path)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("This qualification requires Windows ARM64")
    root = args.output.resolve()
    if root.is_relative_to(ROOT):
        parser.error("--output must be outside the checkout")
    root.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "passed": False,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "qualification_script_sha256": sha(Path(__file__)),
        "checks": [],
        "pcm": {},
    }

    def run(name, command, cwd=root, env=None):
        print("+", name, flush=True)
        with (root / f"{name}.log").open("x", encoding="utf-8") as log:
            print(json.dumps([str(x) for x in command]), file=log, flush=True)
            subprocess.run(
                [str(x) for x in command],
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        report["checks"].append(name)

    try:
        run(
            "rust-build",
            [
                "cargo",
                "build",
                "--locked",
                "--release",
                "-p",
                "mlsr-ffi",
                "--features",
                "qnn",
            ],
            cwd=ROOT,
        )
        native, include = root / "native/lib", root / "native/include"
        native.mkdir(parents=True)
        include.mkdir(parents=True)
        for name in ("mlsr_ffi.dll", "mlsr_ffi.dll.lib"):
            shutil.copy2(ROOT / "target/release" / name, native)
        bridge = Path(os.environ["MLSR_QNN_BRIDGE_DIR"])
        for name in ("mlsr_qnn_bridge.dll", "qnn_executorch_backend.dll"):
            shutil.copy2(bridge / name, native)
        shutil.copy2(ROOT / "include/mlsr.h", include)
        report["native_files"] = {p.name: sha(p) for p in native.iterdir()}
        models = root / "models"
        models.mkdir()
        for name in ("supertonic3-qnn-superfast.yaml", "catalog-qnn-experimental.yaml"):
            shutil.copy2(ROOT / "model-manifests" / name, models)
        manifest = models / "supertonic3-qnn-superfast.yaml"
        report["manifest_sha256"] = sha(manifest)
        bundle = args.bundle.resolve(strict=True)

        def external_paths(value):
            return [
                p
                for p in value.split(os.pathsep)
                if p and not Path(p).resolve().is_relative_to(ROOT)
            ]

        env = dict(os.environ)
        env.pop("MLSR_TEST_NATIVE_OUTPUT", None)
        env.update(
            MLSR_LIBRARY_PATH=str(native / "mlsr_ffi.dll"),
            MLSR_QNN_BRIDGE_DIR=str(native),
            MLSR_DLL_DIRECTORIES=os.pathsep.join(
                [str(native), *external_paths(env.get("MLSR_DLL_DIRECTORIES", ""))]
            ),
            PATH=os.pathsep.join(
                [
                    str(native),
                    str(args.cc.resolve().parent),
                    *external_paths(env["PATH"]),
                ]
            ),
            MLSR_TEST_MANIFEST=str(manifest),
            MLSR_TEST_BUNDLE=str(bundle),
            MLSR_TEST_POLICY="qnn_htp",
            MLSR_TEST_CATALOG=str(models / "catalog-qnn-experimental.yaml"),
            MLSR_TEST_VERIFIED_BUNDLE=str(bundle),
            MLSR_TEST_DELIVERY_ID="supertonic3-qnn-superfast-v1",
            MLSR_TEST_DELIVERY_PLATFORM="windows-arm64",
        )
        wheels, installed = root / "wheels", root / "python-package"
        run(
            "wheel-build",
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
            ],
        )
        (wheel,) = wheels.glob("mlsr-*.whl")
        report["wheel_sha256"] = sha(wheel)
        run(
            "wheel-install",
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-compile",
                "--target",
                installed,
                wheel,
            ],
        )
        env["PYTHONPATH"] = str(installed)
        python = args.python.resolve(strict=True)
        # Embedded Python ignores PYTHONPATH. Copy its runtime and replace only
        # the copy's search path, also covering subprocesses spawned by tests.
        pth = list(python.parent.glob("python*._pth"))
        if pth:
            runtime = root / "python-runtime"
            shutil.copytree(python.parent, runtime)
            for file in pth:
                (runtime / file.name).write_text(
                    file.stem + ".zip\n.\n" + str(installed) + "\n", encoding="utf-8"
                )
            python = runtime / python.name
        python_tests = root / "python-tests"
        shutil.copytree(
            ROOT / "bindings/python/tests",
            python_tests,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        example = root / "python-example.py"
        shutil.copy2(ROOT / "bindings/python/examples/developer_api.py", example)
        probe = """
import ctypes, json, platform, sys
from pathlib import Path
import mlsr
from mlsr._ffi import _library
assert Path(mlsr.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
assert platform.machine().upper() == 'ARM64', platform.machine()
lib = _library()
lib.mlsr_features.restype = ctypes.c_uint32
assert lib.mlsr_features() == 45, lib.mlsr_features()
print(json.dumps({'python': sys.version, 'package': mlsr.__file__, 'library': lib._name, 'features': lib.mlsr_features()}))
"""
        run("python-installed-origin", [python, "-c", probe, installed], env=env)
        env.update(
            MLSR_TEST_OUTPUT=str(root / "python-pcm"),
            MLSR_SDK_OUTPUT=str(root / "python-sdk"),
        )
        run(
            "python-tests",
            [python, "-m", "unittest", "discover", "-s", python_tests, "-v"],
            env=env,
        )
        run(
            "python-example",
            [
                python,
                example,
                "--manifest",
                manifest,
                "--bundle",
                bundle,
                "--policy",
                "qnn_htp",
                "--cancel-first",
                "--output",
                root / "python-example.wav",
            ],
            env=env,
        )

        go = root / "go-package"
        shutil.copytree(ROOT / "bindings/go", go)

        def flag(value):
            return '"' + str(value).replace("\\", "/") + '"'

        env.update(
            GOWORK="off",
            CGO_ENABLED="1",
            CC=flag(args.cc.resolve()),
            CGO_CFLAGS=flag("-I" + str(include)),
            CGO_LDFLAGS=flag("-L" + str(native)),
            MLSR_TEST_OUTPUT=str(root / "go-pcm"),
            MLSR_SDK_OUTPUT=str(root / "go-sdk"),
        )
        run("go-origin", [args.go, "list", "-m", "-f", "{{.Dir}}"], cwd=go, env=env)
        run(
            "go-tests",
            [args.go, "test", "-count=1", "-timeout=120s", "-v", "./..."],
            cwd=go,
            env=env,
        )
        run("go-vet", [args.go, "vet", "./..."], cwd=go, env=env)
        run(
            "go-example",
            [
                args.go,
                "run",
                "./examples/developer_api",
                "--manifest",
                manifest,
                "--bundle",
                bundle,
                "--policy",
                "qnn_htp",
                "--output",
                root / "go-example.wav",
            ],
            cwd=go,
            env=env,
        )
        reference = json.loads(args.reference_report.read_text(encoding="utf-8-sig"))[
            "pcm"
        ]
        for name in ("en_M1", "en_F1", "ko_M1", "ko_F1"):
            expected = (root / "python-pcm" / f"{name}.f32").read_bytes()
            digest = hashlib.sha256(expected).hexdigest()
            assert digest == reference[name]["sha256"], (
                f"Prior qualification differs: {name}"
            )
            for language in ("python", "go"):
                assert (
                    root / f"{language}-pcm" / f"{name}.f32"
                ).read_bytes() == expected
                assert pcm(root / f"{language}-sdk" / f"{name}.wav") == expected
            report["pcm"][name] = {
                "sha256": digest,
                "frames": len(expected) // 4,
                "byte_exact": True,
            }
        expected = (root / "python-pcm/en_M1.f32").read_bytes()
        for language in ("python", "go"):
            assert pcm(root / f"{language}-example.wav") == expected
        report["passed"] = True
    except Exception as error:
        report["failure"] = str(error)
        raise
    finally:
        report["logs"] = {p.name: sha(p) for p in root.glob("*.log")}
        (root / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
