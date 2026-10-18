#!/usr/bin/env python3
"""Build/test the Apple adapter and native artifacts without downloaded weights."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "target"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolchain", help="Existing rustup toolchain, e.g. 1.88.0")
    parser.add_argument(
        "--cross",
        action="store_true",
        help="Also link Intel macOS and iOS device/simulator artifacts (targets must be installed)",
    )
    args = parser.parse_args()
    if sys.platform != "darwin":
        raise SystemExit("Core ML checks require macOS and Xcode")
    env = dict(
        os.environ, MACOSX_DEPLOYMENT_TARGET="15.0", IPHONEOS_DEPLOYMENT_TARGET="18.0"
    )
    cargo = ["cargo"] + (["+" + args.toolchain] if args.toolchain else [])

    def run(*command):
        print("+", *command, flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    def rust(*command):
        run(*cargo, *command, "--locked", "--target-dir", str(TARGET))

    def check_deployment(library, platform, minimum):
        command = ["xcrun", "vtool", "-show-build", str(library)]
        print("+", *command, flush=True)
        metadata = subprocess.check_output(command, cwd=ROOT, env=env, text=True)
        print(metadata, end="", flush=True)
        fields = dict(
            re.findall(r"^\s*(platform|minos)\s+(\S+)\s*$", metadata, re.MULTILINE)
        )
        version = fields.get("minos", "").split(".")
        expected = minimum.split(".")
        while len(version) < len(expected):
            version.append("0")
        while len(expected) < len(version):
            expected.append("0")
        if fields.get("platform") != platform or version != expected:
            raise RuntimeError(
                f"Unexpected deployment metadata for {library}: {fields}"
            )

    rust(
        "test", "-p", "mlsr", "-p", "mlsr-ffi", "-p", "mlsr-cli", "--features", "coreml"
    )
    rust(
        "build", "--release", "-p", "mlsr-ffi", "-p", "mlsr-cli", "--features", "coreml"
    )
    run(str(TARGET / "release/mlsr"), "--help")
    check_deployment(TARGET / "release/mlsr", "MACOS", "15.0")
    rust(
        "build",
        "--release",
        "-p",
        "mlsr",
        "--example",
        "coreml_ffi_probe",
        "--features",
        "coreml",
    )
    examples = TARGET / "release/examples"
    shared = examples / "libcoreml_ffi_probe.dylib"
    static = examples / "libcoreml_ffi_probe.a"
    check_deployment(shared, "MACOS", "15.0")
    check_deployment(TARGET / "release/libmlsr_ffi.dylib", "MACOS", "15.0")
    resource_dir = subprocess.check_output(
        ["xcrun", "clang", "-print-resource-dir"], text=True
    ).strip()
    native_links = [
        "-L" + str(Path(resource_dir) / "lib/darwin"),
        "-lclang_rt.osx",
        "-lobjc",
    ]
    # Static archives don't propagate Rust's native linkage metadata to C callers.
    for framework in (
        "Foundation",
        "CoreFoundation",
        "CoreGraphics",
        "ImageIO",
        "CoreVideo",
        "CoreML",
    ):
        native_links.extend(["-framework", framework])
    with tempfile.TemporaryDirectory(prefix="mlsr-coreml-ffi-") as temporary:
        temporary = Path(temporary)
        run(
            "xcrun",
            "coremlcompiler",
            "compile",
            str(ROOT / "tests/fixtures/coreml/affine.mlmodel"),
            str(temporary),
        )
        model = temporary / "affine.mlmodelc"
        for name, library, links in [
            ("shared", shared, []),
            ("static", static, native_links),
        ]:
            executable = temporary / name
            run(
                "xcrun",
                "clang",
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-mmacosx-version-min=15.0",
                str(ROOT / "tests/integration/coreml_smoke.c"),
                str(library),
                *links,
                "-o",
                str(executable),
            )
            run(str(executable), str(model))
        run(
            sys.executable,
            str(ROOT / "tests/integration/coreml_ctypes.py"),
            str(shared),
            str(model),
        )
    if args.cross:
        for target, platform, minimum in (
            ("x86_64-apple-darwin", "MACOS", "15.0"),
            ("aarch64-apple-ios", "IOS", "18.0"),
            ("aarch64-apple-ios-sim", "IOSSIMULATOR", "18.0"),
        ):
            rust(
                "build",
                "--release",
                "-p",
                "mlsr-ffi",
                "--features",
                "coreml",
                "--target",
                target,
            )
            # This exported development function really calls Session/predict,
            # ensuring the adapter is linked even before public synthesis FFI exists.
            rust(
                "build",
                "--release",
                "-p",
                "mlsr",
                "--example",
                "coreml_ffi_probe",
                "--features",
                "coreml",
                "--target",
                target,
            )
            check_deployment(
                TARGET / target / "release/examples/libcoreml_ffi_probe.dylib",
                platform,
                minimum,
            )
            check_deployment(
                TARGET / target / "release/libmlsr_ffi.dylib", platform, minimum
            )
    print("Core ML runtime and native packaging checks passed", flush=True)


if __name__ == "__main__":
    main()
