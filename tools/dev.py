#!/usr/bin/env python3
"""Portable development commands. Requires Python 3.11+; no virtualenv needed."""

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
DART = ROOT / "bindings/dart"
GO = ROOT / "bindings/go"
PYTHON = ROOT / "bindings/python"
TARGET = ROOT / "target"
NATIVE = TARGET / "debug"
VERSION = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["package"][
    "version"
]


def run(*args, cwd=ROOT, env=None):
    command = [str(arg) for arg in args]
    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def library_name():
    if sys.platform == "darwin":
        return "libmlsr_ffi.dylib"
    if sys.platform == "win32":
        return "mlsr_ffi.dll"
    if sys.platform.startswith("linux"):
        return "libmlsr_ffi.so"
    raise RuntimeError(f"Native build paths are not configured for {sys.platform}.")


def build():
    # Pin the local output directory so CARGO_TARGET_DIR does not break consumers.
    run("cargo", "build", "--locked", "-p", "mlsr-ffi", "--target-dir", TARGET)


def native_environment(native=NATIVE, include=ROOT / "include"):
    env = dict(os.environ)
    env["MLSR_LIBRARY_PATH"] = str(native / library_name())
    env["MLSR_EXPECTED_VERSION"] = VERSION
    env["CGO_ENABLED"] = "1"
    env["CGO_CFLAGS"] = " ".join(
        filter(None, [env.get("CGO_CFLAGS"), shlex.quote("-I" + str(include))])
    )
    env["CGO_LDFLAGS"] = " ".join(
        filter(None, [env.get("CGO_LDFLAGS"), shlex.quote("-L" + str(native))])
    )
    loader_key = (
        "PATH"
        if sys.platform == "win32"
        else ("DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH")
    )
    env[loader_key] = os.pathsep.join(filter(None, [str(native), env.get(loader_key)]))
    env["PYTHONPATH"] = str(PYTHON / "src")
    return env


def test_c():
    compiler = os.environ.get("CC") or shutil.which("gcc" if os.name == "nt" else "cc")
    if not compiler:
        raise RuntimeError("A C compiler is required for the ABI smoke test.")
    executable = NATIVE / ("ffi-smoke.exe" if os.name == "nt" else "ffi-smoke")
    flags = [] if os.name == "nt" else ["-Wl,-rpath," + str(NATIVE)]
    run(
        compiler,
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Werror",
        ROOT / "tests/integration/ffi_smoke.c",
        "-I",
        ROOT / "include",
        "-L",
        NATIVE,
        "-lmlsr_ffi",
        *flags,
        "-o",
        executable,
    )
    run(executable, VERSION, env=native_environment())


def test_python():
    run(
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        PYTHON / "tests",
        "-v",
        env=native_environment(),
    )


def test_go():
    files = sorted(GO.rglob("*.go"))
    unformatted = subprocess.check_output(["gofmt", "-l", *map(str, files)], text=True)
    if unformatted:
        raise RuntimeError("Run gofmt on:\n" + unformatted)
    env = native_environment()
    # Go's test cache does not track changes to a separately built shared library.
    run("go", "test", "-count=1", "./...", cwd=GO, env=env)
    run("go", "vet", "./...", cwd=GO, env=env)


def dart_get():
    run("dart", "pub", "get", "--enforce-lockfile", cwd=DART)


def test_dart():
    dart_get()
    run(
        "dart",
        "format",
        "--output=none",
        "--set-exit-if-changed",
        "lib",
        "hook",
        "tool",
        "test",
        "example",
        cwd=DART,
    )
    run("dart", "analyze", "--fatal-infos", cwd=DART)
    run("dart", "test", cwd=DART, env=native_environment())


def generate(check=False):
    suffix = ["--", "--check"] if check else []
    run("cargo", "run", "--locked", "-p", "mlsr-codegen", *suffix)
    dart_get()
    suffix = ["--check"] if check else []
    run("dart", "run", "tool/generate_bindings.dart", *suffix, cwd=DART)


def package_python():
    wheels = TARGET / "wheels"
    run(
        sys.executable,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--no-build-isolation",
        "--wheel-dir",
        wheels,
        PYTHON,
    )
    wheel = wheels / f"mlsr-{VERSION}-py3-none-any.whl"
    with tempfile.TemporaryDirectory(prefix="mlsr-wheel-") as directory:
        installed = Path(directory) / "installed"
        run(
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-compile",
            "--target",
            installed,
            wheel,
        )
        env = native_environment()
        env["PYTHONPATH"] = str(installed)
        run(
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            PYTHON / "tests",
            "-v",
            cwd=directory,
            env=env,
        )


def test_standalone():
    """Check package copies and relocated native assets outside the checkout."""
    with tempfile.TemporaryDirectory(
        prefix="mlsr-standalone-package-installation-"
    ) as directory:
        root = Path(directory)
        native = root / "native/lib"
        include = root / "native/include"
        native.mkdir(parents=True)
        include.mkdir(parents=True)
        shutil.copy2(NATIVE / library_name(), native)
        shutil.copy2(ROOT / "include/mlsr.h", include)
        # Keep Windows import libraries alongside the DLL when present.
        for pattern in ("mlsr_ffi*.lib", "*mlsr_ffi*.dll.a"):
            for source in NATIVE.glob(pattern):
                shutil.copy2(source, native)
        env = native_environment(native=native, include=include)

        dart = root / "dart"
        shutil.copytree(
            DART, dart, ignore=shutil.ignore_patterns(".dart_tool", "build")
        )
        with (dart / "pubspec.yaml").open("a", encoding="utf-8") as pubspec:
            pubspec.write(
                "\nhooks:\n  user_defines:\n    mlsr:\n      native-library: "
                + json.dumps(str(native / library_name()))
                + "\n"
            )
        run("dart", "pub", "get", "--enforce-lockfile", cwd=dart)
        run("dart", "test", cwd=dart, env=env)

        go = root / "go"
        shutil.copytree(GO, go)
        run("go", "test", "-count=1", "./...", cwd=go, env=env)


def check():
    run("cargo", "fmt", "--all", "--", "--check")
    run(
        "cargo",
        "clippy",
        "--locked",
        "--workspace",
        "--all-targets",
        "--",
        "-D",
        "warnings",
    )
    run("cargo", "test", "--locked", "--workspace")
    run("cargo", "run", "--locked", "-p", "mlsr-codegen", "--", "--check")
    build()
    test_c()
    test_python()
    test_go()
    test_dart()


def test_model_tools():
    """Validate model metadata and portable tooling, without weights or Core ML."""
    run(sys.executable, "-m", "tools.supertonic3", "validate-manifest")
    run(
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        ROOT / "tools/supertonic3/tests",
        "-v",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "build",
            "check",
            "generate",
            "check-generated",
            "test-c",
            "test-python",
            "test-go",
            "test-dart",
            "test-standalone",
            "package-python",
            "test-model-tools",
        ],
    )
    command = parser.parse_args().command
    if command == "generate":
        generate()
    elif command == "check-generated":
        generate(check=True)
    elif command == "check":
        check()
    elif command == "test-model-tools":
        test_model_tools()
    else:
        build()
        tasks = {
            "test-c": test_c,
            "test-python": test_python,
            "test-go": test_go,
            "test-dart": test_dart,
            "test-standalone": test_standalone,
            "package-python": package_python,
        }
        if command in tasks:
            tasks[command]()


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
