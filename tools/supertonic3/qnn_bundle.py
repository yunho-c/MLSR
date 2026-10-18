"""Prepare the pinned QNN bundle from existing local artifacts; never download."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil

from .bundle import ROOT, ReferenceError, load_yaml

MANIFEST = ROOT / "model-manifests/supertonic3-qnn-superfast.yaml"


def verify_file(path: Path, record: dict) -> None:
    if not path.is_file() or path.stat().st_size != record["size_bytes"]:
        raise ReferenceError(f"Missing file or incorrect size: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != record["sha256"]:
        raise ReferenceError(f"SHA-256 mismatch: {path}")


def verify(directory: Path) -> None:
    manifest = load_yaml(MANIFEST)
    for artifact in manifest["artifacts"].values():
        for record in artifact["files"]:
            verify_file(directory / record["path"], record)


def prepare(source: Path, output: Path) -> None:
    """Verify every source before creating a new, independent bundle directory."""
    manifest = load_yaml(MANIFEST)
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise ReferenceError(f"Destination already exists; use verify or a fresh path: {output}")
    copies = []
    for name, artifact in manifest["artifacts"].items():
        record = artifact["files"][0]
        path = source / manifest["provenance"]["local_sources"][name]
        verify_file(path, record)
        copies.append((path, output / record["path"]))
    output.mkdir(parents=True)
    # Retain incomplete outputs after an IO failure for diagnosis; never overwrite
    # or remove caller data. A failed preparation must use a fresh destination.
    for src, dst in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    shutil.copyfile(MANIFEST, output / "mlsr-bundle.yaml")
    verify(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("prepare")
    command.add_argument("--source", type=Path, required=True, help="Local supertonic-npu checkout")
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("verify")
    command.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.output)
        print(f"Verified local QNN bundle: {args.output.resolve()}")
    else:
        verify(args.bundle)
        print(f"Verified local QNN bundle: {args.bundle.resolve()}")


if __name__ == "__main__":
    main()
