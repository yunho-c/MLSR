"""Regenerate frontend expectations using the pinned upstream implementation.

Run with an existing Python that has NumPy, PyYAML and ONNX Runtime. ONNX Runtime
is imported by upstream helper.py; this command does not load or execute models.
"""

import argparse
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

from .bundle import (
    DEFAULT_CORPUS,
    DEFAULT_MANIFEST,
    ReferenceError,
    canonical_digest,
    digest,
    load_manifest,
    verify_bundle,
    write_yaml,
)
from .reference import load_corpus


def freeze(manifest, corpus, upstream, bundle, output):
    if output.exists():
        raise ReferenceError(f"Output already exists: {output}")
    verify_bundle(manifest, bundle)
    provenance = manifest["provenance"]["upstream_preprocessing"]
    source = upstream / provenance["path"]
    # Verify the source against its pinned Git object, even in a dirty checkout.
    committed = subprocess.check_output(
        [
            "git",
            "-C",
            str(upstream),
            "show",
            f"{provenance['revision']}:{provenance['path']}",
        ]
    )
    if source.read_bytes() != committed:
        raise ReferenceError(
            "Upstream preprocessing file differs from its pinned revision"
        )
    spec = importlib.util.spec_from_file_location("mlsr_upstream_frontend", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    index_path = bundle / manifest["artifacts"]["unicode_indexer"]["path"]
    processor = module.UnicodeProcessor(str(index_path))
    fixture = {
        "schema_version": 1,
        "corpus_sha256": canonical_digest(corpus),
        "provenance": {
            "repository": provenance["repository"],
            "revision": provenance["revision"],
            "file": provenance["path"],
            "source_sha256": hashlib.sha256(committed).hexdigest(),
            "unicode_indexer_sha256": digest(index_path),
            "negative_cases": "MLSR explicit validation policy; not upstream permissive behavior.",
        },
        "cases": {},
    }
    for case in corpus["cases"]:
        if "expected_error" in case:
            fixture["cases"][case["id"]] = {"error": case["expected_error"]}
            continue
        normalized = processor._preprocess_text(case["text"], case["language"])
        ids, _ = processor([case["text"]], [case["language"]])
        if len(normalized) > manifest["contract"]["text_tokens"] or (ids < 0).any():
            raise ReferenceError(
                f"Positive corpus case cannot fit this export: {case['id']}"
            )
        fixture["cases"][case["id"]] = {
            "normalized_text": normalized,
            "tokens": ids[0].tolist(),
        }
    write_yaml(output, fixture)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(
        load_manifest(args.manifest),
        load_corpus(args.corpus),
        args.upstream_root,
        args.bundle,
        args.output,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
