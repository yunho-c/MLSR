"""Prepare pinned Supertonic 3 assets and capture/replay reference fixtures."""

import argparse
import json
import os
from pathlib import Path
import sys

from .bundle import (
    DEFAULT_CORPUS,
    DEFAULT_MANIFEST,
    ROOT,
    ReferenceError,
    digest,
    load_manifest,
    load_yaml,
    prepare_bundle,
    verify_bundle,
    write_yaml,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-manifest")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source", type=Path, action="append", default=[])
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="copy" if os.name == "nt" else "symlink",
    )
    prepare.add_argument(
        "--download",
        action="store_true",
        help="Explicitly allow downloading missing pinned files",
    )
    verify = commands.add_parser("verify-bundle")
    verify.add_argument("--bundle", type=Path, required=True)
    frontend = commands.add_parser("check-frontend")
    frontend.add_argument("--bundle", type=Path, required=True)
    frontend.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    frontend.add_argument(
        "--expected",
        type=Path,
        default=ROOT / "tests/fixtures/supertonic3/frontend.yaml",
    )
    capture_parser = commands.add_parser("capture")
    capture_parser.add_argument("--bundle", type=Path, required=True)
    capture_parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    capture_parser.add_argument(
        "--frontend",
        type=Path,
        default=ROOT / "tests/fixtures/supertonic3/frontend.yaml",
    )
    capture_parser.add_argument("--output", type=Path, required=True)
    capture_parser.add_argument(
        "--compute-policy",
        choices=["cpu_and_neural_engine", "cpu_only"],
        default="cpu_and_neural_engine",
    )
    for name in ("verify-reference", "replay", "replay-rust", "summary"):
        sub = commands.add_parser(name)
        sub.add_argument("--reference", type=Path, required=True)
        sub.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
        if name in ("replay", "replay-rust", "summary"):
            sub.add_argument("--output", type=Path, required=True)
        if name in ("replay", "replay-rust"):
            sub.add_argument("--bundle", type=Path, required=True)
            sub.add_argument("--atol", type=float, default=1e-5)
            sub.add_argument("--rtol", type=float, default=1e-5)
        if name == "replay-rust":
            sub.set_defaults(atol=0.0, rtol=0.0)
            sub.add_argument(
                "--binary",
                type=Path,
                default=ROOT / "target/release/examples/coreml_replay",
            )
            sub.add_argument("--passes", type=int, default=2)
            sub.add_argument("--output-backings", action="store_true")
            sub.add_argument("--timeout", type=float, default=180.0)
        if name == "summary":
            sub.add_argument(
                "--replay-report",
                type=Path,
                help="Attach a successful replay report linked to this exact reference",
            )
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if args.command == "validate-manifest":
        result = {"valid": True, "bundle_id": manifest["id"]}
    elif args.command == "prepare":
        result = prepare_bundle(
            manifest,
            args.source,
            args.output,
            args.mode,
            args.download,
            manifest_bytes=args.manifest.read_bytes(),
        )
    elif args.command == "verify-bundle":
        result = verify_bundle(manifest, args.bundle)
    else:
        from .reference import (
            capture,
            check_frontend,
            compact_summary,
            load_corpus,
            read_indexer,
            replay,
            verify_reference,
        )

        corpus = load_corpus(args.corpus)
        if args.command == "check-frontend":
            verify_bundle(manifest, args.bundle)
            result = {
                "cases": check_frontend(
                    corpus,
                    read_indexer(manifest, args.bundle),
                    args.expected,
                    capacity=manifest["contract"]["text_tokens"],
                )
            }
        elif args.command == "capture":
            report = capture(
                manifest,
                args.bundle,
                corpus,
                args.frontend,
                args.output,
                args.compute_policy,
            )
            result = {
                "reference": str(args.output),
                "audio_cases": len(report["cases"]),
                "frontend_cases": len(report["frontend"]),
                "integrity": "passed",
            }
        elif args.command == "replay-rust":
            from .rust_runtime import replay_rust

            report = replay_rust(
                manifest,
                args.bundle,
                corpus,
                args.reference,
                args.output,
                args.binary,
                args.passes,
                args.output_backings,
                args.atol,
                args.rtol,
                args.timeout,
            )
            result = {
                "passed": report["passed"],
                "tensor_comparisons": report["tensor_comparisons"],
                "all_tensors_exact": report["all_tensors_exact"],
                "backings_reused": report["backings_reused"],
                "report": str(args.output),
            }
        elif args.command == "replay":
            report = replay(
                manifest,
                args.bundle,
                corpus,
                args.reference,
                args.output,
                args.atol,
                args.rtol,
            )
            result = {
                "passed": report["passed"],
                "cases": len(report["cases"]),
                "report": str(args.output),
            }
        else:
            report = verify_reference(manifest, corpus, args.reference)
            result = {"valid": True, "cases": len(report["cases"])}
            if args.command == "summary":
                if args.output.exists():
                    raise ReferenceError(f"Summary already exists: {args.output}")
                summary = compact_summary(report)
                if args.replay_report:
                    evidence = load_yaml(args.replay_report)
                    if (
                        evidence.get("kind", "python_synthesis_replay")
                        != "python_synthesis_replay"
                    ):
                        raise ReferenceError(
                            "A runtime-only replay cannot qualify a complete synthesis replay"
                        )
                    if evidence.get("reference_sha256") != digest(
                        args.reference / "reference.yaml"
                    ):
                        raise ReferenceError(
                            "Replay report refers to a different reference capture"
                        )
                    if (
                        evidence.get("passed") is not True
                        or {c["id"] for c in evidence["cases"]}
                        != {c["id"] for c in report["cases"]}
                        or not all(c["passed"] for c in evidence["cases"])
                    ):
                        raise ReferenceError(
                            "Replay report does not establish a passing complete replay"
                        )
                    summary["validation"]["replay"] = "passed"
                    summary["replay"] = {
                        "report_sha256": digest(args.replay_report),
                        "atol": evidence["atol"],
                        "rtol": evidence["rtol"],
                        "all_tensors_exact": all(
                            t["exact"] for c in evidence["cases"] for t in c["tensors"]
                        ),
                    }
                write_yaml(args.output, summary)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
