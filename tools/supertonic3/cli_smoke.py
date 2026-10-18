"""Run the Rust CLI and independently verify Float32 WAVs against seeded replay."""

import argparse
import hashlib
import math
from pathlib import Path
import struct
import subprocess

from .bundle import (
    DEFAULT_CORPUS,
    DEFAULT_MANIFEST,
    ROOT,
    ReferenceError,
    canonical_digest,
    digest,
    load_manifest,
    load_yaml,
    verify_bundle,
    write_yaml,
)
from .reference import load_corpus, verify_reference


def wav_pcm(path):
    """Read the relevant RIFF contract independently of the Rust/Hound writer."""
    raw = path.read_bytes()
    if (
        len(raw) < 12
        or raw[:4] != b"RIFF"
        or raw[8:12] != b"WAVE"
        or struct.unpack_from("<I", raw, 4)[0] != len(raw) - 8
    ):
        raise ReferenceError("Invalid WAV container")
    offset, chunks = 12, {}
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise ReferenceError("Truncated WAV chunk")
        name, size = struct.unpack_from("<4sI", raw, offset)
        start, end = offset + 8, offset + 8 + size
        if end > len(raw) or name in chunks:
            raise ReferenceError("Truncated or repeated WAV chunk")
        chunks[name] = raw[start:end]
        offset = end + (size % 2)
    if offset != len(raw):
        raise ReferenceError("Missing WAV chunk padding")
    fmt = chunks.get(b"fmt ", b"")
    if len(fmt) < 16 or b"data" not in chunks:
        raise ReferenceError("Missing WAV format/data")
    tag, channels, rate, byte_rate, alignment, bits = struct.unpack_from("<HHIIHH", fmt)
    if tag == 65534:
        float_guid = bytes.fromhex("0300000000001000800000aa00389b71")
        if (
            len(fmt) < 40
            or fmt[24:40] != float_guid
            or struct.unpack_from("<H", fmt, 18)[0] != 32
        ):
            raise ReferenceError("Unsupported extensible WAV format")
    elif tag != 3:
        raise ReferenceError("Expected IEEE Float32 WAV")
    pcm = chunks[b"data"]
    if (
        (channels, rate, byte_rate, alignment, bits) != (1, 44100, 176400, 4, 32)
        or not pcm
        or len(pcm) % 4
    ):
        raise ReferenceError("Unexpected WAV audio contract")
    if not all(math.isfinite(v[0]) for v in struct.iter_unpack("<f", pcm)):
        raise ReferenceError("Nonfinite WAV PCM")
    return pcm


def smoke(manifest_path, bundle, reference, replay_path, binary, output):
    manifest = load_manifest(manifest_path)
    corpus = load_corpus(DEFAULT_CORPUS)
    saved = verify_reference(manifest, corpus, reference)
    replay = load_yaml(replay_path)
    if (
        replay.get("kind") != "rust_synthesis_replay"
        or replay.get("passed") is not True
        or replay.get("manifest_sha256") != canonical_digest(manifest)
        or replay.get("reference_sha256") != digest(reference / "reference.yaml")
    ):
        raise ReferenceError(
            "Require passing synthesis evidence for this bundle/reference"
        )
    receipts = {c["id"]: c for c in replay["seeded_checks"]}
    if set(receipts) != {
        "en_m1_basic",
        "en_f1_basic",
        "ko_m1_basic",
        "ko_f1_basic",
    } or not all(c["passed"] for c in receipts.values()):
        raise ReferenceError("Missing seeded CLI comparison receipts")
    verify_bundle(manifest, bundle)
    binary = binary.resolve(strict=True)
    output = output.resolve()
    output.mkdir()  # Must be a new directory; preserve all previous evidence.
    report_path = output / "report.yaml"
    report = {
        "schema_version": 1,
        "kind": "rust_cli_smoke",
        "passed": False,
        "binary_sha256": digest(binary),
        "synthesis_report_sha256": digest(replay_path),
        "manifest_sha256": canonical_digest(manifest),
        "source_files": {
            p.relative_to(ROOT).as_posix(): digest(p)
            for p in [
                ROOT / "crates/mlsr-cli/Cargo.toml",
                ROOT / "crates/mlsr-cli/src/main.rs",
                Path(__file__),
            ]
        },
        "cases": [],
        "rejections": [],
        "overwrites_rejected": 0,
    }
    write_yaml(report_path, report)
    try:
        with (output / "commands.log").open("x") as log:

            def call(args):
                command = [str(binary), *map(str, args)]
                result = subprocess.run(
                    command, cwd=output, capture_output=True, text=True, timeout=180
                )
                print(command, file=log)
                print(result.stdout, result.stderr, file=log, flush=True)
                return result

            assets = [
                "--manifest",
                manifest_path.resolve(),
                "--bundle",
                bundle.resolve(),
            ]
            verified = call(["verify", *assets])
            if verified.returncode != 0:
                raise ReferenceError(verified.stderr)
            policy = ["--compute-policy", replay["compute_policy"].replace("_", "-")]
            if not replay["output_backings"]:
                policy.append("--no-output-reuse")
            for case in saved["cases"]:
                if case["id"] not in receipts:
                    continue
                wav = output / f"{case['id']}.wav"
                text = ["--text", case["text"]]
                if case["id"] == "ko_f1_basic":
                    source = output / "input.txt"
                    source.write_text(case["text"], encoding="utf-8")
                    text = ["--text-file", source]
                args = [
                    "synthesize",
                    *assets,
                    *policy,
                    *text,
                    "--language",
                    case["language"],
                    "--voice",
                    case["voice"],
                    "--steps",
                    str(case["steps"]),
                    "--speed",
                    str(case["effective_speed"]),
                    "--seed",
                    str(case["seed"]),
                    "--output",
                    wav,
                ]
                result = call(args)
                if result.returncode != 0:
                    raise ReferenceError(result.stderr)
                pcm = wav_pcm(wav)
                checksum = hashlib.sha256(pcm).hexdigest()
                if (
                    len(pcm) // 4 != case["pcm_frames"]
                    or checksum != receipts[case["id"]]["pcm_sha256"]
                ):
                    raise ReferenceError(
                        f"CLI PCM differs from Rust replay: {case['id']}"
                    )
                before = digest(wav)
                refused = call(args)
                if (
                    refused.returncode == 0
                    or "output already exists" not in refused.stderr
                    or digest(wav) != before
                ):
                    raise ReferenceError("CLI overwrote existing evidence")
                report["overwrites_rejected"] += 1
                report["cases"].append(
                    {
                        "id": case["id"],
                        "passed": True,
                        "wav": wav.name,
                        "frames": len(pcm) // 4,
                        "pcm_sha256": checksum,
                        "wav_sha256": before,
                    }
                )
            for case in corpus["cases"]:
                if "expected_error" not in case:
                    continue
                wav = output / f"rejected-{case['id']}.wav"
                result = call(
                    [
                        "synthesize",
                        *assets,
                        *policy,
                        "--text",
                        case["text"],
                        "--language",
                        case["language"],
                        "--voice",
                        case["voice"],
                        "--output",
                        wav,
                    ]
                )
                if (
                    result.returncode == 0
                    or f"{case['expected_error']}:" not in result.stderr
                    or wav.exists()
                ):
                    raise ReferenceError(f"CLI did not reject {case['id']} cleanly")
                report["rejections"].append({"id": case["id"], "passed": True})
        verify_bundle(manifest, bundle)
        verify_reference(manifest, corpus, reference)
        report["passed"] = (
            len(report["cases"]) == 4
            and len(report["rejections"]) == 5
            and report["overwrites_rejected"] == 4
        )
    except BaseException as error:
        report["error"] = str(error)
        report["passed"] = False
        raise
    finally:
        write_yaml(report_path, report)
    if not report["passed"]:
        raise ReferenceError("Incomplete CLI qualification")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/mlsr")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    smoke(
        args.manifest,
        args.bundle,
        args.reference,
        args.replay_report,
        args.binary,
        args.output_dir,
    )
    print(
        "CLI passed: four bit-exact WAVs, five input rejections, four overwrite protections"
    )


if __name__ == "__main__":
    main()
