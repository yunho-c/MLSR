"""Real Core ML voice/language smoke checks and exact legacy-voice regression.

This establishes finite, non-silent PCM and SDK coverage, not intelligibility,
speaker similarity or listening quality. Uses the existing Python interpreter.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

from .bundle import ROOT, digest, load_manifest, verify_bundle, write_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "model-manifests/supertonic3-coreml-fp16-voices.yaml",
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--legacy-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--library", type=Path, default=ROOT / "target/release/libmlsr_ffi.dylib"
    )
    parser.add_argument("--save-all-audio", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    verify_bundle(manifest, args.bundle)
    old_manifest = ROOT / "model-manifests/supertonic3-coreml-fp16.yaml"
    verify_bundle(load_manifest(old_manifest), args.legacy_bundle)
    samples_path = ROOT / "examples/flutter_speak/assets/languages.json"
    samples = json.loads(samples_path.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["MLSR_LIBRARY_PATH"] = str(args.library.resolve())
    sys.path.insert(0, str(ROOT / "bindings/python/src"))
    from mlsr import Request, SupertonicOptions, Tts
    from mlsr import _sdk as ffi

    native = ffi.library()
    handles = native.mlsr_live_handles()
    report = {
        "schema_version": 1,
        "passed": False,
        "scope": "Finite non-silent PCM for every preset/language; not listening or intelligibility qualification",
        "manifest_sha256": digest(args.manifest),
        "legacy_manifest_sha256": digest(old_manifest),
        "library_sha256": digest(args.library),
        "script_sha256": digest(Path(__file__)),
        "samples_sha256": digest(samples_path),
        "host": platform.platform(),
        "compute_policy": "cpu_and_neural_engine",
        "steps": 8,
        "speaking_rate": 1.0,
        "raw_speed": 1.05,
        "seed": 1729,
        "timing_boundary": "Blocking Tts.render complete call, excluding model load and WAV writing; smoke timings are not a benchmark",
        "cases": [],
        "legacy_regression": [],
    }
    try:
        with Tts(args.manifest, args.bundle) as tts:
            voices = tts.voices
            assert [v.model_voice_id for v in voices] == manifest["contract"]["voices"]
            assert all(set(v.languages) == set(samples) for v in voices)
            assert all(v.readiness == "engine_ready" for v in voices)
            report["voices"] = [dataclasses.asdict(v) for v in voices]
            for voice in voices:
                for language in voice.languages:
                    text = samples[language]["sample"]
                    row = {
                        "voice": voice.model_voice_id,
                        "language": language,
                        "text": text,
                        "passed": False,
                    }
                    report["cases"].append(row)
                    try:
                        request = Request(
                            text,
                            locale=language,
                            voice=voice.id,
                            backend=SupertonicOptions(seed=1729),
                        )
                        start = time.perf_counter()
                        clip = tts.render(request)
                        elapsed = time.perf_counter() - start
                        pcm = np.frombuffer(clip.samples, dtype=np.float32)
                        assert (
                            clip.format.sample_rate == 44100
                            and clip.format.channels == 1
                        )
                        assert 0 < clip.frames <= 30 * 44100 and np.isfinite(pcm).all()
                        peak = float(np.max(np.abs(pcm)))
                        rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
                        assert peak > 1e-4 and rms > 1e-5, "silent PCM"
                        row.update(
                            frames=clip.frames,
                            pcm_sha256=hashlib.sha256(
                                clip.samples.tobytes()
                            ).hexdigest(),
                            peak=peak,
                            rms=rms,
                            render_ms=elapsed * 1000,
                        )
                        if (
                            args.save_all_audio
                            or voice == voices[0]
                            or language in ("en", "ko")
                        ):
                            name = f"{voice.model_voice_id}-{language}.wav"
                            clip.write_wav(args.output / name)
                            row["wav"] = name
                        row["passed"] = True
                    except Exception as error:
                        row["error"] = str(error)
                print(
                    f"{voice.model_voice_id}: {sum(r['passed'] for r in report['cases'] if r['voice'] == voice.model_voice_id)}/{len(voice.languages)} passed",
                    flush=True,
                )
        with Tts(old_manifest, args.legacy_bundle) as old:
            for voice in ("M1", "F1"):
                for language in ("en", "ko"):
                    clip = old.render(
                        Request(
                            samples[language]["sample"],
                            locale=language,
                            voice=f"supertonic3:{voice}",
                            backend=SupertonicOptions(seed=1729),
                        )
                    )
                    sha = hashlib.sha256(clip.samples.tobytes()).hexdigest()
                    matching = next(
                        r
                        for r in report["cases"]
                        if r["voice"] == voice and r["language"] == language
                    )
                    report["legacy_regression"].append(
                        {
                            "voice": voice,
                            "language": language,
                            "frames": clip.frames,
                            "pcm_sha256": sha,
                            "exact": matching.get("pcm_sha256") == sha
                            and matching.get("frames") == clip.frames,
                        }
                    )
        report["live_handles_before"] = handles
        report["live_handles_after"] = native.mlsr_live_handles()
        report["passed"] = (
            len(report["cases"]) == 310
            and all(r["passed"] for r in report["cases"])
            and all(r["exact"] for r in report["legacy_regression"])
            and report["live_handles_after"] == handles
        )
    finally:
        write_yaml(args.output / "report.yaml", report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "cases": len(report["cases"]),
                "report": str(args.output / "report.yaml"),
            }
        ),
        flush=True,
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
