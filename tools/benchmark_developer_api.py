#!/usr/bin/env python3
"""Opt-in macOS SDK timing probe with separate PCM, playback and cleanup boundaries.

Run against a separately built native library. Uses no venv, model downloads,
cache eviction or audible playback. Run other qualification suites beforehand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import struct
import subprocess
import time


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def pcm_digest(path):
    data = Path(path).read_bytes()
    offset = 12
    while offset + 8 <= len(data):
        size = struct.unpack_from("<I", data, offset + 4)[0]
        if offset + 8 + size > len(data):
            raise RuntimeError("truncated WAV")
        if data[offset : offset + 4] == b"data":
            return hashlib.sha256(data[offset + 8 : offset + 8 + size]).hexdigest()
        offset += 8 + size + (size & 1)
    raise RuntimeError("missing WAV data")


def milliseconds(started):
    return (time.perf_counter_ns() - started) / 1e6


def stats(values):
    return {
        "count": len(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("this qualification targets macOS Apple Silicon")
    if not 1 <= args.repeats <= 10:
        parser.error("--repeats must be in 1..10")
    args.output.mkdir()
    from mlsr import Request, SpeechOptions, SupertonicOptions, Tts
    from mlsr import _sdk as ffi  # Only registry counts are diagnostic/private.

    lib = ffi.library()
    before = lib.mlsr_live_handles()
    report = {
        "schema_version": 1,
        "passed": False,
        "platform": platform.platform(),
        "hardware": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "power_state": subprocess.check_output(
            ["pmset", "-g", "batt"], text=True
        ).strip(),
        "python": platform.python_version(),
        "muted": True,
        "native_library_sha256": digest(os.environ["MLSR_LIBRARY_PATH"]),
        "manifest_sha256": digest(args.manifest),
        "script_sha256": digest(__file__),
        "timing_boundary": {
            "first_pcm_ms": "synthesize admission through first copied PCM from the public Python/C audio iterator",
            "speech_first_buffered_ms": "speak admission through first nonzero retained PCM snapshot; 2 ms poll interval",
            "speech_first_position_ms": "speak admission through first positive latency-adjusted source position; 2 ms poll interval, not acoustic onset",
            "speech_done_ms": "speak admission through observed native playback drain and quiescent completion",
            "cancel_call_ms": "native cancellation request return; not native flush completion",
            "cancel_to_done_ms": "cancel request through quiescent terminal outcome",
            "initialization": "fresh Python Tts with existing OS compiler/file caches, no system-cache eviction",
            "rss": "Python process peak RSS including Rust/Core ML mappings; excludes other processes and driver allocations",
        },
        "samples": [],
        "cancellations": [],
    }
    tts = None
    try:
        started = time.perf_counter_ns()
        tts = Tts(args.manifest, args.bundle)
        report["initialization_ms"] = milliseconds(started)
        # Warm the inference and native output paths separately; do not fold
        # first output graph startup into the subsequent warm request summary.
        tts.render("Warm up the shared synthesis lane.")
        started = time.perf_counter_ns()
        with tts.speak("Ready.", options=SpeechOptions(volume=0)) as warmup:
            warmup.wait()
        report["first_output_warmup_through_drain_ms"] = milliseconds(started)
        for repeat in range(args.repeats):
            for index, (language, voice) in enumerate(
                (("en", "M1"), ("en", "F1"), ("ko", "M1"), ("ko", "F1"))
            ):
                text = (
                    "Hello from MLSR."
                    if language == "en"
                    else "안녕하세요. 반갑습니다."
                )
                request = Request(
                    text,
                    locale=language,
                    voice="supertonic3:" + voice,
                    backend=SupertonicOptions(seed=1729 + index),
                )
                row = {
                    "repeat": repeat,
                    "language": language,
                    "voice": voice,
                    "seed": 1729 + index,
                    "steps": 8,
                    "speaking_rate": 1.0,
                    "effective_speed": 1.05,
                }
                started = time.perf_counter_ns()
                frames, hashed = 0, hashlib.sha256()
                with tts.synthesize(request) as job:
                    with job.audio() as audio:
                        for chunk in audio:
                            if frames == 0:
                                row["first_pcm_ms"] = milliseconds(started)
                            if chunk.start_frame != frames:
                                raise RuntimeError("PCM frame discontinuity")
                            frames += chunk.audio.frames
                            hashed.update(chunk.audio.samples.tobytes())
                    completion = job.wait()
                    if (
                        completion.outcome.kind != "completed"
                        or completion.frames != frames
                    ):
                        raise RuntimeError("synthesis did not complete")
                    row["synthesis_done_ms"] = milliseconds(started)
                row.update(frames=frames, pcm_sha256=hashed.hexdigest())
                started = time.perf_counter_ns()
                with tts.speak(request, options=SpeechOptions(volume=0)) as speech:
                    deadline = time.monotonic() + 60
                    while True:
                        state = speech.snapshot
                        if state.buffered_until_frame > 0:
                            row.setdefault(
                                "speech_first_buffered_ms", milliseconds(started)
                            )
                        if state.position_frame > 0:
                            row.setdefault(
                                "speech_first_position_ms", milliseconds(started)
                            )
                        if state.completion:
                            state.completion.result()
                            if (
                                state.completion.outcome.kind != "completed"
                                or state.position_frame != frames
                            ):
                                raise RuntimeError(
                                    "native playback did not drain to the final frame"
                                )
                            break
                        if time.monotonic() > deadline:
                            raise TimeoutError("native playback timed out")
                        time.sleep(0.002)
                    row["speech_done_ms"] = milliseconds(started)
                    row["output_latency_seconds"] = state.output_latency_seconds
                    if repeat == 0:
                        wav = args.output / f"{language}_{voice}.wav"
                        tts.write_speech(speech, wav)
                        if pcm_digest(wav) != row["pcm_sha256"]:
                            raise RuntimeError("speech and advanced stream PCM differ")
                if tts.retention_stats.used_bytes != 0:
                    raise RuntimeError("retention survived request close")
                report["samples"].append(row)

            request = Request(
                "Read until cancelled. " * 24, backend=SupertonicOptions(seed=1729)
            )
            with tts.speak(request, options=SpeechOptions(volume=0)) as speech:
                deadline = time.monotonic() + 30
                while speech.snapshot.position_frame == 0:
                    if time.monotonic() > deadline:
                        raise TimeoutError("cancellation probe never began playback")
                    time.sleep(0.002)
                started = time.perf_counter_ns()
                speech.cancel()
                call_ms = milliseconds(started)
                completion = speech.wait()
                done_ms = milliseconds(started)
                if completion.outcome.reason != "requested":
                    raise RuntimeError("cancellation lost its reason")
                report["cancellations"].append(
                    {
                        "repeat": repeat,
                        "cancel_call_ms": call_ms,
                        "cancel_to_done_ms": done_ms,
                    }
                )
        report["retention_peak_bytes"] = tts.retention_stats.peak_bytes
        report["retention_after_bytes"] = tts.retention_stats.used_bytes
        tts.close()
        report["handles_before"] = before
        report["handles_after"] = lib.mlsr_live_handles()
        if before != report["handles_after"] or report["retention_after_bytes"] != 0:
            raise RuntimeError("native handle or retention leak")
        report["peak_process_rss_bytes"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        report["summary_ms"] = {
            field: stats([row[field] for row in report["samples"]])
            for field in (
                "first_pcm_ms",
                "synthesis_done_ms",
                "speech_first_buffered_ms",
                "speech_first_position_ms",
                "speech_done_ms",
            )
        }
        report["passed"] = True
    except BaseException as error:
        report["error"] = str(error)
        raise
    finally:
        if tts is not None:
            tts.close()
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary_ms"], indent=2))


if __name__ == "__main__":
    main()
