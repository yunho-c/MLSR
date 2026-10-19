"""Real-device QNN replay; development only, never part of production synthesis.

Compare the Rust pipeline to the existing native reference using identical
saved noise, then compare model stages to pinned canonical unpadded ONNX.
All failures, stage tensors, descriptors and paired Float32 WAVs are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .bundle import ROOT, ReferenceError, load_yaml, write_yaml
from .qnn_bundle import MANIFEST, verify
from .rust_runtime import Worker, decode_tensor_map

FIXTURES = ROOT / "tests/fixtures/supertonic3"
COMPONENTS = ("duration_predictor", "text_encoder", "vector_estimator", "vocoder")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    value = np.ascontiguousarray(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    value.tofile(path)
    return {
        "path": str(path),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": sha(path),
    }


def metrics(actual, expected):
    if (
        actual.shape != expected.shape
        or not np.isfinite(actual).all()
        or not np.isfinite(expected).all()
    ):
        raise ReferenceError("Nonfinite output or comparison shape mismatch")
    a, b = actual.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
    error = a - b
    return {
        "max_abs": float(np.max(np.abs(error))),
        "relative_l2": float(np.linalg.norm(error) / max(np.linalg.norm(b), 1e-30)),
        "byte_exact": actual.dtype == expected.dtype
        and actual.tobytes() == expected.tobytes(),
    }


def spectral(actual, expected):
    def stft(audio):
        frames = np.lib.stride_tricks.sliding_window_view(
            np.pad(audio, (1024, 1024)), 2048
        )[::512]
        return np.abs(np.fft.rfft(frames * np.hanning(2048)))

    a, b = stft(actual), stft(expected)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def plan(raw, speed):
    duration = np.maximum(
        np.float32(0.05),
        np.float32(raw) / np.maximum(np.float32(speed), np.float32(0.05)),
    )
    frames = int(np.float32(duration * np.float32(44100)))
    return duration, frames, (frames + 3071) // 3072


def decode(reply, raw, expected):
    return decode_tensor_map(
        reply["tensors"],
        raw,
        {name: str(value.dtype) for name, value in expected.items()},
    )


def cases(spec, corpus):
    shared = {case["id"]: case for case in corpus["cases"]}
    result = [shared[name].copy() for name in spec["shared_positive_cases"]]
    result += [case.copy() for case in corpus["cases"] if "expected_error" in case]
    for boundary in spec["boundary_cases"]:
        result.append({**shared[boundary["source"]], **boundary})
    for boundary in spec["token_boundaries"]:
        result.append(
            {
                "text": "a" * boundary["repeats"] + ".",
                "language": "en",
                "voice": "M1",
                "seed": 1800,
                **boundary,
            }
        )
    for rejection in spec["option_rejections"]:
        result.append({**shared["en_m1_basic"], **rejection})
    return result


def canonical(sessions, expected, text_count, length, frames, speed, directory, limits):
    import soundfile as sf

    def run(name, inputs):
        return sessions[name].run(
            None, {k: np.ascontiguousarray(v) for k, v in inputs.items()}
        )[0]

    text = {
        "text_ids": expected["runtime_text_ids"][..., :text_count],
        "text_mask": expected["text_mask"][..., :text_count],
    }
    duration = run("duration_predictor", {**text, "style_dp": expected["style_dp"]})
    embedding = run("text_encoder", {**text, "style_ttl": expected["style_ttl"]})
    outputs = {"duration_raw": duration, "text_emb": embedding}
    stages = {
        "duration": metrics(expected["duration_raw"], duration),
        "text_encoder": metrics(expected["text_emb"][..., :text_count], embedding),
    }
    latent = expected["initial_noise"][..., :length].copy()
    actual_previous = latent.copy()
    common = {
        "style_ttl": expected["style_ttl"],
        "latent_mask": np.ones((1, 1, length), np.float32),
        "text_mask": text["text_mask"],
        "total_step": np.array([8], np.float32),
    }
    trajectories = {}
    for step in range(8):
        step_name = f"ve_step_{step:02}"
        step_input = {**common, "current_step": np.array([step], np.float32)}
        isolated = run(
            "vector_estimator",
            {
                **step_input,
                "noisy_latent": actual_previous,
                "text_emb": expected["text_emb"][..., :text_count],
            },
        )
        latent = run(
            "vector_estimator",
            {**step_input, "noisy_latent": latent, "text_emb": embedding},
        )
        actual_previous = expected[step_name][..., :length].copy()
        stages[step_name] = metrics(actual_previous, isolated)
        trajectories[step_name] = metrics(actual_previous, latent)
        outputs[step_name], outputs[step_name + "_isolated"] = latent, isolated
    isolated_pcm = run("vocoder", {"latent": actual_previous}).ravel()[:frames]
    pcm = run("vocoder", {"latent": latent}).ravel()[:frames]
    outputs["pcm"], outputs["vocoder_isolated"] = pcm, isolated_pcm
    stages["vocoder"] = metrics(expected["pcm"], isolated_pcm)
    convergence = spectral(expected["pcm"], pcm)
    checks = {
        name: v["relative_l2"] <= limits["canonical_stage_relative_l2"]
        for name, v in stages.items()
    }
    checks["final_trajectory"] = (
        trajectories["ve_step_07"]["relative_l2"]
        <= limits["canonical_trajectory_relative_l2"]
    )
    checks["spectral_convergence"] = (
        convergence <= limits["canonical_spectral_convergence"]
    )
    for name, array in outputs.items():
        save(directory / f"{name}.raw", array)
    sf.write(directory / "canonical.wav", pcm, 44100, subtype="FLOAT")
    sf.write(directory / "qnn.wav", expected["pcm"], 44100, subtype="FLOAT")
    _, canonical_frames, canonical_length = plan(duration[0], speed)
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "stages": stages,
        "trajectory": trajectories,
        "pcm": metrics(expected["pcm"], pcm),
        "spectral_convergence": convergence,
        "qnn_frames": frames,
        "canonical_independent_frames": canonical_frames,
        "canonical_independent_latent_length": canonical_length,
        "comparison_duration": "QNN duration and same actual noise isolate model numerics; independent ONNX frame count is also recorded",
    }


def replay(args):
    import onnxruntime as ort

    args.output = args.output.resolve()
    args.bundle = args.bundle.resolve(strict=True)
    args.source = args.source.resolve(strict=True)
    args.binary = args.binary.resolve(strict=True)
    verify(args.bundle)
    spec, corpus, frontend = (
        load_yaml(FIXTURES / name)
        for name in ("qnn.yaml", "corpus.yaml", "frontend.yaml")
    )
    helper = args.source / "external/supertonic/py/helper.py"
    # Git's Windows checkout may use CRLF; pin the source blob's LF bytes.
    if (
        hashlib.sha256(helper.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        != frontend["provenance"]["source_sha256"]
    ):
        raise ReferenceError("Upstream helper revision changed")
    sys.path[:0] = [str(args.source / "scripts"), str(helper.parent)]
    from helper import load_text_processor
    from qnn_runtime import NativeRunner

    processor = load_text_processor(str(args.bundle))
    metadata = {
        name: json.loads((args.bundle / "graphs" / f"{name}.json").read_text())
        for name in COMPONENTS
    }
    sessions, onnx_hashes = {}, {}
    for name in COMPONENTS:
        path = args.source / "assets/onnx" / f"{name}.onnx"
        onnx_hashes[name] = sha(path)
        if onnx_hashes[name] != metadata[name]["model_sha256"]:
            raise ReferenceError(f"Canonical ONNX hash mismatch: {name}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        sessions[name] = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / "report.yaml"
    report = {
        "schema_version": 1,
        "adapter_passed": False,
        "canonical_screen_passed": False,
        "acceptance": spec["acceptance"],
        "manifest_sha256": sha(MANIFEST),
        "binary_sha256": sha(args.binary),
        "bridge_sha256": sha(
            Path(os.environ["MLSR_QNN_BRIDGE_DIR"]) / "mlsr_qnn_bridge.dll"
        ),
        "reference_binary_sha256": sha(
            args.source / "build/runner/Release/supertonic_qnn.exe"
        ),
        "onnx_sha256": onnx_hashes,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
        },
        "source_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for p in [
                Path(__file__),
                FIXTURES / "qnn.yaml",
                FIXTURES / "corpus.yaml",
                FIXTURES / "frontend.yaml",
            ]
        },
        "passes_per_backing_mode": 2,
        "cases": [],
        "sessions": [],
    }
    write_yaml(report_path, report)
    references = {}
    try:
        # One process retains all four reference modules. Calls are sequential.
        with (
            NativeRunner(
                args.bundle / "graphs", args.output / "reference.log"
            ) as native,
            ThreadPoolExecutor(max_workers=1) as executor,
        ):

            def infer(name, inputs, directory):
                ordered = [
                    save(directory / f"in_{i}.raw", inputs[entry["name"]])
                    for i, entry in enumerate(metadata[name]["inputs"])
                ]
                output = directory / "output.raw"
                future = executor.submit(
                    native.request,
                    {
                        "command": "infer",
                        "component": name,
                        "inputs": ordered,
                        "output": str(output),
                    },
                )
                try:
                    timing = future.result(timeout=180)
                except BaseException:
                    native.process.terminate()
                    raise
                (directory / "timing.json").write_text(
                    json.dumps(timing), encoding="utf-8"
                )
                return np.fromfile(output, np.float32).reshape(
                    metadata[name]["output"]["shape"]
                )

            for case in cases(spec, corpus):
                directory = args.output / case["id"]
                directory.mkdir()
                record = {"request": case.copy(), "runs": []}
                report["cases"].append(record)
                expected = {}
                expected_error = case.get("expected_error")
                speed = case.get("speed", corpus["defaults"]["speed"])
                # Invalid frontend/options cases are validated by Rust below.
                if not expected_error or "target_latent_length" in case:
                    ids, mask = processor([case["text"]], [case["language"]])
                    normalized = processor._preprocess_text(
                        case["text"], case["language"]
                    )
                    frozen = frontend["cases"].get(case.get("source", case["id"]))
                    if frozen and (
                        normalized != frozen["normalized_text"]
                        or ids.ravel().tolist() != frozen["tokens"]
                    ):
                        raise ReferenceError(
                            "Pinned independent frontend fixtures differ"
                        )
                    text_count = ids.shape[-1]
                    voice = json.loads(
                        (
                            args.bundle / "voice_styles" / f"{case['voice']}.json"
                        ).read_text()
                    )
                    expected = {
                        k: np.array(voice[k]["data"], np.float32).reshape(
                            voice[k]["dims"]
                        )
                        for k in ("style_dp", "style_ttl")
                    }
                    expected["runtime_text_ids"] = np.pad(
                        ids, ((0, 0), (0, 128 - text_count))
                    )
                    expected["text_ids"] = expected["runtime_text_ids"].astype(np.int32)
                    expected["text_mask"] = np.pad(
                        mask, ((0, 0), (0, 0), (0, 128 - text_count))
                    )
                    inputs = {**expected, "text_ids": expected["runtime_text_ids"]}
                    raw = infer(
                        "duration_predictor",
                        inputs,
                        directory / "reference/duration_predictor",
                    )
                    if "target_latent_length" in case:
                        target_seconds = np.float32(
                            (case["target_latent_length"] - 0.5) * 3072 / 44100
                        )
                        speed = float(np.float32(raw[0] / target_seconds))
                    duration, frames, length = plan(raw[0], speed)
                    record["plan"] = {
                        "speed": float(speed),
                        "frames": frames,
                        "latent_length": length,
                        "tokens": text_count,
                        "normalized_text": normalized,
                    }
                    if (
                        "target_latent_length" in case
                        and length != case["target_latent_length"]
                    ):
                        raise ReferenceError(
                            "Boundary speed calibration missed the requested length"
                        )
                    if not expected_error:
                        if not 4 <= length <= 96:
                            raise ReferenceError(
                                f"Shared positive case exceeds QNN shape: {case['id']} L{length}"
                            )
                        noise = np.random.default_rng(case["seed"]).standard_normal(
                            (1, 144, 128), dtype=np.float32
                        )
                        expected.update(
                            duration_raw=raw,
                            duration=np.array([duration], np.float32),
                            initial_noise=noise,
                            latent_mask=np.ones((1, 1, length), np.float32),
                            total_step=np.array([8], np.float32),
                            current_steps=np.arange(8, dtype=np.float32),
                        )
                        expected["ve_initial_latent"] = np.pad(
                            noise[..., :length], ((0, 0), (0, 0), (0, 96 - length))
                        )
                        expected["ve_latent_mask"] = np.pad(
                            expected["latent_mask"], ((0, 0), (0, 0), (0, 96 - length))
                        )
                        expected["text_emb"] = infer(
                            "text_encoder", inputs, directory / "reference/text_encoder"
                        )
                        latent = expected["ve_initial_latent"]
                        for step in range(8):
                            name = f"ve_step_{step:02}"
                            feed = {
                                "noisy_latent": latent,
                                "text_emb": expected["text_emb"],
                                "style_ttl": expected["style_ttl"],
                                "latent_mask": expected["ve_latent_mask"],
                                "text_mask": expected["text_mask"],
                                "current_step": np.array([step], np.float32),
                                "total_step": expected["total_step"],
                            }
                            latent = infer(
                                "vector_estimator", feed, directory / "reference" / name
                            )
                            expected[name] = latent
                        expected["vocoder_input"] = latent
                        expected["wav_raw"] = infer(
                            "vocoder",
                            {"latent": latent},
                            directory / "reference/vocoder",
                        )
                        expected["pcm"] = expected["wav_raw"].ravel()[:frames]
                        record["tensors"] = {
                            name: save(directory / "expected" / f"{name}.raw", value)
                            for name, value in expected.items()
                        }
                        record["canonical"] = canonical(
                            sessions,
                            expected,
                            text_count,
                            length,
                            frames,
                            speed,
                            directory / "canonical",
                            spec["acceptance"],
                        )
                record["speed"] = float(speed)
                references[case["id"]] = expected
                write_yaml(report_path, report)
                print(
                    f"Reference {case['id']}: {expected_error or ('canonical pass=' + str(record['canonical']['passed']))}",
                    flush=True,
                )
        for reuse in (False, True):
            with (args.output / f"rust-reuse-{reuse}.log").open(
                "x", encoding="utf-8"
            ) as log:
                worker = Worker(args.binary, log, 180)
                try:
                    loaded = worker.call(
                        {
                            "command": "load",
                            "manifest": str(MANIFEST),
                            "bundle": str(args.bundle),
                            "compute_policy": "qnn_htp",
                            "output_backings": reuse,
                        }
                    )
                    report["sessions"].append({"reuse": reuse, "load": loaded})
                    for repeat in range(2):
                        for record in report["cases"]:
                            case = record["request"]
                            directory = args.output / case["id"]
                            output = directory / f"rust-{reuse}-{repeat}.raw"
                            request = {
                                "command": "run",
                                "diagnostic": True,
                                "text": case["text"],
                                "language": case["language"],
                                "voice": case["voice"],
                                "steps": case.get("steps", 8),
                                "speed": record["speed"],
                                "seed": case["seed"],
                                "output": str(output),
                            }
                            if "expected_error" not in case:
                                request["noise"] = {
                                    "path": str(
                                        directory / "expected/initial_noise.raw"
                                    ),
                                    "shape": [1, 144, 128],
                                }
                            entry = {"reuse": reuse, "repeat": repeat, "passed": False}
                            record["runs"].append(entry)
                            try:
                                reply = worker.call(request)
                                entry["reply"] = reply
                                if "expected_error" in case:
                                    entry["failure"] = "Unexpected success"
                                else:
                                    expected = references[case["id"]]
                                    actual = decode(
                                        reply, output.read_bytes(), expected
                                    )
                                    entry["tensors"] = {
                                        name: metrics(actual[name], value)
                                        for name, value in expected.items()
                                    }
                                    plan_record, actual_plan = (
                                        record["plan"],
                                        reply["metadata"],
                                    )
                                    entry["metadata_equal"] = actual_plan == {
                                        "normalized_text": plan_record[
                                            "normalized_text"
                                        ],
                                        "tokens": plan_record["tokens"],
                                        "requested_frames": plan_record["frames"],
                                        "pcm_frames": plan_record["frames"],
                                        "latent_length": plan_record["latent_length"],
                                        "latent_bucket": 96,
                                    }
                                    entry["passed"] = (
                                        entry["metadata_equal"]
                                        and all(
                                            v["byte_exact"]
                                            for v in entry["tensors"].values()
                                        )
                                        and reply["ve_backing_reused"] == [reuse] * 8
                                        and reply["stats"]["model_loads"] == 4
                                        and reply["stats"]["live_tensors"]
                                        == (8 if reuse else 6)
                                    )
                            except ReferenceError as error:
                                entry["error"] = str(error)
                                entry["passed"] = "expected_error" in case and case[
                                    "expected_error"
                                ] in str(error)
                            write_yaml(report_path, report)
                    report["sessions"][-1]["final_stats"] = worker.call(
                        {"command": "stats"}
                    )
                    worker.call({"command": "close"})
                    worker.process.stdin.close()
                    worker.process.wait(timeout=30)
                finally:
                    worker.stop()
        report["adapter_passed"] = all(
            run["passed"] for case in report["cases"] for run in case["runs"]
        ) and all(len(case["runs"]) == 4 for case in report["cases"])
        report["canonical_screen_passed"] = all(
            case["canonical"]["passed"]
            for case in report["cases"]
            if "canonical" in case
        )
    except BaseException as error:
        report["failure"] = str(error)
        raise
    finally:
        write_yaml(report_path, report)
    print(
        f"Adapter: {report['adapter_passed']}; canonical screen: {report['canonical_screen_passed']}; {report_path}"
    )
    return report["adapter_passed"] and report["canonical_screen_passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Pinned local supertonic-npu reference checkout",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=ROOT / "target/release/examples/qnn_synthesis_replay.exe",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fresh evidence directory; never overwritten",
    )
    args = parser.parse_args()
    raise SystemExit(0 if replay(args) else 1)


if __name__ == "__main__":
    main()
