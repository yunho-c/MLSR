"""Profile extracted, unchanged PTE contexts with the installed QNN SDK tools.

Uses saved real replay inputs. Detailed instrumentation proves HTP execution;
these measurements must not be substituted for uninstrumented job latency.
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from .bundle import load_yaml, write_yaml
from .qnn_bundle import MANIFEST, verify
from .qnn_replay import COMPONENTS, metrics, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--case", default="en_m1_basic")
    parser.add_argument(
        "--sdk", type=Path, default=Path("C:/Qualcomm/AIStack/QAIRT/2.50.40.260831")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("source", "bundle", "replay", "sdk"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output = args.output.resolve()
    verify(args.bundle)
    report = load_yaml(args.replay / "report.yaml")
    if report["manifest_sha256"] != sha(MANIFEST):
        raise ValueError("Replay used a different manifest")
    case = next(c for c in report["cases"] if c["request"]["id"] == args.case)
    tensors = {}
    for name, desc in case["tensors"].items():
        path = Path(desc["path"])
        if sha(path) != desc["sha256"]:
            raise ValueError(f"Saved input changed: {path}")
        tensors[name] = np.fromfile(path, desc["dtype"]).reshape(desc["shape"])
    sys.path[:0] = [
        str(args.source / "scripts"),
        str(args.source / ".deps"),
        str(args.source / ".deps/executorch"),
    ]
    os.environ["QNN_SDK_ROOT"] = str(args.sdk)
    os.environ["FLATC_EXECUTABLE"] = str(
        args.source / "build/executorch-host/third-party/flatc_ep/bin/flatc.exe"
    )
    with os.add_dll_directory(str(args.sdk / "lib/x86_64-windows-msvc")):
        from executorch.backends.qualcomm.python import PyQnnManagerAdaptor
        from executorch.exir._serialize._program import deserialize_pte_binary
    from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
        flatbuffer_to_option,
    )
    from profile_qnn import summarize_profile
    from qnn_runtime import runtime_environment

    args.output.mkdir(parents=True, exist_ok=False)
    compile_options = {}
    result = {
        "passed": False,
        "manifest_sha256": sha(MANIFEST),
        "replay_report_sha256": sha(args.replay / "report.yaml"),
        "scope": "SDK detailed profiles on exact extracted contexts, real saved inputs, five executions each. Instrumented HTP proof; not production latency.",
        "components": [],
    }
    binaries = args.sdk / "bin/aarch64-windows-msvc"
    result["tools_sha256"] = {
        name: sha(binaries / name)
        for name in (
            "qnn-net-run.exe",
            "qnn-profile-viewer.exe",
            "qnn-context-binary-utility.exe",
        )
    }
    feeds = {
        "duration_predictor": {
            "text_ids": tensors["runtime_text_ids"],
            "text_mask": tensors["text_mask"],
            "style_dp": tensors["style_dp"],
        },
        "text_encoder": {
            "text_ids": tensors["runtime_text_ids"],
            "text_mask": tensors["text_mask"],
            "style_ttl": tensors["style_ttl"],
        },
        "vector_estimator": {
            "noisy_latent": tensors["ve_initial_latent"],
            "text_emb": tensors["text_emb"],
            "style_ttl": tensors["style_ttl"],
            "latent_mask": tensors["ve_latent_mask"],
            "text_mask": tensors["text_mask"],
            "current_step": np.array([0], np.float32),
            "total_step": tensors["total_step"],
        },
        "vocoder": {"latent": tensors["vocoder_input"]},
    }
    expected = {
        "duration_predictor": "duration_raw",
        "text_encoder": "text_emb",
        "vector_estimator": "ve_step_00",
        "vocoder": "wav_raw",
    }
    try:
        for name in COMPONENTS:
            directory = args.output / name
            directory.mkdir()
            path = args.bundle / "graphs" / f"{name}.pte"
            program = deserialize_pte_binary(path.read_bytes()).program
            delegates = program.execution_plan[0].delegates
            if len(delegates) != 1 or delegates[0].id != "QnnBackend":
                raise ValueError("Expected exactly one QNN delegate")
            compile_options[name] = [
                {
                    "key": spec.key,
                    "options": dataclasses.asdict(flatbuffer_to_option(spec.value)),
                }
                for spec in delegates[0].compile_specs
            ]
            (args.output / "compile-options.json").write_text(
                json.dumps(compile_options, indent=2) + "\n", encoding="utf-8"
            )
            data = program.backend_delegate_data[delegates[0].processed.index].data
            context = directory / "context.bin"
            context.write_bytes(PyQnnManagerAdaptor.StripProtocol(data))
            metadata = json.loads(path.with_suffix(".json").read_text())
            paths = []
            for index, entry in enumerate(metadata["inputs"]):
                # qnn-net-run's documented default input format is Float32; it
                # converts to each native graph dtype (including text indices).
                value = feeds[name][entry["name"]]
                converted = value.astype(np.float32)
                if not np.array_equal(value, converted.astype(value.dtype)):
                    raise ValueError("SDK Float32 input conversion is lossy")
                input_path = directory / f"input_{index}.raw"
                converted.tofile(input_path)
                paths.append(f"input_{index}_{entry['name']}_0:={input_path}")
            input_list = directory / "inputs.txt"
            input_list.write_text(" ".join(paths) + "\n", encoding="utf-8")
            command = [
                str(binaries / "qnn-net-run.exe"),
                "--backend",
                str(args.sdk / "lib/aarch64-windows-msvc/QnnHtp.dll"),
                "--retrieve_context",
                str(context),
                "--input_list",
                str(input_list),
                "--output_dir",
                str(directory),
                "--profiling_level",
                "detailed",
                "--perf_profile",
                "burst",
                "--num_inferences",
                "5",
                "--keep_num_outputs",
                "1",
            ]
            with (directory / "run.log").open("x", encoding="utf-8") as log:
                subprocess.run(
                    command,
                    env=runtime_environment(args.sdk),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=180,
                    check=True,
                )
            with (directory / "viewer.log").open("x", encoding="utf-8") as log:
                subprocess.run(
                    [
                        str(binaries / "qnn-profile-viewer.exe"),
                        "--input_log",
                        str(directory / "qnn-profiling-data.log"),
                        "--output",
                        str(directory / "profile.csv"),
                    ],
                    env=runtime_environment(args.sdk),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=60,
                    check=True,
                )
            with (directory / "context-info.log").open("x", encoding="utf-8") as log:
                subprocess.run(
                    [
                        str(binaries / "qnn-context-binary-utility.exe"),
                        "--context_binary",
                        str(context),
                        "--json_file",
                        str(directory / "context-info.json"),
                    ],
                    env=runtime_environment(args.sdk),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=60,
                    check=True,
                )
            summary = summarize_profile(directory / "profile.csv")
            reference = tensors[expected[name]]
            actual = np.fromfile(
                next((directory / "Result_0").glob("*.raw")), np.float32
            ).reshape(reference.shape)
            comparison = metrics(actual, reference)
            passed = summary["total_node_cycles"] > 0 and comparison["byte_exact"]
            result["components"].append(
                {
                    "name": name,
                    "passed": passed,
                    "pte_sha256": sha(path),
                    "context_sha256": sha(context),
                    "command": command,
                    "profile_sha256": sha(directory / "profile.csv"),
                    "qnn_reference": comparison,
                    "profile": summary,
                }
            )
            write_yaml(args.output / "report.yaml", result)
            print(
                f"{name}: {summary['total_node_cycles']:.0f} HTP operator cycles, byte exact: {comparison['byte_exact']}",
                flush=True,
            )
        result["passed"] = all(c["passed"] for c in result["components"])
        result["compile_options_sha256"] = sha(args.output / "compile-options.json")
    except BaseException as error:
        result["failure"] = str(error)
        raise
    finally:
        write_yaml(args.output / "report.yaml", result)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
