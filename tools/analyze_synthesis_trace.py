#!/usr/bin/env python3
"""Attribute recorded ANE model predictions to the traced Rust process and jobs."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.supertonic3.bundle import load_yaml, write_yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    root = ET.parse(path).getroot()
    refs = {e.attrib["id"]: e for e in root.iter() if "id" in e.attrib}

    def resolve(element):
        while element is not None and "ref" in element.attrib:
            element = refs[element.attrib["ref"]]
        return element

    for row in root.findall(".//row"):
        yield {resolve(e).tag: resolve(e) for e in row}, resolve


def intervals(path, pid, subsystem, name):
    pending, result = {}, []
    for row, resolve in rows(path):
        if row.get("subsystem") is None or row["subsystem"].text != subsystem:
            continue
        if row.get("signpost-name") is None or row["signpost-name"].text != name:
            continue
        process = resolve(row.get("process"))
        if process is None or int(resolve(process.find("pid")).text) != pid:
            continue
        thread = resolve(row["thread"])
        key = (
            int(resolve(thread.find("tid")).text),
            int(row["os-signpost-identifier"].text),
        )
        time = int(row["event-time"].text)
        event = row["event-type"].text
        if event == "Begin":
            if key in pending:
                raise ValueError(f"Overlapping marker IDs for {name}")
            pending[key] = time
        elif event == "End":
            if key not in pending or pending[key] >= time:
                raise ValueError(f"Missing or invalid begin marker for {name}")
            result.append({"id": key[1], "start": pending.pop(key), "end": time})
    if pending:
        raise ValueError(f"Incomplete intervals for {name}")
    return sorted(result, key=lambda x: x["start"])


def model(label):
    if label == "Neural Engine Prediction":
        return "unlabeled"
    for prefix, name in [
        ("DurationPredictor_v2_", "duration_predictor"),
        ("TextEncoder_v2_", "text_encoder"),
        ("VectorEstimator_v2_L128_int4_S8_", "fused_ve"),
        ("VectorEstimator_v2_L128_int4_", "ve"),
        ("Vocoder_", "vocoder"),
    ]:
        if label.startswith(prefix):
            return name
    return None


def analyze_condition(root, condition):
    name, workload = condition["name"], condition["workload"]
    pid, requests = workload["process_id"], workload["requests"]
    for file, expected in condition["xml_files"].items():
        if sha(root / file) != expected:
            raise ValueError(f"Changed trace export: {file}")
    if sha(root / f"{name}.yaml") != condition["workload_sha256"]:
        raise ValueError(f"Changed workload report: {name}")
    signs = root / f"{name}-os-signpost.xml"
    jobs = intervals(signs, pid, "dev.mlsr.qualification", "MLSRJob")
    # These are the operating system's own diagnostic markers, not a private API
    # invoked by MLSR. Their availability/names are specific to the recorded OS.
    evaluations = intervals(signs, pid, "com.apple.ane", "_ANEF_MODEL_EVALUATE")
    if (
        len(jobs) != len(requests)
        or [x["id"] for x in jobs] != list(range(1, len(requests) + 1))
        or any(a["end"] >= b["start"] for a, b in zip(jobs, jobs[1:]))
    ):
        raise ValueError("Missing, duplicated, or overlapping request intervals")
    predictions, excluded = [], 0
    for row, _ in rows(root / f"{name}-ane-hw-intervals.xml"):
        label = row["formatted-label"].get("fmt", "")
        if not label.endswith(" Prediction") or model(label) is None:
            excluded += 1
            continue
        start = int(row["start-time"].text)
        end = start + int(row["duration"].text)
        predictions.append(
            {"model": model(label), "label": label, "start": start, "end": end}
        )
    assigned = [Counter() for _ in jobs]
    eval_hits = Counter()
    for event in predictions:
        owners = [
            i
            for i, item in enumerate(jobs)
            if item["start"] <= event["start"] and event["end"] <= item["end"]
        ]
        evaluations_for_event = [
            i
            for i, item in enumerate(evaluations)
            if item["start"] <= event["start"] and event["end"] <= item["end"]
        ]
        if event["model"] == "unlabeled" and not evaluations_for_event:
            # An attached trace may lack model-load name metadata. Its hardware
            # table is still system-wide: unrelated processes are excluded by
            # their lack of an enclosing evaluation from the workload PID.
            excluded += 1
            continue
        if len(owners) != 1 or len(evaluations_for_event) != 1:
            raise ValueError(
                f"Ambiguous or unattributed system-wide ANE event: {event}"
            )
        assigned[owners[0]][event["model"]] += 1
        eval_hits[evaluations_for_event[0]] += 1
    if any(eval_hits[i] != 1 for i in range(len(evaluations))):
        raise ValueError(
            "Process ANE evaluation intervals do not match hardware predictions one-to-one"
        )
    expected = (
        {}
        if workload["compute_policy"] == "cpu"
        else {
            "duration_predictor": 1,
            "text_encoder": 1,
            ("ve" if workload["variant"] == "baseline" else "fused_ve"): (
                8 if workload["variant"] == "baseline" else 1
            ),
        }
    )
    for counts in assigned:
        if "unlabeled" in counts:
            # With attach-after-load, Instruments emits generic hardware labels.
            # Qualify execution/counts, never invent per-model placement from
            # call order. The observed workload is ten baseline or three fused
            # ANE evaluations per job on this OS; future route changes need a new
            # qualification rather than silently broadening this gate.
            expected_count = (
                0
                if workload["compute_policy"] == "cpu"
                else (10 if workload["variant"] == "baseline" else 3)
            )
            if sum(counts.values()) != expected_count:
                raise ValueError(
                    f"Unexpected process-attributed hardware count: {dict(counts)}"
                )
            continue
        # A vocoder hardware interval is optional evidence, never assumed from a
        # compute plan. All other model counts must match the scheduled graph.
        mandatory = {k: v for k, v in counts.items() if k != "vocoder"}
        if mandatory != expected or counts.get("vocoder", 0) > 1:
            raise ValueError(
                f"Unexpected ANE route: {dict(counts)}, expected {expected}"
            )
        if workload["compute_policy"] == "cpu" and counts:
            raise ValueError("CPU-only control emitted a Supertonic ANE prediction")
    total = Counter()
    for counts in assigned:
        total.update(counts)
    return {
        "name": name,
        "process_id": pid,
        "requests": len(jobs),
        "native_predictions": workload["stats"]["predictions"],
        "process_ane_evaluations": len(evaluations),
        "attributed_hardware_predictions": dict(total),
        "excluded_system_intervals": excluded,
        "coreml_activity_rows": sum(
            1 for _ in rows(root / f"{name}-coreml-os-signpost.xml")
        ),
        "vocoder_hardware_execution_observed": total["vocoder"] > 0,
        "model_labels_available": total["unlabeled"] == 0 and sum(total.values()) > 0,
        "jobs": [
            {**job, "hardware_predictions": dict(counts)}
            for job, counts in zip(jobs, assigned)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file")
    capture = load_yaml(args.capture)
    report = {
        "schema_version": 1,
        "kind": "instruments_synthesis_attribution",
        "passed": False,
        "capture_sha256": sha(args.capture),
        "analyzer_sha256": sha(Path(__file__)),
        "scope": "Hardware intervals nested in both the workload PID's ANE calls and Rust job markers; attached traces can omit model names; no utilization or physical-residency claim",
        "conditions": [],
    }
    try:
        if (
            capture.get("passed") is not True
            or {c["name"] for c in capture["conditions"]}
            != {"baseline-ane", "baseline-cpu", "fused8-ane"}
            or len(capture["conditions"]) != 3
        ):
            raise ValueError("Require all three completed capture conditions")
        for condition in capture["conditions"]:
            report["conditions"].append(
                analyze_condition(args.capture.parent, condition)
            )
        conditions = {item["name"]: item["workload"] for item in capture["conditions"]}
        baseline = conditions["baseline-ane"]["requests"][0]
        fused = conditions["fused8-ane"]["requests"][0]
        if (baseline["frames"], baseline["pcm_sha256"]) != (
            fused["frames"],
            fused["pcm_sha256"],
        ):
            raise ValueError(
                "Baseline and fused trace workloads produced different PCM"
            )
        report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        write_yaml(args.output, report)
    print(f"Verified ANE execution and CPU-only control: {args.output}")


if __name__ == "__main__":
    main()
