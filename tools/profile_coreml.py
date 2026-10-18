"""Read public placement plans through an existing Möbius checkout.

No inference, private profiler, cache bypass, model download, or environment
installation. Uses the current interpreter's PyObjC CoreML/Foundation modules.
"""

import argparse
from collections import Counter
import hashlib
import importlib.util
from pathlib import Path
import platform
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mobius", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    module = args.mobius / "tools/coreml-cli/src/coreml_cli/compute_plan.py"
    spec = importlib.util.spec_from_file_location("mobius_compute_plan", module)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = yaml.safe_load(
        (ROOT / "model-manifests/supertonic3-coreml.yaml").read_text()
    )
    variants = yaml.safe_load(
        (ROOT / "model-manifests/supertonic3-coreml-variants.yaml").read_text()
    )
    models = {
        name: a
        for name, a in baseline["artifacts"].items()
        if a["kind"] == "coreml_compiled"
    }
    models["ve_fused8_128"] = variants["artifacts"]["ve_fused8_128"]
    report = {
        "schema_version": 1,
        "kind": "mobius_public_compute_plans",
        "passed": False,
        "platform": platform.platform(),
        "mobius_module_sha256": digest(module),
        "mobius_revision": subprocess.check_output(
            ["git", "-C", str(args.mobius), "rev-parse", "HEAD"], text=True
        ).strip(),
        "scope": "Preferred placement and estimated costs; not execution, utilization or residency",
        "models": [],
    }
    try:
        for name, artifact in models.items():
            print(f"Reading public compute plan: {name}", flush=True)
            for file in artifact["files"]:
                if digest(args.bundle / file["path"]) != file["sha256"]:
                    raise ValueError("Model bytes differ from the pinned manifest")
            plan = loaded.get_compute_plan(
                args.bundle / artifact["path"],
                "cpu_and_neural_engine",
                load_timeout_s=180,
            )
            path = args.output / f"{name}.yaml"
            path.write_text(yaml.safe_dump(plan, sort_keys=False))
            report["models"].append(
                {
                    "name": name,
                    "plan_sha256": digest(path),
                    "preferred_operation_counts": dict(
                        Counter(op["device"] for op in plan["operations"])
                    ),
                    "estimated_cost_distribution": plan["summary"],
                }
            )
        report["passed"] = (
            len(report["models"]) == 7
            and digest(module) == report["mobius_module_sha256"]
        )
    finally:
        (args.output / "report.yaml").write_text(
            yaml.safe_dump(report, sort_keys=False)
        )


if __name__ == "__main__":
    main()
