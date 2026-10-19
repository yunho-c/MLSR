"""Portable negative tests for runtime replay evidence and transport validation."""

import copy
import io
from pathlib import Path
import queue
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tools.supertonic3 import __main__ as cli
from tools.supertonic3.bundle import ReferenceError, load_yaml, write_yaml
from tools.supertonic3.reference import compare_tensors
from tools.supertonic3.rust_runtime import (
    Worker,
    decode_tensor_map,
    decode_outputs,
    output_names,
    replay_rust,
)


class RustRuntimeTests(unittest.TestCase):
    def test_qnn_int64_transport_preserves_full_width_and_checks_byte_count(self):
        values = np.array([-(2**63), 0, 2**63 - 1], dtype="<i8")
        descriptor = {
            "ids": {"shape": [1, 3], "dtype": "int64", "offset": 0, "length": 24}
        }
        decoded = decode_tensor_map(descriptor, values.tobytes(), {"ids": "int64"})
        self.assertEqual(decoded["ids"].tobytes(), values.tobytes())
        with self.assertRaises(ReferenceError):
            decode_tensor_map(descriptor, values.tobytes(), {"ids": "int32"})
        descriptor["ids"]["length"] = 12
        with self.assertRaises(ReferenceError):
            decode_tensor_map(descriptor, values.tobytes(), {"ids": "int64"})

    def payload(self):
        raw = bytearray()
        descriptors = {}
        for name in sorted(output_names(2)):
            data = np.array([1.0, -0.0], dtype="<f4").tobytes()
            descriptors[name] = {
                "shape": [2],
                "dtype": "float32",
                "offset": len(raw),
                "length": len(data),
            }
            raw.extend(data)
        return descriptors, bytes(raw)

    def test_inventory_layout_dtype_and_bounds_are_required(self):
        descriptors, raw = self.payload()
        actual = decode_outputs(descriptors, raw, 2)
        self.assertEqual(set(actual), output_names(2))
        self.assertEqual(
            actual["pcm"].tobytes(), np.array([1.0, -0.0], dtype="<f4").tobytes()
        )
        for change in [
            lambda d: d.pop("pcm"),
            lambda d: d.update(unexpected=d["pcm"]),
            lambda d: d["pcm"].update(dtype="int32"),
            lambda d: d["pcm"].update(shape=[False, 2]),
            lambda d: d["pcm"].update(shape=[0]),
            lambda d: d["pcm"].update(offset=-1),
            lambda d: d["pcm"].update(length=7),
            lambda d: d["pcm"].update(offset=len(raw)),
        ]:
            damaged = copy.deepcopy(descriptors)
            change(damaged)
            with self.assertRaises(ReferenceError):
                decode_outputs(damaged, raw, 2)

    def test_runtime_comparison_preserves_sign_bits_and_detects_drift(self):
        descriptors, raw = self.payload()
        expected = decode_outputs(descriptors, raw, 2)
        actual = {name: arr.copy() for name, arr in expected.items()}
        actual["pcm"][1] = 0.0
        checks = compare_tensors(expected, actual, 0.0, 0.0)
        self.assertFalse(next(c for c in checks if c["tensor"] == "pcm")["passed"])
        actual["pcm"][0] = 2.0
        self.assertFalse(
            all(c["passed"] for c in compare_tensors(expected, actual, 1e-5, 1e-5))
        )

    def test_worker_eof_and_timeout_cannot_report_success(self):
        worker = object.__new__(Worker)
        worker.process = SimpleNamespace(stdin=io.StringIO(), poll=lambda: 1)
        worker.lines = queue.Queue()
        worker.timeout = 0.01
        with self.assertRaisesRegex(ReferenceError, "timed out"):
            worker.call({"command": "load"})
        worker.lines.put(None)
        with self.assertRaisesRegex(ReferenceError, "exited"):
            worker.call({"command": "load"})
        worker.lines.put('{"error": "invalid shape"}\n')
        with self.assertRaisesRegex(ReferenceError, "invalid shape"):
            worker.call({"command": "run"})

    def test_failed_worker_leaves_nonpassing_report_and_is_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "binary"
            binary.write_bytes(b"test binary receipt")
            (root / "reference.yaml").write_text("fixture: placeholder\n")
            output = root / "result.yaml"
            with (
                patch(
                    "tools.supertonic3.rust_runtime.platform.system",
                    return_value="Darwin",
                ),
                patch("tools.supertonic3.rust_runtime.verify_bundle"),
                patch(
                    "tools.supertonic3.rust_runtime.verify_reference",
                    return_value={"compute_policy": "cpu_only", "cases": []},
                ),
                patch("tools.supertonic3.rust_runtime.Worker") as worker,
            ):
                worker.return_value.call.side_effect = RuntimeError(
                    "controlled worker failure"
                )
                with self.assertRaisesRegex(RuntimeError, "controlled worker failure"):
                    replay_rust({"artifacts": {}}, root, {}, root, output, binary)
                worker.return_value.stop.assert_called_once()
                report = load_yaml(output)
                self.assertIs(report["passed"], False)
                self.assertEqual(report["kind"], "rust_runtime_replay")
                before = output.read_bytes()
                with self.assertRaisesRegex(ReferenceError, "already exists"):
                    replay_rust({"artifacts": {}}, root, {}, root, output, binary)
                self.assertEqual(before, output.read_bytes())

    def test_runtime_evidence_cannot_qualify_complete_synthesis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "runtime.yaml"
            write_yaml(evidence, {"kind": "rust_runtime_replay", "passed": True})
            args = [
                "supertonic3",
                "summary",
                "--reference",
                str(root),
                "--output",
                str(root / "summary.yaml"),
                "--replay-report",
                str(evidence),
            ]
            with (
                patch("sys.argv", args),
                patch(
                    "tools.supertonic3.reference.verify_reference",
                    return_value={"cases": []},
                ),
                patch("tools.supertonic3.reference.compact_summary", return_value={}),
            ):
                with self.assertRaisesRegex(ReferenceError, "runtime-only"):
                    cli.main()
            self.assertFalse((root / "summary.yaml").exists())
