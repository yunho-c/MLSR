"""Trace attribution must not qualify partial markers or another process."""

from pathlib import Path
import tempfile
import unittest

from tools.analyze_synthesis_trace import analyze_condition, intervals, model, sha
from tools.supertonic3.bundle import write_yaml


class TraceMarkersTest(unittest.TestCase):
    def parse(self, events, pid=42):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signposts.xml"
            definitions = '<process id="p"><pid>42</pid></process><thread id="t"><tid>17</tid></thread>'
            rows = [
                f'<row><process ref="p"/><thread ref="t"/><event-time>{time}</event-time>'
                f"<event-type>{event}</event-type><os-signpost-identifier>1</os-signpost-identifier>"
                "<subsystem>dev.mlsr.qualification</subsystem><signpost-name>MLSRJob</signpost-name></row>"
                for event, time in events
            ]
            path.write_text("<root>" + definitions + "".join(rows) + "</root>")
            return intervals(path, pid, "dev.mlsr.qualification", "MLSRJob")

    def test_resolves_references_and_requires_the_workload_pid(self):
        self.assertEqual(
            self.parse([("Begin", 10), ("End", 20)]),
            [{"id": 1, "start": 10, "end": 20}],
        )
        self.assertEqual(self.parse([("Begin", 10), ("End", 20)], pid=99), [])

    def test_incomplete_duplicate_and_reversed_markers_fail(self):
        for events in [
            [("Begin", 10)],
            [("End", 20)],
            [("Begin", 10), ("Begin", 11)],
            [("Begin", 20), ("End", 10)],
        ]:
            with self.subTest(events=events), self.assertRaises(ValueError):
                self.parse(events)

    def test_fused_label_is_not_counted_as_eight_single_steps(self):
        self.assertEqual(
            model("VectorEstimator_v2_L128_int4_S8_main__Op2_AneInference Prediction"),
            "fused_ve",
        )
        self.assertEqual(
            model("VectorEstimator_v2_L128_int4_main__Op2_AneInference Prediction"),
            "ve",
        )
        self.assertIsNone(model("AnUnrelatedModel Prediction"))

    def test_unlabeled_hardware_requires_process_calls_and_cpu_control_is_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "baseline-ane"

            def event(kind, time, subsystem, signpost, identifier):
                return (
                    f'<row><process ref="p"/><thread ref="t"/><event-time>{time}</event-time>'
                    f"<event-type>{kind}</event-type><os-signpost-identifier>{identifier}</os-signpost-identifier>"
                    f"<subsystem>{subsystem}</subsystem><signpost-name>{signpost}</signpost-name></row>"
                )

            signs = [
                '<root><process id="p"><pid>42</pid></process><thread id="t"><tid>17</tid></thread>',
                event("Begin", 1, "dev.mlsr.qualification", "MLSRJob", 1),
            ]
            hardware = []
            for i in range(10):
                signs += [
                    event(
                        "Begin",
                        2 + i * 4,
                        "com.apple.ane",
                        "_ANEF_MODEL_EVALUATE",
                        100 + i,
                    ),
                    event(
                        "End",
                        5 + i * 4,
                        "com.apple.ane",
                        "_ANEF_MODEL_EVALUATE",
                        100 + i,
                    ),
                ]
                hardware.append(
                    f'<row><start-time>{3+i*4}</start-time><duration>1</duration><formatted-label fmt="Neural Engine Prediction"/></row>'
                )
            signs += [
                event("End", 100, "dev.mlsr.qualification", "MLSRJob", 1),
                "</root>",
            ]
            (root / f"{name}-os-signpost.xml").write_text("".join(signs))
            (root / f"{name}-coreml-os-signpost.xml").write_text("<root/>")
            hw = root / f"{name}-ane-hw-intervals.xml"
            hw.write_text("<root>" + "".join(hardware) + "</root>")
            workload = {
                "process_id": 42,
                "requests": [{}],
                "compute_policy": "ane",
                "variant": "baseline",
                "stats": {"predictions": 11},
            }
            write_yaml(root / f"{name}.yaml", workload)
            condition = {
                "name": name,
                "workload": workload,
                "workload_sha256": sha(root / f"{name}.yaml"),
                "xml_files": {p.name: sha(p) for p in root.glob("*.xml")},
            }
            actual = analyze_condition(root, condition)
            self.assertEqual(
                actual["attributed_hardware_predictions"], {"unlabeled": 10}
            )
            self.assertFalse(actual["model_labels_available"])
            # Identical hardware events are not allowed to qualify CPU-only use.
            workload["compute_policy"] = "cpu"
            with self.assertRaises(ValueError):
                analyze_condition(root, condition)
            workload["compute_policy"] = "ane"
            # A missing hardware event cannot be hidden by matching other calls.
            hw.write_text("<root>" + "".join(hardware[:-1]) + "</root>")
            condition["xml_files"][hw.name] = sha(hw)
            with self.assertRaises(ValueError):
                analyze_condition(root, condition)


if __name__ == "__main__":
    unittest.main()
