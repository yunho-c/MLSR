"""Evidence gates for complete Rust synthesis, without executing a speech model."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.supertonic3.bundle import load_yaml
from tools.supertonic3.rust_runtime import decode_tensor_map
from tools.supertonic3.synthesis_replay import (
    METADATA,
    check_metadata,
    evidence_complete,
    replay,
)


class SynthesisReplayTests(unittest.TestCase):
    def test_decisions_must_match_including_normalization_frames_and_buckets(self):
        expected = dict(zip(METADATA, ["<en>Hello.</en>", 15, 10000, 4, 128, 10000]))
        self.assertTrue(check_metadata(expected, expected))
        for key in METADATA:
            wrong = dict(expected)
            wrong[key] = None
            self.assertFalse(check_metadata(wrong, expected))

    def test_empty_partial_duplicate_or_failed_evidence_cannot_pass(self):
        saved = {"cases": [{"id": "a"}]}
        corpus = {"cases": [{"id": "bad", "expected_error": "empty_text"}]}
        report = {
            "passes": 2,
            "worker_exit": 0,
            "cases": [
                {
                    "id": "a",
                    "pass": p,
                    "passed": True,
                    "tensors": [{"passed": True, "exact": True}],
                }
                for p in (1, 2)
            ],
            "rejections": [{"id": "bad", "passed": True}],
            "seeded_checks": [{"id": "a", "passed": True}],
        }
        self.assertTrue(evidence_complete(report, saved, corpus, {"a"}))
        for damage in [
            lambda r: r.update(cases=[]),
            lambda r: r["cases"].pop(),
            lambda r: r["cases"].append(r["cases"][0]),
            lambda r: r["cases"][0].update(tensors=[]),
            lambda r: r["cases"][0]["tensors"][0].update(exact=False),
            lambda r: r.update(worker_exit=1),
            lambda r: r.update(rejections=[]),
            lambda r: r["seeded_checks"][0].update(passed=False),
        ]:
            damaged = copy.deepcopy(report)
            damage(damaged)
            self.assertFalse(evidence_complete(damaged, saved, corpus, {"a"}))

    def test_transport_keeps_integer_and_float_types_distinct(self):
        integer = np.array([0, -1, 2147483647], dtype="<i4")
        desc = {
            "text_ids": {"shape": [1, 3], "dtype": "int32", "offset": 0, "length": 12}
        }
        decoded = decode_tensor_map(desc, integer.tobytes(), {"text_ids": "int32"})
        self.assertEqual(decoded["text_ids"].tobytes(), integer.tobytes())
        with self.assertRaisesRegex(ValueError, "descriptor"):
            decode_tensor_map(desc, integer.tobytes(), {"text_ids": "float32"})

    def test_failed_worker_preserves_nonpassing_report_and_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "binary").write_bytes(b"stub binary")
            (root / "manifest.yaml").write_text("fixture: true\n")
            (root / "reference.yaml").write_text("fixture: true\n")
            saved = {
                "compute_policy": "cpu_only",
                "cases": [
                    {"id": id}
                    for id in (
                        "en_m1_basic",
                        "en_f1_basic",
                        "ko_m1_basic",
                        "ko_f1_basic",
                    )
                ],
            }
            with patch(
                "tools.supertonic3.synthesis_replay.platform.system",
                return_value="Darwin",
            ), patch(
                "tools.supertonic3.synthesis_replay.load_manifest", return_value={}
            ), patch(
                "tools.supertonic3.synthesis_replay.load_corpus",
                return_value={"cases": []},
            ), patch(
                "tools.supertonic3.synthesis_replay.verify_bundle"
            ), patch(
                "tools.supertonic3.synthesis_replay.verify_reference",
                return_value=saved,
            ), patch(
                "tools.supertonic3.synthesis_replay.Worker"
            ) as worker:
                worker.return_value.call.side_effect = RuntimeError(
                    "controlled failure"
                )
                with self.assertRaisesRegex(RuntimeError, "controlled failure"):
                    replay(
                        root / "manifest.yaml",
                        root,
                        root / "corpus.yaml",
                        root,
                        root / "out.yaml",
                        root / "binary",
                    )
                worker.return_value.stop.assert_called_once()
            self.assertIs(load_yaml(root / "out.yaml")["passed"], False)
