"""Portable checks for the experiment's comparison and replay boundaries.

These validate tensor transport and evidence bookkeeping, not speech quality.
Real-model qualification is separately recorded in docs/internal.
"""

import unittest

import numpy as np

from tools.supertonic3.bundle import ReferenceError
from tools.supertonic3.context_probe import (
    LATENT_BUCKETS,
    context_bucket,
    official_cases,
    pad_last,
    replay,
)
from tools.supertonic3.quality_probe import (
    encode_normalized,
    joined_pcm,
    metrics,
    noise_for,
)


class ContextProbeTests(unittest.TestCase):
    def test_routing_uses_exported_buckets_even_for_short_inputs(self):
        for length in (4, 127, 128, 129, 255, 256, 257, 511, 512):
            bucket = context_bucket(length)
            self.assertIn(bucket, LATENT_BUCKETS)
            self.assertGreaterEqual(bucket, length)
            self.assertFalse(any(length <= other < bucket for other in LATENT_BUCKETS))
        for length in (0, 3, 513):
            with self.assertRaises(ReferenceError):
                context_bucket(length)

    def test_encoding_rejects_truncation_and_unknown_characters(self):
        ids, mask = encode_normalized("AB", list(range(128)), 4)
        np.testing.assert_array_equal(ids, [[65, 66, 0, 0]])
        np.testing.assert_array_equal(mask, [[[1, 1, 0, 0]]])
        for text, capacity in [("AB", 1), ("", 4), ("한", 4)]:
            with self.assertRaises(ReferenceError):
                encode_normalized(text, list(range(128)), capacity)

    def test_padding_preserves_noise_bytes_and_never_truncates(self):
        noise = noise_for("hello", "M1", 4, 0)
        extended = pad_last(noise, 8)
        self.assertEqual(noise.tobytes(), extended[..., :4].tobytes())
        self.assertFalse(extended[..., 4:].any())
        with self.assertRaises(ReferenceError):
            pad_last(noise, 3)

    def test_noise_is_stable_for_unchanged_inputs(self):
        original = noise_for("hello", "M1", 4, 0)
        noise_for("unrelated chunk", "M1", 4, 0)
        np.testing.assert_array_equal(original, noise_for("hello", "M1", 4, 0))
        for text, voice, seed in [
            ("hello!", "M1", 0),
            ("hello", "F4", 0),
            ("hello", "M1", 1),
        ]:
            self.assertFalse(np.array_equal(original, noise_for(text, voice, 4, seed)))

    def test_join_adds_only_requested_inter_chunk_silence(self):
        chunks = [np.array([1, 2], np.float32), np.array([3], np.float32)]
        np.testing.assert_array_equal(joined_pcm(chunks), [1, 2, 3])
        np.testing.assert_array_equal(joined_pcm(chunks, 2), [1, 2, 0, 0, 3])
        with self.assertRaises(ReferenceError):
            joined_pcm([])

    def test_invalid_comparisons_do_not_silently_broadcast_or_pass(self):
        for actual, expected in [
            (np.zeros(2), np.zeros(3)),
            (np.array([np.nan]), np.zeros(1)),
        ]:
            with self.assertRaises(ReferenceError):
                metrics(actual, expected)

    def test_incomplete_or_duplicate_reference_plans_are_rejected(self):
        valid = {"id": "M1-official", "policy": "official", "chunks": [{"id": 0}]}
        self.assertEqual(official_cases({"cases": [valid]}), [valid])
        for cases in [
            [],
            [valid, valid],
            [{**valid, "chunks": []}],
            [{**valid, "chunks": [{"id": 1}]}],
        ]:
            with self.assertRaises(ReferenceError):
                official_cases({"cases": cases})

    def test_replay_freezes_noise_and_duration_and_crops_before_vocoder(self):
        # A tensor-transport spy, deliberately not a fake speech-quality oracle.
        class Spy:
            def __init__(self):
                self.inputs = []

            def predict(self, name, feeds):
                self.inputs.append((name, feeds))
                if name == "duration_predictor":
                    return np.array([99], np.float32)
                if name == "text_encoder":
                    return np.zeros((1, 256, 8), np.float32)
                if name == "vector_estimator":
                    return feeds["noisy_latent"].copy()
                return np.arange(4 * 3072, dtype=np.float32).reshape(1, -1)

        saved = {
            "style_dp": np.zeros((1, 8, 16), np.float32),
            "style_ttl": np.zeros((1, 50, 256), np.float32),
            "frames": np.array(9000),
            "text_ids": np.array([[1, 2]], np.int64),
            "text_mask": np.ones((1, 1, 2), np.float32),
            "initial_noise": noise_for("test", "M1", 4, 7),
            "latent_mask": np.ones((1, 1, 4), np.float32),
        }
        engine = Spy()
        result = replay(engine, saved, 8, 8, steps=2)
        self.assertEqual(result["pcm"].shape, (9000,))
        self.assertEqual(float(result["raw_duration"][0]), 99)
        self.assertEqual(engine.inputs[-1][1]["latent"].shape, (1, 144, 4))
        self.assertEqual(
            engine.inputs[-1][1]["latent"].tobytes(), saved["initial_noise"].tobytes()
        )
        for name, feeds in engine.inputs:
            if name == "vector_estimator":
                self.assertFalse(feeds["noisy_latent"][..., 4:].any())
                np.testing.assert_array_equal(
                    feeds["latent_mask"], [[[1, 1, 1, 1, 0, 0, 0, 0]]]
                )


if __name__ == "__main__":
    unittest.main()
