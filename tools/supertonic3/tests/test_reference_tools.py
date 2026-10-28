"""Portable integrity/contract tests. Real inference is a separate opt-in command."""

import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
import yaml
from unittest.mock import patch

import numpy as np

from tools.supertonic3.bundle import (
    ROOT,
    ReferenceError,
    asset_url,
    load_manifest,
    load_yaml,
    prepare_bundle,
    relative_path,
    verify_bundle,
    validate_voice_style,
    write_yaml,
)
from tools.supertonic3.frontend import (
    InputError,
    duration_frames,
    encode_text,
    select_bucket,
)
from tools.supertonic3.reference import (
    check_frontend,
    compare_tensors,
    load_corpus,
    tensor_record,
    write_float_wav,
)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load_manifest()

    def changed_manifest(self, change):
        value = copy.deepcopy(self.manifest)
        change(value)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.yaml"
            write_yaml(path, value)
            with self.assertRaises(ReferenceError):
                load_manifest(path)

    def test_published_manifest_has_pinned_urls(self):
        for artifact in self.manifest["artifacts"].values():
            for entry in artifact["files"]:
                url = asset_url(self.manifest, entry["path"])
                self.assertIn("/resolve/5cd12d1c4cb524e704ce40ba1048e36c2e7ea2fa/", url)

    def test_local_fp16_export_has_consistent_axes_and_cannot_fake_a_download(self):
        manifest = load_manifest(ROOT / "model-manifests/supertonic3-coreml-fp16.yaml")
        self.assertEqual(manifest["contract"]["text_tokens"], 384)
        with self.assertRaisesRegex(ReferenceError, "no published download"):
            asset_url(manifest, "tts.json")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.yaml"
            manifest["artifacts"]["ve_128"]["inputs"]["text_emb"]["shape"][-1] = 128
            write_yaml(path, manifest)
            with self.assertRaisesRegex(ReferenceError, "Tensor interface"):
                load_manifest(path)

    def test_expanded_voices_preserve_graphs_and_only_download_pinned_styles(self):
        base = load_manifest(ROOT / "model-manifests/supertonic3-coreml-fp16.yaml")
        expanded = load_manifest(
            ROOT / "model-manifests/supertonic3-coreml-fp16-voices.yaml"
        )
        self.assertEqual(len(expanded["contract"]["voices"]), 10)
        self.assertEqual(expanded["voice_names"]["F4"], "Olivia")
        for key, artifact in base["artifacts"].items():
            self.assertEqual(expanded["artifacts"][key], artifact)
        self.assertEqual(
            asset_url(expanded, "voice_styles/F4.json"),
            "https://huggingface.co/spaces/Supertone/supertonic-3/resolve/4402acca094e2ff2b3dff446e0bf915f3fccc462/assets/voice_styles/F4.json",
        )
        for path in ("tts.json", expanded["artifacts"]["ve_128"]["files"][0]["path"]):
            with self.assertRaisesRegex(ReferenceError, "no published download"):
                asset_url(expanded, path)
        expanded["provenance"]["voice_styles"]["revision"] = "main"
        with self.assertRaisesRegex(ReferenceError, "must pin"):
            asset_url(expanded, "voice_styles/F4.json")

    def test_voice_declarations_and_names_must_agree_with_artifacts(self):
        for voices in ([], ["M1", "M1"], ["../M1"], [42], ["M1", "F4"]):
            self.changed_manifest(lambda m: m["contract"].update(voices=voices))
        self.changed_manifest(lambda m: m.update(voice_names={"M1": "Alex"}))
        self.changed_manifest(lambda m: m.update(voice_names={"M1": "", "F1": "Sarah"}))
        self.changed_manifest(
            lambda m: m["artifacts"]["voice_F1"].update(kind="coreml_compiled")
        )

    def test_voice_style_contract_rejects_invalid_shape_and_nonfinite_values(self):
        value = {
            key: {"type": "float32", "dims": shape, "data": np.zeros(shape).tolist()}
            for key, shape in (("style_ttl", [1, 50, 256]), ("style_dp", [1, 8, 16]))
        }
        validate_voice_style(value)
        for mutation in (
            lambda v: v["style_ttl"].update(dims=[1, 50, 255]),
            lambda v: v["style_ttl"].update(dims=[True, 50, 256]),
            lambda v: v["style_dp"].update(type="float16"),
            lambda v: v["style_dp"]["data"][0][0].__setitem__(0, float("nan")),
            lambda v: v["style_dp"]["data"][0][0].__setitem__(0, 1e100),
            lambda v: v["style_dp"].update(data=[]),
        ):
            changed = copy.deepcopy(value)
            mutation(changed)
            with self.assertRaises(ReferenceError):
                validate_voice_style(changed)

    def test_expanded_frontend_matches_independent_upstream_fixture(self):
        corpus = load_corpus(ROOT / "tests/fixtures/supertonic3/fp16-corpus.yaml")
        path = ROOT / "tests/fixtures/supertonic3/fp16-frontend.yaml"
        expected = load_yaml(path)
        indexer = [-1] * 65536
        for case in expected["cases"].values():
            for character, token in zip(
                case.get("normalized_text", ""), case.get("tokens", [])
            ):
                indexer[ord(character)] = token
        self.assertEqual(
            len(check_frontend(corpus, indexer, path, capacity=384)),
            len(corpus["cases"]),
        )

    def test_duplicate_yaml_keys_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.yaml"
            path.write_text("model: first\nmodel: second\n")
            with self.assertRaisesRegex(ReferenceError, "duplicate"):
                load_yaml(path)

    def test_non_mapping_yaml_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "list.yaml"
            path.write_text("- first\n- second\n")
            with self.assertRaises(ReferenceError):
                load_yaml(path)

    def test_unpinned_revision_rejected(self):
        self.changed_manifest(lambda m: m["source"].update(revision="main"))

    def test_boolean_schema_rejected(self):
        self.changed_manifest(lambda m: m.update(schema_version=True))

    def test_wrong_tensor_interface_rejected(self):
        self.changed_manifest(
            lambda m: m["artifacts"]["text_encoder"]["inputs"]["text_ids"].update(
                shape=[1, 129]
            )
        )

    def test_unexpected_graph_input_rejected(self):
        self.changed_manifest(
            lambda m: m["artifacts"]["ve_128"]["inputs"].update(
                unexpected={"dtype": "float32", "shape": [1]}
            )
        )

    def test_missing_artifact_rejected(self):
        self.changed_manifest(lambda m: m["artifacts"].pop("ve_512"))

    def test_wrong_hash_rejected(self):
        self.changed_manifest(
            lambda m: m["artifacts"]["tts_config"]["files"][0].update(
                sha256="unverified"
            )
        )

    def test_unsafe_paths_rejected(self):
        for path in ("../x", "/tmp/x", "a/../b", "a//b", "./x", "C:/x", "a\\b", ""):
            with self.subTest(path=path), self.assertRaises(ReferenceError):
                relative_path(path)


class BundleTests(unittest.TestCase):
    """Small payloads exercise file integrity only; no fake models are executed."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        files = {
            "model.mlmodelc/payload.bin": b"file-integrity test payload",
            "tts.json": json.dumps(
                {
                    "ae": {"sample_rate": 44100, "base_chunk_size": 512},
                    "ttl": {"chunk_compress_factor": 6, "latent_dim": 24},
                }
            ).encode(),
        }
        artifacts = {}
        for name, data in files.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            model = name.endswith(".bin")
            artifacts["model" if model else "tts_config"] = {
                "kind": "coreml_compiled" if model else "configuration",
                "path": "model.mlmodelc" if model else name,
                "files": [
                    {
                        "path": name,
                        "size_bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                ],
            }
        self.manifest = {
            "id": "file-integrity-test",
            "source": {
                "repository": "https://huggingface.co/example/model",
                "revision": "a" * 40,
            },
            "artifacts": artifacts,
        }

    def tearDown(self):
        self.temporary.cleanup()

    def test_prepare_checks_local_files_without_network(self):
        output = self.root / "prepared"
        with patch("tools.supertonic3.bundle.urlopen") as network:
            report = prepare_bundle(self.manifest, [self.source], output, mode="copy")
        network.assert_not_called()
        self.assertEqual(report["files"], 2)
        verify_bundle(self.manifest, output)

    def test_prepare_never_overwrites_existing_output(self):
        output = self.root / "prepared"
        output.mkdir()
        sentinel = output / "keep"
        sentinel.write_text("preserve")
        with self.assertRaises(ReferenceError):
            prepare_bundle(self.manifest, [self.source], output, mode="copy")
        self.assertEqual(sentinel.read_text(), "preserve")

    def test_prepare_preserves_original_yaml_bytes(self):
        original = (
            "# retain comments and formatting\n" + yaml.safe_dump(self.manifest)
        ).encode()
        output = self.root / "exact-yaml"
        prepare_bundle(
            self.manifest, [self.source], output, mode="copy", manifest_bytes=original
        )
        self.assertEqual((output / "mlsr-bundle.yaml").read_bytes(), original)
        bad_output = self.root / "mismatched-yaml"
        with self.assertRaisesRegex(ReferenceError, "differ"):
            prepare_bundle(
                self.manifest,
                [self.source],
                bad_output,
                manifest_bytes=b"id: different\n",
            )
        self.assertFalse(bad_output.exists())

    def test_missing_source_does_not_download_implicitly(self):
        with patch("tools.supertonic3.bundle.urlopen") as network:
            with self.assertRaisesRegex(ReferenceError, "requires --download"):
                prepare_bundle(self.manifest, [], self.root / "prepared", mode="copy")
        network.assert_not_called()
        self.assertFalse((self.root / "prepared").exists())

    def test_explicit_download_still_checks_bytes(self):
        def response(url, timeout):
            filename = url.split("/resolve/" + "a" * 40 + "/")[1]
            return io.BytesIO((self.source / filename).read_bytes())

        with patch("tools.supertonic3.bundle.urlopen", side_effect=response) as network:
            output = self.root / "downloaded"
            prepare_bundle(self.manifest, [], output, download=True)
        self.assertEqual(network.call_count, 2)
        verify_bundle(self.manifest, output)

    def test_bad_download_is_never_installed(self):
        with patch(
            "tools.supertonic3.bundle.urlopen",
            side_effect=lambda *a, **k: io.BytesIO(b"wrong"),
        ):
            output = self.root / "downloaded"
            with self.assertRaises(ReferenceError):
                prepare_bundle(self.manifest, [], output, download=True)
        self.assertFalse(output.exists())

    def test_corrupted_source_is_rejected(self):
        path = self.source / "model.mlmodelc/payload.bin"
        path.write_bytes(b"x" * path.stat().st_size)
        with self.assertRaises(ReferenceError):
            verify_bundle(self.manifest, self.source)

    def test_unlisted_model_file_is_rejected(self):
        (self.source / "model.mlmodelc/unlisted.bin").write_bytes(b"unexpected")
        with self.assertRaises(ReferenceError):
            verify_bundle(self.manifest, self.source)

    def test_finder_metadata_is_excluded(self):
        (self.source / "model.mlmodelc/.DS_Store").write_bytes(b"finder")
        verify_bundle(self.manifest, self.source)

    def test_later_source_mutation_is_detected_through_symlink(self):
        output = self.root / "prepared"
        try:
            prepare_bundle(self.manifest, [self.source], output, mode="symlink")
        except OSError as error:
            self.skipTest(f"Symlinks unavailable: {error}")
        (self.source / "model.mlmodelc/payload.bin").write_bytes(b"changed")
        with self.assertRaises(ReferenceError):
            verify_bundle(self.manifest, output)


class FrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = ROOT / "tests/fixtures/supertonic3/frontend.yaml"
        cls.expected = load_yaml(cls.fixture)
        # Reconstruct only the real lookup entries used by the frozen corpus.
        # Unknown entries remain unsupported; this never feeds model inference.
        cls.indexer = [-1] * 65536
        for case in cls.expected["cases"].values():
            if "tokens" in case:
                for scalar, token in zip(
                    case["normalized_text"], case["tokens"], strict=True
                ):
                    old = cls.indexer[ord(scalar)]
                    if old not in (-1, token):
                        raise AssertionError("Inconsistent upstream token fixture")
                    cls.indexer[ord(scalar)] = token

    def test_all_upstream_derived_frontend_cases(self):
        results = check_frontend(load_corpus(), self.indexer, self.fixture)
        self.assertEqual(len(results), 19)

    def test_exact_text_limit_includes_language_tags(self):
        normalized, ids, mask = encode_text("a" * 118 + ".", "en", self.indexer)
        self.assertEqual(len(normalized), 128)
        self.assertEqual(ids.shape, (1, 128))
        self.assertEqual(mask.sum(), 128)
        with self.assertRaises(InputError) as caught:
            encode_text("a" * 119 + ".", "en", self.indexer)
        self.assertEqual(caught.exception.code, "text_too_long")

    def test_normalization_expansion_is_checked(self):
        with self.assertRaises(InputError) as caught:
            encode_text("한" * 60, "ko", self.indexer)
        self.assertEqual(caught.exception.code, "text_too_long")

    def test_padding_is_zero_and_mask_tracks_tokens(self):
        normalized, ids, mask = encode_text("Hello.", "en", self.indexer)
        self.assertTrue(np.all(ids[0, len(normalized) :] == 0))
        self.assertTrue(np.all(mask[0, 0, : len(normalized)] == 1))
        self.assertTrue(np.all(mask[0, 0, len(normalized) :] == 0))

    def test_bucket_boundaries(self):
        for length, expected in [
            (1, 128),
            (127, 128),
            (128, 128),
            (129, 256),
            (255, 256),
            (256, 256),
            (257, 512),
            (512, 512),
        ]:
            with self.subTest(length=length):
                self.assertEqual(select_bucket(length), expected)
        for length in (0, -1, 513, True):
            with self.subTest(length=length), self.assertRaises(ReferenceError):
                select_bucket(length)

    def test_invalid_durations_rejected_before_allocating(self):
        self.assertEqual(duration_frames(1.0), 44100)
        for value in (0, -1, float("nan"), float("inf"), 1000):
            with self.subTest(value=value), self.assertRaises(ReferenceError):
                duration_frames(value)


class OutputTests(unittest.TestCase):
    def test_float_wav_keeps_exact_pcm(self):
        values = np.array([-0.5, 0.0, 0.25], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            write_float_wav(path, values, 44100)
            data = path.read_bytes()
        self.assertEqual(data[:4], b"RIFF")
        self.assertEqual(struct.unpack_from("<I", data, 4)[0], len(data) - 8)
        self.assertEqual(
            struct.unpack_from("<HHIIHH", data, 20), (3, 1, 44100, 176400, 4, 32)
        )
        self.assertEqual(data[36:40], b"fact")
        self.assertEqual(data[48:52], b"data")
        self.assertEqual(data[56:], values.astype("<f4").tobytes())

    def test_integer_replay_is_exact_even_with_loose_float_tolerance(self):
        left = {"tokens": np.array([1], dtype=np.int32)}
        right = {"tokens": np.array([2], dtype=np.int32)}
        self.assertFalse(compare_tensors(left, right, 100, 100)[0]["passed"])

    def test_float_replay_reports_exact_and_tolerant_matches(self):
        left = {"values": np.array([1.0], dtype=np.float32)}
        right = {"values": np.array([1.0001], dtype=np.float32)}
        self.assertFalse(compare_tensors(left, right, 0, 0)[0]["passed"])
        comparison = compare_tensors(left, right, 0.001, 0)[0]
        self.assertTrue(comparison["passed"])
        self.assertFalse(comparison["exact"])

    def test_changed_shape_or_missing_intermediate_fails(self):
        left = {"a": np.ones((1, 2), dtype=np.float32)}
        right = {"a": np.ones((2, 1), dtype=np.float32)}
        self.assertFalse(compare_tensors(left, right, 1, 1)[0]["passed"])
        with self.assertRaises(ReferenceError):
            compare_tensors(left, {}, 0, 0)

    def test_signed_zero_is_not_reported_as_bit_exact(self):
        left = {"values": np.array([0.0], dtype=np.float32)}
        right = {"values": np.array([-0.0], dtype=np.float32)}
        comparison = compare_tensors(left, right, 0, 0)[0]
        self.assertFalse(comparison["exact"])
        self.assertFalse(comparison["passed"])
        self.assertTrue(compare_tensors(left, right, 1e-5, 0)[0]["passed"])

    def test_nonfinite_fixture_is_rejected(self):
        with self.assertRaises(ReferenceError):
            tensor_record(np.array([float("nan")], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
