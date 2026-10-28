"""Strict YAML manifests and checksum-verified, opt-in asset preparation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from urllib.parse import quote
from urllib.request import urlopen

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "model-manifests/supertonic3-coreml.yaml"
DEFAULT_CORPUS = ROOT / "tests/fixtures/supertonic3/corpus.yaml"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"[0-9a-f]{40}\Z")
VOICE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


class ReferenceError(ValueError):
    """An input or artifact does not satisfy the reference contract."""


class UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loading, also rejecting ambiguous duplicate mapping keys."""


def _mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, (str, int)) or key in result:
            raise ReferenceError(f"Invalid or duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping
)


def load_yaml(path: Path) -> dict:
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ReferenceError(f"Invalid YAML in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReferenceError(f"Expected a YAML mapping in {path}")
    return value


def write_yaml(path: Path, value: dict) -> None:
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_digest(value) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def relative_path(value) -> str:
    if not isinstance(value, str):
        raise ReferenceError("Artifact paths must be strings")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or ":" in value
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ReferenceError(f"Unsafe relative path: {value!r}")
    return value


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    manifest = load_yaml(path)
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
    ):
        raise ReferenceError("Unsupported model manifest schema_version")
    if manifest.get("runtime", {}).get("kind") != "coreml":
        raise ReferenceError("This reference tool requires a Core ML manifest")
    source = manifest.get("source", {})
    if not REVISION.fullmatch(str(source.get("revision", ""))):
        raise ReferenceError("Model source must pin a full Git revision")
    repository = source.get("repository", "")
    if not re.fullmatch(
        r"https://huggingface\.co/(?:spaces/)?[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
        repository,
    ):
        raise ReferenceError("Expected a Hugging Face model repository URL")
    contract = manifest.get("contract", {})
    voices = contract.get("voices")
    if (
        not isinstance(voices, list)
        or not 1 <= len(voices) <= 64
        or any(not isinstance(v, str) or not VOICE_ID.fullmatch(v) for v in voices)
        or len(set(voices)) != len(voices)
    ):
        raise ReferenceError("Expected 1..=64 unique safe voice IDs")
    names = manifest.get("voice_names", {})
    if (
        not isinstance(names, dict)
        or (names and set(names) != set(voices))
        or any(
            not isinstance(v, str)
            or not v.strip()
            or len(v.encode("utf-8")) > 128
            or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in v)
            for v in names.values()
        )
    ):
        raise ReferenceError("Voice names must match the declared voices")
    expected = {
        "sample_rate": 44100,
        "channels": 1,
        "batch_size": 1,
        "latent_channels": 144,
        "samples_per_latent": 3072,
    }
    if any(contract.get(k) != v for k, v in expected.items()):
        raise ReferenceError("Unsupported Supertonic 3 tensor/audio contract")
    capacity = contract.get("text_tokens")
    if type(capacity) is not int or capacity not in (128, 384):
        raise ReferenceError("Expected a qualified 128 or 384 token text window")
    if contract.get("latent_buckets") != [128, 256, 512]:
        raise ReferenceError("Expected the pinned 128/256/512 VE buckets")
    if contract.get("vocoder_latent_range") != [4, 512]:
        raise ReferenceError("Unexpected vocoder shape range")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ReferenceError("Manifest must contain artifacts")
    seen = set()
    for name, artifact in artifacts.items():
        root = relative_path(artifact.get("path"))
        files = artifact.get("files")
        if not isinstance(files, list) or not files:
            raise ReferenceError(f"No locked files for {name}")
        for entry in files:
            filename = relative_path(entry.get("path"))
            if filename != root and not filename.startswith(root + "/"):
                raise ReferenceError(f"File lies outside artifact {name}: {filename}")
            if filename in seen:
                raise ReferenceError(f"Duplicate artifact file: {filename}")
            seen.add(filename)
            if not SHA256.fullmatch(str(entry.get("sha256", ""))):
                raise ReferenceError(f"Invalid SHA-256 for {filename}")
            if type(entry.get("size_bytes")) is not int or entry["size_bytes"] < 1:
                raise ReferenceError(f"Invalid byte count for {filename}")
    required = {
        "text_encoder",
        "duration_predictor",
        "vocoder",
        "ve_128",
        "ve_256",
        "ve_512",
        "tts_config",
        "unicode_indexer",
    } | {f"voice_{voice}" for voice in voices}
    if set(artifacts) != required:
        raise ReferenceError(f"Expected artifacts: {sorted(required)}")
    for voice in voices:
        artifact = artifacts[f"voice_{voice}"]
        if (
            artifact.get("kind") != "voice_style"
            or len(artifact["files"]) != 1
            or artifact["files"][0]["path"] != artifact["path"]
            or artifact["files"][0]["size_bytes"] > 4 * 1024 * 1024
            or artifact.get("inputs") is not None
            or artifact.get("outputs") is not None
            or artifact.get("latent_bucket") is not None
        ):
            raise ReferenceError(f"Invalid voice style artifact: {voice}")
    for name in (
        "text_encoder",
        "duration_predictor",
        "vocoder",
        "ve_128",
        "ve_256",
        "ve_512",
    ):
        model = artifacts[name]
        if model.get("kind") != "coreml_compiled" or not model["path"].endswith(
            ".mlmodelc"
        ):
            raise ReferenceError(f"Expected compiled Core ML artifact: {name}")
        for direction in ("inputs", "outputs"):
            interface = model.get(direction)
            if not isinstance(interface, dict) or not interface:
                raise ReferenceError(f"Missing {direction} for {name}")
            for tensor in interface.values():
                if tensor.get("dtype") not in ("float32", "int32"):
                    raise ReferenceError(f"Unsupported tensor dtype in {name}")
                if not isinstance(tensor.get("shape"), list):
                    raise ReferenceError(f"Missing tensor shape in {name}")

    # This tool validates this particular Supertonic export, not arbitrary
    # model architectures. Catch contract edits before invoking native code.
    def f32(shape):
        return {"dtype": "float32", "shape": shape}

    text = {
        "text_ids": {"dtype": "int32", "shape": [1, capacity]},
        "text_mask": f32([1, 1, capacity]),
    }
    interfaces = {
        "duration_predictor": (
            {**text, "style_dp": f32([1, 8, 16])},
            {"duration": f32([1])},
        ),
        "text_encoder": (
            {**text, "style_ttl": f32([1, 50, 256])},
            {"text_emb": f32([1, 256, capacity])},
        ),
        "vocoder": ({"latent": f32([1, 144, "L"])}, {"wav": f32([1, "3072*L"])}),
    }
    for bucket in (128, 256, 512):
        latent = f32([1, 144, bucket])
        interfaces[f"ve_{bucket}"] = (
            {
                "noisy_latent": latent,
                "text_emb": f32([1, 256, capacity]),
                "style_ttl": f32([1, 50, 256]),
                "latent_mask": f32([1, 1, bucket]),
                "text_mask": f32([1, 1, capacity]),
                "current_step": f32([1]),
                "total_step": f32([1]),
            },
            {"denoised_latent": latent},
        )
    for name, (inputs, outputs) in interfaces.items():
        if artifacts[name]["inputs"] != inputs or artifacts[name]["outputs"] != outputs:
            raise ReferenceError(
                f"Tensor interface disagrees with the pinned export: {name}"
            )
    return manifest


def locked_files(manifest: dict):
    for artifact in manifest["artifacts"].values():
        yield from artifact["files"]


def file_matches(path: Path, entry: dict) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == entry["size_bytes"]
        and digest(path) == entry["sha256"]
    )


def verify_bundle(manifest: dict, root: Path) -> dict:
    failures = [
        entry["path"]
        for entry in locked_files(manifest)
        if not file_matches(root / entry["path"], entry)
    ]
    # Extra files inside a compiled model could alter its loading behavior.
    # Finder metadata is explicitly outside the model-content contract.
    for artifact in manifest["artifacts"].values():
        if artifact["kind"] == "coreml_compiled":
            directory = root / artifact["path"]
            expected = {entry["path"] for entry in artifact["files"]}
            if directory.is_dir():
                actual = {
                    p.relative_to(root).as_posix()
                    for p in directory.rglob("*")
                    if p.is_file() and p.name != ".DS_Store"
                }
                failures.extend(sorted(actual - expected))
    if failures:
        raise ReferenceError(
            "Missing, changed, or unexpected bundle files:\n" + "\n".join(failures)
        )
    for artifact in manifest["artifacts"].values():
        if artifact["kind"] == "voice_style":
            validate_voice_style(json.loads((root / artifact["path"]).read_text()))
    config = json.loads(
        (root / manifest["artifacts"]["tts_config"]["path"]).read_text()
    )
    if (
        config["ae"]["sample_rate"] != 44100
        or config["ae"]["base_chunk_size"] != 512
        or config["ttl"]["chunk_compress_factor"] != 6
        or config["ttl"]["latent_dim"] != 24
    ):
        raise ReferenceError("tts.json disagrees with the tensor/audio contract")
    entries = list(locked_files(manifest))
    return {
        "bundle_id": manifest["id"],
        "manifest_sha256": canonical_digest(manifest),
        "files": len(entries),
        "size_bytes": sum(e["size_bytes"] for e in entries),
    }


def validate_voice_style(value: dict) -> None:
    """The shared Supertonic conditioning contract, independent of preset ID."""
    if not isinstance(value, dict):
        raise ReferenceError("Expected a voice style object")
    for key, shape in (("style_ttl", [1, 50, 256]), ("style_dp", [1, 8, 16])):
        style = value.get(key, {})
        if not isinstance(style, dict):
            raise ReferenceError(f"Invalid voice style tensor: {key}")
        data = style.get("data")
        if (
            style.get("dims") != shape
            or any(type(dim) is not int for dim in style["dims"])
            or style.get("type") != "float32"
            or not isinstance(data, list)
            or len(data) != shape[0]
            or any(
                not isinstance(batch, list)
                or len(batch) != shape[1]
                or any(
                    not isinstance(row, list)
                    or len(row) != shape[2]
                    or any(
                        type(v) not in (int, float)
                        or not math.isfinite(v)
                        or abs(v) > 3.4028234663852886e38
                        for v in row
                    )
                    for row in batch
                )
                for batch in data
            )
        ):
            raise ReferenceError(f"Invalid voice style tensor: {key}")


def asset_url(manifest: dict, path: str) -> str:
    provenance = manifest.get("provenance", {})
    # Locally exported graphs remain local. A separately pinned upstream source
    # can supply only the explicitly declared small voice-style files.
    voice_source = provenance.get("voice_styles")
    if voice_source is not None and not isinstance(voice_source, dict):
        raise ReferenceError("Expected a pinned voice source mapping")
    if voice_source and any(
        a["kind"] == "voice_style" and a["path"] == path
        for a in manifest["artifacts"].values()
    ):
        repository = voice_source.get("repository", "")
        revision = voice_source.get("revision", "")
        if not re.fullmatch(
            r"https://huggingface\.co/(?:spaces/)?[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
            repository,
        ) or not REVISION.fullmatch(revision):
            raise ReferenceError(
                "Voice source must pin a Hugging Face repository and revision"
            )
        prefix = relative_path(voice_source.get("path_prefix"))
        return f"{repository}/resolve/{revision}/{prefix}/{quote(relative_path(path), safe='/')}"
    if provenance.get("local_export"):
        raise ReferenceError(
            "This local Core ML export has no published download; provide verified local assets"
        )
    source = manifest["source"]
    return f"{source['repository']}/resolve/{source['revision']}/{quote(relative_path(path), safe='/')}"


def prepare_bundle(
    manifest: dict,
    sources: list[Path],
    output: Path,
    mode: str = "symlink",
    download: bool = False,
    *,
    manifest_bytes: bytes | None = None,
) -> dict:
    """Build a new bundle; preserve supplied YAML bytes for file-hash identity."""
    if manifest_bytes is None:
        # Programmatic fixtures may supply only a mapping. Production CLI calls
        # pass the original file bytes, which optional model variants pin.
        manifest_bytes = yaml.safe_dump(
            manifest, sort_keys=False, allow_unicode=True
        ).encode("utf-8")
    try:
        if yaml.load(manifest_bytes, Loader=UniqueKeyLoader) != manifest:
            raise ReferenceError("Original manifest bytes differ from parsed manifest")
    except yaml.YAMLError as error:
        raise ReferenceError("Invalid original manifest bytes") from error
    if output.exists() or output.is_symlink():
        raise ReferenceError(f"Output already exists; use verify-bundle: {output}")
    if mode not in ("symlink", "copy"):
        raise ReferenceError(f"Unknown preparation mode: {mode}")
    selected = []
    for entry in locked_files(manifest):
        candidate = next(
            (
                root / entry["path"]
                for root in sources
                if file_matches(root / entry["path"], entry)
            ),
            None,
        )
        if candidate is None and not download:
            raise ReferenceError(
                f"No matching local file: {entry['path']}; downloading requires --download"
            )
        selected.append((entry, candidate))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for entry, source in selected:
            destination = staging / entry["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source is None:
                with urlopen(
                    asset_url(manifest, entry["path"]), timeout=60
                ) as response:
                    with destination.open("xb") as stream:
                        shutil.copyfileobj(response, stream)
            elif mode == "symlink":
                destination.symlink_to(source.resolve())
            elif mode == "copy":
                shutil.copyfile(source, destination)
            else:
                raise ReferenceError(f"Unknown preparation mode: {mode}")
        result = verify_bundle(manifest, staging)
        (staging / "mlsr-bundle.yaml").write_bytes(manifest_bytes)
        # A concurrent destination must not be replaced, even if it is empty.
        output.mkdir()
        try:
            for child in staging.iterdir():
                child.rename(output / child.name)
        except BaseException:
            shutil.rmtree(output)
            raise
        return result
    finally:
        shutil.rmtree(staging)
