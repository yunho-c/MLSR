"""Independently compare QNN SDK Float32 WAV payloads with legacy binding PCM.

Uses only the standard library, including on the native ARM64 embedded Python.
"""

import hashlib
import json
import struct
import sys
from pathlib import Path


def pcm(path):
    raw = path.read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE" or len(raw) < 12:
        raise ValueError(f"Invalid WAV: {path}")
    if struct.unpack_from("<I", raw, 4)[0] != len(raw) - 8:
        raise ValueError(f"Invalid RIFF length: {path}")
    chunks, offset = {}, 12
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise ValueError(f"Truncated WAV chunk: {path}")
        name, size = struct.unpack_from("<4sI", raw, offset)
        start, end = offset + 8, offset + 8 + size
        if end > len(raw) or name in chunks:
            raise ValueError(f"Invalid WAV chunk: {path}")
        chunks[name] = raw[start:end]
        offset = end + size % 2
    if offset != len(raw):
        raise ValueError(f"Invalid WAV padding: {path}")
    fmt, samples = chunks[b"fmt "], chunks[b"data"]
    tag, channels, rate, byte_rate, alignment, bits = struct.unpack_from("<HHIIHH", fmt)
    if tag == 65534:
        if len(fmt) != 40 or fmt[24:40] != bytes.fromhex(
            "0300000000001000800000aa00389b71"
        ):
            raise ValueError(f"Expected extensible Float32 WAV: {path}")
    elif tag != 3:
        raise ValueError(f"Expected Float32 WAV: {path}")
    if (channels, rate, byte_rate, alignment, bits) != (1, 44100, 176400, 4, 32):
        raise ValueError(f"Unexpected PCM format: {path}")
    if not samples or len(samples) % 4:
        raise ValueError(f"Invalid sample count: {path}")
    return samples


def main():
    root = Path(sys.argv[1])
    report = {"pcm": {}, "examples": {}}
    for name in ("en_M1", "en_F1", "ko_M1", "ko_F1"):
        expected = (root / "python-pcm" / f"{name}.f32").read_bytes()
        for consumer in ("c", "python", "go", "dart"):
            if pcm(root / f"{consumer}-sdk" / f"{name}.wav") != expected:
                raise ValueError(f"SDK/legacy PCM differs: {consumer}/{name}")
        report["pcm"][name] = {
            "sha256": hashlib.sha256(expected).hexdigest(),
            "frames": len(expected) // 4,
            "four_sdk_consumers_and_legacy_byte_exact": True,
        }
    expected = (root / "python-pcm/en_M1.f32").read_bytes()
    for language in ("python", "go", "dart"):
        path = root / f"{language}-sdk-example.wav"
        if pcm(path) != expected:
            raise ValueError(f"SDK example PCM differs: {language}")
        report["examples"][language] = {
            "frames": len(expected) // 4,
            "wav_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "pcm_byte_exact": True,
        }
    (root / "sdk-pcm.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("Four QNN SDK consumers and three examples match legacy PCM byte for byte")


if __name__ == "__main__":
    main()
