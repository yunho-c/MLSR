"""Supertonic's reference text processing, with explicit input-limit failures.

Normalization is adapted from Supertone's MIT-licensed py/helper.py. See
THIRD_PARTY_NOTICES.md. This is development tooling, not the future Rust engine.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np

from .bundle import ReferenceError


LANGUAGES = "en ko ja ar bg cs da de el es et fi fr hi hr hu id it lt lv nl pl pt ro ru sk sl sv tr uk vi".split()
EMOJI = re.compile(
    "[\U0001f300-\U0001f64f\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f\U0001f780-\U0001f7ff\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff"
    "\u2600-\u26ff\u2700-\u27bf\U0001f1e6-\U0001f1ff]+"
)
REPLACEMENTS = {
    "–": "-",
    "‑": "-",
    "—": "-",
    "_": " ",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "´": "'",
    "`": "'",
    "[": " ",
    "]": " ",
    "|": " ",
    "/": " ",
    "#": " ",
    "→": " ",
    "←": " ",
}


class InputError(ReferenceError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def normalize_text(text: str, language: str) -> str:
    if language not in LANGUAGES:
        raise InputError("unsupported_language", f"Unsupported language: {language}")
    if not isinstance(text, str) or not text.strip():
        raise InputError("empty_text", "Text must contain speech content")
    text = unicodedata.normalize("NFKD", text)
    text = EMOJI.sub("", text)
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)
    text = re.sub(r"[♥☆♡©\\]", "", text)
    for old, new in {
        "@": " at ",
        "e.g.,": "for example, ",
        "i.e.,": "that is, ",
    }.items():
        text = text.replace(old, new)
    for punctuation in (",", ".", "!", "?", ";", ":", "'"):
        text = text.replace(" " + punctuation, punctuation)
    for mark in ('"', "'", "`"):
        while mark * 2 in text:
            text = text.replace(mark * 2, mark)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise InputError("empty_text", "Normalization removed all speech content")
    if not re.search(r"[.!?;:,'\"')\]}…。」』】〉》›»]$", text):
        text += "."
    return f"<{language}>{text}</{language}>"


def encode_text(text: str, language: str, indexer: list[int], limit: int = 128):
    normalized = normalize_text(text, language)
    if len(normalized) > limit:
        raise InputError(
            "text_too_long",
            f"{len(normalized)} normalized tokens exceed {limit}; no truncation",
        )
    tokens = []
    for scalar in normalized:
        point = ord(scalar)
        if point >= len(indexer) or indexer[point] < 0:
            raise InputError("unsupported_character", f"No token for U+{point:04X}")
        tokens.append(indexer[point])
    ids = np.zeros((1, limit), dtype=np.int32)
    mask = np.zeros((1, 1, limit), dtype=np.float32)
    ids[0, : len(tokens)] = tokens
    mask[0, 0, : len(tokens)] = 1
    return normalized, ids, mask


def select_bucket(length: int, buckets=(128, 256, 512)) -> int:
    if type(length) is not int or length < 1:
        raise ReferenceError("Latent length must be positive")
    for bucket in buckets:
        if length <= bucket:
            return bucket
    raise ReferenceError(f"Latent length {length} exceeds supported buckets")


def duration_frames(duration: float, sample_rate: int = 44100) -> int:
    # The existing Swift driver performs this multiplication in Float32.
    frames = np.float32(duration) * np.float32(sample_rate)
    if not np.isfinite(frames) or frames <= 0 or frames > 512 * 3072:
        raise ReferenceError(f"Duration outside the supported model range: {duration}")
    return int(frames)
