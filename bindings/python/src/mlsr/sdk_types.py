"""Owned developer API values. Text offsets name their encoding explicitly."""

from __future__ import annotations

import array
from dataclasses import dataclass, field
from ._synthesis import SynthesisError


@dataclass(frozen=True)
class Failure:
    request_id: int
    category: str
    stage: str
    backend: str | None
    message: str


class TtsError(SynthesisError):
    """Stable category and optional request/stage/backend failure context."""

    def __init__(self, code: str, message: str, failure: Failure | None = None):
        super().__init__(code, message)
        self.message = message
        self.failure = failure
        self.category = failure.category if failure else code
        self.stage = failure.stage if failure else None
        self.request_id = failure.request_id if failure else None
        self.backend = failure.backend if failure else None


@dataclass(frozen=True)
class Outcome:
    kind: str
    reason: str | None = None
    error: Failure | None = None

    @classmethod
    def _parse(cls, data):
        return cls(
            data["kind"],
            data.get("reason"),
            Failure(**data["error"]) if data.get("error") else None,
        )


@dataclass(frozen=True)
class Completion:
    request_id: int
    frames: int
    outcome: Outcome

    @classmethod
    def _parse(cls, data):
        return cls(data["request_id"], data["frames"], Outcome._parse(data["outcome"]))

    def result(self) -> Completion:
        """Cancellation is an outcome; failure raises its typed error."""
        if self.outcome.error:
            failure = self.outcome.error
            raise TtsError(failure.category, failure.message, failure)
        return self


@dataclass(frozen=True)
class Span:
    """Half-open range. The containing field specifies frames, UTF-8 or UTF-16."""

    start: int
    end: int


@dataclass(frozen=True)
class TextRange:
    segment_id: int
    sentence_id: int
    utf8: Span
    utf16: Span

    @classmethod
    def _parse(cls, data):
        return cls(
            data["segment_id"],
            data["sentence_id"],
            Span(**data["utf8"]),
            Span(**data["utf16"]),
        )

    def text(self, original: str) -> str:
        """Slice original Python text safely using the reported UTF-8 range."""
        return original.encode("utf-8")[self.utf8.start : self.utf8.end].decode("utf-8")


@dataclass(frozen=True)
class AudioFormat:
    sample_rate: int
    channels: int

    def seconds(self, frames: int) -> float:
        return frames / self.sample_rate


@dataclass(frozen=True)
class WavFile:
    path: str
    frames: int
    format: AudioFormat

    @classmethod
    def _parse(cls, data):
        return cls(data["path"], data["frames"], AudioFormat(**data["format"]))


@dataclass(frozen=True)
class AudioClip:
    """Python-owned interleaved Float32 PCM, independent of every native handle.

    Do not resize or mutate samples concurrently with write_wav/awrite_wav.
    """

    samples: array.array = field(repr=False)
    format: AudioFormat = AudioFormat(44100, 1)

    def __post_init__(self):
        if self.samples.typecode != "f" or self.samples.itemsize != 4:
            raise ValueError("samples must be an array('f') of 32-bit floats")
        if self.format.channels < 1 or len(self.samples) % self.format.channels:
            raise ValueError("PCM must contain complete interleaved frames")

    @property
    def frames(self) -> int:
        return len(self.samples) // self.format.channels

    @property
    def duration(self) -> float:
        return self.format.seconds(self.frames)

    def write_wav(self, path) -> WavFile:
        from . import _sdk

        return _sdk.write_pcm(self, path)

    async def awrite_wav(self, path) -> WavFile:
        from .sdk import _offload

        return await _offload(lambda: self.write_wav(path))


@dataclass(frozen=True)
class AudioChunk:
    request_id: int
    start_frame: int
    audio: AudioClip
    text_range: TextRange


@dataclass(frozen=True)
class VoiceCapabilities:
    pcm: bool
    duration_prediction: bool
    segment_alignment: bool
    word_alignment: bool


@dataclass(frozen=True)
class Voice:
    id: str
    model_voice_id: str
    display_name: str
    model: str
    bundle_id: str
    revision: str
    manifest_sha256: str
    backend: str
    languages: tuple[str, ...]
    qualified_languages: tuple[str, ...]
    requires_network: bool
    readiness: str
    capabilities: VoiceCapabilities

    @classmethod
    def _parse(cls, data):
        return cls(
            **{
                **data,
                "languages": tuple(data["languages"]),
                "qualified_languages": tuple(data["qualified_languages"]),
                "capabilities": VoiceCapabilities(**data["capabilities"]),
            }
        )


@dataclass(frozen=True)
class ResolvedVoice:
    voice: Voice
    requested_locale: str
    language: str
    used_fallback: bool

    @classmethod
    def _parse(cls, data):
        return cls(**{**data, "voice": Voice._parse(data["voice"])})


@dataclass(frozen=True)
class SupertonicOptions:
    steps: int = 8
    seed: int = 0


@dataclass(frozen=True)
class Request:
    """Immutable synthesis input. Normal speaking_rate=1.0 (not raw model speed)."""

    text: str
    locale: str = "en"
    voice: str = ""
    locale_fallback: str = "strict"
    speaking_rate: float = 1.0
    backend: SupertonicOptions = SupertonicOptions()


@dataclass(frozen=True)
class TtsOptions:
    output_mode: str = "default"  # default, headless, host_managed
    max_requests: int = 16
    max_text_bytes: int = 256 * 1024
    queue_frames: int = 44100
    chunk_frames: int = 4096
    max_segment_frames: int = 512 * 3072
    max_retained_stores: int = 128
    max_retained_bytes: int = 128 * 1024 * 1024


@dataclass(frozen=True)
class SpeechOptions:
    playback_rate: float = 1.0
    volume: float = 1.0
    paused: bool = False
    retain_audio: bool = True
    lookahead_frames: int = 88200


@dataclass(frozen=True)
class OutputCapabilities:
    max_chunk_frames: int
    pause_resume: bool
    volume: bool
    rate_range: tuple[float, float] | None
    rebase: bool
    host_managed_policy: bool
    route_events: bool
    interruption_events: bool

    @classmethod
    def _parse(cls, data):
        return cls(
            **{
                **data,
                "rate_range": tuple(data["rate_range"]) if data["rate_range"] else None,
            }
        )


@dataclass(frozen=True)
class RetentionStats:
    used_bytes: int
    peak_bytes: int
    live_stores: int
    peak_stores: int


@dataclass(frozen=True)
class SynthesisState:
    revision: int
    request_id: int
    phase: str
    cancellation_requested: bool
    plan_ready: bool
    generated_frames: int
    consumed_frames: int
    queued_frames: int
    peak_queued_frames: int
    format: AudioFormat
    estimated_frames: int | None
    duration_frames: int | None
    completion: Completion | None

    @classmethod
    def _parse(cls, data):
        return cls(
            **{
                **data,
                "format": AudioFormat(**data["format"]),
                "completion": (
                    Completion._parse(data["completion"])
                    if data["completion"]
                    else None
                ),
            }
        )


@dataclass(frozen=True)
class SpeechState:
    revision: int
    request_id: int
    synthesis: str
    playback: str
    format: AudioFormat
    position_frame: int
    buffered_until_frame: int
    seekable_ranges: tuple[Span, ...]
    estimated_frames: int | None
    duration_frames: int | None
    playback_rate: float
    volume: float
    active_text_range: TextRange | None
    output_latency_seconds: float | None
    cancellation: str | None
    completion: Completion | None

    @classmethod
    def _parse(cls, data):
        return cls(
            **{
                **data,
                "format": AudioFormat(**data["format"]),
                "seekable_ranges": tuple(
                    Span(**span) for span in data["seekable_ranges"]
                ),
                "active_text_range": (
                    TextRange._parse(data["active_text_range"])
                    if data["active_text_range"]
                    else None
                ),
                "completion": (
                    Completion._parse(data["completion"])
                    if data["completion"]
                    else None
                ),
            }
        )

    @property
    def position(self) -> float:
        return self.format.seconds(self.position_frame)

    @property
    def duration(self) -> float | None:
        return (
            None
            if self.duration_frames is None
            else self.format.seconds(self.duration_frames)
        )


@dataclass(frozen=True)
class TextSegment:
    id: int
    sentence_id: int
    utf8: Span
    utf16: Span
    original: str
    normalized: str


@dataclass(frozen=True)
class TextSentence:
    id: int
    utf8: Span
    utf16: Span
    segments: Span


@dataclass(frozen=True)
class TextPlan:
    original: str
    language: str
    manifest_sha256: str
    segments: tuple[TextSegment, ...]
    sentences: tuple[TextSentence, ...]

    @classmethod
    def _parse(cls, data):
        return cls(
            data["original"],
            data["language"],
            data["manifest_sha256"],
            tuple(
                TextSegment(
                    **{**s, "utf8": Span(**s["utf8"]), "utf16": Span(**s["utf16"])}
                )
                for s in data["segments"]
            ),
            tuple(
                TextSentence(
                    **{
                        **s,
                        "utf8": Span(**s["utf8"]),
                        "utf16": Span(**s["utf16"]),
                        "segments": Span(**s["segments"]),
                    }
                )
                for s in data["sentences"]
            ),
        )
