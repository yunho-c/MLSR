"""Shared Rust developer facade with synchronous and asyncio surfaces.

Native output never sends PCM through Python. Audio/state iterators pull on
demand; context managers deterministically release consumers even on early exit.
"""

from __future__ import annotations

import asyncio
import ctypes as C
from dataclasses import dataclass
import os
import threading
import time
import weakref
from . import _sdk as ffi
from .sdk_types import (
    AudioChunk,
    AudioClip,
    Completion,
    OutputCapabilities,
    Request,
    ResolvedVoice,
    RetentionStats,
    SpeechOptions,
    SpeechState,
    SynthesisState,
    TextPlan,
    TextRange,
    TtsError,
    TtsOptions,
    Voice,
    WavFile,
)


async def _join(task):
    # A second task.cancel() must not abandon an in-flight native owner/consumer.
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    return task.result()


async def _offload(function, *, on_cancel=None, discard=None):
    task = asyncio.create_task(asyncio.to_thread(function))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            if on_cancel:
                on_cancel()
        except Exception:
            pass  # A concurrent explicit close can win the cancellation race.
        finally:
            try:
                result = await _join(task)
                if discard:
                    await _join(asyncio.create_task(asyncio.to_thread(discard, result)))
            except Exception:
                pass  # Preserve task cancellation after native cleanup/quiescence.
        raise


def _fallback(lib, handle, forget):
    # The library reference is retained by the finalizer. Explicit close/aclose is
    # preferred; GC timing is intentionally not a completion guarantee.
    try:
        (lib.mlsr_sdk_forget if forget else lib.mlsr_sdk_release)(handle)
    except Exception:
        pass


class _Resource:
    def _init(self, lib, handle, *, forget=False):
        self._lib, self._handle = lib, handle
        self._lock = threading.Lock()
        self._cleanup = weakref.finalize(self, _fallback, lib, handle, forget)

    def _get(self):
        with self._lock:
            if not self._handle:
                raise TtsError("handle_closed", "This SDK handle is closed")
            return self._handle

    def close(self):
        """Cancel/join owned work and release this handle; idempotent, blocking."""
        with self._lock:
            handle, self._handle = self._handle, 0
            self._cleanup.detach()
        if handle and self._lib.mlsr_sdk_release(handle) != 0:
            raise TtsError("release_failed", "SDK handle release failed")

    async def aclose(self):
        """Join native work off the event-loop thread, even if cleanup is cancelled."""
        await _offload(self.close)

    def __enter__(self):
        self._get()
        return self

    def __exit__(self, *_):
        self.close()

    async def __aenter__(self):
        self._get()
        return self

    async def __aexit__(self, *_):
        await self.aclose()


def voices() -> tuple[Voice, ...]:
    """Discover declared voices without loading models or opening output."""
    lib = ffi.library()
    return _voices(lib, 0)


def _voices(lib, handle):
    data = ffi.metadata(lib, "voice_catalog", handle)[1]["data"]
    return tuple(Voice._parse(voice) for voice in data["voices"])


def resolve_voice(locale="en", voice="", *, fallback="strict") -> ResolvedVoice:
    """Resolve against the built-in catalog. Readiness remains catalog_only."""
    return _resolve(ffi.library(), 0, locale, voice, fallback)


def _resolve(lib, handle, locale, voice, fallback):
    locale_arg, locale_owner = ffi.old.string(locale)
    voice_arg, voice_owner = ffi.old.string(voice)
    policy = ffi.enum(fallback, ("strict", "same_language"), "locale fallback")
    data = ffi.metadata(lib, "voice_resolve", handle, locale_arg, voice_arg, policy)[1][
        "data"
    ]
    return ResolvedVoice._parse(data)


class _Speaker(_Resource):
    @property
    def voices(self) -> tuple[Voice, ...]:
        return _voices(self._lib, self._get())

    def resolve_voice(
        self, locale="en", voice="", *, fallback="strict"
    ) -> ResolvedVoice:
        return _resolve(self._lib, self._get(), locale, voice, fallback)

    def speak(
        self, request: Request | str, *, options=SpeechOptions(), mode="replace"
    ) -> Speech:
        """Accept immediately. Output/model work runs in Rust; default replaces.

        Discarding the returned Python object does not cancel fire-and-forget
        speech. Keep it to observe completion or use its controls.
        """
        if isinstance(request, str):
            request = Request(request)
        settings, owners = ffi.request(self._lib, request)
        output = ffi.speech_options(self._lib, options, mode)
        handle = ffi.out(
            self._lib, "speak", self._get(), C.byref(settings), C.byref(output)
        )[1]
        return Speech(self, handle)


class Tts(_Speaker):
    """Persistent facade; independent controllers share its bounded inference lane."""

    def __init__(
        self,
        manifest,
        bundle,
        *,
        options=TtsOptions(),
        compute_policy="cpu_and_neural_engine",
        reuse_output_buffers=True,
        execution_variant=0,
    ):
        lib = ffi.library()
        handle = ffi.open_tts(
            lib,
            manifest,
            bundle,
            options,
            compute_policy,
            reuse_output_buffers,
            execution_variant,
        )
        self._init(lib, handle)
        self._controller = None
        self._controller_lock = threading.Lock()

    @classmethod
    async def open_async(cls, manifest, bundle, **options) -> Tts:
        """Load off the event loop; cancellation waits for load and closes its result."""
        return await _offload(
            lambda: cls(manifest, bundle, **options),
            discard=lambda owner: owner.close(),
        )

    @property
    def output_capabilities(self) -> OutputCapabilities | None:
        data = ffi.metadata(self._lib, "sdk_snapshot", self._get())[1]["data"][
            "output_capabilities"
        ]
        return OutputCapabilities._parse(data) if data else None

    @property
    def retention_stats(self) -> RetentionStats:
        data = ffi.metadata(self._lib, "sdk_snapshot", self._get())[1]["data"][
            "retention"
        ]
        return RetentionStats(**data)

    @property
    def controller(self) -> SpeechController:
        """The persistent default controller also used by Tts.speak."""
        with self._controller_lock:
            if self._controller is None:
                self._controller = self._new_controller(False)
            return self._controller

    def _new_controller(self, independent):
        handle = ffi.out(self._lib, "tts_controller", self._get(), int(independent))[1]
        return SpeechController(self, handle)

    def create_controller(self) -> SpeechController:
        """Create a separate replace/enqueue scope, bounded by the same facade."""
        return self._new_controller(True)

    def synthesize(self, request: Request | str) -> SynthesisJob:
        """Create a lazy job. Snapshot, states and completion never start it."""
        if isinstance(request, str):
            request = Request(request)
        settings, owners = ffi.request(self._lib, request)
        handle = ffi.out(self._lib, "tts_synthesize", self._get(), C.byref(settings))[1]
        return SynthesisJob(self, handle)

    def collect(self, job: SynthesisJob, *, max_bytes=32 * 1024 * 1024) -> AudioClip:
        """Claim one job's sole consumer and collect bounded PCM. Blocking."""
        handle = ffi.out(
            self._lib,
            "tts_collect",
            self._get(),
            job._get(),
            ffi._uint(max_bytes, 64, "max_bytes"),
        )[1]
        return ffi.read_audio(self._lib, handle)[0]

    async def acollect(
        self, job: SynthesisJob, *, max_bytes=32 * 1024 * 1024
    ) -> AudioClip:
        return await _offload(
            lambda: self.collect(job, max_bytes=max_bytes), on_cancel=job.cancel
        )

    def render(
        self, request: Request | str, *, max_bytes=32 * 1024 * 1024
    ) -> AudioClip:
        with self.synthesize(request) as job:
            return self.collect(job, max_bytes=max_bytes)

    async def arender(
        self, request: Request | str, *, max_bytes=32 * 1024 * 1024
    ) -> AudioClip:
        async with self.synthesize(request) as job:
            return await self.acollect(job, max_bytes=max_bytes)

    def _write(self, kind, source, path, max_frames):
        path_arg, path_owner = ffi.old.string(os.fspath(path))
        data = ffi.metadata(
            self._lib,
            "tts_write_" + kind,
            self._get(),
            source._get(),
            path_arg,
            ffi._uint(max_frames, 64, "max_frames"),
        )[1]["data"]
        return WavFile._parse(data)

    def write_job(
        self, job: SynthesisJob, path, *, max_frames=30 * 60 * 44100
    ) -> WavFile:
        """Stream a job to a new WAV; job.cancel remains independent of this call."""
        return self._write("job", job, path, max_frames)

    async def awrite_job(
        self, job: SynthesisJob, path, *, max_frames=30 * 60 * 44100
    ) -> WavFile:
        return await _offload(
            lambda: self.write_job(job, path, max_frames=max_frames),
            on_cancel=job.cancel,
        )

    def render_to_file(
        self, request: Request | str, path, *, max_frames=30 * 60 * 44100
    ) -> WavFile:
        with self.synthesize(request) as job:
            return self.write_job(job, path, max_frames=max_frames)

    async def arender_to_file(
        self, request: Request | str, path, *, max_frames=30 * 60 * 44100
    ) -> WavFile:
        async with self.synthesize(request) as job:
            return await self.awrite_job(job, path, max_frames=max_frames)

    def write_speech(
        self, speech: Speech, path, *, max_frames=30 * 60 * 44100
    ) -> WavFile:
        """Export a completed retained recording without another inference pass."""
        return self._write("speech", speech, path, max_frames)

    async def awrite_speech(
        self, speech: Speech, path, *, max_frames=30 * 60 * 44100
    ) -> WavFile:
        """Run retained export off-loop. Cancellation waits for file finalization."""
        return await _offload(
            lambda: self.write_speech(speech, path, max_frames=max_frames)
        )

    def close(self):
        try:
            super().close()
        finally:
            with self._controller_lock:
                if self._controller is not None:
                    self._controller.close()


class SpeechController(_Speaker):
    def __init__(self, owner, handle):
        self._owner = owner
        self._init(owner._lib, handle)

    @property
    def capabilities(self) -> OutputCapabilities:
        return OutputCapabilities._parse(
            ffi.metadata(self._lib, "sdk_snapshot", self._get())[1]["data"]
        )


class _RequestHandle(_Resource):
    def __init__(self, owner, handle):
        self._owner = owner
        self._init(owner._lib, handle, forget=True)
        self.id = self._lib.mlsr_sdk_id(handle)

    @property
    def snapshot(self) -> SpeechState | SynthesisState:
        data = ffi.metadata(self._lib, "sdk_snapshot", self._get())[1]
        return self._state_type._parse(data["data"])

    @property
    def plan(self) -> TextPlan | None:
        _, data = ffi.metadata(self._lib, "text_plan", self._get())
        return TextPlan._parse(data["data"]) if data else None

    def states(self) -> StateStream:
        """Independent coalesced observer; closing it never cancels the request."""
        handle = ffi.out(self._lib, "sdk_states", self._get())[1]
        return StateStream(self, handle)

    def wait(self, timeout: float | None = None) -> Completion | None:
        """Wait in seconds without consuming PCM or starting lazy work.

        Returns None on timeout, cancellation as an outcome, and raises TtsError
        on terminal failure. A successful speech waits for actual output drain.
        """
        if timeout is not None and (timeout < 0 or not timeout < float("inf")):
            raise ValueError("timeout must be nonnegative finite seconds or None")
        end = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = (
                0.1 if end is None else min(0.1, max(0.0, end - time.monotonic()))
            )
            _, data = ffi.metadata(
                self._lib, "sdk_wait", self._get(), round(remaining * 1000)
            )
            if data:
                return Completion._parse(data["data"]).result()
            if end is not None and time.monotonic() >= end:
                return None

    async def wait_async(self, timeout: float | None = None) -> Completion | None:
        """Task cancellation requests native cancellation and waits for quiescence."""
        return await _offload(lambda: self.wait(timeout), on_cancel=self.cancel)


class SynthesisJob(_RequestHandle):
    _state_type = SynthesisState

    def start(self):
        ffi.command(self._lib, "synthesis_start", self._get())

    def cancel(self):
        ffi.command(self._lib, "synthesis_cancel", self._get(), 0)

    def audio(self) -> AudioStream:
        """Claim/start the sole PCM consumer. Use with/async with on early exit."""
        handle = ffi.out(self._lib, "synthesis_audio", self._get())[1]
        return AudioStream(self, handle)


@dataclass(frozen=True)
class Navigation:
    kind: str  # boundary, seeked, started
    frame: int | None = None
    speech: Speech | None = None


class Speech(_RequestHandle):
    _state_type = SpeechState

    def _command(self, command, value=0.0):
        ffi.command(self._lib, "speech_command", self._get(), command, value)

    def cancel(self):
        self._command(0)

    def pause(self):
        self._command(1)

    def resume(self):
        self._command(2)

    def set_volume(self, value: float):
        self._command(3, value)

    def set_playback_rate(self, value: float):
        self._command(4, value)

    def seek(self, frame: int):
        ffi.command(
            self._lib, "speech_seek", self._get(), ffi._uint(frame, 64, "frame")
        )

    def replay_from(
        self, frame=0, *, options=SpeechOptions(), mode="replace"
    ) -> Speech:
        settings = ffi.speech_options(self._lib, options, mode)
        handle = ffi.out(
            self._lib,
            "speech_replay",
            self._get(),
            ffi._uint(frame, 64, "frame"),
            C.byref(settings),
        )[1]
        return Speech(self._owner, handle)

    def _navigate(self, target):
        new_speech, output, error = ffi.H(), ffi.H(), ffi.H()
        ffi.check(
            self._lib,
            self._lib.mlsr_speech_navigate(
                self._get(),
                target,
                C.byref(new_speech),
                C.byref(output),
                C.byref(error),
            ),
            error,
        )
        try:
            data = ffi.read_json(self._lib, output.value)["data"]
            speech = Speech(self._owner, new_speech.value) if new_speech.value else None
            return Navigation(data["kind"], data.get("frame"), speech)
        except BaseException:
            self._lib.mlsr_sdk_release(new_speech)
            raise

    def previous_sentence(self) -> Navigation:
        return self._navigate(-1)

    def next_sentence(self) -> Navigation:
        return self._navigate(-2)

    def navigate_to_sentence(self, sentence_id: int) -> Navigation:
        return self._navigate(ffi._uint(sentence_id, 32, "sentence_id"))


_END = object()


class _PullStream(_Resource):
    def __init__(self, owner, handle):
        self._owner = owner
        self._init(owner._lib, handle)
        self._ended = False

    def poll(self, timeout_ms=100):
        """Pull one value, None on timeout; StopIteration after terminal/close."""
        if self._ended:
            raise StopIteration
        value = self._poll(ffi._uint(timeout_ms, 32, "timeout_ms"))
        if value is _END:
            self._ended = True
            raise StopIteration
        return value

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            value = self.poll()
            if value is not None:
                return value

    def __aiter__(self):
        return self

    def close(self):
        self._ended = True
        super().close()

    async def __anext__(self):
        def pull():
            try:
                return self.poll(50)
            except StopIteration:
                return _END

        while True:
            # On task cancellation wait for this bounded poll before closing, so
            # no returned owned chunk/metadata is orphaned by asyncio.to_thread.
            try:
                value = await _offload(pull)
            except asyncio.CancelledError:
                await self.aclose()
                raise
            if value is _END:
                raise StopAsyncIteration
            if value is not None:
                return value


class StateStream(_PullStream):
    """Pull-only state observation; no queue grows while the consumer is paused."""

    def _poll(self, timeout):
        status, data = ffi.metadata(self._lib, "state_poll", self._get(), timeout)
        if status == 2:
            return _END
        if data is None:
            return None
        return self._owner._state_type._parse(data["data"])


class AudioStream(_PullStream):
    """Single PCM consumer. Closing an unfinished stream cancels consumer_dropped."""

    completion: Completion | None = None

    def _poll(self, timeout):
        audio, completion, error = ffi.H(), ffi.H(), ffi.H()
        status = ffi.check(
            self._lib,
            self._lib.mlsr_audio_stream_poll(
                self._get(),
                timeout,
                C.byref(audio),
                C.byref(completion),
                C.byref(error),
            ),
            error,
        )
        if status == 2:
            return _END
        if status == 1:
            return None
        if completion.value:
            self.completion = Completion._parse(
                ffi.read_json(self._lib, completion.value)["data"]
            )
            self._ended = True
            self.completion.result()
            return _END
        clip, data = ffi.read_audio(self._lib, audio.value)
        return AudioChunk(
            data["request_id"], data["start_frame"], clip, TextRange._parse(data)
        )
