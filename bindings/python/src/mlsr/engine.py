"""Context-managed synthesis. Blocking iteration releases the GIL in ctypes.

Use a worker thread from asynchronous applications. Iteration pulls one event
at a time; there is no hidden Python audio queue. Samples are Python-owned.
"""

import array
import ctypes as C
from dataclasses import dataclass, field
import os
from . import _synthesis as ffi
from ._synthesis import SynthesisError


@dataclass(frozen=True)
class SynthesisEvent:
    """Prediction/audio/terminal event. text_span indexes original UTF-8 bytes."""

    kind: str
    job_id: int
    segment_id: int
    text_span: tuple[int, int]
    start_frame: int
    frames: int
    predicted_frames: int
    sample_rate: int
    channels: int
    completed_segments: int
    samples: array.array = field(repr=False)
    error: SynthesisError | None = None

    @property
    def terminal(self) -> bool:
        return self.kind in ("completed", "cancelled", "failed")


def _uint(value, bits, name):
    if not isinstance(value, int) or not 0 <= value < 1 << bits:
        raise ValueError(f"{name} must fit uint{bits}")
    return value


class Engine:
    """Persistent engine. Explicit close/context exit cancels and joins work."""

    def __init__(
        self,
        manifest,
        bundle,
        *,
        compute_policy="cpu_and_neural_engine",
        reuse_output_buffers=True,
        execution_variant=0,
    ):
        self._handle = 0
        self._lib = ffi.library()
        settings = ffi.EngineOptions()
        self._lib.mlsr_engine_options_default(C.byref(settings))
        policies = {"cpu_only": 0, "cpu_and_neural_engine": 1}
        if compute_policy not in policies:
            raise ValueError("unsupported compute_policy")
        settings.compute_policy = policies[compute_policy]
        settings.reuse_output_buffers = int(bool(reuse_output_buffers))
        settings.execution_variant = _uint(execution_variant, 32, "execution_variant")
        manifest_arg, manifest_owner = ffi.string(os.fspath(manifest))
        bundle_arg, bundle_owner = ffi.string(os.fspath(bundle))
        output, error = ffi.H(), ffi.H()
        ffi.check(
            self._lib,
            self._lib.mlsr_engine_open(
                manifest_arg,
                bundle_arg,
                C.byref(settings),
                C.byref(output),
                C.byref(error),
            ),
            error,
        )
        self._handle = output.value

    def start(
        self,
        text: str,
        *,
        language="en",
        voice="M1",
        steps=8,
        speed=1.05,
        seed=0,
        queue_frames=44100,
        chunk_frames=4096,
        max_segment_frames=1572864,
    ):
        """Start one job. Its input is copied; a competing job raises engine_busy."""
        if not self._handle:
            raise SynthesisError("engine_closed", "Engine is closed")
        settings = ffi.JobOptions()
        self._lib.mlsr_job_options_default(C.byref(settings))
        for name, value, bits in (
            ("steps", steps, 32),
            ("seed", seed, 64),
            ("queue_frames", queue_frames, 32),
            ("chunk_frames", chunk_frames, 32),
            ("max_segment_frames", max_segment_frames, 32),
        ):
            setattr(settings, name, _uint(value, bits, name))
        settings.speed = speed
        args = [ffi.string(value) for value in (text, language, voice)]
        output, error = ffi.H(), ffi.H()
        ffi.check(
            self._lib,
            self._lib.mlsr_job_start(
                self._handle,
                *(item[0] for item in args),
                C.byref(settings),
                C.byref(output),
                C.byref(error),
            ),
            error,
        )
        return Job(self._lib, output.value)

    def close(self):
        """Idempotent. Jobs remain valid to observe their terminal outcome."""
        handle, self._handle = self._handle, 0
        if handle and self._lib.mlsr_engine_release(handle) != 0:
            raise SynthesisError("release_failed", "Engine release failed")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Job:
    """Single-consumer iterator. Cancel is nonblocking; close cancels and waits."""

    def __init__(self, library, handle):
        self._lib, self._handle, self._ended = library, handle, False
        self.id = library.mlsr_job_id(handle)

    def poll(self, timeout_ms=100) -> SynthesisEvent | None:
        """Return an event, None on timeout, or raise StopIteration after terminal."""
        if self._ended or not self._handle:
            raise StopIteration
        output, error = ffi.H(), ffi.H()
        status = ffi.check(
            self._lib,
            self._lib.mlsr_job_poll(
                self._handle,
                _uint(timeout_ms, 32, "timeout_ms"),
                C.byref(output),
                C.byref(error),
            ),
            error,
        )
        if status == 1:
            return None
        if status == 2:
            self._ended = True
            raise StopIteration
        event = SynthesisEvent(**ffi.event(self._lib, output))
        self._ended = event.terminal
        return event

    def cancel(self):
        if self._handle:
            error = ffi.H()
            ffi.check(
                self._lib,
                self._lib.mlsr_job_cancel(self._handle, C.byref(error)),
                error,
            )

    def wait(self, timeout_ms=100) -> bool:
        """Wait for quiescence, without draining audio. A stalled job needs cancel."""
        if not self._handle:
            return True
        error = ffi.H()
        return (
            ffi.check(
                self._lib,
                self._lib.mlsr_job_wait(
                    self._handle, _uint(timeout_ms, 32, "timeout_ms"), C.byref(error)
                ),
                error,
            )
            == 0
        )

    def close(self):
        handle, self._handle = self._handle, 0
        self._ended = True
        if handle and self._lib.mlsr_job_release(handle) != 0:
            raise SynthesisError("release_failed", "Job release failed")

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            event = self.poll()
            if event is not None:
                return event

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
