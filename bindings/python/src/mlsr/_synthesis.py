"""Private ctypes job ABI. No native pointers escape this module."""

import array
import ctypes as C
import functools
from ._ffi import _library, LibraryLoadError

H = C.c_uint64


class String(C.Structure):
    _fields_ = [("data", C.c_void_p), ("len", C.c_size_t)]


class EngineOptions(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint32),
        ("compute_policy", C.c_uint32),
        ("reuse_output_buffers", C.c_uint32),
        ("execution_variant", C.c_uint32),
    ]


class JobOptions(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint32),
        ("steps", C.c_uint32),
        ("speed", C.c_float),
        ("reserved", C.c_uint32),
        ("seed", H),
        ("queue_frames", C.c_uint32),
        ("chunk_frames", C.c_uint32),
        ("max_segment_frames", C.c_uint32),
        ("reserved2", C.c_uint32),
    ]


class ErrorInfo(C.Structure):
    _fields_ = [("code", String), ("message", String)]


class EventInfo(C.Structure):
    _fields_ = [
        ("kind", C.c_uint32),
        ("sample_rate", C.c_uint32),
        ("channels", C.c_uint32),
        ("segment_id", C.c_uint32),
        ("job_id", H),
        ("text_start", H),
        ("text_end", H),
        ("start_frame", H),
        ("frames", H),
        ("predicted_frames", H),
        ("completed_segments", C.c_uint32),
        ("reserved", C.c_uint32),
        ("samples", C.c_void_p),
        ("sample_count", C.c_size_t),
        ("error", ErrorInfo),
    ]


class SynthesisError(RuntimeError):
    """Native failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@functools.lru_cache(maxsize=1)
def library():
    lib = _library()
    signatures = {
        "mlsr_features": ([], C.c_uint32),
        "mlsr_live_handles": ([], H),
        "mlsr_engine_options_default": ([C.POINTER(EngineOptions)], C.c_int32),
        "mlsr_job_options_default": ([C.POINTER(JobOptions)], C.c_int32),
        "mlsr_engine_open": (
            [String, String, C.POINTER(EngineOptions), C.POINTER(H), C.POINTER(H)],
            C.c_int32,
        ),
        "mlsr_engine_shutdown": ([H, C.POINTER(H)], C.c_int32),
        "mlsr_engine_release": ([H], C.c_int32),
        "mlsr_job_start": (
            [
                H,
                String,
                String,
                String,
                C.POINTER(JobOptions),
                C.POINTER(H),
                C.POINTER(H),
            ],
            C.c_int32,
        ),
        "mlsr_job_id": ([H], H),
        "mlsr_job_poll": ([H, C.c_uint32, C.POINTER(H), C.POINTER(H)], C.c_int32),
        "mlsr_job_cancel": ([H, C.POINTER(H)], C.c_int32),
        "mlsr_job_wait": ([H, C.c_uint32, C.POINTER(H)], C.c_int32),
        "mlsr_job_release": ([H], C.c_int32),
        "mlsr_event_read": (
            [H, C.POINTER(EventInfo), C.c_size_t, C.POINTER(H)],
            C.c_int32,
        ),
        "mlsr_event_release": ([H], C.c_int32),
        "mlsr_error_read": ([H, C.POINTER(ErrorInfo), C.c_size_t], C.c_int32),
        "mlsr_error_release": ([H], C.c_int32),
    }
    try:
        for name, (arguments, result) in signatures.items():
            function = getattr(lib, name)
            function.argtypes, function.restype = arguments, result
    except AttributeError as error:
        raise LibraryLoadError(
            "Native MLSR library lacks the synthesis job API"
        ) from error
    if not lib.mlsr_features() & 1:
        raise LibraryLoadError("Native MLSR library lacks synthesis support")
    return lib


def string(value: str):
    data = value.encode("utf-8")
    owner = C.create_string_buffer(data)
    return String(C.cast(owner, C.c_void_p), len(data)), owner


def decode(value: String) -> str:
    return C.string_at(value.data, value.len).decode("utf-8") if value.len else ""


def check(lib, status, error):
    if status >= 0:
        return status
    try:
        info = ErrorInfo()
        if (
            error.value
            and lib.mlsr_error_read(error, C.byref(info), C.sizeof(info)) == 0
        ):
            raise SynthesisError(decode(info.code), decode(info.message))
        raise SynthesisError("native_error", "Native call failed without a diagnostic")
    finally:
        lib.mlsr_error_release(error)


def event(lib, handle):
    try:
        info, error = EventInfo(), H()
        check(
            lib,
            lib.mlsr_event_read(handle, C.byref(info), C.sizeof(info), C.byref(error)),
            error,
        )
        if info.kind not in range(1, 6) or info.sample_count > 65536:
            raise SynthesisError("invalid_event", "Unexpected native event layout")
        samples = array.array("f")
        if info.sample_count:
            samples.frombytes(C.string_at(info.samples, info.sample_count * 4))
        return dict(
            kind=("predicted", "audio", "completed", "cancelled", "failed")[
                info.kind - 1
            ],
            job_id=info.job_id,
            segment_id=info.segment_id,
            text_span=(info.text_start, info.text_end),
            start_frame=info.start_frame,
            frames=info.frames,
            predicted_frames=info.predicted_frames,
            sample_rate=info.sample_rate,
            channels=info.channels,
            completed_segments=info.completed_segments,
            samples=samples,
            error=(
                SynthesisError(decode(info.error.code), decode(info.error.message))
                if info.kind == 5
                else None
            ),
        )
    finally:
        lib.mlsr_event_release(handle)
