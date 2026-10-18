"""Private developer C ABI. Native PCM/JSON views never escape this module."""

import array
import ctypes as C
import functools
import json
import os
from . import _synthesis as old
from .engine import _uint
from .sdk_types import AudioClip, AudioFormat, Failure, TtsError, WavFile

H, String = old.H, old.String
P = C.POINTER


class ModelSpec(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint32),
        ("catalog", String),
        ("id", String),
        ("platform", String),
    ]


class TtsOptions(C.Structure):
    _fields_ = [
        (name, C.c_uint32)
        for name in (
            "struct_size",
            "output_mode",
            "max_requests",
            "max_text_bytes",
            "queue_frames",
            "chunk_frames",
            "max_segment_frames",
            "max_retained_stores",
        )
    ] + [("max_retained_bytes", H), ("reserved", H)]


class Request(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint32),
        ("locale_fallback", C.c_uint32),
        ("speaking_rate", C.c_float),
        ("supertonic_steps", C.c_uint32),
        ("seed", H),
        ("text", String),
        ("locale", String),
        ("voice", String),
    ]


class SpeechOptions(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint32),
        ("mode", C.c_uint32),
        ("playback_rate", C.c_float),
        ("volume", C.c_float),
        ("paused", C.c_uint32),
        ("retain_audio", C.c_uint32),
        ("lookahead_frames", C.c_uint32),
        ("reserved", C.c_uint32),
    ]


class PcmView(C.Structure):
    _fields_ = [
        ("sample_rate", C.c_uint32),
        ("channels", C.c_uint32),
        ("frames", H),
        ("samples", C.c_void_p),
        ("sample_count", C.c_size_t),
    ]


@functools.lru_cache(maxsize=1)
def library():
    lib = old.library()
    if lib.mlsr_features() & 8 == 0:
        raise old.LibraryLoadError("Native MLSR library lacks developer SDK schema v1")
    signatures = {
        "model_open": [String, P(ModelSpec), P(H), P(H)],
        "model_release": [H],
        "tts_options_default": [P(TtsOptions)],
        "request_default": [P(Request)],
        "speech_options_default": [P(SpeechOptions)],
        "tts_open": [String, String, P(old.EngineOptions), P(TtsOptions), P(H), P(H)],
        "tts_from_model": [H, P(old.EngineOptions), P(TtsOptions), P(H), P(H)],
        "voice_catalog": [H, P(H), P(H)],
        "voice_resolve": [H, String, String, C.c_uint32, P(H), P(H)],
        "json_read": [H, P(String), P(H)],
        "error_details": [H, P(H), P(H)],
        "sdk_release": [H],
        "sdk_forget": [H],
        "tts_synthesize": [H, P(Request), P(H), P(H)],
        "synthesis_start": [H, P(H)],
        "synthesis_cancel": [H, C.c_uint32, P(H)],
        "synthesis_audio": [H, P(H), P(H)],
        "audio_stream_poll": [H, C.c_uint32, P(H), P(H), P(H)],
        "sdk_snapshot": [H, P(H), P(H)],
        "sdk_request": [H, P(H), P(H)],
        "sdk_wait": [H, C.c_uint32, P(H), P(H)],
        "sdk_states": [H, P(H), P(H)],
        "state_poll": [H, C.c_uint32, P(H), P(H)],
        "text_plan": [H, P(H), P(H)],
        "tts_controller": [H, C.c_uint32, P(H), P(H)],
        "speak": [H, P(Request), P(SpeechOptions), P(H), P(H)],
        "speech_command": [H, C.c_uint32, C.c_float, P(H)],
        "speech_seek": [H, H, P(H)],
        "speech_replay": [H, H, P(SpeechOptions), P(H), P(H)],
        "speech_navigate": [H, C.c_int64, P(H), P(H), P(H)],
        "tts_collect": [H, H, H, P(H), P(H)],
        "audio_read": [H, P(PcmView), C.c_size_t, P(H)],
        "tts_write_job": [H, H, String, H, P(H), P(H)],
        "tts_write_speech": [H, H, String, H, P(H), P(H)],
        "pcm_write_wav": [P(PcmView), String, P(H), P(H)],
    }
    try:
        for name, args in signatures.items():
            function = getattr(lib, "mlsr_" + name)
            function.argtypes, function.restype = args, C.c_int32
        lib.mlsr_sdk_id.argtypes, lib.mlsr_sdk_id.restype = [H], H
    except AttributeError as error:
        raise old.LibraryLoadError("Incomplete MLSR developer SDK") from error
    return lib


def check(lib, status, error):
    if status >= 0:
        return status
    details = None
    info = old.ErrorInfo()
    code, message = "native_error", "Native call failed without a diagnostic"
    try:
        if (
            error.value
            and lib.mlsr_error_read(error, C.byref(info), C.sizeof(info)) == 0
        ):
            code, message = old.decode(info.code), old.decode(info.message)
            output, nested = H(), H()
            try:
                if lib.mlsr_error_details(error, C.byref(output), C.byref(nested)) == 0:
                    # Read using the old error decoder to avoid recursive SDK errors.
                    value = read_json(lib, output.value, error_context=True)
                    details = value["data"].get("failure")
            finally:
                lib.mlsr_error_release(nested)
        raise TtsError(code, message, Failure(**details) if details else None)
    finally:
        lib.mlsr_error_release(error)


def read_json(lib, handle, *, error_context=False):
    try:
        view, error = String(), H()
        status = lib.mlsr_json_read(handle, C.byref(view), C.byref(error))
        (old.check if error_context else check)(lib, status, error)
        data = json.loads(old.decode(view))
        if data.get("schema_version") != 1 or not isinstance(data.get("type"), str):
            raise TtsError("unsupported_schema", "Unexpected native metadata envelope")
        return data
    finally:
        lib.mlsr_sdk_release(handle)


def out(lib, name, *args):
    output, error = H(), H()
    status = check(
        lib, getattr(lib, "mlsr_" + name)(*args, C.byref(output), C.byref(error)), error
    )
    return status, output.value


def metadata(lib, name, *args):
    status, handle = out(lib, name, *args)
    return status, read_json(lib, handle) if handle else None


def command(lib, name, *args):
    error = H()
    check(lib, getattr(lib, "mlsr_" + name)(*args, C.byref(error)), error)


def enum(value, choices, name):
    try:
        return choices.index(value)
    except ValueError:
        raise ValueError(f"unsupported {name}: {value!r}") from None


def request(lib, value):
    settings = Request()
    lib.mlsr_request_default(C.byref(settings))
    settings.locale_fallback = enum(
        value.locale_fallback, ("strict", "same_language"), "locale fallback"
    )
    settings.speaking_rate = value.speaking_rate
    settings.supertonic_steps = _uint(value.backend.steps, 32, "steps")
    settings.seed = _uint(value.backend.seed, 64, "seed")
    owners = [old.string(s) for s in (value.text, value.locale, value.voice)]
    settings.text, settings.locale, settings.voice = (s[0] for s in owners)
    return settings, owners


def speech_options(lib, value, mode):
    settings = SpeechOptions()
    lib.mlsr_speech_options_default(C.byref(settings))
    settings.mode = enum(mode, ("replace", "enqueue"), "speech mode")
    settings.playback_rate, settings.volume = value.playback_rate, value.volume
    settings.paused, settings.retain_audio = bool(value.paused), bool(
        value.retain_audio
    )
    settings.lookahead_frames = _uint(value.lookahead_frames, 32, "lookahead_frames")
    return settings


def open_tts(
    lib,
    manifest,
    bundle,
    options,
    compute_policy,
    reuse_output_buffers,
    execution_variant,
    *,
    model_handle=None,
):
    settings = TtsOptions()
    lib.mlsr_tts_options_default(C.byref(settings))
    settings.output_mode = enum(
        options.output_mode, ("default", "headless", "host_managed"), "output mode"
    )
    for name in (
        "max_requests",
        "max_text_bytes",
        "queue_frames",
        "chunk_frames",
        "max_segment_frames",
        "max_retained_stores",
        "max_retained_bytes",
    ):
        setattr(
            settings,
            name,
            _uint(
                getattr(options, name), 64 if name == "max_retained_bytes" else 32, name
            ),
        )
    engine = old.EngineOptions()
    lib.mlsr_engine_options_default(C.byref(engine))
    engine.compute_policy = enum(
        compute_policy, ("cpu_only", "cpu_and_neural_engine"), "compute policy"
    )
    engine.reuse_output_buffers = bool(reuse_output_buffers)
    engine.execution_variant = _uint(execution_variant, 32, "execution_variant")
    if model_handle is not None:
        return out(
            lib, "tts_from_model", model_handle, C.byref(engine), C.byref(settings)
        )[1]
    manifest_arg, manifest_owner = old.string(os.fspath(manifest))
    bundle_arg, bundle_owner = old.string(os.fspath(bundle))
    return out(
        lib, "tts_open", manifest_arg, bundle_arg, C.byref(engine), C.byref(settings)
    )[1]


def read_audio(lib, handle):
    """Copy local-only PCM before releasing its handle, including on errors."""
    try:
        info, error = PcmView(), H()
        check(
            lib,
            lib.mlsr_audio_read(handle, C.byref(info), C.sizeof(info), C.byref(error)),
            error,
        )
        if (
            info.sample_count > 1024 * 1024 * 1024 // 4
            or info.frames * info.channels != info.sample_count
        ):
            raise TtsError("invalid_audio", "Unexpected native PCM layout")
        samples = array.array("f")
        samples.frombytes(C.string_at(info.samples, info.sample_count * 4))
        data = metadata(lib, "sdk_snapshot", handle)[1]["data"]
        return AudioClip(samples, AudioFormat(info.sample_rate, info.channels)), data
    finally:
        lib.mlsr_sdk_release(handle)


def write_pcm(clip, path):
    lib = library()
    path_arg, path_owner = old.string(os.fspath(path))
    # from_buffer pins the array against resize until the synchronous call ends.
    pinned = (C.c_float * len(clip.samples)).from_buffer(clip.samples)
    info = PcmView(
        clip.format.sample_rate,
        clip.format.channels,
        clip.frames,
        C.cast(pinned, C.c_void_p),
        len(clip.samples),
    )
    _, envelope = metadata(lib, "pcm_write_wav", C.byref(info), path_arg)
    return WavFile._parse(envelope["data"])
