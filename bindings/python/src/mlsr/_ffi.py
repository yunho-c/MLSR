"""The only Python module that talks to the MLSR C ABI."""

import ctypes
import ctypes.util
import functools
import os

_SUPPORTED_ABI = 1


class LibraryLoadError(RuntimeError):
    """The native library is missing, invalid, or ABI-incompatible."""


@functools.lru_cache(maxsize=1)
def _library() -> ctypes.CDLL:
    path = os.environ.get("MLSR_LIBRARY_PATH")
    if not path:
        path = ctypes.util.find_library("mlsr_ffi")
    if not path:
        raise LibraryLoadError(
            "MLSR's native library was not found. Set MLSR_LIBRARY_PATH to the "
            "absolute path of libmlsr_ffi.so, libmlsr_ffi.dylib, or mlsr_ffi.dll."
        )
    try:
        library = ctypes.CDLL(path)
        library.mlsr_abi_version.argtypes = []
        library.mlsr_abi_version.restype = ctypes.c_uint32
        library.mlsr_version.argtypes = []
        library.mlsr_version.restype = ctypes.c_char_p
    except (OSError, AttributeError) as error:
        raise LibraryLoadError(
            f"Cannot load the MLSR C ABI from {path!r}: {error}"
        ) from error
    actual = library.mlsr_abi_version()
    if actual != _SUPPORTED_ABI:
        raise LibraryLoadError(
            f"Unsupported MLSR ABI {actual}; this binding requires {_SUPPORTED_ABI}."
        )
    return library


def abi_version() -> int:
    """Return the native ABI version after checking binding compatibility."""
    return _library().mlsr_abi_version()


def native_version() -> str:
    """Return the loaded Rust implementation's version."""
    value = _library().mlsr_version()
    if value is None:
        raise LibraryLoadError("mlsr_version returned a null pointer.")
    return value.decode("utf-8")
