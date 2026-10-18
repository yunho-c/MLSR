"""Exercise the development Core ML shared library from foreign caller threads."""

from concurrent.futures import ThreadPoolExecutor
import ctypes
from pathlib import Path
import sys

library = ctypes.CDLL(str(Path(sys.argv[1]).resolve()))
probe = library.mlsr_coreml_probe
probe.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t]
probe.restype = ctypes.c_int


def call(path):
    diagnostic = ctypes.create_string_buffer(512)
    code = probe(path, diagnostic, len(diagnostic))
    return code, diagnostic.value.decode("utf-8")


model = str(Path(sys.argv[2]).resolve()).encode("utf-8")
with ThreadPoolExecutor(max_workers=2) as executor:
    for result in executor.map(call, [model] * 4):
        assert result == (0, ""), result
for invalid in [None, b"/mlsr-deliberately-missing.mlmodelc", b"\xff"]:
    code, message = call(invalid)
    assert code != 0 and message, (code, message)
print("Core ML ctypes caller passed")
