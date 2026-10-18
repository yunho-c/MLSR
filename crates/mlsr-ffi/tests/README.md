C ABI version coverage lives in `tests/integration/ffi_smoke.c`; real synthesis
coverage is in `tests/integration/synthesis_smoke.c`. The latter checks ordered
audio, busy rejection, cancellation from a pthread, retained event lifetime and
handle cleanup through both static and shared libraries. Rust unit tests cover
invalid handles, layouts/UTF-8 and panic containment. Installed language tests
are orchestrated by `tools/check_synthesis_bindings.py`; Rust tests alone do not
verify C calling conventions or dynamic library packaging.
