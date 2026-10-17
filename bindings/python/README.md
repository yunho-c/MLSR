# MLSR for Python

An initial `ctypes` binding to MLSR's Rust implementation. The current API only
reports native library and ABI versions; speech synthesis is not implemented.

Requires Python 3.10+ and a separately built MLSR shared library. Importing `mlsr`
does not load native code. The first API call loads `MLSR_LIBRARY_PATH` if set,
otherwise it looks for `mlsr_ffi` on the system library search path. Set the
environment variable before the first call; the loaded library is cached.

```python
import mlsr

print(mlsr.native_version())
```

From a repository checkout, run `python3 tools/dev.py test-python` at the root.
`python3 tools/dev.py package-python` builds a wheel and tests an installed copy
outside the source package. The wheel currently contains Python code only and
requires the external native library; bundled native wheels are future work.

Licensed under Apache-2.0; see `LICENSE`.
