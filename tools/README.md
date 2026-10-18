# Development tools

Run `python3 tools/dev.py --help` for available commands. The runner uses only the
Python standard library and invokes the existing Cargo, Dart, Go, and C tools.
It does not create a Python virtual environment or change global toolchains.

`codegen/` pins cbindgen and generates the C header explicitly. The Dart generator
is in `bindings/dart/tool/generate_bindings.dart`. Both support checking whether
the committed output is current without overwriting it.

`test-standalone` copies the Dart and Go packages, header, and shared library to
a temporary directory and tests them there. This checks the prebuilt-library
path and catches dependencies on the checkout layout, including macOS library
relocation failures.

`package-python` additionally requires pip, setuptools 77+, and wheel in the
interpreter used to invoke the tool. It installs the built wrapper into a
temporary directory, runs its native integration tests there, and removes that
temporary installation when finished. Native libraries are still supplied
separately; nothing is published.

`supertonic3/` contains development-only YAML manifest validation, verified asset
preparation, and real Core ML reference capture/replay. See its
[README](supertonic3/README.md). `test-model-tools` runs the portable checks
without building Rust, loading weights, or requiring Core ML; it additionally
needs the packages in `supertonic3/requirements.txt`.

`check_coreml.py` tests the optional Apple runtime without downloading weights.
It compiles the tiny affine model with Xcode, runs Rust tests, builds static/shared
native artifacts, and executes real inference from C and Python ctypes callers.
`--cross` additionally links Intel macOS and iOS device/simulator artifacts.
It requires installed targets/toolchains and never installs them automatically.
The same command tests/builds the Core ML CLI. Full Supertonic synthesis replay
uses opt-in external assets via `python3 -m tools.supertonic3.synthesis_replay`;
see the [synthesis documentation](../docs/supertonic3.md).
