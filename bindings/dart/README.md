# MLSR for Dart

`dart:ffi` bindings to MLSR's Rust library, generated with `ffigen`. Requires
Dart 3.11+. The scaffold exposes version queries; synthesis is not implemented.

```sh
dart pub get --enforce-lockfile
dart run example/version.dart
dart test
```

In the repository checkout, the build hook compiles the Rust library in release
mode and bundles it as a native code asset. Rust and a native linker must be
installed. The hook currently supports host builds on macOS, glibc Linux, and
Windows with the MSVC Rust toolchain, for x64/arm64. See the root platform document
for the distinction between configured targets and validated targets.

Outside the checkout, supply a matching prebuilt library through the consuming
application's `pubspec.yaml`:

```yaml
hooks:
  user_defines:
    mlsr:
      native-library: native/libmlsr_ffi.dylib
```

Use the corresponding `.so` or `.dll` on Linux or Windows. The path is relative
to the containing pubspec. It must match the target architecture and ABI; no
automatic download is configured. Mobile, browser, and cross-compilation support
remain future work. The package is marked `publish_to: none` during initialization.

Regenerate bindings from the repository root with
`python3 tools/dev.py generate`. `tool/generate_bindings.dart` uses ffigen's
programmatic API; its older YAML configuration format is deprecated upstream.
Normal builds use the checked-in generated Dart file and do not need libclang.

Licensed under Apache-2.0; see `LICENSE`.
