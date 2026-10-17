# MLSR for Go

A `cgo` binding to MLSR's shared C ABI. Requires Go 1.22+, a C compiler, the
generated `mlsr.h`, and the Rust shared library. The scaffold exposes version
queries; synthesis is not implemented yet.

From the repository root, run `python3 tools/dev.py test-go`. The development
tool supplies the include, link, and runtime library paths. The Go module itself
does not depend on paths outside its package, so external consumers can install
the native header/library and configure `CGO_CFLAGS` and `CGO_LDFLAGS` themselves.

For manual builds, set `CGO_ENABLED=1`, add the header directory with `-I` to
`CGO_CFLAGS`, add the library directory with `-L` to `CGO_LDFLAGS`, and make the
shared library available to the OS loader. Linking `mlsr_ffi` is handled by the
binding. Cross-compilation also requires a target C toolchain and target Rust
library. Builds with `CGO_ENABLED=0` are not supported.

The module path is `github.com/yunho-c/MLSR/bindings/go`. Future release tags for
this nested module must use the `bindings/go/` prefix, such as
`bindings/go/v0.1.0`.

Licensed under Apache-2.0; see `LICENSE`.
