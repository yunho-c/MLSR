C ABI smoke coverage lives in `tests/integration/ffi_smoke.c` and in each language
package. Add handle lifetime, cancellation, and error-path tests here when those
APIs are implemented. Rust tests alone do not verify C calling conventions or
dynamic library packaging.
