package mlsr

/*
#cgo !windows LDFLAGS: -lmlsr_ffi
// MSVC emits both a static .lib and a DLL import .dll.lib. Select the import
// library explicitly so MinGW does not pull the MSVC Rust static runtime in.
#cgo windows LDFLAGS: -l:mlsr_ffi.dll.lib
#include "mlsr.h"
*/
import "C"

func nativeABIVersion() uint32 {
	return uint32(C.mlsr_abi_version())
}

func nativeVersion() string {
	return C.GoString(C.mlsr_version())
}
