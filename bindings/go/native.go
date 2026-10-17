package mlsr

/*
#cgo LDFLAGS: -lmlsr_ffi
#include "mlsr.h"
*/
import "C"

func nativeABIVersion() uint32 {
	return uint32(C.mlsr_abi_version())
}

func nativeVersion() string {
	return C.GoString(C.mlsr_version())
}
