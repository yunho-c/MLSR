// Package mlsr provides Go bindings to the MLSR Rust text-to-speech library.
// Synthesis uses a persistent Rust engine and bounded pull-based jobs.
package mlsr

import "fmt"

// SupportedABIVersion is the C ABI version understood by this binding.
const SupportedABIVersion uint32 = 1

// ABIVersion reports the ABI version of the linked native library.
func ABIVersion() uint32 {
	return nativeABIVersion()
}

// Version reports the Rust implementation's version after checking its ABI.
func Version() (string, error) {
	if actual := ABIVersion(); actual != SupportedABIVersion {
		return "", fmt.Errorf("mlsr: unsupported ABI %d; require %d", actual, SupportedABIVersion)
	}
	return nativeVersion(), nil
}
