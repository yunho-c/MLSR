package mlsr_test

import (
	"os"
	"testing"

	"github.com/yunho-c/MLSR/bindings/go"
)

func TestLinkedRustLibrary(t *testing.T) {
	version, err := mlsr.Version()
	if err != nil {
		t.Fatal(err)
	}
	expected := os.Getenv("MLSR_EXPECTED_VERSION")
	if expected == "" {
		expected = "0.1.0"
	}
	if version != expected {
		t.Fatalf("native version = %q; want %q", version, expected)
	}
	if abi := mlsr.ABIVersion(); abi != mlsr.SupportedABIVersion {
		t.Fatalf("ABI = %d; want %d", abi, mlsr.SupportedABIVersion)
	}
}
