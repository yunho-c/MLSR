package main

import (
	"fmt"
	"log"

	"github.com/yunho-c/MLSR/bindings/go"
)

func main() {
	version, err := mlsr.Version()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("MLSR %s (C ABI %d)\n", version, mlsr.ABIVersion())
}
