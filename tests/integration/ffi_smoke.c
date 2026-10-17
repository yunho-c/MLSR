#include "mlsr.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (mlsr_abi_version() != 1) {
        fputs("Unsupported MLSR ABI\n", stderr);
        return 1;
    }
    const char *version = mlsr_version();
    if (version == NULL || argc != 2 || strcmp(version, argv[1]) != 0) {
        fputs("Native library version does not match the workspace\n", stderr);
        return 1;
    }
    printf("MLSR %s (C ABI %u)\n", version, (unsigned)mlsr_abi_version());
    return 0;
}
