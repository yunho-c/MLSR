/* Development ABI probe: real inference through shared AND static Rust builds. */
#include <stddef.h>
#include <stdio.h>
#include <string.h>

int mlsr_coreml_probe(const char *path, char *diagnostic, size_t capacity);

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    char diagnostic[256];
    if (mlsr_coreml_probe(argv[1], diagnostic, sizeof diagnostic) != 0) {
        fprintf(stderr, "%s\n", diagnostic);
        return 1;
    }
    if (diagnostic[0] != '\0') return 3;
    if (mlsr_coreml_probe(NULL, diagnostic, sizeof diagnostic) == 0) return 4;
    if (strstr(diagnostic, "null model path") == NULL) return 5;
    char guarded[8];
    memset(guarded, 'x', sizeof guarded);
    if (mlsr_coreml_probe(NULL, guarded + 2, 3) == 0) return 6;
    if (guarded[1] != 'x' || guarded[4] != '\0' || guarded[5] != 'x') return 7;
    if (mlsr_coreml_probe(NULL, NULL, 0) == 0) return 8;
    if (mlsr_coreml_probe(argv[1], NULL, 0) != 0) return 9;
    puts("Core ML C caller passed");
    return 0;
}
