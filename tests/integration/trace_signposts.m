// Development-only Instruments markers. This is not part of MLSR's runtime or ABI.
// Build with the active Apple SDK so the public os_signpost macros choose their
// supported implementation. Called serially by coreml_trace's main thread.
#include <os/signpost.h>
#include <stdint.h>

void mlsr_trace_mark(uint64_t request, uint8_t begin) {
  static os_log_t log;
  if (!log) log = os_log_create("dev.mlsr.qualification", OS_LOG_CATEGORY_POINTS_OF_INTEREST);
  if (begin) os_signpost_interval_begin(log, (os_signpost_id_t)request, "MLSRJob");
  else os_signpost_interval_end(log, (os_signpost_id_t)request, "MLSRJob");
}
