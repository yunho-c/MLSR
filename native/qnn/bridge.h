// Private C boundary. No ExecuTorch/QNN type escapes this library.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 1 = Float32, 2 = Int32, 3 = Int64. Contiguous row-major only.
typedef struct MlsrQnnTensor {
  void* data;
  uint64_t bytes;
  int32_t dims[8];
  uint32_t rank;
  uint32_t dtype;
} MlsrQnnTensor;

__declspec(dllexport) uint32_t mlsr_qnn_bridge_version(void);
// Diagnostic sequence: odd during forward, even after completion. Lock-free
// observation allows tests to prove cancellation happened within a native call.
__declspec(dllexport) uint64_t mlsr_qnn_activity(void);
__declspec(dllexport) int mlsr_qnn_open(const char* utf8_path, void** model,
                                      uint32_t* input_count, char* error, size_t capacity);
__declspec(dllexport) int mlsr_qnn_info(void* model, uint32_t input,
    uint32_t index, MlsrQnnTensor* info, char* error, size_t capacity);
// All views and their storage remain alive through completion. The caller
// serializes every operation on a model. No in-flight abort is supported.
__declspec(dllexport) int mlsr_qnn_execute(void* model, const MlsrQnnTensor* inputs,
    uint32_t count, MlsrQnnTensor* output, char* error, size_t capacity);
__declspec(dllexport) void mlsr_qnn_close(void* model);

#ifdef __cplusplus
}
#endif
