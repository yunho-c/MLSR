#include "bridge.h"
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/platform/runtime.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <malloc.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

using executorch::extension::BufferDataLoader;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::extension::make_tensor_ptr;
using executorch::aten::ScalarType;
using executorch::runtime::EValue;
using executorch::runtime::Error;

// Export from the pinned Windows ExecuTorch QNN backend build. Windows DLLs
// have separate static registries; registration must target this bridge's
// executorch_core instance. See the pinned upstream QnnExecuTorch.h export.
extern "C" __declspec(dllimport) void QnnExecuTorchBackendRegister(void* register_fn);

namespace {
std::mutex runtime_mutex;
std::once_flag runtime_once;
std::atomic<uint64_t> activity{0};
struct Activity {
  Activity() { activity.fetch_add(1); }
  ~Activity() { activity.fetch_add(1); }
};
void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
void checked(Error value, const char* operation) {
  if (value != Error::Ok)
    throw std::runtime_error(std::string(operation) + ": ExecuTorch status " + std::to_string(int(value)));
}
template <typename F> int guarded(char* error, size_t capacity, F&& action) noexcept {
  try {
    action();
    return 0;
  } catch (const std::exception& e) {
    if (error && capacity) {
      const auto size = std::min(capacity - 1, std::strlen(e.what()));
      std::memcpy(error, e.what(), size);
      error[size] = '\0';
    }
  } catch (...) {
    if (error && capacity) {
      const char* message = "Unknown native exception";
      const auto size = std::min(capacity - 1, std::strlen(message));
      std::memcpy(error, message, size);
      error[size] = '\0';
    }
  }
  return -1;
}
uint32_t dtype(ScalarType type) {
  switch (type) {
    case ScalarType::Float: return 1;
    case ScalarType::Int: return 2;
    case ScalarType::Long: return 3;
    default: throw std::runtime_error("Unsupported public tensor dtype");
  }
}
ScalarType scalar(uint32_t type) {
  switch (type) {
    case 1: return ScalarType::Float;
    case 2: return ScalarType::Int;
    case 3: return ScalarType::Long;
    default: throw std::runtime_error("Invalid tensor dtype");
  }
}
struct AlignedFree { void operator()(void* p) const noexcept { _aligned_free(p); } };
struct Model {
  // Reverse destruction order: module/context first, then borrowed view
  // descriptors and finally the aligned program bytes used by its data loader.
  std::unique_ptr<void, AlignedFree> program;
  size_t program_bytes = 0;
  std::vector<TensorPtr> inputs;
  std::unique_ptr<Module> module;

  explicit Model(const char* path) {
    require(path != nullptr, "Null model path");
    std::ifstream file(std::filesystem::u8path(path), std::ios::binary | std::ios::ate);
    require(file.good(), "Cannot open PTE file");
    const auto size = file.tellg();
    require(size > 0 && size <= 1024LL * 1024 * 1024, "Invalid PTE size (maximum 1 GiB)");
    program.reset(_aligned_malloc(static_cast<size_t>(size), 128));
    require(program != nullptr, "PTE allocation failed");
    file.seekg(0);
    file.read(static_cast<char*>(program.get()), size);
    require(file.good(), "Cannot read complete PTE file");
    program_bytes = static_cast<size_t>(size);
    load();
  }

  void load() {
    if (module) return;
    auto replacement = std::make_unique<Module>(
        std::make_unique<BufferDataLoader>(program.get(), program_bytes));
    checked(replacement->load_forward(), "load forward");
    auto meta = replacement->method_meta("forward");
    require(meta.ok() && meta->num_outputs() == 1 && meta->num_inputs() <= 32,
            "Expected one forward tensor output and at most 32 inputs");
    module = std::move(replacement);
  }

  MlsrQnnTensor info(bool input, uint32_t index) {
    load();
    auto meta = module->method_meta("forward");
    require(meta.ok(), "Missing forward metadata");
    auto result = input ? meta->input_tensor_meta(index) : meta->output_tensor_meta(index);
    require(result.ok(), "Missing tensor metadata");
    auto sizes = result->sizes();
    require(sizes.size() > 0 && sizes.size() <= 8, "Unsupported tensor rank");
    MlsrQnnTensor value{};
    value.bytes = result->nbytes();
    value.dtype = dtype(result->scalar_type());
    value.rank = static_cast<uint32_t>(sizes.size());
    auto order = result->dim_order();
    for (size_t i = 0; i < sizes.size(); ++i) {
      require(sizes[i] > 0 && order[i] == i, "Expected static row-major tensor layout");
      value.dims[i] = sizes[i];
    }
    return value;
  }
};
struct ResetOnFailure {
  Model& model;
  bool success = false;
  ~ResetOnFailure() {
    if (!success) {
      // A partial set_inputs / allocation / execution failure may leave either
      // old or new borrowed inputs in Module. Tear it down while BOTH generations
      // of descriptors and Rust allocations are still alive. Reload on next use.
      model.module.reset();
      model.inputs.clear();
    }
  }
};
void validate(const MlsrQnnTensor& supplied, const MlsrQnnTensor& expected) {
  require(supplied.data && supplied.rank == expected.rank && supplied.dtype == expected.dtype &&
          supplied.bytes == expected.bytes, "Tensor dtype, rank or byte count mismatch");
  for (uint32_t i = 0; i < expected.rank; ++i)
    require(supplied.dims[i] == expected.dims[i], "Tensor shape mismatch");
  const auto address = reinterpret_cast<uintptr_t>(supplied.data);
  require(address % (supplied.dtype == 3 ? 8 : 4) == 0, "Unaligned tensor data");
  require(supplied.bytes <= std::numeric_limits<uintptr_t>::max() - address, "Tensor address overflow");
}
} // namespace

uint32_t mlsr_qnn_bridge_version() { return 1; }
uint64_t mlsr_qnn_activity() { return activity.load(); }
int mlsr_qnn_open(const char* path, void** output, uint32_t* count, char* error, size_t capacity) {
  if (output) *output = nullptr;
  return guarded(error, capacity, [&] {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    std::call_once(runtime_once, [] {
      executorch::runtime::runtime_init();
      QnnExecuTorchBackendRegister(reinterpret_cast<void*>(&executorch::runtime::register_backend));
    });
    require(output && count, "Null model result");
    auto model = std::make_unique<Model>(path);
    *count = static_cast<uint32_t>(model->module->method_meta("forward")->num_inputs());
    *output = model.release();
  });
}
int mlsr_qnn_info(void* ptr, uint32_t input, uint32_t index, MlsrQnnTensor* output,
                  char* error, size_t capacity) {
  return guarded(error, capacity, [&] {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    require(ptr && output && input <= 1, "Invalid metadata request");
    *output = static_cast<Model*>(ptr)->info(input == 1, index);
  });
}
int mlsr_qnn_execute(void* ptr, const MlsrQnnTensor* inputs, uint32_t count,
                     MlsrQnnTensor* output, char* error, size_t capacity) {
  return guarded(error, capacity, [&] {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    require(ptr && inputs && output, "Null execution argument");
    auto& model = *static_cast<Model*>(ptr);
    std::vector<TensorPtr> views;
    std::vector<TensorPtr> previous_views;
    std::vector<EValue> values;
    // Declared after descriptors: this guard destroys the failed module FIRST.
    ResetOnFailure reset{model};
    model.load();
    auto meta = model.module->method_meta("forward");
    require(meta.ok() && meta->num_inputs() == count, "Input count mismatch");
    validate(*output, model.info(false, 0));
    for (uint32_t i = 0; i < count; ++i) {
      validate(inputs[i], model.info(true, i));
      const auto a = reinterpret_cast<uintptr_t>(inputs[i].data);
      const auto b = reinterpret_cast<uintptr_t>(output->data);
      require(a + inputs[i].bytes <= b || b + output->bytes <= a, "Input/output storage alias");
      views.push_back(make_tensor_ptr(
          std::vector<int32_t>(inputs[i].dims, inputs[i].dims + inputs[i].rank),
          inputs[i].data, scalar(inputs[i].dtype)));
      values.emplace_back(*views.back());
    }
    // Rust retains the corresponding owned input allocations with this model.
    // Old descriptors remain alive until forward has replaced all inputs.
    previous_views = std::move(model.inputs);
    model.inputs = std::move(views);
    auto result = [&] {
      Activity executing;
      return model.module->forward(values);
    }();
    checked(result.error(), "forward");
    require(result->size() == 1 && result->at(0).isTensor(), "Expected one tensor output");
    const auto& tensor = result->at(0).toTensor();
    require(dtype(tensor.scalar_type()) == output->dtype && tensor.nbytes() == output->bytes &&
            tensor.dim() == output->rank, "Unexpected output storage");
    size_t stride = 1;
    for (uint32_t i = output->rank; i-- > 0;) {
      require(tensor.size(i) == output->dims[i] && tensor.strides()[i] == stride,
              "Output is not contiguous row-major");
      stride *= static_cast<size_t>(output->dims[i]);
    }
    // Explicit adapter copy. The output must survive subsequent forward calls.
    // Reusable output allocations are owned by the Rust worker, not the module.
    std::memcpy(output->data, tensor.const_data_ptr(), static_cast<size_t>(output->bytes));
    reset.success = true;
  });
}
void mlsr_qnn_close(void* ptr) {
  std::lock_guard<std::mutex> lock(runtime_mutex);
  delete static_cast<Model*>(ptr);
}
