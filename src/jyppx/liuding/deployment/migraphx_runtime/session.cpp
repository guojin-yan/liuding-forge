// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "../../foundation/nested_error.hpp"

#include <jyppx/liuding/deployment/migraphx_runtime/session.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime_api.h>
#include <limits>
#include <migraphx/migraphx.h>
#include <migraphx/version.h>
#include <mutex>
#include <new>
#include <string_view>
#include <utility>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace jyppx::liuding::migraphx_runtime {
namespace {
static_assert(MIGRAPHX_VERSION_MAJOR == 2 && MIGRAPHX_VERSION_MINOR == 18 &&
              MIGRAPHX_VERSION_PATCH == 0);
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);

ErrorDetails Details(std::string operation) {
  return {.component = "migraphx_runtime.session",
          .operation = std::move(operation)};
}
template <class Function>
decltype(auto) Boundary(const char* operation, Function&& function) {
  try {
    return std::forward<Function>(function)();
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    ::jyppx::liuding::detail::throw_nested_error(ResourceExhausted(
        "MIGraphX host allocation failed", Details(operation)));
  } catch (const std::exception&) {
    ::jyppx::liuding::detail::throw_nested_error(
        BackendError("MIGraphX adapter failed", Details(operation)));
  }
}
void Check(migraphx_status status, const char* operation) {
  if (status != migraphx_status_success) {
    auto details = Details(operation);
    details.actual = std::to_string(status);
    throw BackendError("MIGraphX C API failed", std::move(details));
  }
}
void Hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    auto details = Details(operation);
    details.actual = std::to_string(status);
    throw DeviceError("HIP device operation failed", std::move(details));
  }
}
class DeviceGuard {
 public:
  explicit DeviceGuard(int selected) {
    int count = 0;
    Hip(hipGetDeviceCount(&count), "device_count");
    if (selected >= count)
      throw DeviceError("Requested HIP device is unavailable",
                        Details("device_id"));
    Hip(hipGetDevice(&previous_), "get_device");
    Hip(hipSetDevice(selected), "set_device");
  }
  ~DeviceGuard() { (void)hipSetDevice(previous_); }
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

 private:
  int previous_ = 0;
};
template <class Handle, auto Destroy>
struct Owner {
  Handle value{};
  Owner() = default;
  Owner(const Owner&) = delete;
  Owner& operator=(const Owner&) = delete;
  ~Owner() {
    if (value) Destroy(value);
  }
};
std::size_t Elements(std::span<const std::size_t> shape, std::uint64_t bound) {
  if (shape.empty() || shape.size() > 8U)
    throw InvalidArgument("MIGraphX requires rank 1..8", Details("shape"));
  std::size_t count = 1U;
  const auto maximum = std::min<std::uint64_t>(
      bound / sizeof(float),
      std::numeric_limits<std::size_t>::max() / sizeof(float));
  for (const auto dimension : shape) {
    if (dimension == 0)
      throw InvalidArgument("MIGraphX requires positive dimensions",
                            Details("shape"));
    if (dimension > maximum / count)
      throw ResourceExhausted("MIGraphX tensor exceeds its byte bound",
                              Details("shape"));
    count *= dimension;
  }
  return count;
}
std::vector<std::size_t> Shape(const_migraphx_shape_t shape,
                               std::uint64_t bound) {
  const std::size_t* lengths = nullptr;
  std::size_t rank = 0, count = 0, bytes = 0;
  bool standard = false;
  migraphx_shape_datatype_t type{};
  Check(migraphx_shape_type(&type, shape), "shape_type");
  Check(migraphx_shape_standard(&standard, shape), "shape_standard");
  Check(migraphx_shape_lengths(&lengths, &rank, shape), "shape_lengths");
  if (!standard || type != migraphx_shape_float_type || rank < 1U ||
      rank > 8U || lengths == nullptr)
    throw CompatibilityError("MIGraphX requires dense fixed FP32 tensors",
                             Details("signature"));
  std::vector<std::size_t> result(lengths, lengths + rank);
  const auto expected = Elements(result, bound);
  Check(migraphx_shape_elements(&count, shape), "shape_elements");
  Check(migraphx_shape_bytes(&bytes, shape), "shape_bytes");
  if (count != expected || bytes != expected * sizeof(float))
    throw CompatibilityError("MIGraphX tensor size mismatch",
                             Details("signature"));
  return result;
}
struct Trace {
  std::vector<std::string> names;
  bool failed = false;
};
using InstructionRun = migraphx_status (*)(
    migraphx_arguments_t*, migraphx_program_t, migraphx_program_parameters_t,
    migraphx_status (*)(const char*, void*), void*);
InstructionRun InstructionNamesRun() {
  // Query only the C API already selected by the application. The extension
  // is additive; an unpatched SDK continues to use its original trace API.
  if (std::getenv("MIGRAPHX_TRACE_EVAL") != nullptr) return nullptr;
  constexpr auto symbol = "migraphx_program_run_instruction_names_v1";
#ifdef _WIN32
  const auto module = GetModuleHandleW(L"migraphx_c.dll");
  return module
             ? reinterpret_cast<InstructionRun>(GetProcAddress(module, symbol))
             : nullptr;
#else
  Dl_info owner{};
  if (!dladdr(reinterpret_cast<const void*>(&migraphx_program_run_trace),
              &owner) ||
      owner.dli_fname == nullptr)
    return nullptr;
  const auto module = dlopen(owner.dli_fname, RTLD_LAZY | RTLD_NOLOAD);
  if (!module) return nullptr;
  auto* address = dlsym(module, symbol);
  Dl_info extension{};
  if (address &&
      (!dladdr(address, &extension) || extension.dli_fbase != owner.dli_fbase))
    address = nullptr;
  (void)dlclose(module);
  return reinterpret_cast<InstructionRun>(address);
#endif
}
struct Finish {
  bool completed = false;
  ~Finish() {
    // The experimental C context getter allocates an opaque handle without a
    // matching public destroy API in 2.18.0. Fence only incomplete/error runs
    // through HIP so borrowed inputs cannot escape with queued work.
    if (!completed) (void)hipDeviceSynchronize();
  }
};
migraphx_status RecordName(const char* name, void* data) noexcept {
  auto& trace = *static_cast<Trace*>(data);
  try {
    if (name == nullptr || trace.names.size() >= 10000U) {
      trace.failed = true;
      return migraphx_status_unknown_error;
    }
    trace.names.emplace_back(name);
    return migraphx_status_success;
  } catch (...) {
    trace.failed = true;
    return migraphx_status_unknown_error;
  }
}
migraphx_status Record(migraphx_trace_info_t info, void* data) noexcept {
  const char* name = nullptr;
  const auto status = migraphx_trace_info_get_name(&name, info);
  if (status != migraphx_status_success) {
    static_cast<Trace*>(data)->failed = true;
    return status;
  }
  return RecordName(name, data);
}
bool Any(const Trace& trace, std::string_view prefix) {
  return std::any_of(trace.names.begin(), trace.names.end(),
                     [&](const auto& name) {
                       return std::string_view(name).starts_with(prefix);
                     });
}
}  // namespace

class Session::Impl {
 public:
  Impl(std::span<const std::byte> model, const ArtifactDigest& digest,
       const SessionOptions& options)
      : options_(options) {
    validate_artifact_digest(digest);
    if (model.empty() || options_.device_id < 0 ||
        options_.input_name.empty() || options_.input_name.size() > 255U ||
        options_.input_name.find('\0') != std::string::npos ||
        options_.max_model_bytes == 0 || options_.max_input_bytes == 0 ||
        options_.max_output_bytes == 0 || options_.max_batch_size == 0)
      throw InvalidArgument("Invalid MIGraphX session options or empty model",
                            Details("load"));
    input_elements_ = Elements(options_.input_shape, options_.max_input_bytes);
    if (options_.input_shape.front() > options_.max_batch_size)
      throw ResourceExhausted("MIGraphX batch exceeds its bound",
                              Details("load"));
    if (model.size() > options_.max_model_bytes)
      throw ResourceExhausted("MIGraphX model exceeds its byte bound",
                              Details("load"));
    if (sha256_hex(model) != digest.hex)
      throw CompatibilityError("MIGraphX model SHA-256 mismatch",
                               Details("load"));
    DeviceGuard device(options_.device_id);
    Owner<migraphx_program_t, migraphx_program_destroy> program;
    Owner<migraphx_onnx_options_t, migraphx_onnx_options_destroy> onnx;
    Check(migraphx_onnx_options_create(&onnx.value), "onnx_options");
    Check(migraphx_onnx_options_set_input_parameter_shape(
              onnx.value, options_.input_name.c_str(),
              options_.input_shape.data(), options_.input_shape.size()),
          "specialization");
    const auto parsed = migraphx_parse_onnx_buffer(&program.value, model.data(),
                                                   model.size(), onnx.value);
    if (parsed != migraphx_status_success)
      throw CompatibilityError("MIGraphX rejected the ONNX model",
                               Details("parse"));
    ValidateInput(program.value);
    output_shape_ = Output(program.value);
    Owner<migraphx_target_t, migraphx_target_destroy> target;
    Owner<migraphx_compile_options_t, migraphx_compile_options_destroy> compile;
    Check(migraphx_target_create(&target.value, "gpu"), "target_gpu");
    Check(migraphx_compile_options_create(&compile.value), "compile_options");
    Check(migraphx_compile_options_set_offload_copy(compile.value, true),
          "offload_copy");
    Check(migraphx_compile_options_set_fast_math(compile.value, false),
          "fast_math");
    Check(
        migraphx_compile_options_set_exhaustive_tune_flag(compile.value, false),
        "exhaustive_tune");
    Check(migraphx_program_compile(program.value, target.value, compile.value),
          "compile_gpu");
    ValidateInput(program.value);
    if (Output(program.value) != output_shape_)
      throw CompatibilityError("Compiled MIGraphX output signature changed",
                               Details("compile"));
    program_ = std::exchange(program.value, nullptr);
    instruction_run_ = InstructionNamesRun();
  }
  ~Impl() {
    int previous = 0;
    const bool restore = hipGetDevice(&previous) == hipSuccess;
    (void)hipSetDevice(options_.device_id);
    if (program_) (void)migraphx_program_destroy(program_);
    if (restore) (void)hipSetDevice(previous);
  }
  RunResult Run(std::span<const float> values) const {
    if (values.size() != input_elements_ ||
        !std::all_of(values.begin(), values.end(),
                     [](float x) { return std::isfinite(x); }))
      throw InvalidArgument("MIGraphX input count or finite-value mismatch",
                            Details("run"));
    std::lock_guard slot(mutex_);
    DeviceGuard device(options_.device_id);
    Owner<migraphx_program_parameter_shapes_t,
          migraphx_program_parameter_shapes_destroy>
        shapes;
    Check(migraphx_program_get_parameter_shapes(&shapes.value, program_),
          "input_shapes");
    const_migraphx_shape_t input = nullptr;
    Check(migraphx_program_parameter_shapes_get(&input, shapes.value,
                                                options_.input_name.c_str()),
          "input_shape");
    Owner<migraphx_argument_t, migraphx_argument_destroy> argument;
    Check(migraphx_argument_create(&argument.value, input,
                                   const_cast<float*>(values.data())),
          "host_input");
    Owner<migraphx_program_parameters_t, migraphx_program_parameters_destroy>
        parameters;
    Check(migraphx_program_parameters_create(&parameters.value), "parameters");
    Check(migraphx_program_parameters_add(
              parameters.value, options_.input_name.c_str(), argument.value),
          "input_binding");
    Trace trace;
    Owner<migraphx_arguments_t, migraphx_arguments_destroy> results;
    Finish finish;
    const auto status =
        instruction_run_
            ? instruction_run_(&results.value, program_, parameters.value,
                               RecordName, &trace)
            : migraphx_program_run_trace(&results.value, program_,
                                         parameters.value, Record, &trace);
    if (trace.failed)
      throw ResourceExhausted("MIGraphX trace storage failed", Details("run"));
    Check(status, "run_gpu");
    if (!(Any(trace, "gpu::hip_gemm") || Any(trace, "gpu::convolution") ||
          Any(trace, "gpu::code_object")) ||
        !Any(trace, "hip::copy_from_gpu") || !Any(trace, "hip::sync_stream") ||
        Any(trace, "cpu::") || Any(trace, "ref::"))
      throw CompatibilityError(
          "MIGraphX did not execute the required synchronized GPU route",
          Details("run"));
    std::size_t count = 0;
    Check(migraphx_arguments_size(&count, results.value), "output_count");
    if (count != 1U)
      throw CompatibilityError("MIGraphX requires one output", Details("run"));
    const_migraphx_argument_t result = nullptr;
    const_migraphx_shape_t shape = nullptr;
    Check(migraphx_arguments_get(&result, results.value, 0), "output");
    Check(migraphx_argument_shape(&shape, result), "output_shape");
    if (Shape(shape, options_.max_output_bytes) != output_shape_)
      throw CompatibilityError("MIGraphX output shape changed", Details("run"));
    char* data = nullptr;
    Check(migraphx_argument_buffer(&data, result), "host_output");
    if (data == nullptr)
      throw BackendError("MIGraphX returned no host output", Details("run"));
    RunResult output{.shape = output_shape_,
                     .values = std::vector<float>(
                         Elements(output_shape_, options_.max_output_bytes)),
                     .instructions = std::move(trace.names)};
    std::memcpy(output.values.data(), data,
                output.values.size() * sizeof(float));
    if (!std::all_of(output.values.begin(), output.values.end(),
                     [](float x) { return std::isfinite(x); }))
      throw NumericalError("MIGraphX output is nonfinite", Details("run"));
    finish.completed = true;
    return output;
  }
  SessionOptions options_;
  std::vector<std::size_t> output_shape_;

 private:
  void ValidateInput(migraphx_program_t program) const {
    Owner<migraphx_program_parameter_shapes_t,
          migraphx_program_parameter_shapes_destroy>
        shapes;
    Check(migraphx_program_get_parameter_shapes(&shapes.value, program),
          "input_shapes");
    std::size_t count = 0;
    Check(migraphx_program_parameter_shapes_size(&count, shapes.value),
          "input_count");
    if (count != 1U)
      throw CompatibilityError("MIGraphX requires one input",
                               Details("signature"));
    const char* name = nullptr;
    Check(migraphx_program_parameter_shapes_names(&name, shapes.value),
          "input_name");
    if (name == nullptr || name != options_.input_name)
      throw CompatibilityError("MIGraphX input name mismatch",
                               Details("signature"));
    const_migraphx_shape_t shape = nullptr;
    Check(migraphx_program_parameter_shapes_get(&shape, shapes.value, name),
          "input_shape");
    if (Shape(shape, options_.max_input_bytes) != options_.input_shape)
      throw CompatibilityError("MIGraphX input shape mismatch",
                               Details("signature"));
  }
  std::vector<std::size_t> Output(migraphx_program_t program) const {
    Owner<migraphx_shapes_t, migraphx_shapes_destroy> outputs;
    Check(migraphx_program_get_output_shapes(&outputs.value, program),
          "output_shapes");
    std::size_t count = 0;
    Check(migraphx_shapes_size(&count, outputs.value), "output_count");
    if (count != 1U)
      throw CompatibilityError("MIGraphX requires one output",
                               Details("signature"));
    const_migraphx_shape_t shape = nullptr;
    Check(migraphx_shapes_get(&shape, outputs.value, 0), "output_shape");
    return Shape(shape, options_.max_output_bytes);
  }
  std::size_t input_elements_ = 0;
  migraphx_program_t program_ = nullptr;
  InstructionRun instruction_run_ = nullptr;
  mutable std::mutex mutex_;
};

Session::Session(std::span<const std::byte> model, const ArtifactDigest& digest,
                 const SessionOptions& options)
    : impl_(Boundary("load", [&] {
        return std::make_unique<Impl>(model, digest, options);
      })) {}
Session::~Session() = default;
const SessionOptions& Session::options() const noexcept {
  return impl_->options_;
}
std::span<const std::size_t> Session::output_shape() const noexcept {
  return impl_->output_shape_;
}
RunResult Session::run(std::span<const float> values) const {
  return Boundary("run", [&] { return impl_->Run(values); });
}
}  // namespace jyppx::liuding::migraphx_runtime
