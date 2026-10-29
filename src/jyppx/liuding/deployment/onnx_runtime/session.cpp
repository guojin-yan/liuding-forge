// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/deployment/onnx_runtime/session.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <new>
#include <onnxruntime_cxx_api.h>
#include <shared_mutex>
#include <thread>
#include <utility>

namespace jyppx::liuding::onnx_runtime {
namespace {
constexpr const char* kRuntimeVersion = "1.23.2";
constexpr const char* kComponent = "onnx_runtime.session";

ErrorDetails Details(std::string operation) {
  return {.component = kComponent, .operation = std::move(operation)};
}

template <typename Function>
decltype(auto) Boundary(const char* operation, Function&& function) {
  try {
    return std::forward<Function>(function)();
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    std::throw_with_nested(ResourceExhausted("ONNX Runtime allocation failed",
                                             Details(operation)));
  } catch (const Ort::Exception&) {
    if (std::string_view(operation) == "load")
      std::throw_with_nested(
          CompatibilityError("ONNX Runtime rejected the model or an operator",
                             Details(operation)));
    std::throw_with_nested(
        BackendError("ONNX Runtime operation failed", Details(operation)));
  } catch (const std::exception&) {
    std::throw_with_nested(
        BackendError("ONNX Runtime operation failed", Details(operation)));
  }
}

void CheckOperation(
    std::stop_token stop,
    std::optional<std::chrono::steady_clock::time_point> deadline) {
  if (stop.stop_requested())
    throw Cancelled("ONNX Runtime inference cancelled", Details("run"));
  if (deadline && std::chrono::steady_clock::now() >= *deadline)
    throw DeadlineExceeded("ONNX Runtime inference deadline reached",
                           Details("run"));
}

void ValidateOptions(const SessionOptions& options) {
  if (options.cpu_threads == 0U || options.cpu_threads > 64U ||
      options.max_model_bytes == 0U || options.max_input_bytes == 0U ||
      options.max_output_bytes == 0U || options.max_batch_size == 0U ||
      options.max_batch_size >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    throw InvalidArgument("Invalid ONNX Runtime session limits",
                          Details("options"));
  if (options.provider.name.empty() || options.provider.name.size() > 128U ||
      options.provider.name.find('\0') != std::string::npos ||
      options.provider.options.size() > 128U)
    throw InvalidArgument("Invalid ONNX Runtime provider selection",
                          Details("provider"));
  for (const auto& [key, value] : options.provider.options) {
    if (key.empty() || key.size() > 256U || value.size() > 4096U ||
        key.find('\0') != std::string::npos ||
        value.find('\0') != std::string::npos)
      throw InvalidArgument("Invalid ONNX Runtime provider options",
                            Details("provider"));
  }
  const auto& prefix = options.profile_prefix.native();
  if (!prefix.empty() &&
      (prefix.size() > 4096U ||
       std::find(prefix.begin(), prefix.end(), 0) != prefix.end()))
    throw InvalidArgument("Invalid ONNX Runtime profile prefix",
                          Details("options"));
}

constexpr std::array<std::string_view, 8U> kGenericProviders{
    "QNNExecutionProvider",     "XnnpackExecutionProvider",
    "WebNNExecutionProvider",   "WebGpuExecutionProvider",
    "AzureExecutionProvider",   "JsExecutionProvider",
    "VitisAIExecutionProvider", "CoreMLExecutionProvider"};

void AppendProvider(Ort::SessionOptions& settings,
                    const SessionOptions& options) {
  const auto& provider = options.provider;
  if (!provider_append_supported(provider.name))
    throw CompatibilityError(
        "ONNX Runtime provider has no supported append API in this adapter",
        Details("provider"));
  const auto providers = Ort::GetAvailableProviders();
  if (std::find(providers.begin(), providers.end(), provider.name) ==
      providers.end())
    throw CompatibilityError("Requested ONNX Runtime provider is unavailable",
                             Details("provider"));
  if (provider.name == "CPUExecutionProvider") {
    if (!provider.options.empty())
      throw InvalidArgument(
          "CPU provider options are controlled by session limits",
          Details("provider"));
    // CPU is ORT's default provider. Explicitly appending it together with
    // disable_cpu_ep_fallback is contradictory in the qualified ORT version.
    return;
  }
  if (!options.allow_cpu_fallback)
    settings.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
  std::vector<const char*> keys;
  std::vector<const char*> values;
  for (const auto& [key, value] : provider.options) {
    keys.push_back(key.c_str());
    values.push_back(value.c_str());
  }
  const auto& api = Ort::GetApi();
  if (provider.name == "CUDAExecutionProvider") {
    OrtCUDAProviderOptionsV2* pointer = nullptr;
    Ort::ThrowOnError(api.CreateCUDAProviderOptions(&pointer));
    std::unique_ptr<OrtCUDAProviderOptionsV2,
                    decltype(api.ReleaseCUDAProviderOptions)>
        native(pointer, api.ReleaseCUDAProviderOptions);
    Ort::ThrowOnError(api.UpdateCUDAProviderOptions(
        native.get(), keys.data(), values.data(), keys.size()));
    Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_CUDA_V2(
        settings, native.get()));
  } else if (provider.name == "TensorrtExecutionProvider") {
    OrtTensorRTProviderOptionsV2* pointer = nullptr;
    Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&pointer));
    std::unique_ptr<OrtTensorRTProviderOptionsV2,
                    decltype(api.ReleaseTensorRTProviderOptions)>
        native(pointer, api.ReleaseTensorRTProviderOptions);
    Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(
        native.get(), keys.data(), values.data(), keys.size()));
    Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(
        settings, native.get()));
  } else if (provider.name == "OpenVINOExecutionProvider") {
    Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_OpenVINO_V2(
        settings, keys.data(), values.data(), keys.size()));
  } else {
    Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider(
        settings, provider.name.c_str(), keys.data(), values.data(),
        keys.size()));
  }
}

std::size_t Elements(std::span<const std::int64_t> shape,
                     std::uint64_t maximum_bytes) {
  std::uint64_t elements = 1U;
  const auto bound = std::min<std::uint64_t>(
      maximum_bytes / sizeof(float),
      std::numeric_limits<std::size_t>::max() / sizeof(float));
  for (const auto dimension : shape) {
    if (dimension <= 0)
      throw InvalidArgument("ONNX Runtime dimensions must be positive",
                            Details("shape"));
    const auto size = static_cast<std::uint64_t>(dimension);
    if (size > bound || elements > bound / size)
      throw ResourceExhausted(
          "ONNX Runtime tensor exceeds its allocation bound", Details("shape"));
    elements *= size;
  }
  return static_cast<std::size_t>(elements);
}

TensorSignature Signature(const Ort::Session& session, bool input) {
  Ort::AllocatorWithDefaultOptions allocator;
  auto type =
      input ? session.GetInputTypeInfo(0U) : session.GetOutputTypeInfo(0U);
  if (type.GetONNXType() != ONNX_TYPE_TENSOR)
    throw CompatibilityError("ONNX Runtime requires dense tensor I/O",
                             Details("signature"));
  auto tensor = type.GetTensorTypeAndShapeInfo();
  if (tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    throw CompatibilityError("ONNX Runtime requires FP32 input and output",
                             Details("signature"));
  auto name = input ? session.GetInputNameAllocated(0U, allocator)
                    : session.GetOutputNameAllocated(0U, allocator);
  TensorSignature result{
      .name = name.get(), .shape = tensor.GetShape(), .batch_symbol = {}};
  if (result.name.empty() || result.name.size() > 4096U ||
      result.shape.empty() || result.shape.size() > 8U)
    throw CompatibilityError("ONNX Runtime signature exceeds supported bounds",
                             Details("signature"));
  const auto symbols = tensor.GetSymbolicDimensions();
  for (std::size_t index = 0U; index < result.shape.size(); ++index) {
    const auto dimension = result.shape[index];
    if (dimension > 0) continue;
    if (index != 0U || dimension != -1 || symbols[index] == nullptr ||
        *symbols[index] == '\0')
      throw CompatibilityError(
          "ONNX Runtime supports only a named dynamic leading batch dimension",
          Details("signature"));
    result.batch_symbol = symbols[index];
    if (result.batch_symbol.size() > 4096U)
      throw CompatibilityError("ONNX Runtime batch symbol exceeds its bound",
                               Details("signature"));
  }
  return result;
}

class RunInterrupt final {
 public:
  RunInterrupt(Ort::RunOptions& options, std::stop_token stop,
               std::optional<std::chrono::steady_clock::time_point> deadline)
      : options_(options), cancellation_(stop, [this] { Terminate(); }) {
    if (deadline) {
      timer_ = std::jthread([this, at = *deadline](std::stop_token token) {
        std::unique_lock lock(mutex_);
        condition_.wait_until(lock, token, at, [] { return false; });
        if (!token.stop_requested()) Terminate();
      });
    }
  }

  ~RunInterrupt() {
    if (timer_.joinable()) {
      timer_.request_stop();
      condition_.notify_all();
      timer_.join();
    }
  }

 private:
  void Terminate() noexcept {
    // A callback must not throw across std::stop_callback or a worker boundary.
    if (auto* error = Ort::GetApi().RunOptionsSetTerminate(options_))
      Ort::GetApi().ReleaseStatus(error);
  }
  Ort::RunOptions& options_;
  std::mutex mutex_;
  std::condition_variable_any condition_;
  std::jthread timer_;
  std::stop_callback<std::function<void()>> cancellation_;
};
}  // namespace

std::vector<std::string> available_providers() {
  return Boundary("providers", [] { return Ort::GetAvailableProviders(); });
}

bool provider_append_supported(std::string_view name) noexcept {
  return name == "CPUExecutionProvider" || name == "CUDAExecutionProvider" ||
         name == "TensorrtExecutionProvider" ||
         name == "OpenVINOExecutionProvider" ||
         std::find(kGenericProviders.begin(), kGenericProviders.end(), name) !=
             kGenericProviders.end();
}

class Session::Impl {
 public:
  Impl(std::span<const std::byte> bytes, const ArtifactDigest& digest,
       const SessionOptions& limits)
      : options(limits),
        identity(digest),
        version(OrtGetApiBase()->GetVersionString()),
        environment(ORT_LOGGING_LEVEL_ERROR, "liuding-forge-onnx-runtime") {
    if (version != kRuntimeVersion)
      throw CompatibilityError("ONNX Runtime runtime version is not qualified",
                               Details("runtime"));
    Ort::SessionOptions settings;
    settings.SetIntraOpNumThreads(static_cast<int>(options.cpu_threads));
    settings.SetInterOpNumThreads(1);
    settings.SetExecutionMode(ORT_SEQUENTIAL);
    settings.SetGraphOptimizationLevel(
        options.disable_graph_optimizations ? ORT_DISABLE_ALL : ORT_ENABLE_ALL);
    settings.DisableMemPattern();
    settings.DisableCpuMemArena();
    settings.AddConfigEntry("session.intra_op.allow_spinning", "0");
    settings.AddConfigEntry("session.load_model_format", "ONNX");
    AppendProvider(settings, options);
    if (!options.profile_prefix.empty())
      settings.EnableProfiling(options.profile_prefix.c_str());
    session = Ort::Session(environment, bytes.data(), bytes.size(), settings);
    if (session.GetInputCount() != 1U || session.GetOutputCount() != 1U ||
        session.GetOverridableInitializerCount() != 0U)
      throw CompatibilityError(
          "ONNX Runtime requires exactly one input and output",
          Details("signature"));
    input = Signature(session, true);
    output = Signature(session, false);
    if (input.shape.front() != output.shape.front() ||
        input.batch_symbol != output.batch_symbol)
      throw CompatibilityError("ONNX Runtime output batch must match its input",
                               Details("signature"));
    auto input_minimum = input.shape;
    auto output_minimum = output.shape;
    if (input_minimum.front() == -1) input_minimum.front() = 1;
    if (output_minimum.front() == -1) output_minimum.front() = 1;
    (void)Elements(input_minimum, options.max_input_bytes);
    (void)Elements(output_minimum, options.max_output_bytes);
    if (static_cast<std::uint64_t>(input_minimum.front()) >
        options.max_batch_size)
      throw ResourceExhausted("ONNX Runtime static batch exceeds its bound",
                              Details("signature"));
  }

  SessionOptions options;
  ArtifactDigest identity;
  std::string version;
  Ort::Env environment;
  Ort::Session session{nullptr};
  TensorSignature input;
  TensorSignature output;
  mutable std::shared_mutex activity;
  std::optional<std::filesystem::path> profile;
};

Session::Session(std::span<const std::byte> model,
                 const ArtifactDigest& expected_digest,
                 const SessionOptions& options) {
  Boundary("load", [&] {
    validate_artifact_digest(expected_digest);
    ValidateOptions(options);
    if (model.empty())
      throw InvalidArgument("ONNX Runtime model bytes are empty",
                            Details("load"));
    if (model.size() > options.max_model_bytes)
      throw ResourceExhausted("ONNX Runtime model exceeds its byte bound",
                              Details("load"));
    if (sha256_hex(model) != expected_digest.hex)
      throw CompatibilityError("ONNX Runtime model digest mismatch",
                               Details("load"));
    impl_ = std::make_unique<Impl>(model, expected_digest, options);
  });
}

Session::~Session() = default;
const TensorSignature& Session::input() const noexcept { return impl_->input; }
const TensorSignature& Session::output() const noexcept {
  return impl_->output;
}
const std::string& Session::runtime_version() const noexcept {
  return impl_->version;
}
const ArtifactDigest& Session::model_digest() const noexcept {
  return impl_->identity;
}
const ExecutionProvider& Session::execution_provider() const noexcept {
  return impl_->options.provider;
}
bool Session::cpu_fallback_allowed() const noexcept {
  return impl_->options.allow_cpu_fallback;
}

TensorOutput Session::run(
    std::span<const float> values, std::span<const std::int64_t> shape,
    std::stop_token stop,
    std::optional<std::chrono::steady_clock::time_point> deadline) const {
  return Boundary("run", [&] {
    CheckOperation(stop, deadline);
    std::shared_lock lock(impl_->activity);
    CheckOperation(stop, deadline);
    if (shape.size() != impl_->input.shape.size())
      throw InvalidArgument("ONNX Runtime input rank mismatch", Details("run"));
    for (std::size_t index = 0U; index < shape.size(); ++index) {
      if (shape[index] <= 0 || (impl_->input.shape[index] != -1 &&
                                shape[index] != impl_->input.shape[index]))
        throw InvalidArgument("ONNX Runtime input shape mismatch",
                              Details("run"));
    }
    if (static_cast<std::uint64_t>(shape.front()) >
        impl_->options.max_batch_size)
      throw ResourceExhausted("ONNX Runtime batch exceeds its bound",
                              Details("run"));
    const auto count = Elements(shape, impl_->options.max_input_bytes);
    if (count != values.size())
      throw InvalidArgument("ONNX Runtime input element count mismatch",
                            Details("run"));
    if (!std::all_of(values.begin(), values.end(),
                     [](float value) { return std::isfinite(value); }))
      throw InvalidArgument("ONNX Runtime input contains nonfinite values",
                            Details("run"));
    TensorOutput result{.shape = impl_->output.shape, .values = {}};
    if (result.shape.front() == -1) result.shape.front() = shape.front();
    result.values.resize(
        Elements(result.shape, impl_->options.max_output_bytes));
    auto memory =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // ORT's C API is not const-correct for inputs; graph inputs are read-only.
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory, const_cast<float*>(values.data()), values.size(), shape.data(),
        shape.size());
    auto output_tensor = Ort::Value::CreateTensor<float>(
        memory, result.values.data(), result.values.size(), result.shape.data(),
        result.shape.size());
    const char* input_names[] = {impl_->input.name.c_str()};
    const char* output_names[] = {impl_->output.name.c_str()};
    Ort::RunOptions options;
    RunInterrupt interrupt(options, stop, deadline);
    try {
      impl_->session.Run(options, input_names, &input_tensor, 1U, output_names,
                         &output_tensor, 1U);
    } catch (const Ort::Exception&) {
      CheckOperation(stop, deadline);
      throw;
    }
    CheckOperation(stop, deadline);
    if (!std::all_of(result.values.begin(), result.values.end(),
                     [](float value) { return std::isfinite(value); }))
      throw NumericalError("ONNX Runtime output contains nonfinite values",
                           Details("run"));
    return result;
  });
}

std::filesystem::path Session::end_profiling() {
  return Boundary("profile", [&] {
    std::unique_lock lock(impl_->activity);
    if (impl_->options.profile_prefix.empty())
      throw InvalidArgument("ONNX Runtime profiling was not enabled",
                            Details("profile"));
    if (!impl_->profile) {
      Ort::AllocatorWithDefaultOptions allocator;
      auto filename = impl_->session.EndProfilingAllocated(allocator);
      if (!filename || *filename.get() == '\0')
        throw BackendError("ONNX Runtime profiler returned no filename",
                           Details("profile"));
      impl_->profile = std::filesystem::u8path(filename.get());
    }
    return *impl_->profile;
  });
}

}  // namespace jyppx::liuding::onnx_runtime
