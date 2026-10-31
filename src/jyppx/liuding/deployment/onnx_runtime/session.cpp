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
#include <charconv>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <new>
#include <onnxruntime_cxx_api.h>
#include <set>
#include <shared_mutex>
#include <system_error>
#include <thread>
#include <utility>

namespace jyppx::liuding::onnx_runtime {
namespace {
constexpr const char* kRuntimeVersion = "1.23.2";
constexpr const char* kComponent = "onnx_runtime.session";

ErrorDetails Details(std::string operation) {
  return {.component = kComponent, .operation = std::move(operation)};
}

ErrorDetails ProviderDetails(const ExecutionProvider& provider,
                             std::string operation, std::string actual) {
  auto details = Details(std::move(operation));
  details.expected = provider.name;
  details.actual = actual.substr(0U, 4096U);
  details.hint =
      "Check the staged ONNX Runtime provider SDK, requested device, driver "
      "and provider dependencies (CUDA/cuDNN for CUDA).";
  return details;
}

template <typename Function>
void ProviderCall(const ExecutionProvider& provider, const char* operation,
                  Function&& function) {
  try {
    std::forward<Function>(function)();
  } catch (const Ort::Exception& error) {
    auto details = ProviderDetails(provider, operation, error.what());
    if (std::string_view(operation) == "provider.options" &&
        error.GetOrtErrorCode() == ORT_INVALID_ARGUMENT) {
      details.hint = "Use provider options documented by the loaded ORT SDK.";
      std::throw_with_nested(InvalidArgument(
          "ONNX Runtime rejected provider options", std::move(details)));
    }
    std::throw_with_nested(DeviceError(
        "ONNX Runtime provider initialization failed", std::move(details)));
  }
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

void ValidateCudaOptions(const ExecutionProvider& provider) {
  if (provider.name != "CUDAExecutionProvider") return;
  const auto reject = [&](std::string_view key, std::string_view value) {
    auto details = ProviderDetails(provider, "provider.options",
                                   std::string(key) + "=" + std::string(value));
    details.hint =
        "Use a nonnegative device, positive arena budget and documented CUDA "
        "options; this adapter requires default-stream copies and owns no "
        "external streams, allocators or CUDA Graph buffers.";
    throw InvalidArgument("Unsupported or invalid CUDA provider option",
                          std::move(details));
  };
  for (const auto& [key, value] : provider.options) {
    if (key == "device_id" || key == "gpu_mem_limit") {
      std::uint64_t parsed = 0U;
      const auto result =
          std::from_chars(value.data(), value.data() + value.size(), parsed);
      const auto maximum = key == "device_id"
                               ? static_cast<std::uint64_t>(
                                     std::numeric_limits<std::int32_t>::max())
                               : static_cast<std::uint64_t>(
                                     std::numeric_limits<std::size_t>::max());
      if (value.empty() || result.ec != std::errc{} ||
          result.ptr != value.data() + value.size() || parsed > maximum ||
          (key == "gpu_mem_limit" && parsed == 0U))
        reject(key, value);
    } else if (key == "enable_cuda_graph") {
      // CUDA Graph requires stable device addresses and disallows concurrent
      // Run calls, unlike this adapter's borrowed host I/O contract.
      if (value != "0") reject(key, value);
    } else if (key == "do_copy_in_default_stream") {
      // ORT documents possible races when provider copies use separate streams.
      if (value != "1") reject(key, value);
    } else if (key == "user_compute_stream" ||
               key == "has_user_compute_stream" ||
               key == "gpu_external_alloc" || key == "gpu_external_free" ||
               key == "gpu_external_empty_cache") {
      reject(key, value);
    } else if (key == "use_tf32" || key == "cudnn_conv_use_max_workspace") {
      if (value != "0" && value != "1") reject(key, value);
    } else if (key == "cudnn_conv_algo_search") {
      if (value != "HEURISTIC" && value != "EXHAUSTIVE" && value != "DEFAULT")
        reject(key, value);
    }
  }
}

void ValidateOptions(const SessionOptions& options) {
  if (options.cpu_threads == 0U || options.cpu_threads > 64U ||
      options.max_model_bytes == 0U || options.max_input_bytes == 0U ||
      options.max_output_bytes == 0U || options.max_batch_size == 0U ||
      options.max_io_tensors < 2U || options.max_io_tensors > 4096U ||
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
  ValidateCudaOptions(options.provider);
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
        ProviderDetails(provider, "provider", "No supported append API"));
  const auto providers = Ort::GetAvailableProviders();
  if (std::find(providers.begin(), providers.end(), provider.name) ==
      providers.end())
    throw CompatibilityError("Requested ONNX Runtime provider is unavailable",
                             ProviderDetails(provider, "provider",
                                             "Provider is not compiled into "
                                             "the loaded runtime"));
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
    ProviderCall(provider, "provider.initialize", [&] {
      Ort::ThrowOnError(api.CreateCUDAProviderOptions(&pointer));
    });
    std::unique_ptr<OrtCUDAProviderOptionsV2,
                    decltype(api.ReleaseCUDAProviderOptions)>
        native(pointer, api.ReleaseCUDAProviderOptions);
    ProviderCall(provider, "provider.options", [&] {
      Ort::ThrowOnError(api.UpdateCUDAProviderOptions(
          native.get(), keys.data(), values.data(), keys.size()));
    });
    ProviderCall(provider, "provider.initialize", [&] {
      Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_CUDA_V2(
          settings, native.get()));
    });
  } else if (provider.name == "TensorrtExecutionProvider") {
    OrtTensorRTProviderOptionsV2* pointer = nullptr;
    ProviderCall(provider, "provider.initialize", [&] {
      Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&pointer));
    });
    std::unique_ptr<OrtTensorRTProviderOptionsV2,
                    decltype(api.ReleaseTensorRTProviderOptions)>
        native(pointer, api.ReleaseTensorRTProviderOptions);
    ProviderCall(provider, "provider.options", [&] {
      Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(
          native.get(), keys.data(), values.data(), keys.size()));
    });
    ProviderCall(provider, "provider.initialize", [&] {
      Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(
          settings, native.get()));
    });
  } else if (provider.name == "OpenVINOExecutionProvider") {
    ProviderCall(provider, "provider.initialize", [&] {
      Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_OpenVINO_V2(
          settings, keys.data(), values.data(), keys.size()));
    });
  } else {
    ProviderCall(provider, "provider.initialize", [&] {
      Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider(
          settings, provider.name.c_str(), keys.data(), values.data(),
          keys.size()));
    });
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

void ValidateInput(const TensorInput& input, const TensorSignature& signature,
                   const SessionOptions& options, std::uint64_t& remaining) {
  if (input.shape.size() != signature.shape.size())
    throw InvalidArgument("ONNX Runtime input rank mismatch", Details("run"));
  for (std::size_t index = 0; index < input.shape.size(); ++index) {
    if (input.shape[index] <= 0 ||
        (signature.shape[index] != -1 &&
         input.shape[index] != signature.shape[index]))
      throw InvalidArgument("ONNX Runtime input shape mismatch",
                            Details("run"));
  }
  if (static_cast<std::uint64_t>(input.shape.front()) > options.max_batch_size)
    throw ResourceExhausted("ONNX Runtime batch exceeds its bound",
                            Details("run"));
  const auto count = Elements(input.shape, remaining);
  if (count != input.values.size())
    throw InvalidArgument("ONNX Runtime input element count mismatch",
                          Details("run"));
  remaining -= count * sizeof(float);
  if (!std::all_of(input.values.begin(), input.values.end(),
                   [](float value) { return std::isfinite(value); }))
    throw InvalidArgument("ONNX Runtime input contains nonfinite values",
                          Details("run"));
}

TensorSignature Signature(const Ort::Session& session, bool input,
                          std::size_t position) {
  Ort::AllocatorWithDefaultOptions allocator;
  auto type = input ? session.GetInputTypeInfo(position)
                    : session.GetOutputTypeInfo(position);
  if (type.GetONNXType() != ONNX_TYPE_TENSOR)
    throw CompatibilityError("ONNX Runtime requires dense tensor I/O",
                             Details("signature"));
  auto tensor = type.GetTensorTypeAndShapeInfo();
  if (tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    throw CompatibilityError("ONNX Runtime requires FP32 input and output",
                             Details("signature"));
  auto name = input ? session.GetInputNameAllocated(position, allocator)
                    : session.GetOutputNameAllocated(position, allocator);
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
      // condition_variable_any owns a shared mutex on MSVC.  A normal run
      // has no deadline, so do not construct that wait state unless a timer
      // is actually needed.
      condition_.emplace();
      timer_ = std::jthread([this, at = *deadline](std::stop_token token) {
        std::unique_lock lock(mutex_);
        condition_->wait_until(lock, token, at, [] { return false; });
        if (!token.stop_requested()) Terminate();
      });
    }
  }

  ~RunInterrupt() {
    if (timer_.joinable()) {
      timer_.request_stop();
      condition_->notify_all();
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
  std::optional<std::condition_variable_any> condition_;
  std::jthread timer_;
  std::stop_callback<std::function<void()>> cancellation_;
};
}  // namespace

std::vector<std::string> available_providers() {
  return Boundary("providers", [] { return Ort::GetAvailableProviders(); });
}

ExecutionProvider cuda_execution_provider(const CudaProviderOptions& options) {
  return Boundary("provider.options", [&] {
    ExecutionProvider result{
        .name = "CUDAExecutionProvider",
        .options = {{"device_id", std::to_string(options.device_id)},
                    {"cudnn_conv_algo_search", "HEURISTIC"},
                    {"cudnn_conv_use_max_workspace",
                     options.allow_max_conv_workspace ? "1" : "0"},
                    {"use_tf32", options.allow_tf32 ? "1" : "0"},
                    {"do_copy_in_default_stream", "1"}}};
    if (options.arena_memory_limit_bytes)
      result.options.emplace("gpu_mem_limit",
                             std::to_string(*options.arena_memory_limit_bytes));
    ValidateCudaOptions(result);
    return result;
  });
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
    try {
      session = Ort::Session(environment, bytes.data(), bytes.size(), settings);
    } catch (const Ort::Exception& error) {
      auto details = ProviderDetails(options.provider, "load", error.what());
      if (error.GetOrtErrorCode() == ORT_EP_FAIL)
        std::throw_with_nested(
            DeviceError("ONNX Runtime execution provider failed to initialize",
                        std::move(details)));
      details.hint =
          "Check the self-contained model, operator schemas and coverage "
          "of the requested execution provider.";
      std::throw_with_nested(CompatibilityError(
          "ONNX Runtime could not initialize the requested model/session",
          std::move(details)));
    }
    const auto input_count = session.GetInputCount();
    const auto output_count = session.GetOutputCount();
    if (input_count == 0U || output_count == 0U ||
        session.GetOverridableInitializerCount() != 0U)
      throw CompatibilityError(
          "ONNX Runtime requires inputs/outputs and immutable initializers",
          Details("signature"));
    if (input_count > options.max_io_tensors ||
        output_count > options.max_io_tensors - input_count)
      throw ResourceExhausted("ONNX Runtime I/O count exceeds its bound",
                              Details("signature"));
    inputs.reserve(input_count);
    outputs.reserve(output_count);
    std::set<std::string> input_symbols;
    for (const bool is_input : {true, false}) {
      auto& signatures = is_input ? inputs : outputs;
      auto remaining =
          is_input ? options.max_input_bytes : options.max_output_bytes;
      const auto count = is_input ? input_count : output_count;
      std::set<std::string> names;
      for (std::size_t index = 0; index < count; ++index) {
        auto signature = Signature(session, is_input, index);
        if (!names.insert(signature.name).second)
          throw CompatibilityError("ONNX Runtime has duplicate I/O names",
                                   Details("signature"));
        if (!signature.batch_symbol.empty()) {
          if (is_input) {
            input_symbols.insert(signature.batch_symbol);
          } else if (!input_symbols.contains(signature.batch_symbol)) {
            throw CompatibilityError(
                "ONNX Runtime output batch symbol has no input binding",
                Details("signature"));
          }
        }
        auto minimum = signature.shape;
        if (minimum.front() == -1) minimum.front() = 1;
        remaining -= Elements(minimum, remaining) * sizeof(float);
        if (is_input && static_cast<std::uint64_t>(minimum.front()) >
                            options.max_batch_size)
          throw ResourceExhausted("ONNX Runtime static batch exceeds its bound",
                                  Details("signature"));
        signatures.push_back(std::move(signature));
      }
    }
    for (const auto& input : inputs) input_names.push_back(input.name.c_str());
    for (const auto& output : outputs)
      output_names.push_back(output.name.c_str());
  }

  void Execute(std::span<const Ort::Value> input_tensors,
               std::span<Ort::Value> output_tensors,
               std::span<const TensorOutput> results, std::stop_token stop,
               std::optional<std::chrono::steady_clock::time_point> deadline) {
    CheckOperation(stop, deadline);
    std::shared_lock lock(activity);
    CheckOperation(stop, deadline);
    Ort::RunOptions run_options;
    RunInterrupt interrupt(run_options, stop, deadline);
    try {
      session.Run(run_options, input_names.data(), input_tensors.data(),
                  input_tensors.size(), output_names.data(),
                  output_tensors.data(), output_tensors.size());
    } catch (const Ort::Exception&) {
      CheckOperation(stop, deadline);
      throw;
    }
    CheckOperation(stop, deadline);
    for (const auto& result : results) {
      if (!std::all_of(result.values.begin(), result.values.end(),
                       [](float value) { return std::isfinite(value); }))
        throw NumericalError("ONNX Runtime output contains nonfinite values",
                             Details("run"));
    }
    CheckOperation(stop, deadline);
  }

  SessionOptions options;
  ArtifactDigest identity;
  std::string version;
  Ort::Env environment;
  Ort::Session session{nullptr};
  std::vector<TensorSignature> inputs;
  std::vector<TensorSignature> outputs;
  std::vector<const char*> input_names;
  std::vector<const char*> output_names;
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
std::span<const TensorSignature> Session::inputs() const noexcept {
  return impl_->inputs;
}
std::span<const TensorSignature> Session::outputs() const noexcept {
  return impl_->outputs;
}
const TensorSignature& Session::input() const {
  if (impl_->inputs.size() != 1U)
    throw InvalidArgument("Use inputs() for multiple ONNX Runtime inputs",
                          Details("signature"));
  return impl_->inputs.front();
}
const TensorSignature& Session::output() const {
  if (impl_->outputs.size() != 1U)
    throw InvalidArgument("Use outputs() for multiple ONNX Runtime outputs",
                          Details("signature"));
  return impl_->outputs.front();
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
    if (impl_->inputs.size() != 1U || impl_->outputs.size() != 1U)
      throw InvalidArgument(
          "Use named inputs for a multi-I/O ONNX Runtime graph",
          Details("run"));
    const auto& signature = impl_->inputs.front();
    auto remaining = impl_->options.max_input_bytes;
    ValidateInput({.name = signature.name, .values = values, .shape = shape},
                  signature, impl_->options, remaining);
    const auto& output = impl_->outputs.front();
    TensorOutput result{
        .shape = output.shape, .values = {}, .name = output.name};
    if (!output.batch_symbol.empty()) result.shape.front() = shape.front();
    result.values.resize(
        Elements(result.shape, impl_->options.max_output_bytes));
    auto memory =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // ORT borrows input storage read-only despite its non-const C signature.
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory, const_cast<float*>(values.data()), values.size(), shape.data(),
        shape.size());
    auto output_tensor = Ort::Value::CreateTensor<float>(
        memory, result.values.data(), result.values.size(), result.shape.data(),
        result.shape.size());
    impl_->Execute(std::span(&input_tensor, 1U), std::span(&output_tensor, 1U),
                   std::span(&result, 1U), stop, deadline);
    return result;
  });
}

std::vector<TensorOutput> Session::run(
    std::span<const TensorInput> inputs, std::stop_token stop,
    std::optional<std::chrono::steady_clock::time_point> deadline) const {
  return Boundary("run", [&] {
    CheckOperation(stop, deadline);
    if (inputs.size() != impl_->inputs.size())
      throw InvalidArgument("ONNX Runtime requires every named input once",
                            Details("run"));
    std::vector<const TensorInput*> ordered(inputs.size(), nullptr);
    std::map<std::string_view, std::int64_t> bindings;
    auto input_remaining = impl_->options.max_input_bytes;
    for (const auto& input : inputs) {
      const auto found = std::find_if(
          impl_->inputs.begin(), impl_->inputs.end(),
          [&](const auto& signature) { return signature.name == input.name; });
      if (found == impl_->inputs.end())
        throw InvalidArgument("ONNX Runtime input name is unknown",
                              Details("run"));
      const auto position =
          static_cast<std::size_t>(found - impl_->inputs.begin());
      if (ordered[position] != nullptr)
        throw InvalidArgument("ONNX Runtime input name is duplicated",
                              Details("run"));
      ordered[position] = &input;
      ValidateInput(input, *found, impl_->options, input_remaining);
      if (!found->batch_symbol.empty()) {
        const auto [binding, inserted] =
            bindings.emplace(found->batch_symbol, input.shape.front());
        if (!inserted && binding->second != input.shape.front())
          throw InvalidArgument("ONNX Runtime input batch symbol sizes differ",
                                Details("run"));
      }
    }
    std::vector<TensorOutput> results;
    std::vector<std::size_t> output_counts;
    results.reserve(impl_->outputs.size());
    output_counts.reserve(impl_->outputs.size());
    auto output_remaining = impl_->options.max_output_bytes;
    for (const auto& signature : impl_->outputs) {
      TensorOutput result{
          .shape = signature.shape, .values = {}, .name = signature.name};
      if (!signature.batch_symbol.empty())
        result.shape.front() = bindings.at(signature.batch_symbol);
      const auto count = Elements(result.shape, output_remaining);
      output_remaining -= count * sizeof(float);
      output_counts.push_back(count);
      results.push_back(std::move(result));
    }
    // Check every output before allocating any FP32 output payload.
    for (std::size_t index = 0; index < results.size(); ++index)
      results[index].values.resize(output_counts[index]);
    auto memory =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // ORT's C API is not const-correct for inputs; graph inputs are read-only.
    std::vector<Ort::Value> input_tensors;
    std::vector<Ort::Value> output_tensors;
    input_tensors.reserve(ordered.size());
    output_tensors.reserve(results.size());
    for (std::size_t index = 0; index < ordered.size(); ++index) {
      const auto& input = *ordered[index];
      input_tensors.push_back(Ort::Value::CreateTensor<float>(
          memory, const_cast<float*>(input.values.data()), input.values.size(),
          input.shape.data(), input.shape.size()));
    }
    for (auto& result : results) {
      output_tensors.push_back(Ort::Value::CreateTensor<float>(
          memory, result.values.data(), result.values.size(),
          result.shape.data(), result.shape.size()));
    }
    impl_->Execute(input_tensors, output_tensors, results, stop, deadline);
    return results;
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
