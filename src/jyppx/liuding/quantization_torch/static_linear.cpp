// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/quantization_torch/static_linear.hpp>

#include <torch/version.h>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <string_view>
#include <utility>

namespace jyppx::liuding::quantization_torch {
namespace {

namespace q = quantization;
constexpr double kSafeQuantizerMagnitude = 1073741824.0;  // 2^30, before int32.

ErrorDetails details(std::string operation) {
  return {
      .component = "quantization_torch.static_linear",
      .operation = std::move(operation),
      .hint = "Use the explicit calibrated CPU static INT8 Linear profile."};
}

template <typename Function>
decltype(auto) backend(std::string operation, Function&& function) {
  try {
    return std::forward<Function>(function)();
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    std::throw_with_nested(ResourceExhausted(
        "Official quantized Linear allocation failed", details(operation)));
  } catch (const c10::Error&) {
    std::throw_with_nested(BackendError(
        "Official LibTorch quantized Linear failed", details(operation)));
  }
}

void require_rounding() {
  if (std::fegetround() != FE_TONEAREST)
    throw CompatibilityError("Static INT8 Linear requires nearest-even mode",
                             details("rounding_mode"));
}

bool accepted_engine(std::string_view engine) {
  return engine == "ONEDNN" || engine == "X86";
}

void check_engine(std::string_view expected) {
  // The supported-engine and kernel registries are SDK initialization facts.
  // Avoid allocating capability strings/vectors on each inference call.
  const auto current = at::globalContext().qEngine();
  if (!((expected == "ONEDNN" && current == c10::QEngine::ONEDNN) ||
        (expected == "X86" && current == c10::QEngine::X86)))
    throw CompatibilityError("Official static INT8 engine changed",
                             details("engine"));
}

void check_backend(std::string_view expected) {
  const auto observed = quantized_cpu_capabilities();
  check_engine(expected);
  if (!accepted_engine(expected) ||
      std::find(observed.supported_engines.begin(),
                observed.supported_engines.end(),
                expected) == observed.supported_engines.end() ||
      !observed.linear_prepack_registered || !observed.linear_registered)
    throw NotImplemented(
        "Requested official static INT8 SDK profile is unavailable",
        details("engine"));
}

void check_options(const StaticLinearOptions& options) {
  q::validate_limits(options.artifact_limits);
  if (options.max_batch == 0U || options.max_io_bytes == 0U)
    throw InvalidArgument("Static Linear budgets must be positive",
                          details("limits"));
  if (options.expected_engine && !accepted_engine(*options.expected_engine))
    throw NotImplemented("This static Linear adapter accepts ONEDNN or X86",
                         details("engine"));
}

void freeze_engine(StaticLinearOptions& options) {
  check_options(options);
  const auto current = c10::toString(at::globalContext().qEngine());
  if (!accepted_engine(current))
    throw NotImplemented("Current quantized engine has no adapter profile",
                         details("engine"));
  if (options.expected_engine && *options.expected_engine != current)
    throw CompatibilityError("Current engine differs from requested identity",
                             details("engine"));
  check_backend(current);
  options.expected_engine = current;
}

std::vector<std::byte> snapshot(std::span<const std::byte> bytes,
                                const ArtifactDigest& digest,
                                const StaticLinearOptions& options,
                                std::string_view field) {
  check_options(options);
  validate_artifact_digest(digest);
  if (bytes.empty())
    throw InvalidArgument("Canonical quantization bytes are empty",
                          details(std::string(field)));
  if (bytes.size() > options.artifact_limits.max_artifact_bytes)
    throw ResourceExhausted("Canonical quantization exceeds byte budget",
                            details(std::string(field)));
  std::vector<std::byte> owned(bytes.begin(), bytes.end());
  if (sha256_hex(owned) != digest.hex)
    throw CompatibilityError("Canonical quantization digest differs",
                             details(std::string(field)));
  return owned;
}

void check_output_calibration(const q::CalibrationResult& input,
                              const q::CalibrationResult& output) {
  // Both values have already passed their owner's bounded canonical decoder.
  const auto& a = input.lineage;
  const auto& b = output.lineage;
  if (a.dataset_version != b.dataset_version ||
      a.dataset_digest != b.dataset_digest ||
      a.preprocessing_fingerprint != b.preprocessing_fingerprint ||
      a.seed != b.seed || input.sample_ids != output.sample_ids)
    throw CompatibilityError("Input/output calibration identities differ",
                             details("output_calibration"));
}

struct PreparedLinear {
  StaticLinearInfo info;
  StaticLinearOptions options;
  std::vector<std::int8_t> weights;
  std::vector<float> bias;
};

PreparedLinear prepare(std::size_t inputs, std::size_t outputs,
                       bool per_channel, const ArtifactDigest& digest,
                       const ArtifactDigest& output_digest,
                       const q::CalibrationResult& calibration,
                       const q::CalibrationResult& output_calibration,
                       StaticLinearOptions options) {
  check_output_calibration(calibration, output_calibration);
  // The Q0 owner may accept wider configurable limits; this compute profile
  // deliberately bounds int32 accumulation to 65536 * 128 * 127.
  if (inputs == 0U || outputs == 0U || inputs > 65536U || outputs > 65536U)
    throw InvalidArgument("Static INT8 dimensions must be in [1,65536]",
                          details("dimensions"));
  PreparedLinear result;
  result.info.profile_id =
      per_channel ? "jyppx.linear.ptq.libtorch.u8s8.per_output_channel/v1"
                  : "jyppx.linear.ptq.libtorch.u8s8/v1";
  result.info.engine = *options.expected_engine;
  result.info.libtorch_version = TORCH_VERSION;
  result.info.source_artifact_digest = digest;
  result.info.output_calibration_digest = output_digest;
  result.info.in_features = inputs;
  result.info.out_features = outputs;
  result.info.per_output_channel = per_channel;
  result.info.input_scale =
      q::static_int8_scale(calibration.activation_params.scale);
  result.info.output_scale =
      q::static_int8_scale(output_calibration.activation_params.scale);
  if (128.0 * result.info.output_scale > std::numeric_limits<float>::max())
    throw NumericalError("Quantized output cannot dequantize to finite FP32",
                         details("output_scale"));
  result.info.output_calibration = output_calibration;
  result.options = std::move(options);
  return result;
}

void check_numeric_bounds(const PreparedLinear& prepared) {
  const auto& info = prepared.info;
  for (std::size_t row = 0U; row < info.out_features; ++row) {
    const auto weight_scale =
        info.weight_scales[info.per_output_channel ? row : 0U];
    const double product = static_cast<double>(info.input_scale) * weight_scale;
    static_cast<void>(q::static_int8_scale(product));
    static_cast<void>(q::static_int8_scale(product / info.output_scale));
    std::uint64_t magnitude = 0U;
    for (std::size_t column = 0U; column < info.in_features; ++column)
      magnitude += static_cast<std::uint64_t>(std::abs(
          static_cast<int>(prepared.weights[row * info.in_features + column])));
    const double bound =
        static_cast<double>(magnitude * 128U) * product +
        (prepared.bias.empty()
             ? 0.0
             : std::abs(static_cast<double>(prepared.bias[row])));
    if (!std::isfinite(bound) || bound > std::numeric_limits<float>::max() ||
        bound / info.output_scale > kSafeQuantizerMagnitude)
      throw NumericalError("Static INT8 output exceeds safe requantizer range",
                           details("output_bound"));
  }
}

void check_shape(const Tensor& input, const StaticLinearInfo& info,
                 const StaticLinearOptions& options) {
  if (!input.defined() || !input.device().is_cpu() || input.dim() != 2 ||
      input.size(0) <= 0 ||
      input.size(1) != static_cast<std::int64_t>(info.in_features) ||
      !input.is_contiguous() || input.layout() != at::kStrided ||
      input.is_conj() || input.is_neg() || input.requires_grad())
    throw InvalidArgument("Expected contiguous inference CPU [batch,in] input",
                          details("input"));
  const auto batch = static_cast<std::uint64_t>(input.size(0));
  // Float inputs, quantized inputs, quantized outputs and an explicit final
  // float output. Official packed workspace/allocator caches are not
  // RSS-capped.
  const auto row_bytes = 5ULL * (info.in_features + info.out_features);
  if (batch > options.max_batch || row_bytes > options.max_io_bytes ||
      batch > options.max_io_bytes / row_bytes)
    throw ResourceExhausted("Static INT8 logical I/O budget exceeded",
                            details("input_budget"));
}

}  // namespace

QuantizedCpuCapabilities quantized_cpu_capabilities() {
  QuantizedCpuCapabilities result;
  result.libtorch_version = TORCH_VERSION;
  result.current_engine = c10::toString(at::globalContext().qEngine());
  for (const auto engine : at::Context::supportedQEngines())
    result.supported_engines.emplace_back(c10::toString(engine));
  auto& dispatcher = c10::Dispatcher::singleton();
  const auto pack = dispatcher.findSchema({"quantized::linear_prepack", ""});
  const auto linear = dispatcher.findSchema({"quantized::linear", ""});
  result.linear_prepack_registered =
      pack && pack->hasKernelForDispatchKey(c10::DispatchKey::QuantizedCPU);
  result.linear_registered =
      linear && linear->hasKernelForDispatchKey(c10::DispatchKey::QuantizedCPU);
  return result;
}

struct StaticLinear::Impl {
  explicit Impl(PreparedLinear prepared)
      : info(std::move(prepared.info)), options(std::move(prepared.options)) {
    require_rounding();
    check_backend(info.engine);
    const auto out = static_cast<std::int64_t>(info.out_features);
    const auto in = static_cast<std::int64_t>(info.in_features);
    auto integer_weight = at::empty(
        {out, in}, at::TensorOptions().device(at::kCPU).dtype(at::kChar));
    std::memcpy(integer_weight.data_ptr<std::int8_t>(), prepared.weights.data(),
                prepared.weights.size());
    Tensor quantized_weight;
    if (info.per_output_channel) {
      auto scales = at::empty(
          {out}, at::TensorOptions().device(at::kCPU).dtype(at::kDouble));
      auto zeros = at::zeros(
          {out}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
      auto* scale_values = scales.data_ptr<double>();
      for (std::size_t row = 0U; row < info.out_features; ++row)
        scale_values[row] = info.weight_scales[row];
      quantized_weight = at::_make_per_channel_quantized_tensor(
          integer_weight, scales, zeros, 0);
    } else {
      quantized_weight = at::_make_per_tensor_quantized_tensor(
          integer_weight, info.weight_scales.front(), 0);
    }
    c10::IValue bias;
    if (!prepared.bias.empty()) {
      auto value = at::empty(
          {out}, at::TensorOptions().device(at::kCPU).dtype(at::kFloat));
      std::memcpy(value.data_ptr<float>(), prepared.bias.data(),
                  prepared.bias.size() * sizeof(float));
      bias = std::move(value);
    }
    auto& dispatcher = c10::Dispatcher::singleton();
    auto pack_op =
        dispatcher.findSchemaOrThrow("quantized::linear_prepack", "");
    linear_op = dispatcher.findSchemaOrThrow("quantized::linear", "");
    c10::Stack arguments{quantized_weight, std::move(bias)};
    pack_op.callBoxed(&arguments);
    if (arguments.size() != 1U || !arguments.front().isObject())
      throw BackendError("Official Linear prepack returned invalid state",
                         details("linear_prepack"));
    packed = std::move(arguments.front());
    check_after_backend();
  }

  void check_live() const {
    if (poisoned)
      throw NotReady("Static Linear backend failed; reload packed state",
                     details("backend_state"));
    require_rounding();
    check_engine(info.engine);
  }

  void check_after_backend() const {
    try {
      check_engine(info.engine);
    } catch (...) {
      poisoned = true;
      throw;
    }
  }

  Tensor quantize(const Tensor& input) const {
    check_shape(input, info, options);
    if (input.is_quantized() || input.scalar_type() != at::kFloat)
      throw InvalidArgument("Input quantization requires CPU FP32 values",
                            details("input_dtype"));
    const auto [minimum, maximum] = at::aminmax(input);
    const double low = minimum.item<double>();
    const double high = maximum.item<double>();
    if (!std::isfinite(low) || !std::isfinite(high))
      throw NumericalError("Static INT8 input contains NaN or infinity",
                           details("input_finite"));
    if (std::max(std::abs(low), std::abs(high)) / info.input_scale >
        kSafeQuantizerMagnitude)
      throw NumericalError("Input exceeds safe official quantizer range",
                           details("input_range"));
    return at::quantize_per_tensor(input, info.input_scale,
                                   info.input_zero_point, at::kQUInt8);
  }

  Tensor run(const Tensor& input) const {
    check_shape(input, info, options);
    if (!input.is_quantized() || input.scalar_type() != at::kQUInt8 ||
        input.qscheme() != at::kPerTensorAffine ||
        input.q_scale() != static_cast<double>(info.input_scale) ||
        input.q_zero_point() != info.input_zero_point)
      throw CompatibilityError(
          "Quantized input qparams do not match calibration",
          details("input_qparams"));
    c10::Stack arguments{input, packed, static_cast<double>(info.output_scale),
                         info.output_zero_point};
    try {
      linear_op->callBoxed(&arguments);
    } catch (...) {
      poisoned = true;
      throw;
    }
    if (arguments.size() != 1U || !arguments.front().isTensor()) {
      poisoned = true;
      throw BackendError("Official Linear returned an invalid result",
                         details("linear"));
    }
    auto output = arguments.front().toTensor();
    if (!output.is_quantized() || output.scalar_type() != at::kQUInt8 ||
        !output.device().is_cpu() || output.dim() != 2 ||
        output.size(0) != input.size(0) ||
        output.size(1) != static_cast<std::int64_t>(info.out_features) ||
        output.qscheme() != at::kPerTensorAffine ||
        output.q_scale() != static_cast<double>(info.output_scale) ||
        output.q_zero_point() != info.output_zero_point) {
      poisoned = true;
      throw BackendError("Official Linear output violates static qparams",
                         details("output"));
    }
    check_after_backend();
    return output;
  }

  StaticLinearInfo info;
  StaticLinearOptions options;
  c10::IValue packed;
  std::optional<c10::OperatorHandle> linear_op;
  mutable std::mutex mutex;
  mutable bool poisoned = false;
};

StaticLinear StaticLinear::load_per_tensor(
    std::span<const std::byte> bytes, const ArtifactDigest& expected_digest,
    std::span<const std::byte> output_calibration_bytes,
    const ArtifactDigest& expected_output_digest, StaticLinearOptions options) {
  return backend("load_per_tensor", [&] {
    require_rounding();
    freeze_engine(options);
    const auto owned =
        snapshot(bytes, expected_digest, options, "source_digest");
    const auto output_owned =
        snapshot(output_calibration_bytes, expected_output_digest, options,
                 "output_calibration_digest");
    const auto output_calibration = q::deserialize_calibration_result(
        output_owned, options.artifact_limits);
    auto artifact =
        q::deserialize_linear_artifact(owned, options.artifact_limits);
    auto prepared =
        prepare(artifact.in_features, artifact.out_features, false,
                expected_digest, expected_output_digest, artifact.calibration,
                output_calibration, std::move(options));
    prepared.info.weight_scales = {
        q::static_int8_scale(artifact.weight_params.scale)};
    prepared.weights = std::move(artifact.quantized_weights);
    prepared.bias = std::move(artifact.bias);
    check_numeric_bounds(prepared);
    return StaticLinear(std::make_unique<Impl>(std::move(prepared)));
  });
}

StaticLinear StaticLinear::load_per_output_channel(
    std::span<const std::byte> bytes, const ArtifactDigest& expected_digest,
    std::span<const std::byte> output_calibration_bytes,
    const ArtifactDigest& expected_output_digest, StaticLinearOptions options) {
  return backend("load_per_output_channel", [&] {
    require_rounding();
    freeze_engine(options);
    const auto owned =
        snapshot(bytes, expected_digest, options, "source_digest");
    const auto output_owned =
        snapshot(output_calibration_bytes, expected_output_digest, options,
                 "output_calibration_digest");
    const auto output_calibration = q::deserialize_calibration_result(
        output_owned, options.artifact_limits);
    auto artifact = q::deserialize_per_output_channel_linear_artifact(
        owned, options.artifact_limits);
    auto prepared = prepare(artifact.in_features, artifact.out_features, true,
                            expected_digest, expected_output_digest,
                            artifact.channels.front().calibration,
                            output_calibration, std::move(options));
    prepared.weights.reserve(artifact.in_features * artifact.out_features);
    prepared.info.weight_scales.reserve(artifact.out_features);
    if (!artifact.channels.front().bias.empty())
      prepared.bias.reserve(artifact.out_features);
    for (auto& channel : artifact.channels) {
      prepared.info.weight_scales.push_back(
          q::static_int8_scale(channel.weight_params.scale));
      prepared.weights.insert(prepared.weights.end(),
                              channel.quantized_weights.begin(),
                              channel.quantized_weights.end());
      if (!channel.bias.empty()) prepared.bias.push_back(channel.bias.front());
    }
    check_numeric_bounds(prepared);
    return StaticLinear(std::make_unique<Impl>(std::move(prepared)));
  });
}

StaticLinear::StaticLinear(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
StaticLinear::~StaticLinear() = default;
StaticLinear::StaticLinear(StaticLinear&& other) noexcept = default;
StaticLinear& StaticLinear::operator=(StaticLinear&& other) noexcept = default;

StaticLinear::Impl& StaticLinear::require_impl() const {
  if (!impl_) throw NotReady("Static Linear ownership was moved away");
  return *impl_;
}

Tensor StaticLinear::quantize_input(const Tensor& input) const {
  auto& impl = require_impl();
  std::lock_guard lock(impl.mutex);
  return backend("quantize_input", [&] {
    impl.check_live();
    auto output = impl.quantize(input);
    impl.check_after_backend();
    return output;
  });
}

Tensor StaticLinear::forward_quantized(const Tensor& input) const {
  auto& impl = require_impl();
  std::lock_guard lock(impl.mutex);
  return backend("forward_quantized", [&] {
    impl.check_live();
    return impl.run(input);
  });
}

Tensor StaticLinear::forward(const Tensor& input) const {
  auto& impl = require_impl();
  std::lock_guard lock(impl.mutex);
  return backend("forward", [&] {
    impl.check_live();
    auto output = impl.run(impl.quantize(input));
    return output.dequantize();
  });
}

const StaticLinearInfo& StaticLinear::info() const {
  return require_impl().info;
}

}  // namespace jyppx::liuding::quantization_torch
