// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "integer_matmul.hpp"

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/quantization_onednn/onednn_build_identity.hpp>
#include <jyppx/liuding/quantization_onednn/static_linear.hpp>

#include <torch/version.h>

#include <oneapi/dnnl/dnnl.hpp>
#if DNNL_VERSION_MAJOR != 3 || DNNL_VERSION_MINOR != 12 || \
    DNNL_VERSION_PATCH != 0
#error "This profile requires official oneDNN 3.12.0"
#endif
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SEQ || \
    DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE
#error "This profile requires official SEQ CPU / NONE GPU"
#endif

#include <ATen/ATen.h>
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <string_view>
#include <utility>

namespace jyppx::liuding::quantization_onednn {

namespace {

namespace q = quantization;
constexpr double kSafeQuantizerMagnitude = 1073741824.0;  // 2^30, before int32.

ErrorDetails details(std::string operation) {
  return {
      .component = "quantization_onednn.static_linear",
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
        "Official composite allocation failed", details(operation)));
  } catch (const dnnl::error&) {
    std::throw_with_nested(BackendError("Official composite oneDNN call failed",
                                        details(operation)));
  } catch (const c10::Error&) {
    std::throw_with_nested(BackendError(
        "Official ATen composite operation failed", details(operation)));
  }
}

void require_rounding() {
  if (std::fegetround() != FE_TONEAREST)
    throw CompatibilityError("Static INT8 Linear requires nearest-even mode",
                             details("rounding_mode"));
}

void check_options(const StaticLinearOptions& options) {
  q::validate_limits(options.artifact_limits);
  if (options.max_batch == 0U || options.max_io_bytes == 0U)
    throw InvalidArgument("Static Linear budgets must be positive",
                          details("limits"));
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
  // The Q0 owner may accept wider configurable limits. This profile bounds
  // dimensions to 65536 and separately enforces the shared helper's
  // conservative 127*sum(abs(int64(weight))) <= 2^24 exact-component bound,
  // including -128.
  if (inputs == 0U || outputs == 0U || inputs > 65536U || outputs > 65536U)
    throw InvalidArgument("Static INT8 dimensions must be in [1,65536]",
                          details("dimensions"));
  PreparedLinear result;
  result.info.profile_id =
      per_channel ? "jyppx.linear.ptq.onednn_aten.avx2.u8s8.7plus1.per_output_"
                    "channel/v1"
                  : "jyppx.linear.ptq.onednn_aten.avx2.u8s8.7plus1/v1";
  result.info.backend = "official oneDNN matmul + ATen INT64 AVX2 7+1";
  result.info.implementation = "gemm:jit";
  result.info.onednn_version = "3.12.0";
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
    const auto sw = info.weight_scales[info.per_output_channel ? row : 0U];
    const double product = static_cast<double>(info.input_scale) * sw;
    static_cast<void>(q::static_int8_scale(product));
    static_cast<void>(q::static_int8_scale(product / info.output_scale));
    std::uint64_t magnitude = 0U;
    // Metadata bound only; the runtime's K-dimensional dot is never
    // handwritten.
    for (std::size_t column = 0U; column < info.in_features; ++column)
      magnitude += static_cast<std::uint64_t>(std::abs(
          static_cast<int>(prepared.weights[row * info.in_features + column])));
    // Every actual F32 stage is also checked below. A finite combined sx*sw
    // alone does not establish that float(Z)*sx remains finite.
    const double stage1 =
        static_cast<double>(magnitude) * 128.0 * info.input_scale;
    const double stage2 = stage1 * sw;
    const double stage3 =
        stage2 + (prepared.bias.empty()
                      ? 0.0
                      : std::abs(static_cast<double>(prepared.bias[row])));
    const double stage4 = stage3 / info.output_scale;
    const double stage5 = stage4 + 128.0;
    const double limit = std::numeric_limits<float>::max();
    if (!std::isfinite(stage1) || !std::isfinite(stage2) ||
        !std::isfinite(stage3) || !std::isfinite(stage4) ||
        !std::isfinite(stage5) || stage1 > limit || stage2 > limit ||
        stage3 > limit || stage4 > limit || stage5 > limit ||
        stage4 > kSafeQuantizerMagnitude)
      throw NumericalError(
          "Composite output exceeds an intermediate F32/requantizer bound",
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
  // Early logical I/O check. The shared helper additionally accounts for
  // split/S32/INT64 buffers, persistent compensation, client scalars, actual
  // scratch and conservative AVX2 packing reserve before batch allocation.
  const auto row_bytes = 5ULL * (info.in_features + info.out_features);
  if (batch > options.max_batch || row_bytes > options.max_io_bytes ||
      batch > options.max_io_bytes / row_bytes)
    throw ResourceExhausted("Static INT8 logical I/O budget exceeded",
                            details("input_budget"));
}

}  // namespace

struct StaticLinear::Impl {
  explicit Impl(PreparedLinear prepared)
      : info(std::move(prepared.info)),
        options(std::move(prepared.options)),
        bias(std::move(prepared.bias)) {
    require_rounding();
    detail::require_sdk();
    // 4-byte FP32 input + 1-byte quantized input; 1-byte quantized output +
    // 4-byte final dequantization. MxN epilogue uses bounded stack scalars,
    // included in the helper's 256-byte fixed allowance, no float batch copy.
    const auto client_rows = 5ULL * (info.in_features + info.out_features);
    const auto client_fixed =
        static_cast<std::uint64_t>(info.weight_scales.size() + bias.size()) *
        sizeof(float);
    integer = std::make_unique<detail::IntegerMatmul>(
        std::move(prepared.weights), info.in_features, info.out_features,
        options.max_batch, options.max_io_bytes, client_fixed, client_rows);
  }

  void check_live() const {
    if (poisoned)
      throw NotReady("Official composite failed; reload the runtime");
    require_rounding();
    detail::require_sdk();
  }
  void check_after_backend() const {
    try {
      detail::require_sdk();
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
    const double low = minimum.item<double>(), high = maximum.item<double>();
    if (!std::isfinite(low) || !std::isfinite(high))
      throw NumericalError("Static INT8 input contains NaN or infinity",
                           details("input_finite"));
    if (std::max(std::abs(low), std::abs(high)) / info.input_scale >
        kSafeQuantizerMagnitude)
      throw NumericalError("Input exceeds safe official quantizer range",
                           details("input_range"));
    // Validate complete live batch/workspace budget before producing q-input.
    try {
      integer->prepare_batch(input.size(0));
      return at::quantize_per_tensor(input, info.input_scale,
                                     info.input_zero_point, at::kQUInt8);
    } catch (...) {
      poisoned = true;
      throw;
    }
  }

  Tensor run(const Tensor& input) const {
    check_shape(input, info, options);
    if (!input.is_quantized() || input.scalar_type() != at::kQUInt8 ||
        input.qscheme() != at::kPerTensorAffine ||
        input.q_scale() != static_cast<double>(info.input_scale) ||
        input.q_zero_point() != 128 || info.input_zero_point != 128 ||
        info.output_zero_point != 128)
      throw CompatibilityError(
          "Quantized input qparams do not match calibration",
          details("input_qparams"));
    try {
      static_assert(sizeof(c10::quint8) == sizeof(std::uint8_t));
      // Borrow the exact logical input bytes; from_blob neither changes nor
      // requantizes them. Input stays live/immutable through both stream waits.
      auto raw = at::from_blob(
          const_cast<c10::quint8*>(input.const_data_ptr<c10::quint8>()),
          {input.size(0), static_cast<std::int64_t>(info.in_features)},
          at::TensorOptions().device(at::kCPU).dtype(at::kByte));
      const auto& combined = integer->execute(raw);
      const auto* accumulator = combined.const_data_ptr<std::int64_t>();
      auto output = at::_empty_affine_quantized(
          {input.size(0), static_cast<std::int64_t>(info.out_features)},
          at::TensorOptions().device(at::kCPU).dtype(at::kQUInt8),
          info.output_scale, info.output_zero_point);
      auto* destination = output.data_ptr<c10::quint8>();
      // MxN output processing only. The K reduction already completed in two
      // official oneDNN calls and official ATen INT64 composition above.
      // Preserve the sequential reference order; no combined sx*sw or FMA,
      // no per-component floating conversion before integer reconstruction.
      for (std::int64_t m = 0; m < input.size(0); ++m) {
        for (std::size_t n = 0; n < info.out_features; ++n) {
          const auto index =
              static_cast<std::size_t>(m) * info.out_features + n;
          volatile float converted = static_cast<float>(accumulator[index]);
          volatile float sx_scaled = converted * info.input_scale;
          volatile float sw_scaled =
              sx_scaled * info.weight_scales[info.per_output_channel ? n : 0U];
          volatile float biased = sw_scaled + (bias.empty() ? 0.0F : bias[n]);
          volatile float divided = biased / info.output_scale;
          volatile float shifted = divided + 128.0F;
          if (!std::isfinite(static_cast<float>(converted)) ||
              !std::isfinite(static_cast<float>(sx_scaled)) ||
              !std::isfinite(static_cast<float>(sw_scaled)) ||
              !std::isfinite(static_cast<float>(biased)) ||
              !std::isfinite(static_cast<float>(divided)) ||
              !std::isfinite(static_cast<float>(shifted)))
            throw NumericalError(
                "Composite epilogue produced a nonfinite F32 stage",
                details("output_finite"));
          const auto clipped =
              std::clamp(static_cast<float>(shifted), 0.0F, 255.0F);
          destination[index].val_ =
              static_cast<std::uint8_t>(std::nearbyint(clipped));
        }
      }
      require_rounding();
      detail::require_sdk();
      return output;
    } catch (...) {
      poisoned = true;
      throw;
    }
  }

  StaticLinearInfo info;
  StaticLinearOptions options;
  std::vector<float> bias;
  std::unique_ptr<detail::IntegerMatmul> integer;
  mutable std::mutex mutex;
  mutable bool poisoned = false;
};

StaticLinear StaticLinear::load_per_tensor(
    std::span<const std::byte> bytes, const ArtifactDigest& expected_digest,
    std::span<const std::byte> output_calibration_bytes,
    const ArtifactDigest& expected_output_digest, StaticLinearOptions options) {
  return backend("load_per_tensor", [&] {
    require_rounding();
    detail::require_sdk();
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
    detail::require_sdk();
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

std::map<std::string, std::string> StaticLinear::dependency_identity() {
  detail::require_sdk();
  const auto* version = dnnl_version();
  auto checked_hash = [](const char* value) {
    const std::string result(value);
    if (result.size() != 64U ||
        result.find_first_not_of("0123456789abcdef") != std::string::npos)
      throw CompatibilityError(
          "Compiled oneDNN provenance has an invalid SHA-256");
    return result;
  };
  return {{"backend", "official oneDNN C++ matmul + ATen INT64 AVX2 7+1"},
          {"execution_api",
           "two oneDNN matmul calls + ATen INT64; not quantized::linear"},
          {"implementation_each_branch", "gemm:jit"},
          {"component_calls_per_forward", "2"},
          {"composition", "q=(q&127)+128*(q>>7); Z=Cl+128*Ch-128*sum(w)"},
          {"component_precision_guard", "127*sum(abs(int64(w)))<=2^24"},
          {"compiled_onednn_version", "3.12.0"},
          {"compiled_libtorch_version", TORCH_VERSION},
          {"library_sha256", checked_hash(build_identity::library_sha256)},
          {"build_configuration_sha256",
           checked_hash(build_identity::build_configuration_sha256)},
          {"source_commit", build_identity::source_commit},
          {"source_archive_sha256",
           checked_hash(build_identity::source_archive_sha256)},
          {"runtime_version", std::to_string(version->major) + "." +
                                  std::to_string(version->minor) + "." +
                                  std::to_string(version->patch)},
          {"runtime_source_hash", version->hash ? version->hash : ""},
          {"cpu_runtime", "SEQ"},
          {"gpu_runtime", "NONE"},
          {"effective_cpu_isa", "AVX2"},
          {"blas_vendor", "NONE"},
          {"provenance_scope",
           "compiled SDK identity; no runtime build/cache files required"}};
}

}  // namespace jyppx::liuding::quantization_onednn
