// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/deployment/onnx_export/static_int8_linear.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/quantization/quantization.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <string>
#include <vector>

#ifdef JYPPX_LIUDING_HAS_ONNX_RUNTIME
#include <jyppx/liuding/deployment/onnx_runtime/session.hpp>
#endif

namespace {

namespace ex = jyppx::liuding::onnx_export;
namespace q = jyppx::liuding::quantization;

q::QuantizedLinearArtifact Source(bool bias = true) {
  const std::array<float, 3U> values{-127.0F, 0.0F, 127.0F};
  q::CalibrationSession calibration(
      {.dataset_version = "native-static-int8-test/v1",
       .dataset_digest = {.hex = std::string(64U, 'a')},
       .preprocessing_fingerprint = "identity-fp32/v1",
       .seed = 20261029U});
  calibration.observe("train/0", values);
  const std::array<float, 6U> weights{127.0F, -127.0F, 1.0F,
                                      -1.0F,  2.0F,    -3.0F};
  const std::array<float, 2U> biases{0.25F, -0.5F};
  return q::quantize_linear({.in_features = 3U,
                             .out_features = 2U,
                             .weights = weights,
                             .bias = bias ? std::span<const float>(biases)
                                          : std::span<const float>{}},
                            calibration.finalize());
}

ex::ExportOptions Options(bool dynamic = true) {
  ex::ExportOptions result;
  result.opset = 17U;
  result.dynamic_batch = dynamic;
  result.minimum_batch = 1;
  result.maximum_batch = 64;
  return result;
}

TEST(NativeStaticInt8Export, ExactSourceIdentityAndIntegerGraph) {
  const auto source = Source();
  const auto bytes = q::serialize_linear_artifact(source);
  const jyppx::liuding::ArtifactDigest digest{
      .hex = jyppx::liuding::sha256_hex(bytes)};
  const std::array<std::int64_t, 2U> shape{1, 3};
  const auto result =
      ex::export_static_int8_linear(bytes, digest, shape, Options());
  EXPECT_EQ(result.source_artifact_digest, digest);
  EXPECT_EQ(result.profile_id, "jyppx.linear.ptq.ort.u8u8/v1");
  EXPECT_EQ(result.artifact.digest.hex,
            jyppx::liuding::sha256_hex(result.artifact.onnx));
  EXPECT_EQ(result.artifact.opset, 17U);
  EXPECT_FALSE(result.artifact.external_data);
  EXPECT_EQ(result.artifact.input_shape, (std::vector<std::int64_t>{1, 3}));
  EXPECT_EQ(result.artifact.output_shape, (std::vector<std::int64_t>{1, 2}));
  EXPECT_NE(std::find(result.artifact.operators.begin(),
                      result.artifact.operators.end(), "onnx::MatMulInteger"),
            result.artifact.operators.end());
  for (const auto& operation : result.artifact.operators) {
    EXPECT_NE(operation, "onnx::MatMul");
    EXPECT_NE(operation, "onnx::Gemm");
    EXPECT_NE(operation, "onnx::DequantizeLinear");
  }
  const auto repeated =
      ex::export_static_int8_linear(bytes, digest, shape, Options());
  EXPECT_EQ(repeated.artifact.onnx, result.artifact.onnx);
  EXPECT_EQ(q::serialize_linear_artifact(source), bytes);
}

TEST(NativeStaticInt8Export, OrtGraphIsExplicitAndHasIndependentIdentity) {
  const std::array<std::int64_t, 2U> shape{1, 3};
  for (const bool biased : {false, true}) {
    SCOPED_TRACE(biased);
    const auto bytes = q::serialize_linear_artifact(Source(biased));
    const jyppx::liuding::ArtifactDigest digest{
        .hex = jyppx::liuding::sha256_hex(bytes)};
    const auto portable =
        ex::export_static_int8_linear(bytes, digest, shape, Options());
    const auto explicit_portable =
        ex::export_static_int8_linear(bytes, digest, shape, Options(), {},
                                      ex::StaticInt8LinearGraph::portable);
    EXPECT_EQ(portable.artifact.onnx, explicit_portable.artifact.onnx);
    const auto fused = ex::export_static_int8_linear(
        bytes, digest, shape, Options(), {},
        ex::StaticInt8LinearGraph::ort_matmul_integer_to_float);
    EXPECT_EQ(fused.source_artifact_digest, digest);
    EXPECT_EQ(fused.profile_id,
              "jyppx.linear.ptq.ort.u8u8.matmul_integer_to_float/v1");
    EXPECT_NE(fused.artifact.digest, portable.artifact.digest);
    EXPECT_EQ(fused.artifact.digest.hex,
              jyppx::liuding::sha256_hex(fused.artifact.onnx));
    auto operators = std::vector<std::string>{
        "com.microsoft::MatMulIntegerToFloat", "onnx::QuantizeLinear"};
    if (biased) operators.push_back("onnx::Add");
    std::sort(operators.begin(), operators.end());
    EXPECT_EQ(fused.artifact.operators, operators);
    EXPECT_EQ(fused.artifact.input_shape, portable.artifact.input_shape);
    EXPECT_EQ(fused.artifact.output_shape, portable.artifact.output_shape);
    EXPECT_FALSE(fused.artifact.external_data);
    const auto repeated = ex::export_static_int8_linear(
        bytes, digest, shape, Options(), {},
        ex::StaticInt8LinearGraph::ort_matmul_integer_to_float);
    EXPECT_EQ(repeated.artifact.onnx, fused.artifact.onnx);
    auto exact_options = Options();
    exact_options.max_output_bytes = fused.artifact.onnx.size();
    EXPECT_NO_THROW(ex::export_static_int8_linear(
        bytes, digest, shape, exact_options, {},
        ex::StaticInt8LinearGraph::ort_matmul_integer_to_float));
    --exact_options.max_output_bytes;
    EXPECT_THROW(ex::export_static_int8_linear(
                     bytes, digest, shape, exact_options, {},
                     ex::StaticInt8LinearGraph::ort_matmul_integer_to_float),
                 jyppx::liuding::ResourceExhausted);
  }
}

TEST(NativeStaticInt8Export, RejectsUnknownGraphContract) {
  const auto bytes = q::serialize_linear_artifact(Source());
  const std::array<std::int64_t, 2U> shape{1, 3};
  EXPECT_THROW(ex::export_static_int8_linear(
                   bytes, {.hex = jyppx::liuding::sha256_hex(bytes)}, shape,
                   Options(), {}, static_cast<ex::StaticInt8LinearGraph>(255)),
               jyppx::liuding::InvalidArgument);
}

TEST(NativeStaticInt8Export,
     RejectsIdentityTruncationVersionAndCapabilityForgery) {
  const auto bytes = q::serialize_linear_artifact(Source());
  const jyppx::liuding::ArtifactDigest digest{
      .hex = jyppx::liuding::sha256_hex(bytes)};
  const std::array<std::int64_t, 2U> shape{1, 3};
  EXPECT_THROW(ex::export_static_int8_linear(
                   bytes, {.hex = std::string(64U, '0')}, shape, Options()),
               jyppx::liuding::CompatibilityError);
  auto malformed = bytes;
  malformed.pop_back();
  EXPECT_THROW(ex::export_static_int8_linear(
                   malformed, {.hex = jyppx::liuding::sha256_hex(malformed)},
                   shape, Options()),
               jyppx::liuding::Error);
  auto false_claim = Source();
  false_claim.format.compute_quantized = true;
  EXPECT_THROW(q::serialize_linear_artifact(false_claim),
               jyppx::liuding::InvalidArgument);
  auto changed_version = bytes;
  changed_version[8U] = std::byte{2U};
  EXPECT_THROW(
      ex::export_static_int8_linear(
          changed_version, {.hex = jyppx::liuding::sha256_hex(changed_version)},
          shape, Options()),
      jyppx::liuding::CompatibilityError);
}

TEST(NativeStaticInt8Export, RejectsWrongShapeUnsupportedOpsetAndOutputBudget) {
  const auto bytes = q::serialize_linear_artifact(Source());
  const jyppx::liuding::ArtifactDigest digest{
      .hex = jyppx::liuding::sha256_hex(bytes)};
  const std::array<std::int64_t, 2U> wrong_shape{1, 4};
  const std::array<std::int64_t, 2U> shape{1, 3};
  EXPECT_THROW(
      ex::export_static_int8_linear(bytes, digest, wrong_shape, Options()),
      jyppx::liuding::InvalidArgument);
  const std::array<std::int64_t, 1U> wrong_rank{3};
  EXPECT_THROW(
      ex::export_static_int8_linear(bytes, digest, wrong_rank, Options()),
      jyppx::liuding::InvalidArgument);
  auto options = Options();
  options.opset = 16U;
  EXPECT_THROW(ex::export_static_int8_linear(bytes, digest, shape, options),
               jyppx::liuding::InvalidArgument);
  options = Options();
  options.max_output_bytes = 1U;
  EXPECT_THROW(ex::export_static_int8_linear(bytes, digest, shape, options),
               jyppx::liuding::ResourceExhausted);
  q::QuantizationLimits limits;
  limits.max_artifact_bytes = bytes.size() - 1U;
  EXPECT_THROW(
      ex::export_static_int8_linear(bytes, digest, shape, Options(), limits),
      jyppx::liuding::ResourceExhausted);
}

#ifdef JYPPX_LIUDING_HAS_ONNX_RUNTIME

std::vector<float> IntegerOracle(const q::QuantizedLinearArtifact& source,
                                 std::span<const std::int32_t> input,
                                 std::size_t batch) {
  std::vector<float> output(batch * source.out_features);
  for (std::size_t row = 0U; row < batch; ++row) {
    for (std::size_t out = 0U; out < source.out_features; ++out) {
      std::int64_t accumulated = 0;
      for (std::size_t in = 0U; in < source.in_features; ++in)
        accumulated += static_cast<std::int64_t>(std::clamp(
                           input[row * source.in_features + in], -128, 127)) *
                       source.quantized_weights[out * source.in_features + in];
      output[row * source.out_features + out] =
          static_cast<float>(accumulated) +
          (source.bias.empty() ? 0.0F : source.bias[out]);
    }
  }
  return output;
}

TEST(NativeStaticInt8Runtime,
     SaturationAndTiesEvenAgreeWithIndependentInt64Oracle) {
  namespace rt = jyppx::liuding::onnx_runtime;
  for (const bool biased : {false, true}) {
    const auto source = Source(biased);
    const auto bytes = q::serialize_linear_artifact(source);
    const std::array<std::int64_t, 2U> shape{4, 3};
    const auto exported = ex::export_static_int8_linear(
        bytes, {.hex = jyppx::liuding::sha256_hex(bytes)}, shape, Options());
    const std::array<float, 12U> input{-1000.0F, -3.5F,  -2.5F,  -1.5F,
                                       -0.5F,    0.5F,   1.5F,   2.5F,
                                       3.5F,     127.0F, 128.0F, 1000.0F};
    const std::array<std::int32_t, 12U> rounded{-1000, -4, -2, -2,  0,   0,
                                                2,     2,  4,  127, 128, 1000};
    const auto expected = IntegerOracle(source, rounded, 4U);
    for (const bool disable_optimization : {false, true}) {
      rt::SessionOptions runtime_options;
      runtime_options.provider.name = "CPUExecutionProvider";
      runtime_options.disable_graph_optimizations = disable_optimization;
      rt::Session session(exported.artifact.onnx, exported.artifact.digest,
                          runtime_options);
      EXPECT_EQ(session.runtime_version(), "1.23.2");
      EXPECT_EQ(session.execution_provider().name, "CPUExecutionProvider");
      const auto actual = session.run(input, shape);
      ASSERT_EQ(actual.values.size(), expected.size());
      for (std::size_t i = 0U; i < expected.size(); ++i)
        EXPECT_FLOAT_EQ(actual.values[i], expected[i]);
      const std::array<std::int64_t, 2U> single_shape{1, 3};
      EXPECT_NO_THROW(session.run(std::span(input).first(3U), single_shape));
      auto nonfinite = input;
      nonfinite[0U] = std::numeric_limits<float>::quiet_NaN();
      EXPECT_THROW(session.run(nonfinite, shape),
                   jyppx::liuding::InvalidArgument);
    }
  }
}

TEST(NativeStaticInt8Runtime,
     FullRangeUnsignedWeightsAvoidAvx2SignedSaturation) {
  namespace rt = jyppx::liuding::onnx_runtime;
  auto source = Source(false);
  source.in_features = 128U;
  source.out_features = 2U;
  source.quantized_weights.assign(256U, 127);
  std::fill(source.quantized_weights.begin() + 128U,
            source.quantized_weights.end(), -127);
  const auto bytes = q::serialize_linear_artifact(source);
  const std::array<std::int64_t, 2U> shape{4, 128};
  const auto exported = ex::export_static_int8_linear(
      bytes, {.hex = jyppx::liuding::sha256_hex(bytes)}, shape, Options());
  const std::vector<float> input(4U * 128U, 127.0F);
  const std::vector<std::int32_t> integer_input(input.size(), 127);
  const auto expected = IntegerOracle(source, integer_input, 4U);
  rt::Session session(exported.artifact.onnx, exported.artifact.digest);
  const auto actual = session.run(input, shape);
  ASSERT_EQ(actual.values.size(), expected.size());
  for (std::size_t i = 0U; i < expected.size(); ++i)
    EXPECT_FLOAT_EQ(actual.values[i], expected[i]);
}

TEST(NativeStaticInt8Runtime, OrtFusedPreservesSaturationAndExactBiasRounding) {
  namespace rt = jyppx::liuding::onnx_runtime;
  const std::array<float, 12U> input{-1000.0F, -3.5F,  -2.5F,  -1.5F,
                                     -0.5F,    0.5F,   1.5F,   2.5F,
                                     3.5F,     127.0F, 128.0F, 1000.0F};
  const std::array<std::int32_t, 12U> rounded{-1000, -4, -2, -2,  0,   0,
                                              2,     2,  4,  127, 128, 1000};
  for (const bool biased : {false, true}) {
    SCOPED_TRACE(biased);
    const auto source = Source(biased);
    const auto bytes = q::serialize_linear_artifact(source);
    for (const std::int64_t example_batch : {1, 4}) {
      SCOPED_TRACE(example_batch);
      const std::array<std::int64_t, 2U> example_shape{example_batch, 3};
      const auto exported = ex::export_static_int8_linear(
          bytes, {.hex = jyppx::liuding::sha256_hex(bytes)}, example_shape,
          Options(), {},
          ex::StaticInt8LinearGraph::ort_matmul_integer_to_float);
      for (const bool disable_optimization : {false, true}) {
        SCOPED_TRACE(disable_optimization);
        rt::SessionOptions options;
        options.disable_graph_optimizations = disable_optimization;
        rt::Session session(exported.artifact.onnx, exported.artifact.digest,
                            options);
        // Reuse one Session across shrinking and growing batches, including
        // the CLI's batch-one export example and the original failing four.
        for (const std::size_t batch : {4U, 1U, 5U}) {
          SCOPED_TRACE(batch);
          std::vector<float> values(batch * 3U);
          std::vector<std::int32_t> integers(values.size());
          for (std::size_t i = 0U; i < values.size(); ++i) {
            values[i] = input[i % input.size()];
            integers[i] = rounded[i % rounded.size()];
          }
          const auto expected = IntegerOracle(source, integers, batch);
          const std::array<std::int64_t, 2U> shape{
              static_cast<std::int64_t>(batch), 3};
          const auto actual = session.run(values, shape);
          EXPECT_EQ(actual.shape, (std::vector<std::int64_t>{shape[0], 2}));
          ASSERT_EQ(actual.values.size(), expected.size());
          for (std::size_t i = 0U; i < expected.size(); ++i)
            EXPECT_EQ(std::bit_cast<std::uint32_t>(actual.values[i]),
                      std::bit_cast<std::uint32_t>(expected[i]));
        }
      }
    }
  }
}

TEST(NativeStaticInt8Runtime, OrtFusedNonunitScalarScaleMatchesInt64Oracle) {
  namespace rt = jyppx::liuding::onnx_runtime;
  q::CalibrationSession calibration(
      {.dataset_version = "native-nonunit-int8-test/v1",
       .dataset_digest = {.hex = std::string(64U, 'b')},
       .preprocessing_fingerprint = "identity-fp32/v1",
       .seed = 20261031U});
  const std::array<float, 3U> samples{-37.75F, 0.0F, 37.75F};
  calibration.observe("train/0", samples);
  const auto calibrated = calibration.finalize();
  const std::array<float, 6U> weights{37.125F, -37.125F, 0.0F,
                                      37.125F, 0.0F,     -37.125F};
  const std::array<float, 2U> bias{0.25F, -0.5F};
  const std::array<std::int32_t, 12U> integers{-128, -127, -3,  -1,  0,    1,
                                               3,    63,   127, 127, -128, 42};
  const std::array<std::int64_t, 2U> shape{4, 3};
  for (const bool biased : {false, true}) {
    SCOPED_TRACE(biased);
    const auto source =
        q::quantize_linear({.in_features = 3U,
                            .out_features = 2U,
                            .weights = weights,
                            .bias = biased ? std::span<const float>(bias)
                                           : std::span<const float>{}},
                           calibrated);
    ASSERT_EQ(source.quantized_weights,
              (std::vector<std::int8_t>{127, -127, 0, 127, 0, -127}));
    const auto bytes = q::serialize_linear_artifact(source);
    const auto activation_scale =
        static_cast<float>(calibrated.activation_params.scale);
    const auto weight_scale = static_cast<float>(source.weight_params.scale);
    const auto product_scale = static_cast<float>(
        static_cast<double>(activation_scale) * weight_scale);
    std::array<float, integers.size()> input{};
    for (std::size_t i = 0U; i < input.size(); ++i)
      input[i] = static_cast<float>(integers[i]) * activation_scale;
    std::array<float, 8U> expected{};
    for (std::size_t row = 0U; row < 4U; ++row) {
      for (std::size_t out = 0U; out < 2U; ++out) {
        std::int64_t accumulator = 0;
        for (std::size_t in = 0U; in < 3U; ++in)
          accumulator += static_cast<std::int64_t>(integers[row * 3U + in]) *
                         source.quantized_weights[out * 3U + in];
        // The volatile store makes the oracle's product rounding explicit.
        volatile float scaled = static_cast<float>(accumulator) * product_scale;
        expected[row * 2U + out] = biased ? scaled + bias[out] : scaled;
      }
    }
    for (const auto graph :
         {ex::StaticInt8LinearGraph::portable,
          ex::StaticInt8LinearGraph::ort_matmul_integer_to_float}) {
      SCOPED_TRACE(static_cast<unsigned>(graph));
      const auto exported = ex::export_static_int8_linear(
          bytes, {.hex = jyppx::liuding::sha256_hex(bytes)}, shape, Options(),
          {}, graph);
      for (const bool disable_optimization : {false, true}) {
        SCOPED_TRACE(disable_optimization);
        rt::SessionOptions options;
        options.disable_graph_optimizations = disable_optimization;
        rt::Session session(exported.artifact.onnx, exported.artifact.digest,
                            options);
        const auto actual = session.run(input, shape);
        ASSERT_EQ(actual.values.size(), expected.size());
        for (std::size_t i = 0U; i < expected.size(); ++i)
          EXPECT_EQ(std::bit_cast<std::uint32_t>(actual.values[i]),
                    std::bit_cast<std::uint32_t>(expected[i]));
      }
    }
  }
}

TEST(NativeStaticInt8Runtime,
     OrtFusedFullUnsignedRangeAndLargeAccumulatorMatchInt64Oracle) {
  namespace rt = jyppx::liuding::onnx_runtime;
  auto source = Source(false);
  source.in_features = 4097U;
  source.out_features = 2U;
  source.quantized_weights.assign(source.in_features * 2U, 127);
  std::fill(source.quantized_weights.begin() + source.in_features,
            source.quantized_weights.end(), -127);
  const auto bytes = q::serialize_linear_artifact(source);
  const std::array<std::int64_t, 2U> shape{3, 4097};
  const auto exported = ex::export_static_int8_linear(
      bytes, {.hex = jyppx::liuding::sha256_hex(bytes)}, shape, Options(), {},
      ex::StaticInt8LinearGraph::ort_matmul_integer_to_float);
  std::vector<float> input(3U * source.in_features, 127.0F);
  std::fill(input.begin() + source.in_features,
            input.begin() + 2U * source.in_features, -128.0F);
  for (std::size_t i = 2U * source.in_features; i < input.size(); ++i)
    input[i] = i % 2U == 0U ? -127.0F : 127.0F;
  std::vector<std::int32_t> integers(input.size());
  std::transform(input.begin(), input.end(), integers.begin(),
                 [](float value) { return static_cast<std::int32_t>(value); });
  const auto expected = IntegerOracle(source, integers, 3U);
  for (const bool disable_optimization : {false, true}) {
    SCOPED_TRACE(disable_optimization);
    rt::SessionOptions options;
    options.disable_graph_optimizations = disable_optimization;
    rt::Session session(exported.artifact.onnx, exported.artifact.digest,
                        options);
    const auto actual = session.run(input, shape);
    ASSERT_EQ(actual.values.size(), expected.size());
    for (std::size_t i = 0U; i < expected.size(); ++i)
      EXPECT_EQ(std::bit_cast<std::uint32_t>(actual.values[i]),
                std::bit_cast<std::uint32_t>(expected[i]));
  }
}

#endif

}  // namespace
