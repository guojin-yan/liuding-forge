// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#include <jyppx/liuding/deployment/onnx_export/static_int8_linear.hpp>
#include <jyppx/liuding/deployment/onnx_runtime/session.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/quantization/quantization.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace {
namespace q = jyppx::liuding::quantization;
namespace ex = jyppx::liuding::onnx_export;
namespace rt = jyppx::liuding::onnx_runtime;

q::CalibrationResult Calibration(std::uint64_t seed = 20261030U) {
  q::CalibrationSession session(
      {.dataset_version = "per-output-fixture/v1",
       .dataset_digest = {.hex = std::string(64U, 'a')},
       .preprocessing_fingerprint = "identity-fp32/v1",
       .seed = seed});
  const std::array<float, 3U> values{-127.0F, 0.0F, 127.0F};
  session.observe("train/0", values);
  return session.finalize();
}

q::PerOutputChannelLinearArtifact Source(bool biased = true) {
  const std::array<float, 6U> weights{127.0F, -127.0F, 1.0F,
                                      0.125F, -0.25F,  0.375F};
  const std::array<float, 2U> bias{0.25F, -0.5F};
  return q::quantize_linear_per_output_channel(
      {.in_features = 3U,
       .out_features = 2U,
       .weights = weights,
       .bias =
           biased ? std::span<const float>(bias) : std::span<const float>{}},
      Calibration());
}

TEST(PerOutputChannelLinear, RowScalesRoundtripAndExactAggregateBudget) {
  const auto source = Source();
  ASSERT_EQ(source.channels.size(), 2U);
  EXPECT_DOUBLE_EQ(source.channels[0].weight_params.scale, 1.0);
  EXPECT_DOUBLE_EQ(source.channels[1].weight_params.scale, 0.375 / 127.0);
  EXPECT_EQ(source.channels[1].quantized_weights,
            (std::vector<std::int8_t>{42, -85, 127}));
  const auto bytes = q::serialize_per_output_channel_linear_artifact(source);
  const auto restored =
      q::deserialize_per_output_channel_linear_artifact(bytes);
  EXPECT_EQ(q::serialize_per_output_channel_linear_artifact(restored), bytes);
  q::QuantizationLimits limits;
  limits.max_artifact_bytes = bytes.size();
  EXPECT_EQ(q::serialize_per_output_channel_linear_artifact(source, limits),
            bytes);
  --limits.max_artifact_bytes;
  EXPECT_THROW(q::serialize_per_output_channel_linear_artifact(source, limits),
               jyppx::liuding::ResourceExhausted);
  EXPECT_THROW(q::deserialize_per_output_channel_linear_artifact(bytes, limits),
               jyppx::liuding::ResourceExhausted);
  limits = {};
  limits.max_features = 1U;
  EXPECT_THROW(q::validate_per_output_channel_linear_artifact(source, limits),
               jyppx::liuding::InvalidArgument);
}

TEST(PerOutputChannelLinear, RejectsMissingShapeBiasAndCalibrationMismatch) {
  auto source = Source();
  source.channels.pop_back();
  EXPECT_THROW(q::validate_per_output_channel_linear_artifact(source),
               jyppx::liuding::InvalidArgument);
  source = Source();
  source.channels[1].bias.clear();
  EXPECT_THROW(q::validate_per_output_channel_linear_artifact(source),
               jyppx::liuding::InvalidArgument);
  source = Source();
  source.channels[1].calibration = Calibration(1U);
  EXPECT_THROW(q::validate_per_output_channel_linear_artifact(source),
               jyppx::liuding::InvalidArgument);
  source = Source();
  source.channels[1].calibration.sample_ids[0] = "train/1";
  EXPECT_THROW(q::validate_per_output_channel_linear_artifact(source),
               jyppx::liuding::InvalidArgument);
  source = Source();
  source.channels[1].in_features = 2U;
  source.channels[1].quantized_weights.pop_back();
  EXPECT_THROW(q::validate_per_output_channel_linear_artifact(source),
               jyppx::liuding::InvalidArgument);
}

TEST(PerOutputChannelLinear,
     AggregateBudgetPrecedesPayloadCopyAndQuantization) {
  const auto source = Source();
  const auto bytes = q::serialize_per_output_channel_linear_artifact(source);
  q::QuantizationLimits limits;
  limits.max_artifact_bytes = bytes.size() - 1U;
  // A NaN would fail row quantization. The aggregate preflight must reject the
  // complete repeated calibration budget before visiting any borrowed weight.
  const std::array<float, 6U> weights{
      std::numeric_limits<float>::quiet_NaN(), 0, 0, 0, 0, 0};
  const std::array<float, 2U> bias{0, 0};
  EXPECT_THROW(q::quantize_linear_per_output_channel({.in_features = 3U,
                                                      .out_features = 2U,
                                                      .weights = weights,
                                                      .bias = bias},
                                                     Calibration(), limits),
               jyppx::liuding::ResourceExhausted);
}

TEST(PerOutputChannelLinear, RejectsReorderedTruncatedVersionAndTrailingBytes) {
  const auto bytes = q::serialize_per_output_channel_linear_artifact(Source());
  for (std::size_t length = 0U; length < bytes.size(); ++length)
    EXPECT_THROW(q::deserialize_per_output_channel_linear_artifact(
                     std::span(bytes).first(length)),
                 jyppx::liuding::Error)
        << length;
  auto changed = bytes;
  changed[36U] = std::byte{1U};  // First encoded row must have index zero.
  EXPECT_THROW(q::deserialize_per_output_channel_linear_artifact(changed),
               jyppx::liuding::SerializationError);
  changed = bytes;
  changed[8U] = std::byte{2U};
  EXPECT_THROW(q::deserialize_per_output_channel_linear_artifact(changed),
               jyppx::liuding::CompatibilityError);
  changed = bytes;
  changed.push_back(std::byte{0U});
  EXPECT_THROW(q::deserialize_per_output_channel_linear_artifact(changed),
               jyppx::liuding::SerializationError);
}

TEST(PerOutputChannelInt8, AggregateIdentityBindsRowOrderAndBias) {
  auto source = Source();
  const auto bytes = q::serialize_per_output_channel_linear_artifact(source);
  const jyppx::liuding::ArtifactDigest digest{
      .hex = jyppx::liuding::sha256_hex(bytes)};
  const std::array<std::int64_t, 2U> shape{1, 3};
  std::swap(source.channels[0], source.channels[1]);
  const auto reordered =
      q::serialize_per_output_channel_linear_artifact(source);
  EXPECT_THROW(ex::export_static_int8_linear_per_output_channel(reordered,
                                                                digest, shape),
               jyppx::liuding::CompatibilityError);
  source = Source();
  source.channels[1].bias[0] += 1.0F;
  const auto changed_bias =
      q::serialize_per_output_channel_linear_artifact(source);
  EXPECT_THROW(ex::export_static_int8_linear_per_output_channel(changed_bias,
                                                                digest, shape),
               jyppx::liuding::CompatibilityError);
}

TEST(PerOutputChannelInt8, CpuIntegerMatchesIndependentInt64Oracle) {
  for (const bool biased : {false, true}) {
    const auto bytes =
        q::serialize_per_output_channel_linear_artifact(Source(biased));
    const std::array<std::int64_t, 2U> shape{4, 3};
    const auto exported = ex::export_static_int8_linear_per_output_channel(
        bytes, {.hex = jyppx::liuding::sha256_hex(bytes)}, shape);
    EXPECT_EQ(exported.profile_id,
              "jyppx.linear.ptq.ort.u8u8.per_output_channel/v1");
    EXPECT_NE(
        std::find(exported.artifact.operators.begin(),
                  exported.artifact.operators.end(), "onnx::MatMulInteger"),
        exported.artifact.operators.end());
    for (const auto& op : exported.artifact.operators) {
      EXPECT_NE(op, "onnx::MatMul");
      EXPECT_NE(op, "onnx::Gemm");
      EXPECT_NE(op, "onnx::DequantizeLinear");
    }
    // Explicit nearest-even/saturated integers, independent of product
    // quantizer.
    const std::array<float, 12U> input{-1000, -3.5F, -2.5F, -1.5F, -0.5F, 0.5F,
                                       1.5F,  2.5F,  3.5F,  127,   128,   1000};
    const std::array<std::int64_t, 12U> integer{-128, -4, -2, -2,  0,   0,
                                                2,    2,  4,  127, 127, 127};
    constexpr std::array<std::array<std::int64_t, 3U>, 2U> weights{
        {{127, -127, 1}, {42, -85, 127}}};
    const std::array<float, 2U> scales{1.0F, static_cast<float>(0.375 / 127.0)};
    for (const bool disable_optimization : {false, true}) {
      rt::SessionOptions options;
      options.provider.name = "CPUExecutionProvider";
      options.allow_cpu_fallback = false;
      options.disable_graph_optimizations = disable_optimization;
      rt::Session session(exported.artifact.onnx, exported.artifact.digest,
                          options);
      EXPECT_EQ(session.runtime_version(), "1.23.2");
      const auto actual = session.run(input, shape);
      ASSERT_EQ(actual.values.size(), 8U);
      for (std::size_t row = 0U; row < 4U; ++row) {
        for (std::size_t out = 0U; out < 2U; ++out) {
          std::int64_t accumulator = 0;
          for (std::size_t in = 0U; in < 3U; ++in)
            accumulator += integer[row * 3U + in] * weights[out][in];
          const auto expected = static_cast<float>(accumulator) * scales[out] +
                                (biased ? (out == 0U ? 0.25F : -0.5F) : 0.0F);
          EXPECT_FLOAT_EQ(actual.values[row * 2U + out], expected);
        }
      }
      const std::array<std::int64_t, 2U> one{1, 3};
      EXPECT_NO_THROW(session.run(std::span(input).first(3U), one));
    }
  }
}
}  // namespace
