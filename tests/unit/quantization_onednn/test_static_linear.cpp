// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "integer_matmul.hpp"

#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/quantization_onednn/static_linear.hpp>

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <future>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace j = jyppx::liuding;
namespace q = j::quantization;
namespace qo = j::quantization_onednn;

struct FixtureData {
  q::QuantizedLinearArtifact tensor;
  q::PerOutputChannelLinearArtifact channel;
  q::CalibrationResult output;
  std::vector<std::byte> output_bytes;
};

FixtureData fixture(bool per_channel = false, bool with_bias = true) {
  const q::CalibrationLineage lineage{
      .dataset_version = "training-calibration-fixture/v1",
      .dataset_digest = {.hex = std::string(64U, 'b')},
      .preprocessing_fingerprint = "fixture-fp32/v1",
      .seed = 19U};
  q::CalibrationSession input_session(lineage), output_session(lineage);
  const std::array<std::array<float, 4U>, 2U> training_rows{
      {{15.875F, -15.875F, -15.375F, 15.375F},
       {-15.875F, 15.875F, 15.875F, -15.875F}}};
  for (std::size_t i = 0U; i < training_rows.size(); ++i) {
    const auto& x = training_rows[i];
    const std::array<float, 2U> y{x[0] - x[2] + (with_bias ? 0.5F : 0.F),
                                  (per_channel ? 0.25F : 1.0F) * (x[1] - x[3]) -
                                      (with_bias ? 0.5F : 0.F)};
    const auto id = "training-" + std::to_string(i);
    input_session.observe(id, x);
    output_session.observe(id, y);  // FP32 outputs; no evaluation labels.
  }
  FixtureData result;
  auto& tensor = result.tensor;
  tensor.in_features = 4U;
  tensor.out_features = 2U;
  tensor.weight_params.scale = 0.125;
  tensor.quantized_weights = {8, 0, -8, 0, 0, 8, 0, -8};
  if (with_bias) tensor.bias = {0.5F, -0.5F};
  tensor.calibration = input_session.finalize();
  result.output = output_session.finalize();
  result.output_bytes = q::serialize_calibration_result(result.output);
  result.channel.in_features = 4U;
  result.channel.out_features = 2U;
  for (std::size_t row = 0U; row < 2U; ++row) {
    auto item = tensor;
    item.out_features = 1U;
    item.weight_params.scale = row == 0U ? 0.125 : 0.03125;
    item.quantized_weights.assign(
        tensor.quantized_weights.begin() + row * 4U,
        tensor.quantized_weights.begin() + (row + 1U) * 4U);
    if (with_bias) item.bias = {tensor.bias[row]};
    result.channel.channels.push_back(std::move(item));
  }
  return result;
}

j::ArtifactDigest digest(std::span<const std::byte> bytes) {
  return {.hex = j::sha256_hex(bytes)};
}

std::uint8_t quantized_scalar(double value, double scale) {
  const double rounded = std::nearbyint(value / scale) + 128.0;
  return static_cast<std::uint8_t>(std::clamp(rounded, 0.0, 255.0));
}

// Independent arithmetic oracle: no official quantize, dequantize or Linear.
std::vector<std::uint8_t> oracle(std::span<const float> x, std::size_t rows,
                                 const FixtureData& data,
                                 const qo::StaticLinearInfo& info) {
  std::vector<std::uint8_t> result;
  for (std::size_t batch = 0U; batch < rows; ++batch) {
    for (std::size_t row = 0U; row < info.out_features; ++row) {
      std::int64_t accumulator = 0;
      for (std::size_t column = 0U; column < info.in_features; ++column) {
        const auto input =
            static_cast<std::int64_t>(quantized_scalar(
                x[batch * info.in_features + column], info.input_scale)) -
            128;
        const auto weight =
            info.per_output_channel
                ? data.channel.channels[row].quantized_weights[column]
                : data.tensor
                      .quantized_weights[row * info.in_features + column];
        accumulator += input * static_cast<std::int64_t>(weight);
      }
      const auto scale = info.weight_scales[info.per_output_channel ? row : 0U];
      const double biased =
          static_cast<double>(accumulator) * info.input_scale * scale +
          (data.tensor.bias.empty() ? 0.F : data.tensor.bias[row]);
      result.push_back(quantized_scalar(biased, info.output_scale));
    }
  }
  return result;
}

class OfficialComposedLinear : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(std::fegetround(), FE_TONEAREST);
    ASSERT_NO_THROW(qo::detail::require_sdk());
  }
};

TEST_F(OfficialComposedLinear, PerTensorExactIntegerOracleAndQuantizedOutput) {
  const auto data = fixture();
  const auto bytes = q::serialize_linear_artifact(data.tensor);
  auto runtime = qo::StaticLinear::load_per_tensor(
      bytes, digest(bytes), data.output_bytes, digest(data.output_bytes));
  const auto input = at::tensor({1.F, 2.F, 3.F, 4.F}).reshape({1, 4});
  const auto quantized =
      runtime.forward_quantized(runtime.quantize_input(input));
  EXPECT_TRUE(quantized.is_quantized());
  EXPECT_EQ(quantized.scalar_type(), at::kQUInt8);
  EXPECT_DOUBLE_EQ(quantized.q_scale(), 0.25);
  EXPECT_EQ(quantized.q_zero_point(), 128);
  EXPECT_TRUE(at::equal(quantized.dequantize(),
                        at::tensor({-1.5F, -2.5F}).reshape({1, 2})));
  EXPECT_TRUE(at::equal(runtime.forward(input), quantized.dequantize()));
  EXPECT_EQ(runtime.info().source_artifact_digest, digest(bytes));
  EXPECT_EQ(runtime.info().output_calibration_digest,
            digest(data.output_bytes));
  EXPECT_EQ(runtime.info().output_calibration.sample_ids,
            data.output.sample_ids);
  EXPECT_EQ(runtime.info().implementation, "gemm:jit");
  EXPECT_EQ(runtime.info().onednn_version, "3.12.0");
  EXPECT_NE(runtime.info().backend.find("ATen"), std::string::npos);
  EXPECT_FALSE(runtime.info().per_output_channel);
}

TEST_F(OfficialComposedLinear,
       PerChannelDistinctScalesAndRequantizationOracle) {
  const auto data = fixture(true);
  const auto bytes =
      q::serialize_per_output_channel_linear_artifact(data.channel);
  auto runtime = qo::StaticLinear::load_per_output_channel(
      bytes, digest(bytes), data.output_bytes, digest(data.output_bytes));
  const std::vector<float> values{1.F,     2.F,     3.F,      4.F,
                                  0.0625F, 0.1875F, -0.0625F, -0.1875F,
                                  -200.F,  200.F,   200.F,    -200.F};
  const auto input = at::tensor(values).reshape({3, 4});
  const auto actual = runtime.forward_quantized(runtime.quantize_input(input));
  const auto integers = actual.int_repr().contiguous();
  const auto expected = oracle(values, 3U, data, runtime.info());
  ASSERT_EQ(integers.numel(), static_cast<std::int64_t>(expected.size()));
  for (std::size_t i = 0U; i < expected.size(); ++i)
    EXPECT_EQ(integers.data_ptr<std::uint8_t>()[i], expected[i])
        << "Exact binary-scale output at " << i;
  EXPECT_TRUE(runtime.info().per_output_channel);
  EXPECT_FLOAT_EQ(runtime.info().input_scale, 0.125F);
  EXPECT_FLOAT_EQ(runtime.info().output_scale, 0.25F);
  ASSERT_EQ(runtime.info().weight_scales.size(), 2U);
  EXPECT_FLOAT_EQ(runtime.info().weight_scales[0], 0.125F);
  EXPECT_FLOAT_EQ(runtime.info().weight_scales[1], 0.03125F);
  EXPECT_TRUE(at::equal(actual.dequantize().slice(0, 0, 1),
                        at::tensor({-1.5F, -1.0F}).reshape({1, 2})));
}

TEST_F(OfficialComposedLinear, InputNearestEvenAndSaturationAreExplicit) {
  const auto data = fixture();
  const auto bytes = q::serialize_linear_artifact(data.tensor);
  auto runtime = qo::StaticLinear::load_per_tensor(
      bytes, digest(bytes), data.output_bytes, digest(data.output_bytes));
  const auto input = at::tensor({0.0625F, 0.1875F, -0.0625F, -0.1875F, -200.F,
                                 200.F, 0.F, 1.F})
                         .reshape({2, 4});
  const auto integers = runtime.quantize_input(input).int_repr().contiguous();
  const std::array<std::uint8_t, 8U> expected{128, 130, 128, 126,
                                              0,   255, 128, 136};
  ASSERT_EQ(integers.numel(), static_cast<std::int64_t>(expected.size()));
  for (std::size_t i = 0U; i < expected.size(); ++i)
    EXPECT_EQ(integers.data_ptr<std::uint8_t>()[i], expected[i]);
}

TEST_F(OfficialComposedLinear,
       FrozenOutputCalibrationAndBothDigestsAreRequired) {
  const auto data = fixture();
  const auto bytes = q::serialize_linear_artifact(data.tensor);
  auto wrong_digest = digest(bytes);
  wrong_digest.hex[0] = wrong_digest.hex[0] == '0' ? '1' : '0';
  EXPECT_THROW(
      qo::StaticLinear::load_per_tensor(bytes, wrong_digest, data.output_bytes,
                                        digest(data.output_bytes)),
      j::CompatibilityError);
  auto wrong_output_digest = digest(data.output_bytes);
  wrong_output_digest.hex[0] = wrong_output_digest.hex[0] == '0' ? '1' : '0';
  EXPECT_THROW(
      qo::StaticLinear::load_per_tensor(bytes, digest(bytes), data.output_bytes,
                                        wrong_output_digest),
      j::CompatibilityError);
  auto reordered = data.output;
  std::reverse(reordered.sample_ids.begin(), reordered.sample_ids.end());
  const auto reordered_bytes = q::serialize_calibration_result(reordered);
  EXPECT_THROW(
      qo::StaticLinear::load_per_tensor(bytes, digest(bytes), reordered_bytes,
                                        digest(reordered_bytes)),
      j::CompatibilityError);
  auto foreign = data.output;
  foreign.lineage.seed++;
  const auto foreign_bytes = q::serialize_calibration_result(foreign);
  EXPECT_THROW(qo::StaticLinear::load_per_tensor(
                   bytes, digest(bytes), foreign_bytes, digest(foreign_bytes)),
               j::CompatibilityError);
  auto corrupt = data.output_bytes;
  corrupt.front() = std::byte{0};
  EXPECT_THROW(qo::StaticLinear::load_per_tensor(bytes, digest(bytes), corrupt,
                                                 digest(corrupt)),
               j::SerializationError);
  EXPECT_THROW(
      qo::StaticLinear::load_per_tensor(bytes, digest(bytes), {}, digest({})),
      j::InvalidArgument);
  EXPECT_THROW(qo::StaticLinear::load_per_tensor(bytes, digest(bytes), bytes,
                                                 digest(bytes)),
               j::SerializationError);
  EXPECT_THROW(
      qo::StaticLinear::load_per_output_channel(
          bytes, digest(bytes), data.output_bytes, digest(data.output_bytes)),
      j::Error);
}

TEST_F(OfficialComposedLinear,
       InvalidInputsFailWithoutFloatFallbackAndRemainUsable) {
  const auto data = fixture();
  const auto bytes = q::serialize_linear_artifact(data.tensor);
  qo::StaticLinearOptions options;
  options.max_batch = 1U;
  auto runtime =
      qo::StaticLinear::load_per_tensor(bytes, digest(bytes), data.output_bytes,
                                        digest(data.output_bytes), options);
  EXPECT_THROW(runtime.forward(at::ones({1, 3})), j::InvalidArgument);
  EXPECT_THROW(runtime.forward(at::ones({2, 4})), j::ResourceExhausted);
  EXPECT_THROW(runtime.forward(at::ones({1, 4}, at::kDouble)),
               j::InvalidArgument);
  EXPECT_THROW(runtime.forward(at::ones({1, 8}).slice(1, 0, 8, 2)),
               j::InvalidArgument);
  auto nonfinite = at::ones({1, 4});
  nonfinite[0][0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(runtime.forward(nonfinite), j::NumericalError);
  nonfinite[0][0] = std::numeric_limits<float>::infinity();
  EXPECT_THROW(runtime.forward(nonfinite), j::NumericalError);
  nonfinite[0][0] = std::numeric_limits<float>::max();
  EXPECT_THROW(runtime.forward(nonfinite), j::NumericalError);
  EXPECT_THROW(runtime.forward_quantized(at::ones({1, 4})),
               j::CompatibilityError);
  const auto wrong_qparams =
      at::quantize_per_tensor(at::ones({1, 4}), 1.0, 128, at::kQUInt8);
  EXPECT_THROW(runtime.forward_quantized(wrong_qparams), j::CompatibilityError);
  EXPECT_NO_THROW(runtime.forward(at::zeros({1, 4})));
}

TEST_F(OfficialComposedLinear,
       ConcurrentCallsSerializePackedCacheAndOwnOutputs) {
  const auto data = fixture(true);
  const auto bytes =
      q::serialize_per_output_channel_linear_artifact(data.channel);
  auto runtime = qo::StaticLinear::load_per_output_channel(
      bytes, digest(bytes), data.output_bytes, digest(data.output_bytes));
  const auto invoke = [&] {
    return runtime.forward(at::tensor({1.F, 2.F, 3.F, 4.F}).reshape({1, 4}));
  };
  auto first = std::async(std::launch::async, invoke);
  auto second = std::async(std::launch::async, invoke);
  auto a = first.get();
  const auto b = second.get();
  EXPECT_TRUE(at::equal(a, b));
  a.fill_(100.0F);
  EXPECT_FALSE(at::equal(a, b));
  auto moved = std::move(runtime);
  EXPECT_THROW(runtime.info(), j::NotReady);
  EXPECT_NO_THROW(moved.forward(at::zeros({1, 4})));
}

TEST_F(OfficialComposedLinear, OutputRequantizationAndBothSaturationEnds) {
  const auto data = fixture();
  const auto bytes = q::serialize_linear_artifact(data.tensor);
  auto runtime = qo::StaticLinear::load_per_tensor(
      bytes, digest(bytes), data.output_bytes, digest(data.output_bytes));
  const std::vector<float> values{200.F,  200.F, -200.F, -200.F, -200.F, -200.F,
                                  200.F,  200.F, 0.125F, 0.F,    0.F,    0.F,
                                  0.375F, 0.F,   0.F,    0.F};
  const auto output = runtime.forward_quantized(
      runtime.quantize_input(at::tensor(values).reshape({4, 4})));
  const auto integers = output.int_repr().contiguous();
  const auto expected = oracle(values, 4U, data, runtime.info());
  ASSERT_EQ(integers.numel(), static_cast<std::int64_t>(expected.size()));
  for (std::size_t i = 0U; i < expected.size(); ++i)
    EXPECT_EQ(integers.data_ptr<std::uint8_t>()[i], expected[i]);
  EXPECT_EQ(integers.data_ptr<std::uint8_t>()[0], 255U);
  EXPECT_EQ(integers.data_ptr<std::uint8_t>()[3], 0U);
}

TEST_F(OfficialComposedLinear, OptionalBiasAndRuntimeReloadFromOwnedBytes) {
  for (const bool per_channel : {false, true}) {
    const auto data = fixture(per_channel, false);
    auto bytes =
        per_channel
            ? q::serialize_per_output_channel_linear_artifact(data.channel)
            : q::serialize_linear_artifact(data.tensor);
    auto output_bytes = data.output_bytes;
    const auto source_sha = digest(bytes), output_sha = digest(output_bytes);
    const auto load = [&] {
      return per_channel ? qo::StaticLinear::load_per_output_channel(
                               bytes, source_sha, output_bytes, output_sha)
                         : qo::StaticLinear::load_per_tensor(
                               bytes, source_sha, output_bytes, output_sha);
    };
    auto first = load();
    auto reloaded = load();
    bytes.assign(bytes.size(), std::byte{0});
    output_bytes.assign(output_bytes.size(), std::byte{0});
    const std::vector<float> values{1.F, 2.F, 3.F, 4.F};
    const auto x = at::tensor(values).reshape({1, 4});
    const auto a = first.forward_quantized(first.quantize_input(x));
    const auto b = reloaded.forward_quantized(reloaded.quantize_input(x));
    EXPECT_TRUE(at::equal(a.int_repr(), b.int_repr()));
    const auto expected = oracle(values, 1U, data, first.info());
    const auto integer = a.int_repr();
    ASSERT_EQ(integer.numel(), static_cast<std::int64_t>(expected.size()));
    for (std::size_t i = 0U; i < expected.size(); ++i)
      EXPECT_EQ(integer.data_ptr<std::uint8_t>()[i], expected[i]);
    EXPECT_EQ(reloaded.info().output_calibration_digest, output_sha);
  }
}

TEST_F(OfficialComposedLinear, ArtifactAndIoBudgetsFailExplicitly) {
  const auto data = fixture();
  const auto bytes = q::serialize_linear_artifact(data.tensor);
  qo::StaticLinearOptions options;
  options.artifact_limits.max_artifact_bytes = bytes.size() - 1U;
  EXPECT_THROW(
      qo::StaticLinear::load_per_tensor(bytes, digest(bytes), data.output_bytes,
                                        digest(data.output_bytes), options),
      j::ResourceExhausted);
  options = {};
  options.artifact_limits.max_samples = 1U;
  EXPECT_THROW(
      qo::StaticLinear::load_per_tensor(bytes, digest(bytes), data.output_bytes,
                                        digest(data.output_bytes), options),
      j::ResourceExhausted);
  options = {};
  options.max_batch = 0U;
  EXPECT_THROW(
      qo::StaticLinear::load_per_tensor(bytes, digest(bytes), data.output_bytes,
                                        digest(data.output_bytes), options),
      j::InvalidArgument);
  options = {};
  options.max_io_bytes =
      1U;  // Includes owned arrays and official packing reserve.
  EXPECT_THROW(
      qo::StaticLinear::load_per_tensor(bytes, digest(bytes), data.output_bytes,
                                        digest(data.output_bytes), options),
      j::ResourceExhausted);
  options = {};
  auto bounded =
      qo::StaticLinear::load_per_tensor(bytes, digest(bytes), data.output_bytes,
                                        digest(data.output_bytes), options);
  EXPECT_NO_THROW(bounded.forward(at::zeros({64, 4})));
  EXPECT_THROW(bounded.forward(at::zeros({65, 4})), j::ResourceExhausted);
  EXPECT_NO_THROW(bounded.forward(at::zeros({1, 4})));
}

TEST_F(OfficialComposedLinear,
       RejectsAmbientRoundingWithoutMutatingItOrRuntime) {
  const auto data = fixture();
  const auto bytes = q::serialize_linear_artifact(data.tensor);
  auto runtime = qo::StaticLinear::load_per_tensor(
      bytes, digest(bytes), data.output_bytes, digest(data.output_bytes));
  struct RestoreRounding {
    int value;
    ~RestoreRounding() { static_cast<void>(std::fesetround(value)); }
  } restore{std::fegetround()};
  for (const int mode : {FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(mode), 0);
    EXPECT_THROW(
        qo::StaticLinear::load_per_tensor(
            bytes, digest(bytes), data.output_bytes, digest(data.output_bytes)),
        j::CompatibilityError);
    EXPECT_THROW(runtime.forward(at::zeros({1, 4})), j::CompatibilityError);
    EXPECT_EQ(std::fegetround(), mode);
  }
  ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
  EXPECT_NO_THROW(runtime.forward(at::zeros({1, 4})));
}

FixtureData non_binary_fixture(std::span<const float> weights) {
  const auto lineage = fixture().tensor.calibration.lineage;
  q::CalibrationSession inputs(lineage), outputs(lineage);
  const std::array<float, 2U> bias{0.13F, -0.27F};
  const std::array<std::array<float, 4U>, 3U> training{
      {{-4.F, 3.F, 2.F, -1.F}, {1.F, -2.F, 3.F, 4.F}, {4.F, 2.F, -4.F, -2.F}}};
  for (std::size_t i = 0U; i < training.size(); ++i) {
    auto fp32 = bias;
    for (std::size_t row = 0U; row < fp32.size(); ++row)
      for (std::size_t col = 0U; col < training[i].size(); ++col)
        fp32[row] += training[i][col] * weights[row * 4U + col];
    const auto id = "training-" + std::to_string(i);
    inputs.observe(id, training[i]);
    outputs.observe(id, fp32);
  }
  const q::FloatLinearModel model{
      .in_features = 4U, .out_features = 2U, .weights = weights, .bias = bias};
  FixtureData data;
  const auto calibration = inputs.finalize();
  data.tensor = q::quantize_linear(model, calibration);
  data.channel = q::quantize_linear_per_output_channel(model, calibration);
  data.output = outputs.finalize();
  data.output_bytes = q::serialize_calibration_result(data.output);
  return data;
}

qo::StaticLinear load_fixture(const FixtureData& data, bool per_channel) {
  const auto bytes =
      per_channel
          ? q::serialize_per_output_channel_linear_artifact(data.channel)
          : q::serialize_linear_artifact(data.tensor);
  return per_channel ? qo::StaticLinear::load_per_output_channel(
                           bytes, digest(bytes), data.output_bytes,
                           digest(data.output_bytes))
                     : qo::StaticLinear::load_per_tensor(
                           bytes, digest(bytes), data.output_bytes,
                           digest(data.output_bytes));
}

FixtureData weight_fixture(std::span<const std::int8_t> weights,
                           std::size_t width) {
  auto data = fixture(false, false);
  data.tensor.in_features = width;
  data.tensor.out_features = weights.size() / width;
  data.tensor.quantized_weights.assign(weights.begin(), weights.end());
  data.channel.in_features = width;
  data.channel.out_features = data.tensor.out_features;
  data.channel.channels.clear();
  for (std::size_t row = 0U; row < data.tensor.out_features; ++row) {
    auto channel = data.tensor;
    channel.out_features = 1U;
    channel.quantized_weights.assign(weights.begin() + row * width,
                                     weights.begin() + (row + 1U) * width);
    data.channel.channels.push_back(std::move(channel));
  }
  return data;
}

TEST_F(OfficialComposedLinear, SafePairBoundariesOddWidthsAndRowsMatchOracle) {
  struct Case {
    std::vector<std::int8_t> weights;
    std::size_t width;
  };
  const std::vector<Case> cases{
      {{64, 64, -64, -64}, 2U},
      {{127, 1, -127, -1}, 2U},
      {{127, -127}, 2U},
      {{127, -127}, 1U},  // Distinct rows; each has only one weight.
      {{127, 127}, 1U},   // Same sign across rows must not be combined.
      {{-127, -127}, 1U},
      {{127, 0, 1, -127, -1, 0}, 3U},
      {{0, 0, 0}, 3U}};
  for (const auto& item : cases) {
    const auto data = weight_fixture(item.weights, item.width);
    std::vector<float> values;
    for (const float value : {-16.F, 15.875F, 0.F, 0.125F})
      for (std::size_t col = 0U; col < item.width; ++col)
        values.push_back(value);
    for (const bool per_channel : {false, true}) {
      auto runtime = load_fixture(data, per_channel);
      const auto input = at::tensor(values).reshape(
          {4, static_cast<std::int64_t>(item.width)});
      const auto actual =
          runtime.forward_quantized(runtime.quantize_input(input))
              .int_repr()
              .contiguous();
      const auto expected = oracle(values, 4U, data, runtime.info());
      ASSERT_EQ(actual.numel(), static_cast<std::int64_t>(expected.size()));
      for (std::size_t i = 0U; i < expected.size(); ++i)
        EXPECT_EQ(actual.data_ptr<std::uint8_t>()[i], expected[i]);
    }
  }
}

TEST_F(OfficialComposedLinear, NonPowerOfTwoScalesHaveBoundedOutputLsbError) {
  const std::array<float, 8U> weights{1.1F, 0.F,   -0.9F, 0.F,
                                      0.F,  0.25F, 0.F,   -0.2F};
  const auto data = non_binary_fixture(weights);
  const std::vector<float> values{0.1F, -0.2F, 0.3F,  0.7F,   -8.4F,  3.2F,
                                  1.5F, -2.F,  200.F, -200.F, -200.F, 200.F};
  for (const bool per_channel : {false, true}) {
    auto runtime = load_fixture(data, per_channel);
    const auto actual = runtime
                            .forward_quantized(runtime.quantize_input(
                                at::tensor(values).reshape({3, 4})))
                            .int_repr()
                            .contiguous();
    EXPECT_FLOAT_EQ(
        runtime.info().input_scale,
        static_cast<float>(data.tensor.calibration.activation_params.scale));
    EXPECT_FLOAT_EQ(runtime.info().output_scale,
                    static_cast<float>(data.output.activation_params.scale));
    const auto expected = oracle(values, 3U, data, runtime.info());
    ASSERT_EQ(actual.numel(), static_cast<std::int64_t>(expected.size()));
    for (std::size_t i = 0U; i < expected.size(); ++i) {
      // Official FP32 post-ops and output requantization may move a boundary
      // by one QUInt8 LSB; this is not the ORT floating-output contract.
      EXPECT_LE(std::abs(static_cast<int>(actual.data_ptr<std::uint8_t>()[i]) -
                         expected[i]),
                1)
          << "backend=" << runtime.info().backend << ", output index=" << i;
    }
  }
}

TEST_F(OfficialComposedLinear, DependencyIdentityNeedsNoBuildOrSdkFiles) {
  const auto identity = qo::StaticLinear::dependency_identity();
  EXPECT_EQ(identity.at("compiled_onednn_version"), "3.12.0");
  EXPECT_EQ(identity.at("runtime_version"), "3.12.0");
  EXPECT_EQ(identity.at("effective_cpu_isa"), "AVX2");
  EXPECT_EQ(identity.at("cpu_runtime"), "SEQ");
  EXPECT_EQ(identity.at("gpu_runtime"), "NONE");
  EXPECT_EQ(identity.at("implementation_each_branch"), "gemm:jit");
  EXPECT_EQ(identity.at("component_calls_per_forward"), "2");
  EXPECT_NE(identity.at("execution_api").find("not quantized::linear"),
            std::string::npos);
  for (const auto* field : {"library_sha256", "build_configuration_sha256",
                            "source_archive_sha256"}) {
    const auto& value = identity.at(field);
    EXPECT_EQ(value.size(), 64U) << field;
    EXPECT_EQ(value.find_first_not_of("0123456789abcdef"), std::string::npos);
  }
  EXPECT_EQ(identity.at("source_commit"),
            "80afa71049cd69a3df32adcccb623b12cd7baa22");
  const auto& runtime_hash = identity.at("runtime_source_hash");
  EXPECT_TRUE(runtime_hash.empty() || runtime_hash == "N/A" ||
              runtime_hash == identity.at("source_commit"));
  for (const auto& [key, value] : identity) {
    EXPECT_EQ(key.find("path"), std::string::npos);
    EXPECT_EQ(value.find("AppData"), std::string::npos);
    EXPECT_EQ(value.find("CMakeCache.txt"), std::string::npos);
  }
}

TEST_F(OfficialComposedLinear,
       FullRangePreviouslySaturatingRowsMatchBothFormats) {
  const std::vector<std::vector<std::int8_t>> weights{
      {64, 65},    {-64, -65},  {127, 127},    {-127, -127},
      {-88, -127}, {127, -127}, {127, 1, 127}, {-127, -1, -127}};
  for (const auto& row : weights) {
    const auto data = weight_fixture(row, row.size());
    std::vector<float> values;
    for (const auto byte : {0, 127, 128, 255})
      for (std::size_t k = 0; k < row.size(); ++k)
        values.push_back((byte - 128) * 0.125F);
    for (const bool per_channel : {false, true}) {
      auto runtime = load_fixture(data, per_channel);
      const auto raw = runtime
                           .forward_quantized(runtime.quantize_input(
                               at::tensor(values).reshape(
                                   {4, static_cast<std::int64_t>(row.size())})))
                           .int_repr()
                           .contiguous();
      const auto expected = oracle(values, 4, data, runtime.info());
      ASSERT_EQ(raw.numel(), static_cast<std::int64_t>(expected.size()));
      for (std::size_t i = 0; i < expected.size(); ++i)
        EXPECT_EQ(raw.const_data_ptr<std::uint8_t>()[i], expected[i]);
    }
  }
}

TEST_F(OfficialComposedLinear,
       Original60CenteredAnd120ComponentValuesAreExact) {
  struct Case {
    const char* name;
    std::int64_t rows, inputs, outputs;
    std::vector<std::uint8_t> src;
    std::vector<std::int8_t> weights;  // Physical [N,K], matching the task.
  };
  const std::vector<Case> cases{
      {"u8-s8-extreme-cartesian",
       4,
       1,
       5,
       {0, 127, 128, 255},
       {-128, -127, 0, 126, 127}},
      {"risk-pair-and-positive-negative-pairs",
       4,
       2,
       5,
       {0, 0, 127, 127, 128, 128, 255, 255},
       {-88, -127, -127, -127, 127, 127, -128, -128, 126, 127}},
      {"odd-k-multiple-rows-and-output-channels",
       4,
       3,
       5,
       {0, 127, 128, 255, 128, 0, 127, 255, 128, 128, 127, 255},
       {-88, -127, 127, 127, 127, 127, -128, -127, -128, 126, 0, -127, 0, 0,
        0}}};
  std::size_t compared = 0, component_compared = 0;
  for (const auto& test : cases) {
    SCOPED_TRACE(test.name);
    qo::detail::IntegerMatmul helper(test.weights, test.inputs, test.outputs,
                                     4U, 1U << 20U);
    const auto source = at::from_blob(
        const_cast<std::uint8_t*>(test.src.data()), {test.rows, test.inputs},
        at::TensorOptions().device(at::kCPU).dtype(at::kByte));
    const auto& actual = helper.execute(source);
    EXPECT_EQ(helper.selected_implementation(), "gemm:jit");
    EXPECT_LE(helper.logical_memory_bytes(), 1U << 20U);
    EXPECT_GT(helper.packing_reserve_bytes(), 0U);
    for (std::int64_t m = 0; m < test.rows; ++m) {
      for (std::int64_t n = 0; n < test.outputs; ++n) {
        std::int64_t centered = 0, low = 0, high = 0;
        // Independent test-only K reduction. The shipped helper has none.
        for (std::int64_t k = 0; k < test.inputs; ++k) {
          const auto byte = test.src[m * test.inputs + k];
          const auto weight =
              static_cast<std::int64_t>(test.weights[n * test.inputs + k]);
          centered += (static_cast<std::int64_t>(byte) - 128) * weight;
          low += static_cast<std::int64_t>(byte & 127U) * weight;
          high += static_cast<std::int64_t>(byte >> 7U) * weight;
        }
        const auto i = m * test.outputs + n;
        EXPECT_EQ(actual.const_data_ptr<std::int64_t>()[i], centered);
        EXPECT_EQ(helper.low_s32().const_data_ptr<std::int32_t>()[i], low);
        EXPECT_EQ(helper.high_s32().const_data_ptr<std::int32_t>()[i], high);
        ++compared;
        component_compared += 2;
      }
    }
  }
  EXPECT_EQ(compared, 60U);
  EXPECT_EQ(component_compared, 120U);
}

TEST_F(OfficialComposedLinear,
       ExactComponentRangeBoundaryAndBudgetsFailClosed) {
  std::vector<std::int8_t> boundary(1041, 127);
  boundary.back() = 24;  // 127*132104 == 16777208 <= 2^24.
  EXPECT_NO_THROW(
      (qo::detail::IntegerMatmul(boundary, 1041, 1, 64, 1U << 20U)));
  boundary.back() = 25;  // 127*132105 == 16777335 > 2^24.
  EXPECT_THROW((qo::detail::IntegerMatmul(boundary, 1041, 1, 64, 1U << 20U)),
               j::NumericalError);
  boundary.back() = 127;  // Original K1041 counterexample, 16790289.
  for (const bool per_channel : {false, true}) {
    const auto data = weight_fixture(boundary, boundary.size());
    EXPECT_THROW(load_fixture(data, per_channel), j::NumericalError);
  }
  EXPECT_THROW((qo::detail::IntegerMatmul({127}, 1, 1, 64, 1)),
               j::ResourceExhausted);
}

TEST_F(OfficialComposedLinear, FailedWorkspacePreparationPoisonsUntilReload) {
  const auto data = fixture();
  const auto bytes = q::serialize_linear_artifact(data.tensor);
  qo::StaticLinearOptions options;
  options.max_batch = 4096;
  options.max_io_bytes = 200000;
  auto runtime =
      qo::StaticLinear::load_per_tensor(bytes, digest(bytes), data.output_bytes,
                                        digest(data.output_bytes), options);
  EXPECT_NO_THROW(runtime.forward(at::zeros({1, 4})));
  // Client I/O is only 122880 bytes, but full split/S32/INT64 workspace exceeds
  // the ceiling. Preparation releases the old plan and refuses any execution.
  EXPECT_THROW(runtime.forward(at::zeros({4096, 4})), j::ResourceExhausted);
  EXPECT_THROW(runtime.forward(at::zeros({1, 4})), j::NotReady);
  EXPECT_THROW(runtime.quantize_input(at::zeros({1, 4})), j::NotReady);
  auto reloaded = load_fixture(data, false);
  EXPECT_NO_THROW(reloaded.forward(at::zeros({1, 4})));
  qo::detail::IntegerMatmul helper({127}, 1, 1, 4, 1U << 20U);
  EXPECT_THROW(helper.prepare_batch(5), j::ResourceExhausted);
  EXPECT_THROW(helper.low_s32(), j::NotReady);
  EXPECT_THROW(helper.selected_implementation(), j::NotReady);
}

TEST_F(OfficialComposedLinear,
       IntermediateFloatOverflowCannotBeHiddenBySmallWeightScale) {
  auto data = fixture();
  q::CalibrationSession input(data.tensor.calibration.lineage);
  const auto huge = std::numeric_limits<float>::max() * 0.5F;
  const std::array<float, 4> values{huge, -huge, 0.F, 0.F};
  input.observe("training-0", values);
  input.observe("training-1", values);
  data.tensor.calibration = input.finalize();
  data.tensor.quantized_weights = {127, 127, 0, 0, 0, 127, 127, 0};
  data.tensor.weight_params.scale = 1e-30;
  const auto bytes = q::serialize_linear_artifact(data.tensor);
  // Combined sx*sw is finite; float(Z)*sx can overflow first and must fail.
  EXPECT_THROW(
      qo::StaticLinear::load_per_tensor(bytes, digest(bytes), data.output_bytes,
                                        digest(data.output_bytes)),
      j::NumericalError);
}

TEST_F(OfficialComposedLinear,
       MovedFromObjectsAndShapeChangingConcurrentCallsOwnOutputs) {
  const auto data = fixture(true);
  auto source = load_fixture(data, true);
  auto destination = load_fixture(data, true);
  destination = std::move(source);
  EXPECT_THROW(source.info(), j::NotReady);
  EXPECT_THROW(source.quantize_input(at::zeros({1, 4})), j::NotReady);
  EXPECT_THROW(source.forward(at::zeros({1, 4})), j::NotReady);
  EXPECT_THROW(source.forward_quantized(at::zeros({1, 4})), j::NotReady);
  auto a = std::async(std::launch::async,
                      [&] { return destination.forward(at::zeros({2, 4})); });
  auto b = std::async(std::launch::async,
                      [&] { return destination.forward(at::zeros({7, 4})); });
  auto first = a.get();
  const auto second = b.get();
  EXPECT_EQ(first.size(0), 2);
  EXPECT_EQ(second.size(0), 7);
  const auto saved = second.clone();
  first.fill_(99.F);
  EXPECT_TRUE(at::equal(second, saved));
  EXPECT_NO_THROW(destination.forward(at::zeros({1, 4})));
}

}  // namespace
