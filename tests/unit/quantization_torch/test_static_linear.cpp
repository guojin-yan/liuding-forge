// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/quantization_torch/static_linear.hpp>

#include <ATen/Context.h>
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <future>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace j = jyppx::liuding;
namespace q = j::quantization;
namespace qt = j::quantization_torch;

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
                                 const qt::StaticLinearInfo& info) {
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

class OfficialStaticLinear : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(std::fegetround(), FE_TONEAREST);
    const auto capabilities = qt::quantized_cpu_capabilities();
    if ((capabilities.current_engine != "ONEDNN" &&
         capabilities.current_engine != "X86") ||
        !capabilities.linear_prepack_registered ||
        !capabilities.linear_registered ||
        std::find(capabilities.supported_engines.begin(),
                  capabilities.supported_engines.end(),
                  capabilities.current_engine) ==
            capabilities.supported_engines.end())
      GTEST_SKIP() << "This lane does not expose the ONEDNN/X86 profile; "
                      "a skip grants no platform qualification.";
  }
};

TEST(OfficialQuantizedCapabilities, DoesNotChangeGlobalEngine) {
  const auto before = at::globalContext().qEngine();
  const auto observed = qt::quantized_cpu_capabilities();
  EXPECT_EQ(at::globalContext().qEngine(), before);
  EXPECT_EQ(observed.current_engine, c10::toString(before));
  EXPECT_FALSE(observed.libtorch_version.empty());
}

TEST_F(OfficialStaticLinear, PerTensorExactIntegerOracleAndQuantizedOutput) {
  const auto data = fixture();
  const auto bytes = q::serialize_linear_artifact(data.tensor);
  const auto engine = at::globalContext().qEngine();
  auto runtime = qt::StaticLinear::load_per_tensor(
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
  EXPECT_EQ(at::globalContext().qEngine(), engine);
  EXPECT_EQ(runtime.info().engine, c10::toString(engine));
  EXPECT_FALSE(runtime.info().per_output_channel);
}

TEST_F(OfficialStaticLinear, PerChannelDistinctScalesAndRequantizationOracle) {
  const auto data = fixture(true);
  const auto bytes =
      q::serialize_per_output_channel_linear_artifact(data.channel);
  auto runtime = qt::StaticLinear::load_per_output_channel(
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

TEST_F(OfficialStaticLinear, InputNearestEvenAndSaturationAreExplicit) {
  const auto data = fixture();
  const auto bytes = q::serialize_linear_artifact(data.tensor);
  auto runtime = qt::StaticLinear::load_per_tensor(
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

TEST_F(OfficialStaticLinear, FrozenOutputCalibrationAndBothDigestsAreRequired) {
  const auto data = fixture();
  const auto bytes = q::serialize_linear_artifact(data.tensor);
  auto wrong_digest = digest(bytes);
  wrong_digest.hex[0] = wrong_digest.hex[0] == '0' ? '1' : '0';
  EXPECT_THROW(
      qt::StaticLinear::load_per_tensor(bytes, wrong_digest, data.output_bytes,
                                        digest(data.output_bytes)),
      j::CompatibilityError);
  auto wrong_output_digest = digest(data.output_bytes);
  wrong_output_digest.hex[0] = wrong_output_digest.hex[0] == '0' ? '1' : '0';
  EXPECT_THROW(
      qt::StaticLinear::load_per_tensor(bytes, digest(bytes), data.output_bytes,
                                        wrong_output_digest),
      j::CompatibilityError);
  auto reordered = data.output;
  std::reverse(reordered.sample_ids.begin(), reordered.sample_ids.end());
  const auto reordered_bytes = q::serialize_calibration_result(reordered);
  EXPECT_THROW(
      qt::StaticLinear::load_per_tensor(bytes, digest(bytes), reordered_bytes,
                                        digest(reordered_bytes)),
      j::CompatibilityError);
  auto foreign = data.output;
  foreign.lineage.seed++;
  const auto foreign_bytes = q::serialize_calibration_result(foreign);
  EXPECT_THROW(qt::StaticLinear::load_per_tensor(
                   bytes, digest(bytes), foreign_bytes, digest(foreign_bytes)),
               j::CompatibilityError);
  auto corrupt = data.output_bytes;
  corrupt.front() = std::byte{0};
  EXPECT_THROW(qt::StaticLinear::load_per_tensor(bytes, digest(bytes), corrupt,
                                                 digest(corrupt)),
               j::SerializationError);
  EXPECT_THROW(
      qt::StaticLinear::load_per_tensor(bytes, digest(bytes), {}, digest({})),
      j::InvalidArgument);
  EXPECT_THROW(qt::StaticLinear::load_per_tensor(bytes, digest(bytes), bytes,
                                                 digest(bytes)),
               j::SerializationError);
  EXPECT_THROW(
      qt::StaticLinear::load_per_output_channel(
          bytes, digest(bytes), data.output_bytes, digest(data.output_bytes)),
      j::Error);
  qt::StaticLinearOptions wrong_engine;
  wrong_engine.expected_engine = "FBGEMM";
  EXPECT_THROW(qt::StaticLinear::load_per_tensor(
                   bytes, digest(bytes), data.output_bytes,
                   digest(data.output_bytes), wrong_engine),
               j::NotImplemented);
}

TEST_F(OfficialStaticLinear,
       InvalidInputsFailWithoutFloatFallbackAndRemainUsable) {
  const auto data = fixture();
  const auto bytes = q::serialize_linear_artifact(data.tensor);
  qt::StaticLinearOptions options;
  options.max_batch = 1U;
  auto runtime =
      qt::StaticLinear::load_per_tensor(bytes, digest(bytes), data.output_bytes,
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

TEST_F(OfficialStaticLinear, ConcurrentCallsSerializePackedCacheAndOwnOutputs) {
  const auto data = fixture(true);
  const auto bytes =
      q::serialize_per_output_channel_linear_artifact(data.channel);
  auto runtime = qt::StaticLinear::load_per_output_channel(
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

TEST_F(OfficialStaticLinear,
       ChangedGlobalEngineFailsBeforeOfficialCallWhenAvailable) {
  const auto old = at::globalContext().qEngine();
  const auto& engines = at::Context::supportedQEngines();
  const auto other =
      std::find_if(engines.begin(), engines.end(), [old](auto engine) {
        return engine != old && engine != c10::QEngine::NoQEngine;
      });
  if (other == engines.end())
    GTEST_SKIP() << "No second supported engine; drift test unqualified.";
  const auto data = fixture();
  const auto bytes = q::serialize_linear_artifact(data.tensor);
  auto runtime = qt::StaticLinear::load_per_tensor(
      bytes, digest(bytes), data.output_bytes, digest(data.output_bytes));
  struct RestoreEngine {
    c10::QEngine value;
    ~RestoreEngine() { at::globalContext().setQEngine(value); }
  } restore{old};
  at::globalContext().setQEngine(*other);  // Test-owned isolated process only.
  EXPECT_THROW(runtime.forward(at::zeros({1, 4})), j::CompatibilityError);
}

TEST_F(OfficialStaticLinear, OutputRequantizationAndBothSaturationEnds) {
  const auto data = fixture();
  const auto bytes = q::serialize_linear_artifact(data.tensor);
  auto runtime = qt::StaticLinear::load_per_tensor(
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

TEST_F(OfficialStaticLinear, OptionalBiasAndRuntimeReloadFromOwnedBytes) {
  for (const bool per_channel : {false, true}) {
    const auto data = fixture(per_channel, false);
    auto bytes =
        per_channel
            ? q::serialize_per_output_channel_linear_artifact(data.channel)
            : q::serialize_linear_artifact(data.tensor);
    auto output_bytes = data.output_bytes;
    const auto source_sha = digest(bytes), output_sha = digest(output_bytes);
    const auto load = [&] {
      return per_channel ? qt::StaticLinear::load_per_output_channel(
                               bytes, source_sha, output_bytes, output_sha)
                         : qt::StaticLinear::load_per_tensor(
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

TEST_F(OfficialStaticLinear, ArtifactAndIoBudgetsFailExplicitly) {
  const auto data = fixture();
  const auto bytes = q::serialize_linear_artifact(data.tensor);
  qt::StaticLinearOptions options;
  options.artifact_limits.max_artifact_bytes = bytes.size() - 1U;
  EXPECT_THROW(
      qt::StaticLinear::load_per_tensor(bytes, digest(bytes), data.output_bytes,
                                        digest(data.output_bytes), options),
      j::ResourceExhausted);
  options = {};
  options.artifact_limits.max_samples = 1U;
  EXPECT_THROW(
      qt::StaticLinear::load_per_tensor(bytes, digest(bytes), data.output_bytes,
                                        digest(data.output_bytes), options),
      j::ResourceExhausted);
  options = {};
  options.max_batch = 0U;
  EXPECT_THROW(
      qt::StaticLinear::load_per_tensor(bytes, digest(bytes), data.output_bytes,
                                        digest(data.output_bytes), options),
      j::InvalidArgument);
  options = {};
  options.max_io_bytes = 29U;  // One row needs 5*(4+2) logical bytes.
  auto bounded =
      qt::StaticLinear::load_per_tensor(bytes, digest(bytes), data.output_bytes,
                                        digest(data.output_bytes), options);
  EXPECT_THROW(bounded.forward(at::zeros({1, 4})), j::ResourceExhausted);
  options.max_io_bytes = 30U;
  auto exact =
      qt::StaticLinear::load_per_tensor(bytes, digest(bytes), data.output_bytes,
                                        digest(data.output_bytes), options);
  EXPECT_NO_THROW(exact.forward(at::zeros({1, 4})));
}

TEST_F(OfficialStaticLinear, RejectsAmbientRoundingWithoutMutatingItOrRuntime) {
  const auto data = fixture();
  const auto bytes = q::serialize_linear_artifact(data.tensor);
  auto runtime = qt::StaticLinear::load_per_tensor(
      bytes, digest(bytes), data.output_bytes, digest(data.output_bytes));
  struct RestoreRounding {
    int value;
    ~RestoreRounding() { static_cast<void>(std::fesetround(value)); }
  } restore{std::fegetround()};
  for (const int mode : {FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(mode), 0);
    EXPECT_THROW(
        qt::StaticLinear::load_per_tensor(
            bytes, digest(bytes), data.output_bytes, digest(data.output_bytes)),
        j::CompatibilityError);
    EXPECT_THROW(runtime.forward(at::zeros({1, 4})), j::CompatibilityError);
    EXPECT_EQ(std::fegetround(), mode);
  }
  ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
  EXPECT_NO_THROW(runtime.forward(at::zeros({1, 4})));
}

TEST_F(OfficialStaticLinear,
       ExpectedEngineConstrainsWithoutSwitchingGlobalState) {
  const auto data = fixture();
  const auto bytes = q::serialize_linear_artifact(data.tensor);
  const auto current = at::globalContext().qEngine();
  const auto name = std::string(c10::toString(current));
  qt::StaticLinearOptions options;
  options.expected_engine = name == "ONEDNN" ? "X86" : "ONEDNN";
  EXPECT_THROW(
      qt::StaticLinear::load_per_tensor(bytes, digest(bytes), data.output_bytes,
                                        digest(data.output_bytes), options),
      j::CompatibilityError);
  EXPECT_EQ(at::globalContext().qEngine(), current);
  options.expected_engine = name;
  auto runtime =
      qt::StaticLinear::load_per_tensor(bytes, digest(bytes), data.output_bytes,
                                        digest(data.output_bytes), options);
  EXPECT_EQ(runtime.info().engine, name);
  EXPECT_EQ(at::globalContext().qEngine(), current);
}

TEST_F(OfficialStaticLinear, NonPowerOfTwoScalesHaveBoundedOutputLsbError) {
  const auto lineage = fixture().tensor.calibration.lineage;
  q::CalibrationSession inputs(lineage), outputs(lineage);
  const std::array<float, 8U> weights{1.1F,  0.1F,  -0.9F,  0.3F,
                                      0.03F, 0.25F, -0.17F, -0.2F};
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
  const std::vector<float> values{0.1F, -0.2F, 0.3F,  0.7F,   -8.4F,  3.2F,
                                  1.5F, -2.F,  200.F, -200.F, -200.F, 200.F};
  for (const bool per_channel : {false, true}) {
    const auto bytes =
        per_channel
            ? q::serialize_per_output_channel_linear_artifact(data.channel)
            : q::serialize_linear_artifact(data.tensor);
    auto runtime = per_channel ? qt::StaticLinear::load_per_output_channel(
                                     bytes, digest(bytes), data.output_bytes,
                                     digest(data.output_bytes))
                               : qt::StaticLinear::load_per_tensor(
                                     bytes, digest(bytes), data.output_bytes,
                                     digest(data.output_bytes));
    const auto actual = runtime
                            .forward_quantized(runtime.quantize_input(
                                at::tensor(values).reshape({3, 4})))
                            .int_repr()
                            .contiguous();
    EXPECT_FLOAT_EQ(runtime.info().input_scale,
                    static_cast<float>(calibration.activation_params.scale));
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
          << "engine=" << runtime.info().engine << ", output index=" << i;
    }
  }
}

}  // namespace
