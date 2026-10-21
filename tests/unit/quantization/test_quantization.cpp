// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/quantization/quantization.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace q = jyppx::quantization;

class MemoryState final : public jyppx::StateWriter, public jyppx::StateReader {
 public:
  explicit MemoryState(jyppx::StateManifest manifest)
      : manifest_(std::move(manifest)) {}

  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    const auto field =
        std::find_if(manifest_.fields.begin(), manifest_.fields.end(),
                     [name](const auto& value) { return value.name == name; });
    if (field == manifest_.fields.end() || fields.contains(std::string(name)))
      throw jyppx::SerializationError("Unknown or duplicate test state field");
    if (data.size() > field->max_bytes)
      throw jyppx::ResourceExhausted(
          "Test state field exceeds declared budget");
    fields.emplace(std::string(name),
                   std::vector<std::byte>(data.begin(), data.end()));
  }

  const jyppx::StateManifest& manifest() const noexcept override {
    return manifest_;
  }

  bool contains(std::string_view name) const override {
    return fields.contains(std::string(name));
  }

  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max_bytes) const override {
    const auto found = fields.find(std::string(name));
    if (found == fields.end())
      throw jyppx::SerializationError("Missing test state field");
    if (found->second.size() > max_bytes)
      throw jyppx::ResourceExhausted("Test state exceeds read budget");
    return found->second;
  }

  jyppx::StateManifest manifest_;
  std::map<std::string, std::vector<std::byte>> fields;
};

q::CalibrationLineage lineage() {
  return {.dataset_version = "calibration-fixture/v1",
          .dataset_digest = {.hex = std::string(64U, 'a')},
          .preprocessing_fingerprint = "identity-fp32/v1",
          .seed = 731U};
}

q::CalibrationResult calibration() {
  q::CalibrationSession session(lineage());
  const std::array<float, 2U> first{-1.0F, 2.0F};
  const std::array<float, 2U> second{3.0F, -4.0F};
  session.observe("sample-a", first);
  session.observe("sample-b", second);
  return session.finalize();
}

q::QuantizedLinearArtifact artifact() {
  const std::array<float, 6U> weights{1.0F, -2.0F, 0.0F, 0.25F, 0.5F, -1.0F};
  const std::array<float, 2U> bias{0.25F, -0.5F};
  return q::quantize_linear(
      {.in_features = 3U, .out_features = 2U, .weights = weights, .bias = bias},
      calibration());
}

void expect_same_state(const q::MinMaxObserverState& actual,
                       const q::MinMaxObserverState& expected) {
  EXPECT_EQ(actual.sample_count, expected.sample_count);
  EXPECT_EQ(actual.value_count, expected.value_count);
  EXPECT_EQ(actual.has_finite_value, expected.has_finite_value);
  EXPECT_FLOAT_EQ(actual.minimum, expected.minimum);
  EXPECT_FLOAT_EQ(actual.maximum, expected.maximum);
}

TEST(QuantizationFormat, ClaimsOnlyStorageQuantization) {
  q::QuantizationFormatDescriptor descriptor;
  EXPECT_NO_THROW(q::validate_format_descriptor(descriptor));
  EXPECT_FALSE(descriptor.compute_quantized);
  EXPECT_FALSE(descriptor.qat_supported);
  EXPECT_EQ(descriptor.execution_plane,
            q::QuantizationExecutionPlane::storage_only);
  descriptor.compute_quantized = true;
  EXPECT_THROW(q::validate_format_descriptor(descriptor),
               jyppx::InvalidArgument);
  descriptor.compute_quantized = false;
  descriptor.qat_supported = true;
  EXPECT_THROW(q::validate_format_descriptor(descriptor),
               jyppx::InvalidArgument);
}

TEST(QuantizationReference, SymmetricGoldenAndZeroPolicy) {
  const std::array<float, 5U> values{-127.0F, -64.0F, 0.0F, 64.0F, 127.0F};
  const auto params = q::derive_symmetric_params(values);
  EXPECT_DOUBLE_EQ(params.scale, 1.0);
  EXPECT_EQ(params.zero_point, 0);
  const auto quantized = q::quantize_symmetric(values, params);
  EXPECT_EQ(quantized, (std::vector<std::int8_t>{-127, -64, 0, 64, 127}));
  EXPECT_EQ(q::dequantize_symmetric(quantized, params),
            (std::vector<float>(values.begin(), values.end())));
  const std::array<float, 2U> zeros{0.0F, -0.0F};
  EXPECT_DOUBLE_EQ(q::derive_symmetric_params(zeros).scale, 1.0);
  EXPECT_THROW(q::derive_symmetric_params({}), jyppx::NotReady);
}

TEST(QuantizationReference, RoundToNearestEvenAndSaturate) {
  const std::array<float, 10U> values{-200.0F, -3.5F, -2.5F, -1.5F, -0.5F,
                                      0.5F,    1.5F,  2.5F,  3.5F,  200.0F};
  const std::vector<std::int8_t> expected{-127, -4, -2, -2, 0, 0, 2, 2, 4, 127};
  EXPECT_EQ(q::quantize_symmetric(values, {}), expected);
  const int original = std::fegetround();
  if (original != -1 && std::fesetround(FE_DOWNWARD) == 0) {
    EXPECT_EQ(q::quantize_symmetric(values, {}), expected);
    static_cast<void>(std::fesetround(original));
  }
}

TEST(QuantizationReference, NonfiniteInputsAndMalformedQparamsAreExplicit) {
  const std::array<float, 2U> values{std::numeric_limits<float>::quiet_NaN(),
                                     std::numeric_limits<float>::infinity()};
  EXPECT_THROW(q::derive_symmetric_params(values), jyppx::NotReady);
  EXPECT_THROW(q::quantize_symmetric(values, {}), jyppx::NumericalError);
  q::QuantizationParams params;
  params.scale = 0.0;
  EXPECT_THROW(q::quantize_symmetric({}, params), jyppx::InvalidArgument);
  params.scale = 1.0;
  params.zero_point = 1;
  EXPECT_THROW(q::dequantize_symmetric({}, params), jyppx::InvalidArgument);
  const std::array<std::int8_t, 1U> noncanonical{-128};
  EXPECT_THROW(q::dequantize_symmetric(noncanonical, {}),
               jyppx::InvalidArgument);
}

TEST(QuantizationReference, ErrorReportGolden) {
  const std::array<float, 2U> original{1.0F, -1.0F};
  const std::array<float, 2U> reconstructed{0.5F, -0.5F};
  const auto result = q::compare_reference(original, reconstructed);
  EXPECT_EQ(result.count, 2U);
  EXPECT_EQ(result.nonfinite_inputs, 0U);
  EXPECT_DOUBLE_EQ(result.rmse, 0.5);
  EXPECT_DOUBLE_EQ(result.mae, 0.5);
  EXPECT_DOUBLE_EQ(result.max_absolute_error, 0.5);
  EXPECT_DOUBLE_EQ(result.max_relative_error, 0.5);
  EXPECT_NEAR(result.sqnr_db, 6.020599913279624, 1.0e-12);
  EXPECT_NEAR(result.cosine_similarity, 1.0, 1.0e-12);
  EXPECT_THROW(q::compare_reference(original, {}), jyppx::InvalidArgument);
  const std::array<float, 1U> nan{std::numeric_limits<float>::quiet_NaN()};
  EXPECT_EQ(q::compare_reference(nan, nan).nonfinite_inputs, 1U);
}

TEST(QuantizationObserver,
     CountsNonfiniteAndEmptySamplesButObservesFiniteRange) {
  q::MinMaxObserver observer;
  EXPECT_THROW(observer.finalize(), jyppx::NotReady);
  observer.observe({});
  const std::array<float, 4U> values{std::numeric_limits<float>::quiet_NaN(),
                                     -2.0F, 4.0F,
                                     std::numeric_limits<float>::infinity()};
  observer.observe(values);
  const auto state = observer.state();
  EXPECT_EQ(state.sample_count, 2U);
  EXPECT_EQ(state.value_count, 4U);
  EXPECT_TRUE(state.has_finite_value);
  EXPECT_FLOAT_EQ(state.minimum, -2.0F);
  EXPECT_FLOAT_EQ(state.maximum, 4.0F);
  EXPECT_DOUBLE_EQ(observer.finalize().scale, 4.0 / 127.0);
}

TEST(QuantizationObserver, MergeAndCheckpointContinueExactly) {
  const std::array<float, 3U> first{-2.0F, 1.0F, 3.0F};
  const std::array<float, 2U> second{-4.0F, 0.5F};
  q::MinMaxObserver left;
  q::MinMaxObserver right;
  left.observe(first);
  right.observe(second);
  MemoryState store(left.describe_state());
  left.save_state(store);
  q::MinMaxObserver restored;
  restored.load_state(store, jyppx::RestorePolicy::exact_only);
  expect_same_state(restored.state(), left.state());
  left.merge(right);
  restored.observe(second);
  expect_same_state(restored.state(), left.state());
  EXPECT_DOUBLE_EQ(restored.finalize().scale, 4.0 / 127.0);
}

TEST(QuantizationObserver,
     RestoreRejectsIncompatibleOrMalformedStateAtomically) {
  q::MinMaxObserver source;
  const std::array<float, 2U> values{-2.0F, 4.0F};
  source.observe(values);
  MemoryState good(source.describe_state());
  source.save_state(good);
  q::MinMaxObserver target;
  const std::array<float, 1U> preexisting{7.0F};
  target.observe(preexisting);
  const auto before = target.state();
  for (std::size_t size = 0U; size < good.fields.at("state").size(); ++size) {
    auto malformed = good;
    malformed.fields.at("state").resize(size);
    EXPECT_THROW(target.load_state(malformed, jyppx::RestorePolicy::exact_only),
                 jyppx::SerializationError);
    expect_same_state(target.state(), before);
  }
  auto wrong_id = good;
  wrong_id.manifest_.participant_id = "different.observer";
  EXPECT_THROW(target.load_state(wrong_id, jyppx::RestorePolicy::exact_only),
               jyppx::CompatibilityError);
  expect_same_state(target.state(), before);
  auto invalid_bool = good;
  invalid_bool.fields.at("state")[24U] = std::byte{2U};
  EXPECT_THROW(
      target.load_state(invalid_bool, jyppx::RestorePolicy::exact_only),
      jyppx::SerializationError);
  expect_same_state(target.state(), before);
  auto nan_minimum = good;
  const auto nan_bits =
      std::bit_cast<std::uint32_t>(std::numeric_limits<float>::quiet_NaN());
  for (std::size_t i = 0U; i < sizeof(float); ++i)
    nan_minimum.fields.at("state")[32U + i] =
        static_cast<std::byte>((nan_bits >> (8U * i)) & 0xffU);
  EXPECT_THROW(target.load_state(nan_minimum, jyppx::RestorePolicy::exact_only),
               jyppx::SerializationError);
  expect_same_state(target.state(), before);
}

TEST(QuantizationCalibration,
     PreservesDatasetVersionSamplesPreprocessingAndSeed) {
  const auto result = calibration();
  EXPECT_EQ(result.lineage.dataset_version, "calibration-fixture/v1");
  EXPECT_EQ(result.lineage.dataset_digest.hex, std::string(64U, 'a'));
  EXPECT_EQ(result.lineage.preprocessing_fingerprint, "identity-fp32/v1");
  EXPECT_EQ(result.lineage.seed, 731U);
  EXPECT_EQ(result.sample_ids,
            (std::vector<std::string>{"sample-a", "sample-b"}));
  EXPECT_EQ(result.activation_state.sample_count, 2U);
  EXPECT_EQ(result.activation_state.value_count, 4U);
  EXPECT_DOUBLE_EQ(result.activation_params.scale, 4.0 / 127.0);
}

TEST(QuantizationCalibration, DuplicateAndBudgetRejectionDoNotMutateSession) {
  q::QuantizationLimits limits;
  limits.max_samples = 1U;
  q::CalibrationSession session(lineage(), limits);
  const std::array<float, 1U> value{2.0F};
  session.observe("sample-a", value);
  EXPECT_THROW(session.observe("sample-b", value), jyppx::ResourceExhausted);
  EXPECT_EQ(session.finalize().sample_ids.size(), 1U);
  q::CalibrationSession duplicate(lineage());
  duplicate.observe("sample-a", value);
  EXPECT_THROW(duplicate.observe("sample-a", value), jyppx::InvalidArgument);
  EXPECT_EQ(duplicate.finalize().activation_state.sample_count, 1U);
}

TEST(QuantizationArtifact, CanonicalBinaryRoundTripAndExactBudget) {
  const auto source = artifact();
  const auto bytes = q::serialize_linear_artifact(source);
  ASSERT_GT(bytes.size(), 128U);
  EXPECT_EQ(bytes.front(), std::byte{'J'});
  EXPECT_EQ(bytes[7U], std::byte{0U});
  const auto decoded = q::deserialize_linear_artifact(bytes);
  EXPECT_EQ(decoded.in_features, source.in_features);
  EXPECT_EQ(decoded.out_features, source.out_features);
  EXPECT_EQ(decoded.quantized_weights, source.quantized_weights);
  EXPECT_EQ(decoded.bias, source.bias);
  EXPECT_EQ(decoded.calibration.sample_ids, source.calibration.sample_ids);
  EXPECT_EQ(decoded.calibration.lineage.dataset_digest,
            source.calibration.lineage.dataset_digest);
  EXPECT_EQ(q::serialize_linear_artifact(decoded), bytes);
  q::QuantizationLimits exact;
  exact.max_artifact_bytes = bytes.size();
  EXPECT_EQ(q::serialize_linear_artifact(source, exact), bytes);
  EXPECT_NO_THROW(q::deserialize_linear_artifact(bytes, exact));
  --exact.max_artifact_bytes;
  EXPECT_THROW(q::serialize_linear_artifact(source, exact),
               jyppx::ResourceExhausted);
  EXPECT_THROW(q::deserialize_linear_artifact(bytes, exact),
               jyppx::ResourceExhausted);
}

TEST(QuantizationArtifact,
     RejectsEveryTruncatedPrefixWrongMagicVersionAndTrailingBytes) {
  const auto bytes = q::serialize_linear_artifact(artifact());
  for (std::size_t size = 0U; size < bytes.size(); ++size) {
    EXPECT_THROW(q::deserialize_linear_artifact(std::span(bytes).first(size)),
                 jyppx::Error);
  }
  auto wrong_magic = bytes;
  wrong_magic[0U] = std::byte{'X'};
  EXPECT_THROW(q::deserialize_linear_artifact(wrong_magic),
               jyppx::SerializationError);
  auto wrong_version = bytes;
  wrong_version[8U] = std::byte{2U};
  EXPECT_THROW(q::deserialize_linear_artifact(wrong_version),
               jyppx::CompatibilityError);
  auto trailing = bytes;
  trailing.push_back(std::byte{0U});
  EXPECT_THROW(q::deserialize_linear_artifact(trailing),
               jyppx::SerializationError);
}

TEST(QuantizationArtifact, RejectsShapePayloadMismatchesAndFalseClaims) {
  auto bad = artifact();
  bad.quantized_weights.pop_back();
  EXPECT_THROW(q::serialize_linear_artifact(bad), jyppx::InvalidArgument);
  bad = artifact();
  bad.format.compute_quantized = true;
  EXPECT_THROW(q::serialize_linear_artifact(bad), jyppx::InvalidArgument);
  bad = artifact();
  bad.calibration.activation_params.scale *= 2.0;
  EXPECT_THROW(q::serialize_linear_artifact(bad), jyppx::InvalidArgument);
}

TEST(QuantizationArtifact, RejectsEncodedDimensionAndCountBeforeAllocation) {
  const auto source = artifact();
  const auto bytes = q::serialize_linear_artifact(source);
  // Eight-byte magic, artifact version, descriptor string and fixed descriptor
  // fields.
  const auto dimensions_offset =
      12U + 4U + source.format.format_id.size() + 20U;
  auto huge_dimensions = bytes;
  for (std::size_t i = 0U; i < 8U; ++i)
    huge_dimensions[dimensions_offset + i] = std::byte{0xffU};
  EXPECT_THROW(q::deserialize_linear_artifact(huge_dimensions),
               jyppx::ResourceExhausted);
  const auto count_offset = dimensions_offset + 16U + 16U;
  auto huge_count = bytes;
  for (std::size_t i = 0U; i < 8U; ++i)
    huge_count[count_offset + i] = std::byte{0xffU};
  EXPECT_THROW(q::deserialize_linear_artifact(huge_count),
               jyppx::SerializationError);
}

TEST(QuantizationArtifact, RejectsNoncanonicalSignedZero) {
  auto source = artifact();
  source.bias[0U] = -0.0F;
  auto bytes = q::serialize_linear_artifact(source);
  const auto dimensions_offset =
      12U + 4U + source.format.format_id.size() + 20U;
  const auto bias_offset =
      dimensions_offset + 16U + 16U + 8U + source.quantized_weights.size() + 8U;
  EXPECT_EQ(bytes[bias_offset + 3U], std::byte{0U});
  bytes[bias_offset + 3U] = std::byte{0x80U};
  EXPECT_THROW(q::deserialize_linear_artifact(bytes),
               jyppx::SerializationError);
}

TEST(QuantizationRuntime, DequantizesAndExecutesFp32LinearWithExplicitReport) {
  const auto source = artifact();
  q::StorageQuantizedLinearRuntime runtime(source);
  EXPECT_FALSE(runtime.compute_quantized());
  EXPECT_FALSE(runtime.qat_supported());
  EXPECT_EQ(runtime.execution_path(),
            "storage-quantized-dequantize-to-fp32-linear");
  const std::array<float, 6U> input{1.0F, 2.0F, 3.0F, -1.0F, 0.5F, 2.0F};
  const auto actual = runtime.forward(input, 2U);
  ASSERT_EQ(actual.size(), 4U);
  const std::array<double, 4U> expected{-2.7421259842519685, -2.263779527559055,
                                        -1.7578740157480315,
                                        -2.515748031496063};
  for (std::size_t i = 0U; i < expected.size(); ++i)
    EXPECT_NEAR(actual[i], expected[i], 1.0e-6);
  const std::array<float, 6U> original{1.0F, -2.0F, 0.0F, 0.25F, 0.5F, -1.0F};
  const auto report = runtime.report(original);
  EXPECT_EQ(report.execution_claim, "storage-quantized");
  EXPECT_FALSE(report.compute_quantized);
  EXPECT_FALSE(report.qat);
  EXPECT_EQ(report.fallback_modules, 1U);
  EXPECT_EQ(report.unsupported_modules, 0U);
  EXPECT_EQ(report.artifact_bytes, q::serialize_linear_artifact(source).size());
  const auto json = q::quantization_report_json(report);
  EXPECT_NE(json.find("\"compute_quantized\":false"), std::string::npos);
  EXPECT_NE(json.find("\"qat\":false"), std::string::npos);
  EXPECT_NE(json.find("\"rmse\":"), std::string::npos);
  EXPECT_NE(json.find("\"seed\":731"), std::string::npos);
  EXPECT_EQ(q::quantization_report_json(report), json);
  EXPECT_THROW(runtime.forward({}), jyppx::InvalidArgument);
}

}  // namespace
