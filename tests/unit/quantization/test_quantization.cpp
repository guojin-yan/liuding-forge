// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/quantization/quantization.hpp>

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

namespace q = jyppx::liuding::quantization;

struct RestoreRounding {
  int mode;
  ~RestoreRounding() { static_cast<void>(std::fesetround(mode)); }
};

class MemoryState final : public jyppx::liuding::StateWriter, public jyppx::liuding::StateReader {
 public:
  explicit MemoryState(jyppx::liuding::StateManifest manifest)
      : manifest_(std::move(manifest)) {}

  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    const auto field =
        std::find_if(manifest_.fields.begin(), manifest_.fields.end(),
                     [name](const auto& value) { return value.name == name; });
    if (field == manifest_.fields.end() || fields.contains(std::string(name)))
      throw jyppx::liuding::SerializationError("Unknown or duplicate test state field");
    if (data.size() > field->max_bytes)
      throw jyppx::liuding::ResourceExhausted(
          "Test state field exceeds declared budget");
    fields.emplace(std::string(name),
                   std::vector<std::byte>(data.begin(), data.end()));
  }

  const jyppx::liuding::StateManifest& manifest() const noexcept override {
    return manifest_;
  }

  bool contains(std::string_view name) const override {
    return fields.contains(std::string(name));
  }

  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max_bytes) const override {
    const auto found = fields.find(std::string(name));
    if (found == fields.end())
      throw jyppx::liuding::SerializationError("Missing test state field");
    if (found->second.size() > max_bytes)
      throw jyppx::liuding::ResourceExhausted("Test state exceeds read budget");
    return found->second;
  }

  jyppx::liuding::StateManifest manifest_;
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
               jyppx::liuding::InvalidArgument);
  descriptor.compute_quantized = false;
  descriptor.qat_supported = true;
  EXPECT_THROW(q::validate_format_descriptor(descriptor),
               jyppx::liuding::InvalidArgument);
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
  EXPECT_THROW(q::derive_symmetric_params({}), jyppx::liuding::NotReady);
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

TEST(QuantizationReference,
     ExactHalfStepComparisonsIgnoreAmbientRoundingAndSaturateTinyScales) {
  const int original = std::fegetround();
  ASSERT_NE(original, -1);
  {
    RestoreRounding restore{original};
    q::QuantizationParams adjacent;
    adjacent.scale = std::nextafter(0.4, 0.0);
    const std::array<float, 4U> adjacent_values{-1.0F, 1.0F, 0.0F, -0.0F};
    // The exact ratio 1/nextafter(0.4,0) is greater than 2.5, although a
    // nearest-rounded double division loses that distinction and produces 2.5.
    const std::vector<std::int8_t> adjacent_expected{-3, 3, 0, 0};
    q::QuantizationParams halves;
    halves.scale = 0.5;
    const std::array<float, 8U> half_values{-1.75F, -1.25F, -0.75F, -0.25F,
                                            0.25F,  0.75F,  1.25F,  1.75F};
    const std::vector<std::int8_t> half_expected{-4, -2, -2, 0, 0, 2, 2, 4};
    q::QuantizationParams tiny;
    tiny.scale = std::numeric_limits<double>::denorm_min();
    const std::array<float, 6U> tiny_values{
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::denorm_min(),
        -0.0F,
        0.0F,
        std::numeric_limits<float>::denorm_min(),
        std::numeric_limits<float>::max()};
    const std::vector<std::int8_t> tiny_expected{-127, -127, 0, 0, 127, 127};
    for (const int mode :
         {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
      SCOPED_TRACE(mode);
      ASSERT_EQ(std::fesetround(mode), 0);
      EXPECT_EQ(q::quantize_symmetric(adjacent_values, adjacent),
                adjacent_expected);
      EXPECT_EQ(q::quantize_symmetric(half_values, halves), half_expected);
      EXPECT_EQ(q::quantize_symmetric(tiny_values, tiny), tiny_expected);
      EXPECT_EQ(std::fegetround(), mode);
    }
  }
  EXPECT_EQ(std::fegetround(), original);
}

TEST(QuantizationReference,
     ParameterScalesHaveFixedNearestEvenBitsInEveryRoundingMode) {
  const int original = std::fegetround();
  ASSERT_NE(original, -1);
  {
    RestoreRounding restore{original};
    const std::array<std::pair<std::uint32_t, std::uint64_t>, 10U> cases{{
        {0x00000000U, 0x3ff0000000000000ULL},
        {0x00000001U, 0x3630204081020408ULL},
        {0x007fffffU, 0x37a0204060c18306ULL},
        {0x00800000U, 0x37a0204081020408ULL},
        {0x3eaaaaabU, 0x3f6580560c183061ULL},
        {0x3f7fffffU, 0x3f80204070e1c387ULL},
        {0x3f800000U, 0x3f80204081020408ULL},
        {0x40800000U, 0x3fa0204081020408ULL},
        {0x42fe0000U, 0x3ff0000000000000ULL},
        {0x7f7fffffU, 0x4780204070e1c387ULL},
    }};
    for (const int mode :
         {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
      SCOPED_TRACE(mode);
      ASSERT_EQ(std::fesetround(mode), 0);
      for (const auto& [magnitude, expected] : cases) {
        SCOPED_TRACE(magnitude);
        const std::array<float, 2U> values{
            std::bit_cast<float>(magnitude),
            std::bit_cast<float>(magnitude | 0x80000000U)};
        EXPECT_EQ(std::bit_cast<std::uint64_t>(
                      q::derive_symmetric_params(values).scale),
                  expected);
      }
      q::QuantizationParams largest;
      largest.scale = std::bit_cast<double>(0x4780204070e1c387ULL);
      EXPECT_NO_THROW(q::quantize_symmetric({}, largest));
      largest.scale = std::bit_cast<double>(0x4780204070e1c388ULL);
      EXPECT_THROW(q::quantize_symmetric({}, largest), jyppx::liuding::InvalidArgument);
      EXPECT_EQ(std::fegetround(), mode);
    }
  }
  EXPECT_EQ(std::fegetround(), original);
}

TEST(QuantizationReference, NonfiniteInputsAndMalformedQparamsAreExplicit) {
  const std::array<float, 2U> values{std::numeric_limits<float>::quiet_NaN(),
                                     std::numeric_limits<float>::infinity()};
  EXPECT_THROW(q::derive_symmetric_params(values), jyppx::liuding::NotReady);
  EXPECT_THROW(q::quantize_symmetric(values, {}), jyppx::liuding::NumericalError);
  q::QuantizationParams params;
  params.scale = 0.0;
  EXPECT_THROW(q::quantize_symmetric({}, params), jyppx::liuding::InvalidArgument);
  params.scale = 1.0;
  params.zero_point = 1;
  EXPECT_THROW(q::dequantize_symmetric({}, params), jyppx::liuding::InvalidArgument);
  const std::array<std::int8_t, 1U> noncanonical{-128};
  EXPECT_THROW(q::dequantize_symmetric(noncanonical, {}),
               jyppx::liuding::InvalidArgument);
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
  EXPECT_THROW(q::compare_reference(original, {}), jyppx::liuding::InvalidArgument);
  const std::array<float, 1U> nan{std::numeric_limits<float>::quiet_NaN()};
  EXPECT_EQ(q::compare_reference(nan, nan).nonfinite_inputs, 1U);
}

TEST(QuantizationReference, ErrorReportCountsFinitePairsAndZeroSignalNoise) {
  const std::array<float, 4U> original{
      0.0F, std::numeric_limits<float>::quiet_NaN(), 1.0F,
      std::numeric_limits<float>::infinity()};
  const std::array<float, 4U> reconstructed{
      1.0F, std::numeric_limits<float>::quiet_NaN(), 1.0F, 0.0F};
  const auto mixed = q::compare_reference(original, reconstructed);
  EXPECT_EQ(mixed.count, 2U);
  EXPECT_EQ(mixed.nonfinite_inputs, 2U);
  EXPECT_DOUBLE_EQ(mixed.mae, 0.5);
  EXPECT_DOUBLE_EQ(mixed.rmse, std::sqrt(0.5));
  const std::array<float, 1U> zero{0.0F}, one{1.0F};
  const auto noisy_zero = q::compare_reference(zero, one);
  EXPECT_EQ(noisy_zero.count, 1U);
  EXPECT_EQ(noisy_zero.sqnr_db, -std::numeric_limits<double>::infinity());
  EXPECT_DOUBLE_EQ(q::compare_reference(zero, zero).sqnr_db, 0.0);
  EXPECT_EQ(q::compare_reference(one, one).sqnr_db,
            std::numeric_limits<double>::infinity());
  q::QuantizationReport report;
  report.format_id = q::QuantizationFormatDescriptor{}.format_id;
  report.weight_error = noisy_zero;
  report.calibration = lineage();
  EXPECT_NE(q::quantization_report_json(report).find("\"sqnr_db\":null"),
            std::string::npos);
}

TEST(QuantizationObserver,
     CountsNonfiniteAndEmptySamplesButObservesFiniteRange) {
  q::MinMaxObserver observer;
  EXPECT_THROW(observer.finalize(), jyppx::liuding::NotReady);
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
  restored.load_state(store, jyppx::liuding::RestorePolicy::exact_only);
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
    EXPECT_THROW(target.load_state(malformed, jyppx::liuding::RestorePolicy::exact_only),
                 jyppx::liuding::SerializationError);
    expect_same_state(target.state(), before);
  }
  auto wrong_id = good;
  wrong_id.manifest_.participant_id = "different.observer";
  EXPECT_THROW(target.load_state(wrong_id, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::CompatibilityError);
  expect_same_state(target.state(), before);
  auto invalid_bool = good;
  invalid_bool.fields.at("state")[24U] = std::byte{2U};
  EXPECT_THROW(
      target.load_state(invalid_bool, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::SerializationError);
  expect_same_state(target.state(), before);
  auto nan_minimum = good;
  const auto nan_bits =
      std::bit_cast<std::uint32_t>(std::numeric_limits<float>::quiet_NaN());
  for (std::size_t i = 0U; i < sizeof(float); ++i)
    nan_minimum.fields.at("state")[32U + i] =
        static_cast<std::byte>((nan_bits >> (8U * i)) & 0xffU);
  EXPECT_THROW(target.load_state(nan_minimum, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::SerializationError);
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
  EXPECT_THROW(session.observe("sample-b", value), jyppx::liuding::ResourceExhausted);
  EXPECT_EQ(session.finalize().sample_ids.size(), 1U);
  q::CalibrationSession duplicate(lineage());
  duplicate.observe("sample-a", value);
  EXPECT_THROW(duplicate.observe("sample-a", value), jyppx::liuding::InvalidArgument);
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
               jyppx::liuding::ResourceExhausted);
  EXPECT_THROW(q::deserialize_linear_artifact(bytes, exact),
               jyppx::liuding::ResourceExhausted);
}

TEST(QuantizationArtifact, CanonicalBytesDecodeAcrossAllAmbientRoundingModes) {
  const int original = std::fegetround();
  ASSERT_NE(original, -1);
  {
    RestoreRounding restore{original};
    std::array<std::vector<std::byte>, 2U> canonical;
    for (const int write_mode :
         {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
      SCOPED_TRACE(write_mode);
      ASSERT_EQ(std::fesetround(write_mode), 0);
      const std::array<float, 2U> extreme_values{
          -std::numeric_limits<float>::max(),
          std::numeric_limits<float>::max()};
      q::CalibrationSession extreme_session(lineage());
      extreme_session.observe("extreme-sample", extreme_values);
      const auto extreme = q::quantize_linear({.in_features = 2U,
                                               .out_features = 1U,
                                               .weights = extreme_values,
                                               .bias = {}},
                                              extreme_session.finalize());
      const std::array<std::vector<std::byte>, 2U> encoded{
          q::serialize_linear_artifact(artifact()),
          q::serialize_linear_artifact(extreme)};
      EXPECT_EQ(std::fegetround(), write_mode);
      for (std::size_t i = 0U; i < encoded.size(); ++i) {
        if (canonical[i].empty()) canonical[i] = encoded[i];
        EXPECT_EQ(encoded[i], canonical[i]);
      }
      for (const int read_mode :
           {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        SCOPED_TRACE(read_mode);
        ASSERT_EQ(std::fesetround(read_mode), 0);
        for (const auto& bytes : encoded) {
          q::QuantizedLinearArtifact decoded;
          ASSERT_NO_THROW(decoded = q::deserialize_linear_artifact(bytes));
          EXPECT_EQ(q::serialize_linear_artifact(decoded), bytes);
        }
        EXPECT_EQ(std::fegetround(), read_mode);
      }
    }
  }
  EXPECT_EQ(std::fegetround(), original);
}

TEST(QuantizationArtifact, SourceMetadataAndBudgetFailBeforeWeightConversion) {
  const auto source = artifact();
  const auto bytes = q::serialize_linear_artifact(source);
  const std::array<float, 6U> weights{
      std::numeric_limits<float>::quiet_NaN(), -2.0F, 0.0F, 0.25F, 0.5F, -1.0F};
  const q::FloatLinearModel model{.in_features = source.in_features,
                                  .out_features = source.out_features,
                                  .weights = weights,
                                  .bias = source.bias};
  q::QuantizationLimits limits;
  limits.max_artifact_bytes = bytes.size() - 1U;
  // Invalid numerical weights must not bypass a prior complete byte-budget
  // rejection or cause borrowed bias/calibration metadata to be copied first.
  EXPECT_THROW(q::quantize_linear(model, source.calibration, limits),
               jyppx::liuding::ResourceExhausted);
  limits = {};
  limits.max_samples = source.calibration.sample_ids.size() - 1U;
  EXPECT_THROW(q::quantize_linear(model, source.calibration, limits),
               jyppx::liuding::ResourceExhausted);
  limits = {};
  limits.max_lineage_text_bytes = source.format.format_id.size() - 1U;
  EXPECT_THROW(q::quantize_linear(model, source.calibration, limits),
               jyppx::liuding::ResourceExhausted);
  const std::array<float, 3U> wrong_bias{0.0F, 0.0F, 0.0F};
  auto wrong_shape = model;
  wrong_shape.bias = wrong_bias;
  EXPECT_THROW(q::quantize_linear(wrong_shape, source.calibration),
               jyppx::liuding::InvalidArgument);
}

TEST(QuantizationArtifact,
     RejectsEveryTruncatedPrefixWrongMagicVersionAndTrailingBytes) {
  const auto bytes = q::serialize_linear_artifact(artifact());
  for (std::size_t size = 0U; size < bytes.size(); ++size) {
    EXPECT_THROW(q::deserialize_linear_artifact(std::span(bytes).first(size)),
                 jyppx::liuding::Error);
  }
  auto wrong_magic = bytes;
  wrong_magic[0U] = std::byte{'X'};
  EXPECT_THROW(q::deserialize_linear_artifact(wrong_magic),
               jyppx::liuding::SerializationError);
  auto wrong_version = bytes;
  wrong_version[8U] = std::byte{2U};
  EXPECT_THROW(q::deserialize_linear_artifact(wrong_version),
               jyppx::liuding::CompatibilityError);
  auto trailing = bytes;
  trailing.push_back(std::byte{0U});
  EXPECT_THROW(q::deserialize_linear_artifact(trailing),
               jyppx::liuding::SerializationError);
}

TEST(QuantizationArtifact, RejectsShapePayloadMismatchesAndFalseClaims) {
  auto bad = artifact();
  bad.quantized_weights.pop_back();
  EXPECT_THROW(q::serialize_linear_artifact(bad), jyppx::liuding::InvalidArgument);
  bad = artifact();
  bad.format.compute_quantized = true;
  EXPECT_THROW(q::serialize_linear_artifact(bad), jyppx::liuding::InvalidArgument);
  bad = artifact();
  bad.calibration.activation_params.scale *= 2.0;
  EXPECT_THROW(q::serialize_linear_artifact(bad), jyppx::liuding::InvalidArgument);
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
               jyppx::liuding::ResourceExhausted);
  const auto count_offset = dimensions_offset + 16U + 16U;
  auto huge_count = bytes;
  for (std::size_t i = 0U; i < 8U; ++i)
    huge_count[count_offset + i] = std::byte{0xffU};
  EXPECT_THROW(q::deserialize_linear_artifact(huge_count),
               jyppx::liuding::SerializationError);
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
               jyppx::liuding::SerializationError);
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
  EXPECT_THROW(runtime.forward({}), jyppx::liuding::InvalidArgument);
}

}  // namespace
