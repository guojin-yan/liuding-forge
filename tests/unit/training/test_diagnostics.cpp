// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/core/device.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/training/diagnostics.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <vector>

namespace {
namespace training = jyppx::liuding::training;
using jyppx::liuding::Tensor;

class State final : public jyppx::liuding::StateWriter,
                    public jyppx::liuding::StateReader {
 public:
  explicit State(const training::TrainingDiagnostics& participant)
      : descriptor(participant.describe_state()) {}
  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    fields[std::string(name)] = {data.begin(), data.end()};
  }
  const jyppx::liuding::StateManifest& manifest() const noexcept override {
    return descriptor;
  }
  bool contains(std::string_view name) const override {
    return fields.contains(std::string(name));
  }
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t limit) const override {
    const auto& bytes = fields.at(std::string(name));
    if (bytes.size() > limit)
      throw jyppx::liuding::ResourceExhausted("test field exceeds limit");
    return bytes;
  }
  jyppx::liuding::StateManifest descriptor;
  std::map<std::string, std::vector<std::byte>> fields;
};

TEST(TrainingDiagnostics,
     ScaledDoubleTilesMatchIndependentHypotAndPreserveStrides) {
  for (const auto tile : {1U, 3U, 64U}) {
    auto base = torch::tensor({3.0, 0.0, 4.0, 0.0, 12.0, 0.0}, torch::kFloat64);
    const auto strided = base.slice(0, 0, 6, 2);
    ASSERT_FALSE(strided.is_contiguous());
    const std::array<Tensor, 3> values{strided,
                                       torch::zeros({0}, torch::kFloat32),
                                       torch::tensor({-5.0}, torch::kFloat32)};
    const auto before = base.clone();
    const auto result =
        training::measure_gradient_l2(values, {.tile_elements = tile});
    EXPECT_NEAR(result.l2, std::hypot(13.0, 5.0), 1e-13);
    EXPECT_EQ(result.elements, 4U);
    EXPECT_LE(result.known_scratch_bytes, 8U * tile + 16U);
    EXPECT_TRUE(torch::equal(base, before));
    EXPECT_EQ(strided.stride(0), 2);
  }
}

TEST(TrainingDiagnostics, AvoidsFloat32AndFloat64SquareOverflowAndUnderflow) {
  for (const auto scale : {1e-300, 1e30, 1e300}) {
    const std::array<Tensor, 1> values{
        torch::tensor({3.0 * scale, 4.0 * scale}, torch::kFloat64)};
    const auto result =
        training::measure_gradient_l2(values, {.tile_elements = 1});
    EXPECT_NEAR(result.l2 / scale, 5.0, 1e-14);
  }
  const std::array<Tensor, 1> fp32{torch::tensor({3e30F, 4e30F})};
  EXPECT_NEAR(training::measure_gradient_l2(fp32).l2 / 1e30, 5.0, 1e-6);
}

TEST(TrainingDiagnostics, EmptyAndZeroGradientsHaveZeroNorm) {
  EXPECT_DOUBLE_EQ(training::measure_gradient_l2({}).l2, 0.0);
  const std::array<Tensor, 1> zeros{torch::zeros({2, 3})};
  EXPECT_DOUBLE_EQ(training::measure_gradient_l2(zeros).l2, 0.0);
}

TEST(TrainingDiagnostics, RejectsCompleteInputBoundsAndUnsupportedValues) {
  const std::array<Tensor, 2> values{torch::ones({5}), torch::ones({6})};
  EXPECT_THROW(training::measure_gradient_l2(values, {.max_tensors = 1}),
               jyppx::liuding::ResourceExhausted);
  EXPECT_THROW(training::measure_gradient_l2(values, {.max_elements = 10}),
               jyppx::liuding::ResourceExhausted);
  EXPECT_THROW(training::measure_gradient_l2(
                   values, {.tile_elements = 3, .max_scratch_bytes = 39}),
               jyppx::liuding::ResourceExhausted);
  EXPECT_NO_THROW(training::measure_gradient_l2(
      values,
      {.max_elements = 11, .tile_elements = 3, .max_scratch_bytes = 40}));
  const std::array<Tensor, 1> integral{torch::ones({2}, torch::kInt64)};
  EXPECT_THROW(training::measure_gradient_l2(integral),
               jyppx::liuding::InvalidArgument);
  const std::array<Tensor, 1> undefined{Tensor{}};
  EXPECT_THROW(training::measure_gradient_l2(undefined),
               jyppx::liuding::InvalidArgument);
  const std::array<Tensor, 1> sparse{torch::eye(2).to_sparse()};
  EXPECT_THROW(training::measure_gradient_l2(sparse),
               jyppx::liuding::InvalidArgument);
  for (const auto value : {std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity()}) {
    const std::array<Tensor, 1> nonfinite{
        torch::tensor({value}, torch::kFloat64)};
    EXPECT_THROW(training::measure_gradient_l2(nonfinite),
                 jyppx::liuding::NumericalError);
  }
  const std::array<Tensor, 1> too_large{
      torch::full({2}, std::numeric_limits<double>::max(), torch::kFloat64)};
  EXPECT_THROW(training::measure_gradient_l2(too_large),
               jyppx::liuding::NumericalError);
}

TEST(TrainingDiagnostics, OptionsRejectInvalidIdentityAndLimits) {
  EXPECT_THROW(training::TrainingDiagnostics({.participant_id = "../escape"}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(
      training::TrainingDiagnostics({.gradient_norm = {.tile_elements = 0}}),
      jyppx::liuding::InvalidArgument);
  EXPECT_THROW(training::TrainingDiagnostics(
                   {.gradient_norm = {.max_scratch_bytes = 23}}),
               jyppx::liuding::InvalidArgument);
}

TEST(TrainingDiagnostics,
     CurrentCheckpointValidatesTopologyManifestAndStatisticsAtomically) {
  training::TrainingDiagnostics source;
  State original(source);
  source.save_state(original);
  ASSERT_EQ(original.fields.at("statistics").size(), 184U);
  for (int corruption = 0; corruption < 6; ++corruption) {
    auto changed = original;
    if (corruption == 0) changed.descriptor.schema_version = 2;
    if (corruption == 1) changed.fields.at("topology").back() ^= std::byte{1};
    if (corruption == 2) changed.fields.at("statistics").pop_back();
    if (corruption == 3)
      changed.fields.at("statistics")[16] =
          std::byte{1};  // NaN > observations.
    if (corruption == 4)
      changed.fields.at("statistics")[128] =
          std::byte{1};  // Samples with no attempts.
    if (corruption == 5) {
      const auto value = std::bit_cast<std::array<std::byte, 8>>(
          std::numeric_limits<double>::infinity());
      std::copy(value.begin(), value.end(),
                changed.fields.at("statistics").begin() + 144);
    }
    training::TrainingDiagnostics target;
    EXPECT_THROW(
        target.load_state(changed, jyppx::liuding::RestorePolicy::exact_only),
        jyppx::liuding::Error);
    State after(target);
    target.save_state(after);
    EXPECT_EQ(after.fields, original.fields);
  }
  training::TrainingDiagnostics changed_options({.gradient_norm_every = 2});
  EXPECT_THROW(changed_options.load_state(
                   original, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::CompatibilityError);
  EXPECT_NO_THROW(
      source.load_state(original, jyppx::liuding::RestorePolicy::exact_only));
}

TEST(TrainingDiagnostics, MetricDefinitionsHaveExplicitStableSemantics) {
  const auto bindings = training::training_diagnostic_metric_bindings();
  ASSERT_EQ(bindings.size(), 10U);
  for (const auto& [key, binding] : bindings) {
    SCOPED_TRACE(key);
    EXPECT_NO_THROW(jyppx::liuding::validate_telemetry_metric_definition(
        binding.definition));
    EXPECT_EQ(binding.definition.aggregation,
              jyppx::liuding::TelemetryAggregation::last);
    EXPECT_FALSE(binding.definition.parameters.empty());
  }
}

TEST(TrainingDiagnosticsCuda, NoncontiguousHalfAndDoubleTilesUseRealCuda) {
  if (!jyppx::liuding::cuda_is_available()) GTEST_SKIP() << "CUDA unavailable";
  for (const auto dtype :
       {torch::kHalf, torch::kBFloat16, torch::kFloat32, torch::kFloat64}) {
    auto base =
        torch::tensor({3.0, 0.0, 4.0, 0.0},
                      torch::TensorOptions().dtype(dtype).device(torch::kCUDA));
    const std::array<Tensor, 1> values{base.slice(0, 0, 4, 2)};
    const auto before = base.clone();
    EXPECT_DOUBLE_EQ(
        training::measure_gradient_l2(values, {.tile_elements = 1}).l2, 5.0);
    EXPECT_TRUE(torch::equal(base, before));
  }
}
}  // namespace
