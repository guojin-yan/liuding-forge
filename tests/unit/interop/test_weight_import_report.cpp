// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/interop/weight_import_report.hpp>

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using jyppx::liuding::interop::plan_weight_import;
using jyppx::liuding::interop::SafeTensorDtype;
using jyppx::liuding::interop::WeightImportLimits;
using jyppx::liuding::interop::WeightImportMode;
using jyppx::liuding::interop::WeightImportOptions;
using jyppx::liuding::interop::WeightSkipReason;
using jyppx::liuding::interop::WeightTensorDescriptor;

jyppx::liuding::ArtifactDigest SourceDigest() {
  return {.algorithm = jyppx::liuding::ArtifactDigestAlgorithm::sha256,
          .hex = std::string(64U, 'a')};
}

WeightTensorDescriptor Tensor(std::string name,
                              std::vector<std::uint64_t> shape = {2U},
                              SafeTensorDtype dtype = SafeTensorDtype::f32) {
  return {.name = std::move(name), .dtype = dtype, .shape = std::move(shape)};
}

TEST(WeightImportReportTest, StrictMapsExactKeysAndPreservesOriginalIndices) {
  const std::vector source{Tensor("encoder.weight", {2U, 3U}),
                           Tensor("head.bias", {})};
  const std::vector target{source[1], source[0]};
  const auto plan = plan_weight_import(SourceDigest(), source, target);
  const auto& report = plan.report();
  ASSERT_TRUE(report.ready);
  EXPECT_EQ(report.mode, WeightImportMode::strict);
  EXPECT_EQ(report.source_digest, SourceDigest());
  EXPECT_EQ(report.matched,
            (std::vector<std::string>{"encoder.weight", "head.bias"}));
  EXPECT_EQ(report.source_element_count, 7U);
  EXPECT_EQ(report.matched_element_count, 7U);
  EXPECT_DOUBLE_EQ(report.tensor_coverage, 1.0);
  EXPECT_DOUBLE_EQ(report.element_coverage, 1.0);
  const auto assignments = plan.assignments();
  ASSERT_EQ(assignments.size(), 2U);
  EXPECT_EQ(assignments[0].source_index, 0U);
  EXPECT_EQ(assignments[0].target_index, 1U);
  EXPECT_EQ(assignments[1].source_index, 1U);
  EXPECT_EQ(assignments[1].target_index, 0U);
  EXPECT_TRUE(assignments[0].conversion_rule.empty());
  const auto json = plan.report_json();
  EXPECT_NE(json.find("\"phase\":\"preflight\""), std::string::npos);
  EXPECT_NE(json.find("\"schema_version\":1"), std::string::npos);
  EXPECT_NE(json.find("\"unassigned_destination_action\":\"retain-existing\""),
            std::string::npos);
  EXPECT_EQ(json.find("\"applied\""), std::string::npos);
}

TEST(WeightImportReportTest, CanonicalReportDoesNotDependOnInventoryOrder) {
  std::vector source{Tensor("z"), Tensor("a", {1U, 2U}), Tensor("unused")};
  std::vector target{Tensor("a", {2U, 1U}), Tensor("z"), Tensor("missing")};
  const auto before =
      plan_weight_import(SourceDigest(), source, target).report_json();
  std::reverse(source.begin(), source.end());
  std::reverse(target.begin(), target.end());
  EXPECT_EQ(plan_weight_import(SourceDigest(), source, target).report_json(),
            before);
}

TEST(WeightImportReportTest,
     StrictCollectsAllMismatchesBeforeRejectingEveryAssignment) {
  const std::vector source{Tensor("good"), Tensor("shape", {3U}),
                           Tensor("dtype", {2U}, SafeTensorDtype::f16),
                           Tensor("extra")};
  const std::vector target{Tensor("good"), Tensor("shape", {2U}),
                           Tensor("dtype"), Tensor("absent")};
  const auto plan = plan_weight_import(SourceDigest(), source, target);
  const auto& report = plan.report();
  EXPECT_FALSE(report.ready);
  EXPECT_EQ(report.matched, (std::vector<std::string>{"good"}));
  EXPECT_EQ(report.missing, (std::vector<std::string>{"absent"}));
  EXPECT_EQ(report.unexpected, (std::vector<std::string>{"extra"}));
  ASSERT_EQ(report.shape_mismatches.size(), 1U);
  EXPECT_EQ(report.shape_mismatches[0].key, "shape");
  EXPECT_EQ(report.shape_mismatches[0].source_shape,
            (std::vector<std::uint64_t>{3U}));
  EXPECT_EQ(report.shape_mismatches[0].target_shape,
            (std::vector<std::uint64_t>{2U}));
  ASSERT_EQ(report.dtype_mismatches.size(), 1U);
  EXPECT_EQ(report.dtype_mismatches[0].key, "dtype");
  EXPECT_TRUE(report.dtype_mismatches[0].conversion_rule.empty());
  EXPECT_DOUBLE_EQ(report.tensor_coverage, 0.25);
  EXPECT_THROW(plan.assignments(), jyppx::liuding::CompatibilityError);
  // Metadata and caller topology are read-only even when some earlier key
  // matched. No early per-key assignment can escape the rejected complete plan.
  EXPECT_EQ(source[1].shape, (std::vector<std::uint64_t>{3U}));
  EXPECT_EQ(target[1].shape, (std::vector<std::uint64_t>{2U}));
}

TEST(WeightImportReportTest,
     PartialExplicitlyRecordsConversionsAndRetainedDestination) {
  const std::vector source{Tensor("encoder", {6U}),
                           Tensor("head", {2U}, SafeTensorDtype::f16),
                           Tensor("old_classifier")};
  const std::vector target{Tensor("head"), Tensor("new_classifier", {4U}),
                           Tensor("encoder", {6U})};
  WeightImportOptions options;
  options.mode = WeightImportMode::partial;
  options.minimum_tensor_coverage = 0.6;
  options.conversions = {{.key = "head",
                          .source_dtype = SafeTensorDtype::f16,
                          .target_dtype = SafeTensorDtype::f32,
                          .rule_id = "float-cast.v1"}};
  const auto plan = plan_weight_import(SourceDigest(), source, target, options);
  const auto& report = plan.report();
  ASSERT_TRUE(report.ready);
  EXPECT_TRUE(report.coverage_satisfied);
  EXPECT_DOUBLE_EQ(report.tensor_coverage, 2.0 / 3.0);
  EXPECT_DOUBLE_EQ(report.element_coverage, 8.0 / 12.0);
  EXPECT_EQ(report.unexpected, (std::vector<std::string>{"old_classifier"}));
  ASSERT_EQ(report.conversions.size(), 1U);
  EXPECT_EQ(report.conversions[0].rule_id, "float-cast.v1");
  ASSERT_EQ(report.dtype_mismatches.size(), 1U);
  EXPECT_EQ(report.dtype_mismatches[0].conversion_rule, "float-cast.v1");
  ASSERT_EQ(report.retained_targets.size(), 1U);
  EXPECT_EQ(report.retained_targets[0].key, "new_classifier");
  EXPECT_EQ(report.retained_targets[0].reason, WeightSkipReason::missing);
  ASSERT_EQ(plan.assignments().size(), 2U);
  EXPECT_EQ(plan.assignments()[1].key, "head");
  EXPECT_EQ(plan.assignments()[1].source_index, 1U);
  EXPECT_EQ(plan.assignments()[1].target_index, 0U);
  EXPECT_EQ(plan.assignments()[1].conversion_rule, "float-cast.v1");
}

TEST(WeightImportReportTest, PartialDoesNotImplicitlyCastOrReshape) {
  const std::vector source{Tensor("dtype", {2U}, SafeTensorDtype::bf16),
                           Tensor("shape", {1U, 2U})};
  const std::vector target{Tensor("dtype"), Tensor("shape")};
  WeightImportOptions options;
  options.mode = WeightImportMode::partial;
  const auto plan = plan_weight_import(SourceDigest(), source, target, options);
  ASSERT_TRUE(plan.report().ready);  // Explicit partial with zero minimum.
  EXPECT_TRUE(plan.assignments().empty());
  EXPECT_DOUBLE_EQ(plan.report().tensor_coverage, 0.0);
  EXPECT_DOUBLE_EQ(plan.report().element_coverage, 0.0);
  ASSERT_EQ(plan.report().retained_targets.size(), 2U);
  EXPECT_EQ(plan.report().retained_targets[0].reason,
            WeightSkipReason::dtype_mismatch);
  EXPECT_EQ(plan.report().retained_targets[1].reason,
            WeightSkipReason::shape_mismatch);
  options.minimum_tensor_coverage = 0.1;
  const auto rejected =
      plan_weight_import(SourceDigest(), source, target, options);
  EXPECT_FALSE(rejected.report().coverage_satisfied);
  EXPECT_FALSE(rejected.report().ready);
  EXPECT_THROW(rejected.assignments(), jyppx::liuding::CompatibilityError);
}

TEST(WeightImportReportTest, DuplicateKeysAlwaysRejectEvenExplicitPartial) {
  const std::vector source{Tensor("source_duplicate"),
                           Tensor("source_duplicate"),
                           Tensor("target_duplicate"), Tensor("unique")};
  const std::vector target{Tensor("source_duplicate"),
                           Tensor("target_duplicate"),
                           Tensor("target_duplicate"), Tensor("unique")};
  WeightImportOptions options;
  options.mode = WeightImportMode::partial;
  const auto plan = plan_weight_import(SourceDigest(), source, target, options);
  const auto& report = plan.report();
  EXPECT_FALSE(report.ready);
  EXPECT_EQ(report.duplicate_source,
            (std::vector<std::string>{"source_duplicate"}));
  EXPECT_EQ(report.duplicate_target,
            (std::vector<std::string>{"target_duplicate"}));
  EXPECT_EQ(report.matched, (std::vector<std::string>{"unique"}));
  EXPECT_EQ(report.retained_targets[0].reason,
            WeightSkipReason::duplicate_source);
  EXPECT_EQ(report.retained_targets[1].reason,
            WeightSkipReason::duplicate_target);
  EXPECT_THROW(plan.assignments(), jyppx::liuding::CompatibilityError);
}

TEST(WeightImportReportTest,
     ConversionDeclarationsMustMatchOneRealPairExactly) {
  const std::vector source{Tensor("x", {2U}, SafeTensorDtype::f16)};
  const std::vector target{Tensor("x")};
  WeightImportOptions options;
  options.conversions = {{.key = "x",
                          .source_dtype = SafeTensorDtype::f16,
                          .target_dtype = SafeTensorDtype::f32,
                          .rule_id = "cast.v1"}};
  EXPECT_THROW(plan_weight_import(SourceDigest(), source, target, options),
               jyppx::liuding::InvalidArgument);
  options.mode = WeightImportMode::partial;
  EXPECT_TRUE(plan_weight_import(SourceDigest(), source, target, options)
                  .report()
                  .ready);
  auto invalid = options;
  invalid.conversions[0].key = "stale";
  EXPECT_THROW(plan_weight_import(SourceDigest(), source, target, invalid),
               jyppx::liuding::InvalidArgument);
  invalid = options;
  invalid.conversions[0].source_dtype = SafeTensorDtype::bf16;
  EXPECT_THROW(plan_weight_import(SourceDigest(), source, target, invalid),
               jyppx::liuding::InvalidArgument);
  invalid = options;
  invalid.conversions.push_back(invalid.conversions[0]);
  EXPECT_THROW(plan_weight_import(SourceDigest(), source, target, invalid),
               jyppx::liuding::InvalidArgument);
  invalid = options;
  invalid.conversions[0].rule_id = "arbitrary code()";
  EXPECT_THROW(plan_weight_import(SourceDigest(), source, target, invalid),
               jyppx::liuding::InvalidArgument);
  auto wrong_shape = target;
  wrong_shape[0].shape = {1U, 2U};
  EXPECT_THROW(plan_weight_import(SourceDigest(), source, wrong_shape, options),
               jyppx::liuding::InvalidArgument);
}

TEST(WeightImportReportTest,
     ScalarsEmptyTensorsAndEmptyModelsHaveExplicitCoverage) {
  const std::vector inventory{Tensor("empty", {2U, 0U, 3U}),
                              Tensor("scalar", {})};
  const auto plan = plan_weight_import(SourceDigest(), inventory, inventory);
  EXPECT_TRUE(plan.report().ready);
  EXPECT_EQ(plan.report().target_tensor_count, 2U);
  EXPECT_EQ(plan.report().target_element_count, 1U);
  EXPECT_DOUBLE_EQ(plan.report().element_coverage, 1.0);
  const auto empty = plan_weight_import(SourceDigest(), {}, {});
  EXPECT_TRUE(empty.report().ready);
  EXPECT_TRUE(empty.assignments().empty());
  EXPECT_DOUBLE_EQ(empty.report().tensor_coverage, 1.0);
  EXPECT_DOUBLE_EQ(empty.report().element_coverage, 1.0);
}

TEST(WeightImportReportTest, RejectsInvalidMetadataPoliciesAndDigest) {
  const std::vector valid{Tensor("x")};
  WeightImportOptions options;
  options.mode = static_cast<WeightImportMode>(99);
  EXPECT_THROW(plan_weight_import(SourceDigest(), valid, valid, options),
               jyppx::liuding::InvalidArgument);
  options.mode = WeightImportMode::partial;
  for (const auto value :
       {-0.1, 1.1, (std::numeric_limits<double>::quiet_NaN)(),
        (std::numeric_limits<double>::infinity)()}) {
    options.minimum_tensor_coverage = value;
    EXPECT_THROW(plan_weight_import(SourceDigest(), valid, valid, options),
                 jyppx::liuding::InvalidArgument);
  }
  auto invalid = valid;
  invalid[0].name = std::string("\xc0\xaf");
  EXPECT_THROW(plan_weight_import(SourceDigest(), invalid, valid),
               jyppx::liuding::InvalidArgument);
  invalid[0].name = "";
  EXPECT_THROW(plan_weight_import(SourceDigest(), invalid, valid),
               jyppx::liuding::InvalidArgument);
  invalid[0] = valid[0];
  invalid[0].dtype = static_cast<SafeTensorDtype>(255);
  EXPECT_THROW(plan_weight_import(SourceDigest(), invalid, valid),
               jyppx::liuding::InvalidArgument);
  auto digest = SourceDigest();
  digest.hex[0] = 'A';
  EXPECT_THROW(plan_weight_import(digest, valid, valid),
               jyppx::liuding::InvalidArgument);
}

TEST(WeightImportReportTest, BoundsEveryInventoryAndSerializedReport) {
  const std::vector inventory{Tensor("first", {2U, 2U}),
                              Tensor("second", {3U})};
  const auto constrained = [&](WeightImportLimits limits) {
    EXPECT_THROW(
        plan_weight_import(SourceDigest(), inventory, inventory, {}, limits),
        jyppx::liuding::ResourceExhausted);
  };
  WeightImportLimits limits;
  limits.max_tensors = 1U;
  constrained(limits);
  limits = {};
  limits.max_rank = 1U;
  constrained(limits);
  limits = {};
  limits.max_dimension = 1U;
  constrained(limits);
  limits = {};
  limits.max_tensor_elements = 2U;
  constrained(limits);
  limits = {};
  limits.max_total_elements = 5U;
  constrained(limits);
  limits = {};
  limits.max_string_bytes = 2U;
  constrained(limits);
  limits = {};
  limits.max_total_string_bytes = 8U;
  constrained(limits);
  limits = {};
  limits.max_report_bytes = 16U;
  const auto plan =
      plan_weight_import(SourceDigest(), inventory, inventory, {}, limits);
  EXPECT_THROW(plan.report_json(), jyppx::liuding::ResourceExhausted);
  limits = {};
  limits.max_rank = 0U;
  EXPECT_THROW(
      plan_weight_import(SourceDigest(), inventory, inventory, {}, limits),
      jyppx::liuding::InvalidArgument);
}

TEST(WeightImportReportTest, ChecksShapeAndDtypeByteProductsBeforePlanning) {
  WeightImportLimits limits;
  limits.max_dimension = (std::numeric_limits<std::uint64_t>::max)();
  limits.max_tensor_elements = (std::numeric_limits<std::uint64_t>::max)();
  limits.max_total_elements = (std::numeric_limits<std::uint64_t>::max)();
  const std::vector overflow{
      Tensor("x", {(std::numeric_limits<std::uint64_t>::max)(), 2U})};
  EXPECT_THROW(plan_weight_import(SourceDigest(), overflow, {}, {}, limits),
               jyppx::liuding::InvalidArgument);
  const std::vector byte_overflow{
      Tensor("x", {(std::numeric_limits<std::uint64_t>::max)()},
             SafeTensorDtype::f64)};
  EXPECT_THROW(
      plan_weight_import(SourceDigest(), byte_overflow, {}, {}, limits),
      jyppx::liuding::InvalidArgument);
}

}  // namespace
