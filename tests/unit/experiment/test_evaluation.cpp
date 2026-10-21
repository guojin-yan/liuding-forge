// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/experiment/evaluation.hpp>
#include <jyppx/foundation/error.hpp>

#include <array>
#include <chrono>
#include <gtest/gtest.h>
#include <limits>
#include <stop_token>

namespace {
using namespace jyppx;
using namespace jyppx::experiment;

ArtifactDigest digest(char value) { return {.hex = std::string(64U, value)}; }

EvaluationSuite suite() {
  TelemetryMetricDefinition definition;
  definition.metric_id = "classification.accuracy";
  definition.unit = TelemetryMetricUnit::ratio;
  definition.aggregation = TelemetryAggregation::mean;
  definition.direction = TelemetryMetricDirection::maximize;
  definition.parameters = {{"average", "micro"}};
  definition.context_keys = {"split"};
  return {.schema_version = 1U,
          .suite_id = "classification.smoke",
          .version = 1U,
          .task = "classification",
          .dataset_digest = digest('a'),
          .environment_id = "cpu-f64-deterministic",
          .seed = 71U,
          .rules = {{.rule_id = "validation.accuracy",
                     .definition = definition,
                     .context = {{"split", "validation"}},
                     .minimum_samples = 10U,
                     .minimum = 0.8,
                     .maximum = 1.0,
                     .max_absolute_regression = 0.01,
                     .max_relative_regression = 0.02}}};
}

EvaluationIdentity identity() {
  return {.evaluation_id = "eval-1",
          .run_id = "training-1",
          .model_digest = digest('b'),
          .build_fingerprint = "test-build",
          .adapter_id = "classification.local",
          .adapter_version = 1U,
          .config_digest = digest('c')};
}

TelemetryMetricPoint point(double value = 0.9) {
  const auto value_suite = suite();
  return {.definition = value_suite.rules.front().definition,
          .step = {.kind = TelemetryStepKind::sample, .value = 10U},
          .observation = make_telemetry_metric_value(value),
          .context = value_suite.rules.front().context,
          .sample_count = 10U};
}

QualityReport report(double value = 0.9) {
  return evaluate_observations(suite(), identity(), std::array{point(value)});
}

TEST(EvaluationInputTest, RoundTripRetainsProvenanceAndNonfiniteStatus) {
  EvaluationInput input{
      .schema_version = 1U,
      .identity = identity(),
      .observations = {point(std::numeric_limits<double>::infinity())}};
  const auto wire = evaluation_input_json(input);
  const auto parsed = parse_evaluation_input_json(wire);
  EXPECT_EQ(evaluation_input_json(parsed), wire);
  EXPECT_EQ(parsed.identity.model_digest, input.identity.model_digest);
  ASSERT_EQ(parsed.observations.size(), 1U);
  EXPECT_FALSE(parsed.observations.front().observation.value);
  EXPECT_EQ(evaluate_observations(suite(), parsed.identity, parsed.observations)
                .verdict,
            QualityVerdict::incomplete);
}

TEST(EvaluationInputTest, RejectsWireAmbiguitiesDuplicatesAndSemanticMismatch) {
  EvaluationInput input{
      .schema_version = 1U, .identity = identity(), .observations = {point()}};
  auto wire = evaluation_input_json(input);
  wire.insert(1U, "\"unknown\":true,");
  EXPECT_THROW(parse_evaluation_input_json(wire), SerializationError);
  wire = evaluation_input_json(input);
  wire.insert(1U, "\"schema_version\":1,");
  EXPECT_THROW(parse_evaluation_input_json(wire), SerializationError);
  input.schema_version = 2U;
  EXPECT_THROW(evaluation_input_json(input), CompatibilityError);
  input.schema_version = 1U;
  input.observations.push_back(point());
  EXPECT_THROW(evaluation_input_json(input), InvalidArgument);
  input.observations.pop_back();
  input.observations.front().definition.unit = TelemetryMetricUnit::percent;
  const auto parsed = parse_evaluation_input_json(evaluation_input_json(input));
  EXPECT_THROW(
      evaluate_observations(suite(), parsed.identity, parsed.observations),
      CompatibilityError);
  EvaluationLimits limits;
  limits.max_json_nodes = 2U;
  EXPECT_THROW(
      parse_evaluation_input_json(evaluation_input_json(input), limits),
      ResourceExhausted);
}

TEST(EvaluationSuiteTest,
     CanonicalRoundTripRetainsAllSemanticParametersAndThresholds) {
  const auto original = suite();
  const auto encoded = evaluation_suite_json(original);
  EXPECT_EQ(evaluation_suite_json(parse_evaluation_suite_json(encoded)),
            encoded);
  auto changed = original;
  changed.rules.front().minimum = 0.85;
  EXPECT_NE(evaluation_suite_digest(original),
            evaluation_suite_digest(changed));
  changed = original;
  changed.rules.front().definition.parameters["average"] = "macro";
  EXPECT_NE(evaluation_suite_digest(original),
            evaluation_suite_digest(changed));
}

TEST(EvaluationSuiteTest,
     RejectsUnknownSchemaDuplicateKeysUnknownFieldsAndNegativeVersions) {
  auto encoded = evaluation_suite_json(suite());
  EXPECT_THROW(parse_evaluation_suite_json("{\"version\":1,\"version\":1}"),
               Error);
  encoded.insert(1U, "\"unknown\":true,");
  EXPECT_THROW(parse_evaluation_suite_json(encoded), Error);
  auto invalid = suite();
  invalid.schema_version = 2U;
  EXPECT_THROW(evaluation_suite_json(invalid), Error);
  encoded = evaluation_suite_json(suite());
  const auto index = encoded.find("\"version\":1");
  ASSERT_NE(index, std::string::npos);
  encoded.replace(index, std::string("\"version\":1").size(), "\"version\":-1");
  EXPECT_THROW(parse_evaluation_suite_json(encoded), Error);
}

TEST(EvaluationSuiteTest,
     RejectsUnknownMetricSemanticsDuplicateRulesAndMalformedContext) {
  auto invalid = suite();
  invalid.rules.front().definition.unit = TelemetryMetricUnit::unspecified;
  EXPECT_THROW(validate_evaluation_suite(invalid), Error);
  invalid = suite();
  invalid.rules.push_back(invalid.rules.front());
  EXPECT_THROW(validate_evaluation_suite(invalid), Error);
  invalid = suite();
  invalid.rules.front().context.clear();
  EXPECT_THROW(validate_evaluation_suite(invalid), Error);
  invalid = suite();
  invalid.rules.front().maximum = 0.1;
  EXPECT_THROW(validate_evaluation_suite(invalid), Error);
}

TEST(QualityReportTest,
     DistinguishesFiniteFailureMissingNonfiniteAndInsufficientSamples) {
  EXPECT_EQ(report().verdict, QualityVerdict::pass);
  EXPECT_EQ(report(0.7).verdict, QualityVerdict::fail);
  EXPECT_EQ(evaluate_observations(suite(), identity(), {}).verdict,
            QualityVerdict::incomplete);
  auto measured = point(std::numeric_limits<double>::quiet_NaN());
  EXPECT_EQ(
      evaluate_observations(suite(), identity(), std::array{measured}).verdict,
      QualityVerdict::incomplete);
  measured = point();
  measured.sample_count = std::nullopt;
  EXPECT_EQ(
      evaluate_observations(suite(), identity(), std::array{measured}).verdict,
      QualityVerdict::incomplete);
  measured.sample_count = 9U;
  EXPECT_EQ(
      evaluate_observations(suite(), identity(), std::array{measured}).verdict,
      QualityVerdict::incomplete);
}

TEST(QualityReportTest, RejectsDefinitionMismatchDuplicatesAndForgedVerdicts) {
  auto measured = point();
  measured.definition.unit = TelemetryMetricUnit::percent;
  EXPECT_THROW(evaluate_observations(suite(), identity(), std::array{measured}),
               Error);
  EXPECT_THROW(
      evaluate_observations(suite(), identity(), std::array{point(), point()}),
      Error);
  auto forged = report(0.7);
  forged.verdict = QualityVerdict::pass;
  EXPECT_THROW(quality_report_json(forged), Error);
  forged = report(0.7);
  forged.results.front().verdict = QualityVerdict::pass;
  EXPECT_THROW(quality_report_json(forged), Error);
}

TEST(QualityReportTest, ReportRoundTripPreservesProvenanceAndEscapesHtml) {
  auto value_suite = suite();
  value_suite.suite_id = "<script>alert('x')</script>";
  const auto original =
      evaluate_observations(value_suite, identity(), std::array{point()});
  const auto encoded = quality_report_json(original);
  EXPECT_EQ(quality_report_json(parse_quality_report_json(encoded)), encoded);
  const auto html = quality_report_html(original);
  EXPECT_EQ(html.find("<script>"), std::string::npos);
  EXPECT_NE(html.find("&lt;script&gt;"), std::string::npos);
  EXPECT_NE(html.find("training-1"), std::string::npos);
}

TEST(QualityComparisonTest, AppliesDirectionalAbsoluteAndRelativeTolerances) {
  EXPECT_EQ(compare_quality_reports(report(), report(0.895)).verdict,
            QualityVerdict::pass);
  const auto comparison = compare_quality_reports(report(), report(0.87));
  EXPECT_EQ(comparison.verdict, QualityVerdict::fail);
  ASSERT_TRUE(comparison.results.front().regression);
  EXPECT_NEAR(*comparison.results.front().regression, 0.03, 1e-14);
  EXPECT_NE(comparison.baseline_report_digest,
            comparison.candidate_report_digest);
  EXPECT_EQ(compare_quality_reports(report(0.9), report(0.95)).verdict,
            QualityVerdict::pass);
}

TEST(QualityComparisonTest,
     RejectsDatasetSuiteEnvironmentAndAdapterMismatches) {
  const auto baseline = report();
  auto changed_suite = suite();
  changed_suite.dataset_digest = digest('d');
  auto candidate =
      evaluate_observations(changed_suite, identity(), std::array{point()});
  EXPECT_THROW(compare_quality_reports(baseline, candidate),
               CompatibilityError);
  changed_suite = suite();
  changed_suite.environment_id = "another-workload";
  candidate =
      evaluate_observations(changed_suite, identity(), std::array{point()});
  EXPECT_THROW(compare_quality_reports(baseline, candidate),
               CompatibilityError);
  auto changed_identity = identity();
  ++changed_identity.adapter_version;
  candidate =
      evaluate_observations(suite(), changed_identity, std::array{point()});
  EXPECT_THROW(compare_quality_reports(baseline, candidate),
               CompatibilityError);
}

TEST(EvaluationRunnerTest,
     ValidatesBeforeInvocationAndChecksCancellationAfterAdapterReturn) {
  bool invoked = false;
  auto invalid = suite();
  invalid.rules.clear();
  const auto evaluator = [&](const EvaluationSuite &, std::stop_token,
                             TelemetryDeadline) {
    invoked = true;
    return std::vector{point()};
  };
  EXPECT_THROW(run_evaluation(invalid, identity(), evaluator), Error);
  EXPECT_FALSE(invoked);
  std::stop_source stop;
  stop.request_stop();
  EXPECT_THROW(run_evaluation(suite(), identity(), evaluator, stop.get_token()),
               Error);
  EXPECT_FALSE(invoked);
  EXPECT_EQ(run_evaluation(suite(), identity(), evaluator).verdict,
            QualityVerdict::pass);
  EXPECT_TRUE(invoked);
  EXPECT_THROW(run_evaluation(
                   suite(), identity(), evaluator, {},
                   std::chrono::steady_clock::now() - std::chrono::seconds(1)),
               Error);
  std::stop_source requested_inside;
  const auto cancelled_adapter = [&](const EvaluationSuite &, std::stop_token,
                                     TelemetryDeadline) {
    requested_inside.request_stop();
    return std::vector{point()};
  };
  EXPECT_THROW(run_evaluation(suite(), identity(), cancelled_adapter,
                              requested_inside.get_token()),
               Cancelled);
}

TEST(DataQualityTest,
     DetectsMissingRangeNonfiniteDuplicateLeakageAndLabelConflicts) {
  const std::map<std::string, DataQualityField> schema{
      {"feature", {.required = true, .minimum = 0.0, .maximum = 1.0}}};
  std::array rows{
      DataQualityRow{.sample_id = "sample-a",
                     .split = "train",
                     .content_digest = digest('a'),
                     .fields = {{"feature", 0.5}},
                     .label = "cat"},
      DataQualityRow{.sample_id = "sample-a",
                     .split = "train",
                     .content_digest = digest('a'),
                     .fields = {{"feature", 2.0}},
                     .label = "cat"},
      DataQualityRow{
          .sample_id = "sample-b",
          .split = "validation",
          .content_digest = digest('a'),
          .fields = {{"feature", std::nullopt},
                     {"extra", std::numeric_limits<double>::infinity()}},
          .label = "dog"}};
  const auto evidence = inspect_data_quality(schema, rows);
  EXPECT_EQ(evidence.verdict, QualityVerdict::fail);
  EXPECT_EQ(evidence.checked_rows, 3U);
  EXPECT_EQ(evidence.missing_values, 1U);
  EXPECT_EQ(evidence.nonfinite_values, 1U);
  EXPECT_EQ(evidence.out_of_range_values, 1U);
  EXPECT_EQ(evidence.unknown_fields, 1U);
  EXPECT_EQ(evidence.duplicate_ids, 1U);
  EXPECT_EQ(evidence.duplicate_contents, 1U);
  EXPECT_EQ(evidence.cross_split_leaks, 1U);
  EXPECT_EQ(evidence.conflicting_labels, 1U);
  EXPECT_EQ(inspect_data_quality(schema, {}).verdict,
            QualityVerdict::incomplete);
}

TEST(EvaluationLimitsTest,
     RejectsOversizedJsonObjectsAndDataBeforeGrowingIndexes) {
  EvaluationLimits limits;
  limits.max_json_bytes = 16U;
  EXPECT_THROW(
      parse_evaluation_suite_json(evaluation_suite_json(suite()), limits),
      Error);
  limits = {};
  limits.max_json_nodes = 3U;
  EXPECT_THROW(
      parse_evaluation_suite_json(evaluation_suite_json(suite()), limits),
      Error);
  limits = {};
  limits.max_data_text_bytes = 1U;
  const DataQualityRow row{.sample_id = "sample-a",
                           .split = "train",
                           .content_digest = digest('a'),
                           .fields = {},
                           .label = std::nullopt};
  EXPECT_THROW(
      inspect_data_quality({}, std::array{row}, {}, std::nullopt, limits),
      Error);
}
}  // namespace
