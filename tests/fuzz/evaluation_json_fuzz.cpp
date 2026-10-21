// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "fuzz_support.hpp"
#include "telemetry_seed.hpp"

#include <jyppx/experiment/evaluation.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace {
namespace ex = jyppx::experiment;
namespace fuzz = jyppx::fuzz;

ex::EvaluationLimits limits() {
  return {.max_rules = 32U,
          .max_text_bytes = 512U,
          .max_json_bytes = fuzz::kMaxInputBytes,
          .max_json_nodes = 8192U,
          .max_json_depth = 16U,
          .max_data_rows = 32U,
          .max_data_fields = 16U,
          .max_data_text_bytes = 32U * 1024U,
          .telemetry = fuzz::telemetry_limits()};
}

void exercise(std::span<const std::byte> input) {
  static_cast<void>(fuzz::exercise_json_codec(
      fuzz::text(input), limits(), &ex::EvaluationLimits::max_json_bytes,
      ex::parse_evaluation_suite_json, ex::evaluation_suite_json));
  static_cast<void>(fuzz::exercise_json_codec(
      fuzz::text(input), limits(), &ex::EvaluationLimits::max_json_bytes,
      ex::parse_evaluation_input_json, ex::evaluation_input_json));
  const auto parsed = fuzz::exercise_json_codec(
      fuzz::text(input), limits(), &ex::EvaluationLimits::max_json_bytes,
      ex::parse_quality_report_json, ex::quality_report_json);
  if (!parsed) return;
  auto canonical_limits = limits();
  canonical_limits.max_json_bytes = fuzz::kCanonicalBytes;
  const auto comparison =
      ex::compare_quality_reports(*parsed, *parsed, canonical_limits);
  fuzz::require(
      comparison.baseline_report_digest == comparison.candidate_report_digest &&
          comparison.results.size() == parsed->results.size(),
      "self comparison changed immutable report identity");
  // HTML has a distinct escaped output size; measure it with an independent
  // budget instead of incorrectly requiring the JSON byte budget to fit it.
  const auto html = ex::quality_report_html(*parsed, canonical_limits);
  fuzz::require(html.find("<script") == std::string::npos &&
                    html.find("</script") == std::string::npos,
                "quality report emitted executable unescaped script markup");
}

const std::vector<std::vector<std::byte>>& seeds() {
  static const auto value = [] {
    const ex::EvaluationSuite suite{
        .suite_id = "classification.fuzz",
        .task = "classification",
        .dataset_digest = {.hex = std::string(64U, 'a')},
        .environment_id = "cpu-f64-deterministic",
        .seed = 71U,
        .rules = {{.rule_id = "validation.accuracy",
                   .definition = fuzz::seed_definition(),
                   .context = {{"split", "validation"}},
                   .minimum_samples = 10U,
                   .minimum = 0.8,
                   .maximum = 1.0,
                   .max_absolute_regression = 0.01,
                   .max_relative_regression = 0.02}}};
    const ex::EvaluationIdentity identity{
        .evaluation_id = "eval.fuzz",
        .run_id = "run.fuzz",
        .model_digest = {.hex = std::string(64U, 'b')},
        .build_fingerprint = "fuzz-build",
        .adapter_id = "classification.local",
        .config_digest = {.hex = std::string(64U, 'c')}};
    const ex::EvaluationInput input{.identity = identity,
                                    .observations = {fuzz::seed_point()}};
    const auto report = ex::evaluate_observations(
        suite, identity, input.observations, {}, std::nullopt, limits());
    const auto missing = ex::evaluate_observations(suite, identity, {}, {},
                                                   std::nullopt, limits());
    std::vector<std::vector<std::byte>> result;
    result.push_back(
        fuzz::copy_bytes(ex::evaluation_suite_json(suite, limits())));
    result.push_back(
        fuzz::copy_bytes(ex::evaluation_input_json(input, limits())));
    result.push_back(
        fuzz::copy_bytes(ex::quality_report_json(report, limits())));
    result.push_back(
        fuzz::copy_bytes(ex::quality_report_json(missing, limits())));
    // Seed codecs fail loudly here; initialization cannot silently treat a
    // regression in a valid generated seed as expected malformed input.
    static_cast<void>(
        ex::parse_evaluation_suite_json(fuzz::text(result[0]), limits()));
    static_cast<void>(
        ex::parse_evaluation_input_json(fuzz::text(result[1]), limits()));
    static_cast<void>(
        ex::parse_quality_report_json(fuzz::text(result[2]), limits()));
    static_cast<void>(
        ex::parse_quality_report_json(fuzz::text(result[3]), limits()));
    for (const auto& seed : result) exercise(seed);
    return result;
  }();
  return value;
}
}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data,
                                      std::size_t size) {
  return fuzz::exercise_raw_and_seed(data, size, seeds(), exercise);
}
