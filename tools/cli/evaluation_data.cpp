// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "cli.hpp"

#include <jyppx/experiment/evaluation.hpp>
#include <jyppx/experiment/run_store.hpp>
#include <jyppx/foundation/sha256.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>

namespace jyppx::cli {
namespace {
using namespace experiment;

std::filesystem::path path(std::string_view value) {
  return std::filesystem::u8path(value.begin(), value.end());
}

Json digest_json(const ArtifactDigest& digest) {
  return {{"algorithm", "sha256"}, {"hex", digest.hex}};
}

ArtifactDigest read_digest(const Json& value) {
  fields(value, {"algorithm", "hex"});
  if (value.at("algorithm") != "sha256")
    throw CompatibilityError("Unsupported data digest algorithm");
  ArtifactDigest digest{.hex = value.at("hex").get<std::string>()};
  validate_artifact_digest(digest);
  return digest;
}

int verdict_exit(QualityVerdict verdict) {
  return verdict == QualityVerdict::pass || verdict == QualityVerdict::warn ? 0
                                                                            : 1;
}

// Outputs are owned by one command invocation. Existing identical artifacts
// permit replay; different files are never intentionally replaced.
void write_output(const std::filesystem::path& output, std::string_view bytes) {
  std::error_code status_error;
  const auto status = std::filesystem::symlink_status(output, status_error);
  if (status_error && status_error != std::errc::no_such_file_or_directory)
    throw IoError("Could not inspect output path");
  if (std::filesystem::exists(status)) {
    if (!std::filesystem::is_regular_file(status) ||
        read_text(output) != bytes) {
      throw IoError("Output already exists with different content",
                    "Choose a new output path.");
    }
    return;
  }
  auto parent = output.parent_path();
  if (parent.empty()) parent = ".";
  if (!std::filesystem::is_directory(parent))
    throw IoError("Output parent directory does not exist");
  static std::atomic<std::uint64_t> sequence{0U};
  std::filesystem::path temporary;
  bool reserved = false;
  for (unsigned attempt = 0U; attempt < 32U; ++attempt) {
    const auto clock =
        std::chrono::steady_clock::now().time_since_epoch().count();
    temporary = parent / (".jyppx-output-" + std::to_string(clock) + "-" +
                          std::to_string(sequence.fetch_add(1U)));
    if (std::filesystem::create_directory(temporary)) {
      reserved = true;
      break;
    }
  }
  if (!reserved) throw IoError("Could not reserve output staging directory");
  const auto staged = temporary / "artifact";
  try {
    std::ofstream stream(staged, std::ios::binary | std::ios::trunc);
    if (!stream) throw IoError("Could not open output staging file");
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    stream.flush();
    if (!stream) throw IoError("Could not write complete output artifact");
    stream.close();
    if (!stream) throw IoError("Could not close output staging file");
    // Same-filesystem link creation is atomic and never replaces an existing
    // destination, including one created concurrently after the initial check.
    std::error_code publication_error;
    std::filesystem::create_hard_link(staged, output, publication_error);
    if (publication_error)
      throw IoError("Could not publish output without replacement");
    std::filesystem::remove(staged);
    std::filesystem::remove(temporary);
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(staged, ignored);
    std::filesystem::remove(temporary, ignored);
    throw;
  }
}

void retain_report(const Arguments& arguments, const QualityReport& report,
                   const std::string& canonical, Json& result) {
  result["report_digest"] = digest_json(
      ArtifactDigest{.hex = sha256_hex(std::as_bytes(std::span(canonical)))});
  const auto root = arguments.optional("root");
  if (!root) {
    if (arguments.optional("reference-id") ||
        arguments.optional("attached-ms")) {
      usage("evaluation.persist",
            "Use --root with --reference-id and --attached-ms.");
    }
  } else {
    static_cast<void>(arguments.require("reference-id"));
    static_cast<void>(arguments.require("attached-ms"));
    static_cast<void>(arguments.number(
        "attached-ms", 0U, std::numeric_limits<std::uint64_t>::max()));
    if (!std::filesystem::is_regular_file(path(*root) / "runs.sqlite3"))
      throw IoError("Run storage database does not exist");
  }
  if (const auto output = arguments.optional("output"))
    write_output(path(*output), canonical);
  if (const auto html = arguments.optional("html"))
    write_output(path(*html), quality_report_html(report));
  if (!root) return;
  const auto reference_id = arguments.require("reference-id");
  static_cast<void>(arguments.require("attached-ms"));
  SQLiteRunStore store(path(*root));
  const ArtifactDescriptor descriptor{
      .logical_name = report.identity.evaluation_id + ".quality.json",
      .media_type = "application/vnd.jyppx.quality-report+json;version=1",
      .producer = "jyppx.cli.evaluation/v1",
      .expected_size_bytes = canonical.size()};
  const auto reference = store.put_artifact(
      report.identity.run_id, reference_id, RunArtifactKind::artifact,
      "evaluation.report", std::as_bytes(std::span(canonical)), descriptor,
      arguments.number("attached-ms", 0U,
                       std::numeric_limits<std::uint64_t>::max()));
  result["reference_id"] = reference.reference_id;
}

Json comparison_json(const QualityComparison& comparison) {
  Json results = Json::array();
  for (const auto& rule : comparison.results) {
    results.push_back(
        {{"rule_id", rule.rule_id},
         {"verdict", quality_verdict_name(rule.verdict)},
         {"candidate_minus_baseline", rule.candidate_minus_baseline
                                          ? Json(*rule.candidate_minus_baseline)
                                          : Json(nullptr)},
         {"regression",
          rule.regression ? Json(*rule.regression) : Json(nullptr)},
         {"reason", rule.reason}});
  }
  return {{"schema_version", 1U},
          {"verdict", quality_verdict_name(comparison.verdict)},
          {"baseline_report_digest",
           digest_json(comparison.baseline_report_digest)},
          {"candidate_report_digest",
           digest_json(comparison.candidate_report_digest)},
          {"results", results}};
}

std::optional<double> optional_scalar(const Json& value) {
  if (value.is_null()) return std::nullopt;
  if (!value.is_number())
    throw SerializationError("Data scalar must be a finite number or null");
  const auto result = value.get<double>();
  if (!std::isfinite(result))
    throw SerializationError("Data scalar is outside finite binary64 range");
  return result;
}
}  // namespace

Result evaluation_command(const Arguments& arguments) {
  const auto action = arguments.words()[1];
  if (action == "run") {
    arguments.allow({"suite", "input", "output", "html", "root", "reference-id",
                     "attached-ms"});
    const auto suite = parse_evaluation_suite_json(
        read_text(path(arguments.require("suite"))));
    const auto input = parse_evaluation_input_json(
        read_text(path(arguments.require("input"))));
    const auto report =
        evaluate_observations(suite, input.identity, input.observations);
    const auto canonical = quality_report_json(report);
    Json result{{"verdict", quality_verdict_name(report.verdict)},
                {"report", parse_json(canonical)}};
    retain_report(arguments, report, canonical, result);
    return {result, verdict_exit(report.verdict)};
  }
  if (action == "show") {
    arguments.allow({"report", "html"});
    const auto report =
        parse_quality_report_json(read_text(path(arguments.require("report"))));
    const auto canonical = quality_report_json(report);
    if (const auto html = arguments.optional("html"))
      write_output(path(*html), quality_report_html(report));
    return {Json{{"verdict", quality_verdict_name(report.verdict)},
                 {"report", parse_json(canonical)}},
            verdict_exit(report.verdict)};
  }
  if (action == "compare") {
    arguments.allow({"baseline", "candidate", "output"});
    const auto baseline = parse_quality_report_json(
        read_text(path(arguments.require("baseline"))));
    const auto candidate = parse_quality_report_json(
        read_text(path(arguments.require("candidate"))));
    const auto comparison = compare_quality_reports(baseline, candidate);
    auto result = comparison_json(comparison);
    if (const auto output = arguments.optional("output"))
      write_output(path(*output), result.dump());
    return {result, verdict_exit(comparison.verdict)};
  }
  usage("evaluation",
        "Choose run/show/compare; use run --suite PATH --input PATH.");
}

Result data_command(const Arguments& arguments) {
  const auto action = arguments.words()[1];
  if (action != "inspect" && action != "verify")
    usage("data", "Choose inspect or verify.");
  arguments.allow({"input", "output"});
  const auto input = parse_json(read_text(path(arguments.require("input"))));
  fields(input, {"schema_version", "schema", "rows"});
  if (unsigned_number(input.at("schema_version"),
                      std::numeric_limits<std::uint32_t>::max()) != 1U) {
    throw CompatibilityError("Unsupported data projection schema");
  }
  const EvaluationLimits limits;
  if (!input.at("schema").is_object() || !input.at("rows").is_array())
    throw SerializationError("Expected data schema object and rows array");
  if (input.at("schema").size() > limits.max_data_fields ||
      input.at("rows").size() > limits.max_data_rows) {
    throw ResourceExhausted("Data projection exceeds schema or row limits");
  }
  std::map<std::string, DataQualityField> schema;
  for (const auto& [name, field] : input.at("schema").items()) {
    fields(field, {"required", "minimum", "maximum"});
    schema.emplace(name, DataQualityField{
                             .required = field.at("required").get<bool>(),
                             .minimum = optional_scalar(field.at("minimum")),
                             .maximum = optional_scalar(field.at("maximum"))});
  }
  std::vector<DataQualityRow> rows;
  rows.reserve(input.at("rows").size());
  for (const auto& item : input.at("rows")) {
    fields(item, {"sample_id", "split", "content_digest", "fields", "label"});
    DataQualityRow row{
        .sample_id = item.at("sample_id").get<std::string>(),
        .split = item.at("split").get<std::string>(),
        .content_digest = read_digest(item.at("content_digest")),
        .fields = {},
        .label = item.at("label").is_null()
                     ? std::nullopt
                     : std::optional(item.at("label").get<std::string>())};
    if (!item.at("fields").is_object())
      throw SerializationError("Expected data fields object");
    if (item.at("fields").size() > limits.max_data_fields)
      throw ResourceExhausted("Data row has too many scalar fields");
    for (const auto& [name, scalar] : item.at("fields").items())
      row.fields.emplace(name, optional_scalar(scalar));
    rows.push_back(std::move(row));
  }
  const auto report = inspect_data_quality(schema, rows);
  Json result{{"schema_version", 1U},
              {"verdict", quality_verdict_name(report.verdict)},
              {"checked_rows", report.checked_rows},
              {"missing_values", report.missing_values},
              {"nonfinite_values", report.nonfinite_values},
              {"out_of_range_values", report.out_of_range_values},
              {"unknown_fields", report.unknown_fields},
              {"duplicate_ids", report.duplicate_ids},
              {"duplicate_contents", report.duplicate_contents},
              {"cross_split_leaks", report.cross_split_leaks},
              {"conflicting_labels", report.conflicting_labels}};
  if (const auto output = arguments.optional("output"))
    write_output(path(*output), result.dump());
  return {result, verdict_exit(report.verdict)};
}
}  // namespace jyppx::cli
