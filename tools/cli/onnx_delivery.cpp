// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "cli.hpp"

#ifdef JYPPX_LIUDING_CLI_ONNX_EXPORT
#include <jyppx/liuding/deployment/onnx_export/report.hpp>
#include <jyppx/liuding/experiment/run_store.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#endif

namespace jyppx::liuding::cli {

Result onnx_command(const Arguments& arguments) {
  const auto& words = arguments.words();
  if (words.size() != 3U || words[2] != "register")
    usage("model.onnx", "Use model onnx register with a native export report.");
  arguments.allow(
      {"path", "report", "report-sha256", "root", "id", "reference"});
#ifndef JYPPX_LIUDING_CLI_ONNX_EXPORT
  throw CompatibilityError("Native ONNX export component is disabled",
                           "Enable JYPPX_LIUDING_BUILD_ONNX_EXPORT.");
#else
  const auto bytes = [](std::string_view value) {
    return std::as_bytes(std::span(value.data(), value.size()));
  };
  const ArtifactDigest expected{.hex = arguments.require("report-sha256")};
  validate_artifact_digest(expected);
  const auto report_json = read_text(arguments.require("report"), 4U << 20U);
  if (sha256_hex(bytes(report_json)) != expected.hex)
    throw SerializationError("Native ONNX report SHA-256 mismatch");
  const auto report = onnx_export::deserialize_export_report(report_json);
  if (report.model_size_bytes > (256U << 20U))
    throw ResourceExhausted("ONNX registration model exceeds 256 MiB");
  const auto model =
      read_text(arguments.require("path"),
                static_cast<std::size_t>(report.model_size_bytes));
  onnx_export::verify_export_report_model(report, bytes(model));
  const auto run_id = arguments.require("id");
  const auto reference = arguments.require("reference");
  if (reference.empty() || reference.size() > 240U)
    throw InvalidArgument(
        "ONNX registration reference must contain 1-240 bytes");
  const auto root = std::filesystem::u8path(arguments.require("root"));
  if (!std::filesystem::is_regular_file(root / "runs.sqlite3"))
    throw IoError("ONNX registration requires an existing run store");
  experiment::SQLiteRunStore store(root);
  if (!report.source.run_id || *report.source.run_id == run_id)
    throw InvalidArgument(
        "Use a distinct export run ID and an original source run in the "
        "report");
  const auto source = store.find(*report.source.run_id);
  if (!source || report.source.source_digest != source->lineage.source_digest ||
      report.source.config_digest != source->lineage.config_digest ||
      report.source.data_manifest_digest !=
          source->lineage.data_manifest_digest)
    throw CompatibilityError(
        "ONNX export report does not match the original run lineage");
  const auto previous = store.find(run_id);
  const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
  if (now < 0) throw InvalidArgument("System clock precedes Unix epoch");
  experiment::RunMetadata metadata{
      .run_id = run_id,
      .project_id = source->project_id,
      .experiment_id = source->experiment_id,
      .display_name = "ONNX export " + run_id,
      .status = experiment::RunStatus::running,
      .created_at_unix_ms = previous ? previous->created_at_unix_ms
                                     : static_cast<std::uint64_t>(now),
      .lineage = {},
      .tags = {}};
  metadata.updated_at_unix_ms = metadata.created_at_unix_ms;
  metadata.lineage = {
      .parent_run_id = source->run_id,
      .source_digest = report.source.source_digest,
      .build_fingerprint = build_fingerprint_json(report.build),
      .config_digest = report.source.config_digest,
      .data_manifest_digest = report.source.data_manifest_digest};
  metadata.tags = {{"operation", "onnx.export"},
                   {"onnx_sha256", report.model_digest.hex},
                   {"export_report_sha256", expected.hex}};
  (void)store.start(metadata);
  // Preserve completed training runs; export owns a separate linked lifecycle.
  const std::array uploads{
      experiment::RunArtifactUpload{
          .reference_id = reference + ".onnx",
          .role = "onnx.model",
          .bytes = bytes(model),
          .descriptor = {.logical_name = "model.onnx",
                         .media_type = "application/onnx",
                         .producer = "onnx_export.native",
                         .expected_size_bytes = report.model_size_bytes},
          .expected_digest = report.model_digest,
          .attached_at_unix_ms = metadata.created_at_unix_ms},
      experiment::RunArtifactUpload{
          .reference_id = reference + ".report",
          .role = "onnx.export_report",
          .bytes = bytes(report_json),
          .descriptor = {.logical_name = "onnx-export-report.json",
                         .media_type = "application/json",
                         .producer = "onnx_export.native",
                         .expected_size_bytes = report_json.size()},
          .expected_digest = expected,
          .attached_at_unix_ms = metadata.created_at_unix_ms}};
  const auto receipts = store.put_artifacts(run_id, uploads);
  (void)store.transition(
      run_id, experiment::RunStatus::completed,
      (std::max)(static_cast<std::uint64_t>(now),
                 previous ? previous->updated_at_unix_ms : 0U));
  auto artifacts = Json::array();
  for (const auto& receipt : receipts)
    artifacts.push_back({{"reference_id", receipt.reference_id},
                         {"role", receipt.role},
                         {"sha256", receipt.artifact.digest.hex},
                         {"size_bytes", receipt.artifact.size_bytes}});
  return {Json{{"operation", "model.onnx.register"},
               {"run_id", run_id},
               {"source_run_id", source->run_id},
               {"artifacts", artifacts},
               {"model_integrity_verified", true},
               {"official_checker", "passed"},
               {"runtime_executed", false},
               {"source_authenticated", false}},
          0};
#endif
}

}  // namespace jyppx::liuding::cli
