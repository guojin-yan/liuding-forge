// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "jyppx/liuding/experiment/run.hpp"

#include "jyppx/liuding/foundation/error.hpp"

#include <algorithm>
#include <string>
#include <utility>

namespace jyppx::liuding::experiment {
namespace {

[[noreturn]] void throw_invalid(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
  throw InvalidArgument("Invalid run metadata",
                        ErrorDetails{.component = "experiment.run",
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

[[noreturn]] void throw_limit(std::string operation, std::string expected,
                              std::string actual) {
  throw ResourceExhausted(
      "Run metadata bound exceeded",
      ErrorDetails{
          .component = "experiment.run",
          .operation = std::move(operation),
          .expected = std::move(expected),
          .actual = std::move(actual),
          .hint = "Raise the configured metadata bound or shorten the value."});
}

bool is_valid_status(RunStatus status) noexcept {
  switch (status) {
    case RunStatus::created:
    case RunStatus::running:
    case RunStatus::completed:
    case RunStatus::failed:
    case RunStatus::cancelled:
    case RunStatus::interrupted:
      return true;
  }
  return false;
}

bool has_control_character(std::string_view value) noexcept {
  return std::any_of(value.begin(), value.end(),
                     [](unsigned char character) { return character < 0x20U; });
}

void validate_bounded(std::string_view field, std::string_view value,
                      std::size_t bound, bool required) {
  if (required && value.empty()) {
    throw_invalid(std::string(field), "a nonempty UTF-8 string", "empty",
                  "Provide the stable identity field.");
  }
  if (value.size() > bound) {
    throw_limit(std::string(field),
                "at most " + std::to_string(bound) + " bytes",
                std::to_string(value.size()) + " bytes");
  }
  if (has_control_character(value)) {
    throw_invalid(std::string(field), "text without control characters",
                  std::string(value), "Store display-safe metadata text.");
  }
}

void validate_optional_ref(std::string_view field,
                           const std::optional<std::string>& value,
                           std::size_t bound) {
  if (value) validate_bounded(field, *value, bound, true);
}

}  // namespace

std::string_view run_status_name(RunStatus status) noexcept {
  switch (status) {
    case RunStatus::created:
      return "created";
    case RunStatus::running:
      return "running";
    case RunStatus::completed:
      return "completed";
    case RunStatus::failed:
      return "failed";
    case RunStatus::cancelled:
      return "cancelled";
    case RunStatus::interrupted:
      return "interrupted";
  }
  return "unknown";
}

bool is_terminal_run_status(RunStatus status) noexcept {
  return status == RunStatus::completed || status == RunStatus::failed ||
         status == RunStatus::cancelled || status == RunStatus::interrupted;
}

bool is_valid_run_transition(RunStatus from, RunStatus to) noexcept {
  if (!is_valid_status(from) || !is_valid_status(to)) return false;
  if (from == to) return true;
  if (is_terminal_run_status(from)) return false;
  if (from == RunStatus::created) return to == RunStatus::running;
  return is_terminal_run_status(to);
}

void validate_run_metadata(const RunMetadata& metadata,
                           const RunMetadataLimits& limits) {
  if (limits.max_id_bytes == 0U || limits.max_display_name_bytes == 0U ||
      limits.max_tag_key_bytes == 0U || limits.max_tag_value_bytes == 0U ||
      limits.max_lineage_ref_bytes == 0U) {
    throw_invalid("limits", "positive metadata bounds", "zero bound",
                  "Use positive bounds for every metadata field.");
  }
  if (!is_valid_status(metadata.status)) {
    throw_invalid("status", "a known RunStatus value", "unknown",
                  "Use a status declared by the run lifecycle contract.");
  }
  validate_bounded("run_id", metadata.run_id, limits.max_id_bytes, true);
  validate_bounded("project_id", metadata.project_id, limits.max_id_bytes,
                   true);
  validate_bounded("experiment_id", metadata.experiment_id, limits.max_id_bytes,
                   true);
  validate_bounded("display_name", metadata.display_name,
                   limits.max_display_name_bytes, true);
  if (metadata.updated_at_unix_ms < metadata.created_at_unix_ms) {
    throw_invalid("updated_at_unix_ms", "at least created_at_unix_ms",
                  std::to_string(metadata.updated_at_unix_ms),
                  "Keep run timestamps monotonic.");
  }

  validate_optional_ref("lineage.parent_run_id", metadata.lineage.parent_run_id,
                        limits.max_lineage_ref_bytes);
  validate_optional_ref("lineage.source_digest", metadata.lineage.source_digest,
                        limits.max_lineage_ref_bytes);
  validate_optional_ref("lineage.build_fingerprint",
                        metadata.lineage.build_fingerprint,
                        limits.max_lineage_ref_bytes);
  validate_optional_ref("lineage.config_digest", metadata.lineage.config_digest,
                        limits.max_lineage_ref_bytes);
  validate_optional_ref("lineage.data_manifest_digest",
                        metadata.lineage.data_manifest_digest,
                        limits.max_lineage_ref_bytes);

  if (metadata.tags.size() > limits.max_tag_count) {
    throw_limit("tags",
                "at most " + std::to_string(limits.max_tag_count) + " entries",
                std::to_string(metadata.tags.size()) + " entries");
  }
  for (const auto& [key, value] : metadata.tags) {
    validate_bounded("tags.key", key, limits.max_tag_key_bytes, true);
    validate_bounded("tags.value", value, limits.max_tag_value_bytes, true);
  }
}

}  // namespace jyppx::liuding::experiment
