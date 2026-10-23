// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/checkpoint.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace jyppx::liuding {
namespace {

constexpr std::size_t kMaxIdentifierLength = 128;
constexpr std::size_t kMaxMediaTypeLength = 128;
constexpr std::size_t kMaxDegradationReasonLength = 1024;

bool is_ascii_alphanumeric(char value) noexcept {
  return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') ||
         (value >= '0' && value <= '9');
}

bool is_valid_identifier(std::string_view value) noexcept {
  if (value.empty() || value.size() > kMaxIdentifierLength ||
      !is_ascii_alphanumeric(value.front())) {
    return false;
  }
  return std::all_of(value.begin(), value.end(), [](char character) {
    return is_ascii_alphanumeric(character) || character == '.' ||
           character == '_' || character == '-';
  });
}

bool is_valid_media_type(std::string_view value) noexcept {
  if (value.empty() || value.size() > kMaxMediaTypeLength) {
    return false;
  }
  const auto separator = value.find('/');
  if (separator == std::string_view::npos || separator == 0 ||
      separator + 1 == value.size() ||
      value.find('/', separator + 1) != std::string_view::npos) {
    return false;
  }
  return std::all_of(value.begin(), value.end(), [](char character) {
    return character >= 0x21 && character <= 0x7e;
  });
}

[[noreturn]] void throw_manifest_error(std::string operation,
                                       std::string expected, std::string actual,
                                       std::string hint) {
  throw InvalidArgument("Invalid checkpoint participant manifest",
                        ErrorDetails{.component = "foundation.checkpoint",
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

}  // namespace

std::string_view restore_level_name(RestoreLevel level) noexcept {
  switch (level) {
    case RestoreLevel::exact:
      return "exact";
    case RestoreLevel::bounded:
      return "bounded";
    case RestoreLevel::best_effort:
      return "best_effort";
  }
  return "unknown";
}

std::string_view restore_policy_name(RestorePolicy policy) noexcept {
  switch (policy) {
    case RestorePolicy::exact_only:
      return "exact_only";
    case RestorePolicy::allow_bounded:
      return "allow_bounded";
    case RestorePolicy::allow_best_effort:
      return "allow_best_effort";
  }
  return "unknown";
}

bool restore_policy_accepts(RestorePolicy policy, RestoreLevel level) noexcept {
  const auto policy_value = static_cast<std::uint8_t>(policy);
  const auto level_value = static_cast<std::uint8_t>(level);
  if (policy_value < static_cast<std::uint8_t>(RestorePolicy::exact_only) ||
      policy_value >
          static_cast<std::uint8_t>(RestorePolicy::allow_best_effort) ||
      level_value < static_cast<std::uint8_t>(RestoreLevel::exact) ||
      level_value > static_cast<std::uint8_t>(RestoreLevel::best_effort)) {
    return false;
  }
  return level_value <= policy_value;
}

bool state_fields_equal(std::span<const StateFieldDescriptor> left,
                        std::span<const StateFieldDescriptor> right) {
  if (left.size() != right.size()) return false;
  std::unordered_map<std::string_view, const StateFieldDescriptor*> fields;
  fields.reserve(left.size());
  for (const auto& field : left) {
    if (!fields.emplace(field.name, &field).second) return false;
  }
  for (const auto& field : right) {
    const auto found = fields.find(field.name);
    if (found == fields.end() ||
        found->second->media_type != field.media_type ||
        found->second->max_bytes != field.max_bytes ||
        found->second->required != field.required) {
      return false;
    }
    fields.erase(found);
  }
  return true;
}

void validate_state_manifest(const StateManifest& manifest) {
  if (!is_valid_identifier(manifest.participant_id)) {
    throw_manifest_error(
        "participant_id", "1-128 ASCII letters, digits, '.', '_' or '-'",
        manifest.participant_id,
        "Use a stable logical identifier, not a filesystem path.");
  }
  if (manifest.schema_version == 0) {
    throw_manifest_error("schema_version", "a positive version", "0",
                         "Start participant schemas at version 1.");
  }
  if (restore_level_name(manifest.restore_level) == "unknown") {
    throw_manifest_error(
        "restore_level", "exact, bounded, or best_effort",
        std::to_string(static_cast<std::uint8_t>(manifest.restore_level)),
        "Reject unrecognized persisted enum values.");
  }
  if (manifest.fields.empty()) {
    throw_manifest_error(
        "fields", "at least one declared state field", "empty",
        "Do not register stateless objects as checkpoint participants.");
  }
  if (manifest.restore_level == RestoreLevel::exact &&
      !manifest.degradation_reasons.empty()) {
    throw_manifest_error(
        "degradation_reasons", "empty for an exact participant", "non-empty",
        "Use bounded or best_effort when continuation can degrade.");
  }
  if (manifest.restore_level != RestoreLevel::exact &&
      manifest.degradation_reasons.empty()) {
    throw_manifest_error("degradation_reasons",
                         "at least one reason for non-exact state", "empty",
                         "Explain every declared recovery degradation.");
  }
  for (const auto& reason : manifest.degradation_reasons) {
    if (reason.empty() || reason.size() > kMaxDegradationReasonLength) {
      throw_manifest_error(
          "degradation_reasons", "non-empty reasons of at most 1024 bytes",
          reason.empty() ? "empty" : "too long",
          "Keep degradation diagnostics concise and actionable.");
    }
  }

  std::unordered_set<std::string> field_names;
  field_names.reserve(manifest.fields.size());
  for (const auto& field : manifest.fields) {
    if (!is_valid_identifier(field.name)) {
      throw_manifest_error(
          "fields.name", "a stable logical ASCII identifier", field.name,
          "Do not expose paths or unescaped separators as field names.");
    }
    if (!field_names.insert(field.name).second) {
      throw_manifest_error("fields.name", "unique field names", field.name,
                           "Rename or merge the duplicate state field.");
    }
    if (!is_valid_media_type(field.media_type)) {
      throw_manifest_error(
          "fields.media_type", "one printable ASCII type/subtype value",
          field.media_type,
          "Use a registered or vendor media type without whitespace.");
    }
    if (field.max_bytes == 0) {
      throw_manifest_error(
          "fields.max_bytes", "a positive resource limit", "0",
          "Declare a finite upper bound before reading checkpoint bytes.");
    }
  }
}

void validate_checkpoint_participant(const CheckpointParticipant& participant) {
  const auto identifier = participant.participant_id();
  const auto manifest = participant.describe_state();
  if (identifier != manifest.participant_id) {
    throw_manifest_error(
        "participant_id", "the interface and manifest identifiers to match",
        identifier + " != " + manifest.participant_id,
        "Publish one stable identifier from both participant entry points.");
  }
  validate_state_manifest(manifest);
}

}  // namespace jyppx::liuding
