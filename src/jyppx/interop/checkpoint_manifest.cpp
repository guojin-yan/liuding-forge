// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/interop/checkpoint_manifest.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace jyppx::interop {
namespace {

using Json = nlohmann::json;

constexpr std::string_view kComponent = "interop.checkpoint_manifest";

[[noreturn]] void ThrowInvalid(std::string operation, std::string expected,
                               std::string actual, std::string hint) {
  throw InvalidArgument("Invalid checkpoint artifact manifest",
                        ErrorDetails{.component = std::string(kComponent),
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

[[noreturn]] void ThrowSerialization(std::string operation,
                                     std::string expected, std::string actual,
                                     std::string hint) {
  throw SerializationError("Malformed checkpoint artifact manifest",
                           ErrorDetails{.component = std::string(kComponent),
                                        .operation = std::move(operation),
                                        .expected = std::move(expected),
                                        .actual = std::move(actual),
                                        .hint = std::move(hint)});
}

[[noreturn]] void ThrowCompatibility(std::string operation,
                                     std::string expected, std::string actual,
                                     std::string hint) {
  throw CompatibilityError("Unsupported checkpoint artifact manifest",
                           ErrorDetails{.component = std::string(kComponent),
                                        .operation = std::move(operation),
                                        .expected = std::move(expected),
                                        .actual = std::move(actual),
                                        .hint = std::move(hint)});
}

[[noreturn]] void ThrowResource(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
  throw ResourceExhausted(
      "Checkpoint artifact manifest exceeds a resource limit",
      ErrorDetails{.component = std::string(kComponent),
                   .operation = std::move(operation),
                   .expected = std::move(expected),
                   .actual = std::move(actual),
                   .hint = std::move(hint)});
}

void ValidateLimits(const CheckpointManifestLimits& limits) {
  if (limits.max_manifest_bytes == 0U || limits.max_depth == 0U ||
      limits.max_nodes == 0U || limits.max_string_bytes == 0U ||
      limits.max_total_string_bytes == 0U || limits.max_participants == 0U ||
      limits.max_blobs == 0U || limits.max_path_bytes == 0U ||
      limits.max_total_blob_bytes == 0U) {
    ThrowInvalid("limits", "strictly positive resource limits", "zero",
                 "Set every CheckpointManifestLimits field to a positive "
                 "value.");
  }
}

bool IsContinuation(unsigned char byte) noexcept {
  return (byte & 0xC0U) == 0x80U;
}

bool IsValidUtf8(std::string_view text) noexcept {
  for (std::size_t index = 0; index < text.size();) {
    const auto first = static_cast<unsigned char>(text[index]);
    if (first <= 0x7FU) {
      ++index;
      continue;
    }
    if (first >= 0xC2U && first <= 0xDFU) {
      if (index + 1U >= text.size() ||
          !IsContinuation(static_cast<unsigned char>(text[index + 1U]))) {
        return false;
      }
      index += 2U;
      continue;
    }
    if (first >= 0xE0U && first <= 0xEFU) {
      if (index + 2U >= text.size()) return false;
      const auto second = static_cast<unsigned char>(text[index + 1U]);
      const auto third = static_cast<unsigned char>(text[index + 2U]);
      if (!IsContinuation(second) || !IsContinuation(third) ||
          (first == 0xE0U && second < 0xA0U) ||
          (first == 0xEDU && second >= 0xA0U)) {
        return false;
      }
      index += 3U;
      continue;
    }
    if (first >= 0xF0U && first <= 0xF4U) {
      if (index + 3U >= text.size()) return false;
      const auto second = static_cast<unsigned char>(text[index + 1U]);
      const auto third = static_cast<unsigned char>(text[index + 2U]);
      const auto fourth = static_cast<unsigned char>(text[index + 3U]);
      if (!IsContinuation(second) || !IsContinuation(third) ||
          !IsContinuation(fourth) || (first == 0xF0U && second < 0x90U) ||
          (first == 0xF4U && second >= 0x90U)) {
        return false;
      }
      index += 4U;
      continue;
    }
    return false;
  }
  return true;
}

void CheckedStringAdd(std::size_t& value, std::size_t addition,
                      std::size_t maximum, std::string_view operation) {
  if (addition > maximum || value > maximum - addition) {
    ThrowResource(std::string(operation),
                  "at most " + std::to_string(maximum) + " bytes",
                  "more than the configured bound",
                  "Reduce checkpoint manifest metadata or raise the explicit "
                  "limit.");
  }
  value += addition;
}

void CheckedBlobByteAdd(std::uint64_t& value, std::uint64_t addition,
                        std::uint64_t maximum, std::string_view operation) {
  if (addition > maximum || value > maximum - addition) {
    ThrowResource(std::string(operation),
                  "at most " + std::to_string(maximum) + " bytes",
                  "more than the configured bound",
                  "Split the checkpoint or raise the explicit blob limit.");
  }
  value += addition;
}

void CheckedNodeAdd(std::size_t& value, std::size_t addition,
                    std::size_t maximum) {
  if (addition > maximum || value > maximum - addition) {
    ThrowResource("manifest.nodes",
                  "at most " + std::to_string(maximum) + " nodes",
                  "more than the configured bound",
                  "Reduce participants, fields, degradation reasons, or "
                  "blobs.");
  }
  value += addition;
}

void CountString(std::string_view value, std::string_view operation,
                 const CheckpointManifestLimits& limits,
                 std::size_t& total_string_bytes) {
  if (!IsValidUtf8(value)) {
    ThrowInvalid(std::string(operation), "well-formed UTF-8",
                 "invalid UTF-8 byte sequence",
                 "Encode manifest metadata as strict UTF-8.");
  }
  if (value.size() > limits.max_string_bytes) {
    ThrowResource(
        std::string(operation),
        "at most " + std::to_string(limits.max_string_bytes) + " UTF-8 bytes",
        std::to_string(value.size()) + " bytes",
        "Shorten the manifest metadata value.");
  }
  CheckedStringAdd(total_string_bytes, value.size(),
                   limits.max_total_string_bytes, "strings.total_bytes");
}

bool IsPortableRelativePath(std::string_view path) noexcept {
  if (path.empty() || path.front() == '/' || path.back() == '/' ||
      path.find('\\') != std::string_view::npos ||
      path.find(':') != std::string_view::npos) {
    return false;
  }
  std::size_t start = 0U;
  while (start < path.size()) {
    const auto end = path.find('/', start);
    const auto segment =
        path.substr(start, end == std::string_view::npos ? path.size() - start
                                                         : end - start);
    if (segment.empty() || segment == "." || segment == ".." ||
        !std::all_of(segment.begin(), segment.end(), [](char value) {
          const auto byte = static_cast<unsigned char>(value);
          return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
                 (byte >= '0' && byte <= '9') || value == '.' || value == '_' ||
                 value == '-';
        })) {
      return false;
    }
    if (end == std::string_view::npos) break;
    start = end + 1U;
  }
  return true;
}

bool IsSha256(std::string_view value) noexcept {
  return value.size() == 64U &&
         std::all_of(value.begin(), value.end(), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

void ValidateBuild(const BuildFingerprint& build) {
  validate_capability_manifest(CapabilityManifest{
      .schema_version = CapabilityManifest::format_version,
      .build = build,
      .features = {},
  });
}

void RequireObject(const Json& value, std::string_view operation) {
  if (!value.is_object()) {
    ThrowSerialization(std::string(operation), "a JSON object",
                       value.type_name(), "Use the documented object shape.");
  }
}

void RequireArray(const Json& value, std::string_view operation) {
  if (!value.is_array()) {
    ThrowSerialization(std::string(operation), "a JSON array",
                       value.type_name(), "Use the documented array shape.");
  }
}

void RequireFields(const Json& value, std::string_view operation,
                   std::initializer_list<std::string_view> fields) {
  RequireObject(value, operation);
  std::set<std::string_view> allowed(fields);
  for (const auto& [key, unused] : value.items()) {
    static_cast<void>(unused);
    if (!allowed.contains(key)) {
      ThrowSerialization(std::string(operation) + "." + key,
                         "a documented field", "unknown field",
                         "Remove fields not defined by schema version 1.");
    }
  }
  for (const auto field : fields) {
    if (!value.contains(std::string(field))) {
      ThrowSerialization(std::string(operation) + "." + std::string(field),
                         "a required field", "missing",
                         "Write every schema-v1 manifest field explicitly.");
    }
  }
}

std::string ReadString(const Json& value, std::string_view operation) {
  if (!value.is_string()) {
    ThrowSerialization(std::string(operation), "a JSON string",
                       value.type_name(), "Use a quoted UTF-8 string.");
  }
  return value.get<std::string>();
}

std::uint64_t ReadUnsigned(const Json& value, std::string_view operation) {
  if (!value.is_number_unsigned()) {
    ThrowSerialization(std::string(operation), "an unsigned JSON integer",
                       value.type_name(),
                       "Use a non-negative integer within uint64 range.");
  }
  return value.get<std::uint64_t>();
}

std::uint32_t ReadVersion(const Json& value, std::string_view operation) {
  const auto version = ReadUnsigned(value, operation);
  if (version > std::numeric_limits<std::uint32_t>::max()) {
    ThrowSerialization(std::string(operation), "a uint32 schema version",
                       std::to_string(version),
                       "Use a supported bounded schema version.");
  }
  return static_cast<std::uint32_t>(version);
}

bool ReadBoolean(const Json& value, std::string_view operation) {
  if (!value.is_boolean()) {
    ThrowSerialization(std::string(operation), "a JSON Boolean",
                       value.type_name(), "Use true or false.");
  }
  return value.get<bool>();
}

BuildFingerprint ParseBuild(const Json& value) {
  RequireFields(value, "build",
                {"jyppx_version", "git_commit", "compiler", "cxx_standard",
                 "os_arch", "build_type", "libtorch_version", "libtorch_build",
                 "accelerator_runtime", "dependency_lock_digest"});
  return {
      .jyppx_version =
          ReadString(value.at("jyppx_version"), "build.jyppx_version"),
      .git_commit = ReadString(value.at("git_commit"), "build.git_commit"),
      .compiler = ReadString(value.at("compiler"), "build.compiler"),
      .cxx_standard =
          ReadString(value.at("cxx_standard"), "build.cxx_standard"),
      .os_arch = ReadString(value.at("os_arch"), "build.os_arch"),
      .build_type = ReadString(value.at("build_type"), "build.build_type"),
      .libtorch_version =
          ReadString(value.at("libtorch_version"), "build.libtorch_version"),
      .libtorch_build =
          ReadString(value.at("libtorch_build"), "build.libtorch_build"),
      .accelerator_runtime = ReadString(value.at("accelerator_runtime"),
                                        "build.accelerator_runtime"),
      .dependency_lock_digest = ReadString(value.at("dependency_lock_digest"),
                                           "build.dependency_lock_digest")};
}

RestoreLevel ParseRestoreLevel(const Json& value) {
  const auto name = ReadString(value, "participants.state.restore_level");
  if (name == "exact") return RestoreLevel::exact;
  if (name == "bounded") return RestoreLevel::bounded;
  if (name == "best_effort") return RestoreLevel::best_effort;
  ThrowSerialization("participants.state.restore_level",
                     "exact, bounded, or best_effort", name,
                     "Reject unrecognized persisted restore levels.");
}

StateFieldDescriptor ParseStateField(const Json& value) {
  RequireFields(value, "participants.state.fields[]",
                {"name", "media_type", "max_bytes", "required"});
  return {
      .name = ReadString(value.at("name"), "participants.state.fields[].name"),
      .media_type = ReadString(value.at("media_type"),
                               "participants.state.fields[].media_type"),
      .max_bytes = ReadUnsigned(value.at("max_bytes"),
                                "participants.state.fields[].max_bytes"),
      .required = ReadBoolean(value.at("required"),
                              "participants.state.fields[].required")};
}

StateManifest ParseState(const Json& value) {
  RequireFields(value, "participants.state",
                {"participant_id", "schema_version", "restore_level", "fields",
                 "degradation_reasons"});
  RequireArray(value.at("fields"), "participants.state.fields");
  RequireArray(value.at("degradation_reasons"),
               "participants.state.degradation_reasons");
  StateManifest state{
      .participant_id = ReadString(value.at("participant_id"),
                                   "participants.state.participant_id"),
      .schema_version = ReadVersion(value.at("schema_version"),
                                    "participants.state.schema_version"),
      .restore_level = ParseRestoreLevel(value.at("restore_level")),
      .fields = {},
      .degradation_reasons = {},
  };
  for (const auto& field : value.at("fields")) {
    state.fields.push_back(ParseStateField(field));
  }
  for (const auto& reason : value.at("degradation_reasons")) {
    state.degradation_reasons.push_back(
        ReadString(reason, "participants.state.degradation_reasons[]"));
  }
  return state;
}

ArtifactDigest ParseDigest(const Json& value) {
  RequireFields(value, "participants.blobs[].digest", {"algorithm", "hex"});
  const auto algorithm = ReadString(value.at("algorithm"),
                                    "participants.blobs[].digest.algorithm");
  if (algorithm != "sha256") {
    ThrowSerialization("participants.blobs[].digest.algorithm", "sha256",
                       algorithm,
                       "Use an explicitly supported digest algorithm.");
  }
  return {
      .algorithm = DigestAlgorithm::sha256,
      .hex = ReadString(value.at("hex"), "participants.blobs[].digest.hex")};
}

CheckpointBlobEntry ParseBlob(const Json& value) {
  RequireFields(value, "participants.blobs[]",
                {"field_name", "relative_path", "size_bytes", "digest"});
  return {.field_name = ReadString(value.at("field_name"),
                                   "participants.blobs[].field_name"),
          .relative_path = ReadString(value.at("relative_path"),
                                      "participants.blobs[].relative_path"),
          .size_bytes = ReadUnsigned(value.at("size_bytes"),
                                     "participants.blobs[].size_bytes"),
          .digest = ParseDigest(value.at("digest"))};
}

CheckpointParticipantEntry ParseParticipant(const Json& value) {
  RequireFields(value, "participants[]", {"state", "blobs"});
  RequireArray(value.at("blobs"), "participants[].blobs");
  CheckpointParticipantEntry participant{
      .state = ParseState(value.at("state")),
      .blobs = {},
  };
  for (const auto& blob : value.at("blobs")) {
    participant.blobs.push_back(ParseBlob(blob));
  }
  return participant;
}

Json BuildJson(const BuildFingerprint& build) {
  return Json{{"accelerator_runtime", build.accelerator_runtime},
              {"build_type", build.build_type},
              {"compiler", build.compiler},
              {"cxx_standard", build.cxx_standard},
              {"dependency_lock_digest", build.dependency_lock_digest},
              {"git_commit", build.git_commit},
              {"jyppx_version", build.jyppx_version},
              {"libtorch_build", build.libtorch_build},
              {"libtorch_version", build.libtorch_version},
              {"os_arch", build.os_arch}};
}

Json StateJson(const StateManifest& state) {
  Json fields = Json::array();
  for (const auto& field : state.fields) {
    fields.push_back(Json{{"max_bytes", field.max_bytes},
                          {"media_type", field.media_type},
                          {"name", field.name},
                          {"required", field.required}});
  }
  return Json{
      {"degradation_reasons", state.degradation_reasons},
      {"fields", std::move(fields)},
      {"participant_id", state.participant_id},
      {"restore_level", std::string(restore_level_name(state.restore_level))},
      {"schema_version", state.schema_version}};
}

Json BlobJson(const CheckpointBlobEntry& blob) {
  return Json{{"digest", Json{{"algorithm", std::string(digest_algorithm_name(
                                                blob.digest.algorithm))},
                              {"hex", blob.digest.hex}}},
              {"field_name", blob.field_name},
              {"relative_path", blob.relative_path},
              {"size_bytes", blob.size_bytes}};
}

CheckpointArtifactManifest Canonicalized(CheckpointArtifactManifest manifest) {
  std::sort(manifest.participants.begin(), manifest.participants.end(),
            [](const auto& left, const auto& right) {
              return left.state.participant_id < right.state.participant_id;
            });
  for (auto& participant : manifest.participants) {
    std::sort(participant.state.fields.begin(), participant.state.fields.end(),
              [](const auto& left, const auto& right) {
                return left.name < right.name;
              });
    std::sort(participant.state.degradation_reasons.begin(),
              participant.state.degradation_reasons.end());
    std::sort(participant.blobs.begin(), participant.blobs.end(),
              [](const auto& left, const auto& right) {
                return left.field_name < right.field_name;
              });
  }
  return manifest;
}

}  // namespace

std::string_view digest_algorithm_name(DigestAlgorithm algorithm) noexcept {
  switch (algorithm) {
    case DigestAlgorithm::sha256:
      return "sha256";
  }
  return "unknown";
}

void validate_checkpoint_artifact_manifest(
    const CheckpointArtifactManifest& manifest,
    const CheckpointManifestLimits& limits) {
  try {
    ValidateLimits(limits);
    if (manifest.schema_version != CheckpointArtifactManifest::format_version) {
      ThrowCompatibility(
          "schema_version",
          std::to_string(CheckpointArtifactManifest::format_version),
          std::to_string(manifest.schema_version),
          "Migrate the manifest explicitly before reading its blobs.");
    }
    ValidateBuild(manifest.build);
    if (manifest.participants.empty()) {
      ThrowInvalid("participants", "at least one state participant", "empty",
                   "Do not publish an empty checkpoint artifact.");
    }
    if (manifest.participants.size() > limits.max_participants) {
      ThrowResource(
          "participants",
          "at most " + std::to_string(limits.max_participants) + " entries",
          std::to_string(manifest.participants.size()) + " entries",
          "Split unrelated state into separate artifacts.");
    }
    if (limits.max_depth < 6U) {
      ThrowResource("manifest.depth", "at least 6 for schema version 1",
                    std::to_string(limits.max_depth),
                    "Raise the depth limit to represent the fixed manifest "
                    "shape.");
    }

    std::size_t total_strings = 0U;
    std::size_t total_nodes = 27U;
    if (total_nodes > limits.max_nodes) {
      ThrowResource("manifest.nodes",
                    "at most " + std::to_string(limits.max_nodes) + " nodes",
                    std::to_string(total_nodes) + " fixed schema nodes",
                    "Raise the node limit to represent schema version 1.");
    }
    const std::array<std::string_view, 10U> build_strings{
        manifest.build.jyppx_version,
        manifest.build.git_commit,
        manifest.build.compiler,
        manifest.build.cxx_standard,
        manifest.build.os_arch,
        manifest.build.build_type,
        manifest.build.libtorch_version,
        manifest.build.libtorch_build,
        manifest.build.accelerator_runtime,
        manifest.build.dependency_lock_digest};
    for (const auto value : build_strings) {
      CountString(value, "build", limits, total_strings);
    }

    std::unordered_set<std::string> participant_ids;
    std::unordered_set<std::string> artifact_paths;
    std::size_t total_blobs = 0U;
    std::uint64_t total_blob_bytes = 0U;
    for (const auto& participant : manifest.participants) {
      CheckedNodeAdd(total_nodes, 15U, limits.max_nodes);
      validate_state_manifest(participant.state);
      if (!participant_ids.insert(participant.state.participant_id).second) {
        ThrowInvalid("participants.state.participant_id",
                     "unique participant identities",
                     participant.state.participant_id,
                     "Merge duplicate participant entries before commit.");
      }
      CountString(participant.state.participant_id,
                  "participants.state.participant_id", limits, total_strings);

      std::unordered_map<std::string, const StateFieldDescriptor*> fields;
      fields.reserve(participant.state.fields.size());
      for (const auto& field : participant.state.fields) {
        CheckedNodeAdd(total_nodes, 9U, limits.max_nodes);
        fields.emplace(field.name, &field);
        CountString(field.name, "participants.state.fields.name", limits,
                    total_strings);
        CountString(field.media_type, "participants.state.fields.media_type",
                    limits, total_strings);
      }
      for (const auto& reason : participant.state.degradation_reasons) {
        CheckedNodeAdd(total_nodes, 1U, limits.max_nodes);
        CountString(reason, "participants.state.degradation_reasons", limits,
                    total_strings);
      }

      if (participant.blobs.size() > limits.max_blobs - total_blobs) {
        ThrowResource(
            "participants.blobs",
            "at most " + std::to_string(limits.max_blobs) + " total entries",
            "more than the configured bound",
            "Split the checkpoint into bounded artifacts.");
      }
      total_blobs += participant.blobs.size();
      std::unordered_set<std::string> blob_fields;
      for (const auto& blob : participant.blobs) {
        CheckedNodeAdd(total_nodes, 13U, limits.max_nodes);
        const auto field = fields.find(blob.field_name);
        if (field == fields.end()) {
          ThrowInvalid("participants.blobs.field_name",
                       "a field declared by the participant state manifest",
                       blob.field_name,
                       "Remove undeclared blobs or update the state schema.");
        }
        if (!blob_fields.insert(blob.field_name).second) {
          ThrowInvalid("participants.blobs.field_name",
                       "one blob per declared field", blob.field_name,
                       "Remove the duplicate concrete blob entry.");
        }
        if (blob.size_bytes > field->second->max_bytes) {
          ThrowResource(
              "participants.blobs.size_bytes",
              "at most the field max_bytes " +
                  std::to_string(field->second->max_bytes),
              std::to_string(blob.size_bytes),
              "Reject oversized state before allocating or reading bytes.");
        }
        CheckedBlobByteAdd(total_blob_bytes, blob.size_bytes,
                           limits.max_total_blob_bytes, "blobs.total_bytes");
        CountString(blob.field_name, "participants.blobs.field_name", limits,
                    total_strings);
        if (blob.relative_path.size() > limits.max_path_bytes) {
          ThrowResource(
              "participants.blobs.relative_path",
              "at most " + std::to_string(limits.max_path_bytes) + " bytes",
              std::to_string(blob.relative_path.size()) + " bytes",
              "Use a short portable artifact-relative path.");
        }
        CountString(blob.relative_path, "participants.blobs.relative_path",
                    limits, total_strings);
        if (!IsPortableRelativePath(blob.relative_path)) {
          ThrowInvalid(
              "participants.blobs.relative_path",
              "a normalized relative ASCII path without '.', '..', '\\' or ':'",
              blob.relative_path,
              "Use portable slash-separated artifact path segments.");
        }
        if (!artifact_paths.insert(blob.relative_path).second) {
          ThrowInvalid("participants.blobs.relative_path",
                       "a unique path for every blob", blob.relative_path,
                       "Assign each blob a distinct artifact path.");
        }
        if (digest_algorithm_name(blob.digest.algorithm) == "unknown") {
          ThrowInvalid(
              "participants.blobs.digest.algorithm", "sha256",
              std::to_string(static_cast<std::uint8_t>(blob.digest.algorithm)),
              "Reject unrecognized persisted digest algorithms.");
        }
        CountString(blob.digest.hex, "participants.blobs.digest.hex", limits,
                    total_strings);
        if (!IsSha256(blob.digest.hex)) {
          ThrowInvalid(
              "participants.blobs.digest.hex",
              "64 lowercase hexadecimal characters", blob.digest.hex,
              "Encode the complete SHA-256 digest in lowercase hexadecimal.");
        }
      }
      for (const auto& [name, descriptor] : fields) {
        if (descriptor->required && !blob_fields.contains(name)) {
          ThrowInvalid(
              "participants.blobs",
              "every required state field to have one blob", name,
              "Write all required state before publishing a manifest.");
        }
      }
    }
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    ThrowResource("validation", "bounded validation memory",
                  "allocation failed",
                  "Reduce manifest size or available participant state.");
  }
}

CheckpointArtifactManifest parse_checkpoint_artifact_manifest_json(
    std::string_view json, const CheckpointManifestLimits& limits) {
  try {
    ValidateLimits(limits);
    if (json.size() > limits.max_manifest_bytes) {
      ThrowResource(
          "manifest.bytes",
          "at most " + std::to_string(limits.max_manifest_bytes) + " bytes",
          std::to_string(json.size()) + " bytes",
          "Reject oversized metadata before JSON parsing.");
    }
    if (!IsValidUtf8(json)) {
      ThrowSerialization("manifest.encoding", "well-formed UTF-8",
                         "invalid UTF-8 byte sequence",
                         "Encode the complete manifest as strict UTF-8.");
    }

    std::vector<std::set<std::string>> object_keys;
    std::size_t nodes = 0U;
    std::size_t total_strings = 0U;
    auto callback = [&](int depth, Json::parse_event_t event, Json& parsed) {
      if ((event == Json::parse_event_t::object_start ||
           event == Json::parse_event_t::array_start) &&
          static_cast<std::size_t>(depth + 1) > limits.max_depth) {
        ThrowResource(
            "manifest.depth", "at most " + std::to_string(limits.max_depth),
            std::to_string(depth + 1), "Reduce nested JSON containers.");
      }
      if (event == Json::parse_event_t::object_start) {
        object_keys.emplace_back();
      } else if (event == Json::parse_event_t::key) {
        const auto key = parsed.get<std::string>();
        if (object_keys.empty() || !object_keys.back().insert(key).second) {
          ThrowSerialization("manifest.object_key", "unique object keys", key,
                             "Remove the duplicate JSON field.");
        }
      } else if (event == Json::parse_event_t::object_end) {
        if (!object_keys.empty()) object_keys.pop_back();
      }
      if (event == Json::parse_event_t::object_start ||
          event == Json::parse_event_t::array_start ||
          event == Json::parse_event_t::key ||
          event == Json::parse_event_t::value) {
        if (nodes == limits.max_nodes) {
          ThrowResource("manifest.nodes",
                        "at most " + std::to_string(limits.max_nodes),
                        "more than the configured bound",
                        "Reduce manifest object and array entries.");
        }
        ++nodes;
      }
      if ((event == Json::parse_event_t::key ||
           event == Json::parse_event_t::value) &&
          parsed.is_string()) {
        CountString(parsed.get_ref<const std::string&>(), "manifest.strings",
                    limits, total_strings);
      }
      return true;
    };

    const auto root =
        Json::parse(json.begin(), json.end(), callback, true, false);
    RequireFields(root, "manifest",
                  {"schema_version", "build", "participants"});
    RequireArray(root.at("participants"), "participants");
    CheckpointArtifactManifest manifest{
        .schema_version =
            ReadVersion(root.at("schema_version"), "schema_version"),
        .build = ParseBuild(root.at("build")),
        .participants = {},
    };
    for (const auto& participant : root.at("participants")) {
      manifest.participants.push_back(ParseParticipant(participant));
    }
    validate_checkpoint_artifact_manifest(manifest, limits);
    return manifest;
  } catch (const Error&) {
    throw;
  } catch (const nlohmann::json::parse_error& error) {
    ThrowSerialization(
        "manifest.syntax", "strict RFC 8259 JSON",
        "parser error " + std::to_string(error.id) + " at byte " +
            std::to_string(error.byte),
        "Fix JSON syntax, number range, encoding, or duplicate fields.");
  } catch (const nlohmann::json::exception& error) {
    ThrowSerialization("manifest.value", "schema-v1 JSON values",
                       "parser error " + std::to_string(error.id),
                       "Fix the documented field type or numeric range.");
  } catch (const std::bad_alloc&) {
    ThrowResource("parsing", "bounded parser memory", "allocation failed",
                  "Reduce the encoded manifest size.");
  }
}

std::string checkpoint_artifact_manifest_json(
    const CheckpointArtifactManifest& manifest,
    const CheckpointManifestLimits& limits) {
  try {
    validate_checkpoint_artifact_manifest(manifest, limits);
    const auto canonical = Canonicalized(manifest);
    Json participants = Json::array();
    for (const auto& participant : canonical.participants) {
      Json blobs = Json::array();
      for (const auto& blob : participant.blobs) {
        blobs.push_back(BlobJson(blob));
      }
      participants.push_back(Json{{"blobs", std::move(blobs)},
                                  {"state", StateJson(participant.state)}});
    }
    const Json root{{"build", BuildJson(canonical.build)},
                    {"participants", std::move(participants)},
                    {"schema_version", canonical.schema_version}};
    auto encoded = root.dump();
    if (encoded.size() > limits.max_manifest_bytes) {
      ThrowResource(
          "manifest.bytes",
          "at most " + std::to_string(limits.max_manifest_bytes) + " bytes",
          std::to_string(encoded.size()) + " bytes",
          "Reduce metadata before serializing the manifest.");
    }
    return encoded;
  } catch (const Error&) {
    throw;
  } catch (const nlohmann::json::exception& error) {
    ThrowSerialization("serialization", "schema-v1 JSON",
                       "encoder error " + std::to_string(error.id),
                       "Validate UTF-8 metadata before serialization.");
  } catch (const std::bad_alloc&) {
    ThrowResource("serialization", "bounded encoder memory",
                  "allocation failed", "Reduce manifest metadata.");
  }
}

CapabilityDescriptor make_checkpoint_manifest_capability(
    const CheckpointManifestLimits& limits) {
  ValidateLimits(limits);
  CapabilityDescriptor descriptor{
      .feature_id = "interop.checkpoint_manifest",
      .scope_id = "",
      .feature_version = 1U,
      .status = CapabilityStatus::experimental,
      .implementation = CapabilityImplementation::native,
      .source = CapabilitySource::build,
      .backend = "jyppx",
      .devices = {},
      .dtypes = {},
      .schemas = {"jyppx.checkpoint.artifact_manifest.v1"},
      .operations = {"parse", "serialize", "validate"},
      .limits = {{.name = "max_manifest_bytes",
                  .value = limits.max_manifest_bytes,
                  .unit = "bytes"},
                 {.name = "max_participants",
                  .value = limits.max_participants,
                  .unit = "count"},
                 {.name = "max_blobs",
                  .value = limits.max_blobs,
                  .unit = "count"},
                 {.name = "max_total_blob_bytes",
                  .value = limits.max_total_blob_bytes,
                  .unit = "bytes"}},
      .limitations =
          {"digest values are syntax-checked; blob hashing and atomic "
           "filesystem publication are outside this codec"},
  };
  validate_capability_descriptor(descriptor);
  return descriptor;
}

}  // namespace jyppx::interop
