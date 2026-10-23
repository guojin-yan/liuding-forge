// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_INTEROP_METADATA_VALIDATION_HPP
#define JYPPX_INTEROP_METADATA_VALIDATION_HPP

/** @file metadata_validation.hpp
 * @brief Private bounded metadata helpers; never installed as a public header.
 */

#include <jyppx/liuding/foundation/artifact.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace jyppx::liuding::interop::metadata_detail {

/** @brief Private JSON representation for metadata control messages. */
using Json = nlohmann::json;

/** @brief Builds context without echoing untrusted payloads or source paths. */
inline ErrorDetails Details(std::string_view component,
                            std::string_view operation,
                            std::string_view expected) {
  return {.component = std::string(component),
          .operation = std::string(operation),
          .expected = std::string(expected),
          .actual = "metadata does not satisfy the declared contract",
          .hint = "Use the supported schema and explicit resource limits."};
}

/** @brief Rejects invalid caller metadata without exposing its contents. */
[[noreturn]] inline void Invalid(std::string_view component,
                                 std::string_view operation,
                                 std::string_view expected) {
  throw InvalidArgument("Invalid interop metadata",
                        Details(component, operation, expected));
}

/** @brief Rejects invalid encoded metadata without exposing its contents. */
[[noreturn]] inline void Malformed(std::string_view component,
                                   std::string_view operation,
                                   std::string_view expected) {
  throw SerializationError("Malformed interop metadata",
                           Details(component, operation, expected));
}

/** @brief Reports a configured metadata resource bound. */
[[noreturn]] inline void Resource(std::string_view component,
                                  std::string_view operation,
                                  std::uint64_t maximum) {
  throw ResourceExhausted(
      "Interop metadata resource limit exceeded",
      Details(component, operation, "at most " + std::to_string(maximum)));
}

/** @brief Checks strict UTF-8, including overlong and surrogate rejection. */
inline bool Utf8(std::string_view text) noexcept {
  const auto continuation = [](unsigned char value) {
    return (value & 0xc0U) == 0x80U;
  };
  for (std::size_t index = 0U; index < text.size();) {
    const auto first = static_cast<unsigned char>(text[index]);
    if (first <= 0x7fU) {
      ++index;
      continue;
    }
    const std::size_t count = first >= 0xc2U && first <= 0xdfU   ? 2U
                              : first >= 0xe0U && first <= 0xefU ? 3U
                              : first >= 0xf0U && first <= 0xf4U ? 4U
                                                                 : 0U;
    if (count == 0U || count > text.size() - index) return false;
    const auto second = static_cast<unsigned char>(text[index + 1U]);
    if (!continuation(second) || (first == 0xe0U && second < 0xa0U) ||
        (first == 0xedU && second >= 0xa0U) ||
        (first == 0xf0U && second < 0x90U) ||
        (first == 0xf4U && second >= 0x90U))
      return false;
    for (std::size_t offset = 2U; offset < count; ++offset) {
      if (!continuation(static_cast<unsigned char>(text[index + offset])))
        return false;
    }
    index += count;
  }
  return true;
}

/** @brief Validates a bounded logical identity, never a filesystem path. */
inline void Text(std::string_view text, std::size_t maximum,
                 std::string_view component, std::string_view operation,
                 bool allow_empty = false) {
  if (text.size() > maximum) Resource(component, operation, maximum);
  if ((!allow_empty && text.empty()) || !Utf8(text) ||
      std::any_of(text.begin(), text.end(), [](unsigned char value) {
        return value < 0x20U || value == 0x7fU;
      })) {
    Invalid(component, operation, "bounded UTF-8 without control characters");
  }
}

/** @brief Checks a stable lowercase token used for machine identities. */
inline void Token(std::string_view value, std::string_view component,
                  std::string_view operation) {
  Text(value, 256U, component, operation);
  if (!std::all_of(value.begin(), value.end(), [](unsigned char byte) {
        return (byte >= 'a' && byte <= 'z') || (byte >= '0' && byte <= '9') ||
               byte == '_' || byte == '-' || byte == '.';
      })) {
    Invalid(component, operation, "a stable lowercase machine token");
  }
}

/** @brief Adds a nonnegative count with checked overflow and an upper bound. */
inline void Add(std::uint64_t& value, std::uint64_t amount,
                std::uint64_t maximum, std::string_view component,
                std::string_view operation) {
  if (value > maximum || amount > maximum - value)
    Resource(component, operation, maximum);
  value += amount;
}

/** @brief Requires exactly the declared fields, rejecting unknown keys. */
inline void Fields(const Json& value,
                   std::initializer_list<std::string_view> fields,
                   std::string_view component) {
  if (!value.is_object() || value.size() != fields.size())
    Malformed(component, "json.fields", "exactly the versioned schema fields");
  for (const auto field : fields) {
    if (!value.contains(std::string(field)))
      Malformed(component, "json.fields", "all required schema fields");
  }
}

/** @brief Reads an exact nonnegative uint64 without floating coercion. */
inline std::uint64_t Uint(const Json& value, std::string_view component) {
  if (value.is_number_unsigned()) return value.get<std::uint64_t>();
  if (value.is_number_integer()) {
    const auto number = value.get<std::int64_t>();
    if (number >= 0) return static_cast<std::uint64_t>(number);
  }
  Malformed(component, "json.integer", "a nonnegative uint64 integer");
}

/** @brief Reads a string without coercion. */
inline std::string String(const Json& value, std::string_view component) {
  if (!value.is_string())
    Malformed(component, "json.string", "a UTF-8 JSON string");
  return value.get<std::string>();
}

/** @brief Reads an exact Boolean without integer coercion. */
inline bool Boolean(const Json& value, std::string_view component) {
  if (!value.is_boolean())
    Malformed(component, "json.boolean", "a JSON boolean");
  return value.get<bool>();
}

/** @brief Parses a control message with independent structural bounds. */
inline Json Parse(std::string_view bytes, std::size_t maximum,
                  std::string_view component) {
  if (maximum == 0U || maximum > 64U * 1024U * 1024U)
    Invalid(component, "control.limit", "a positive limit at most 64 MiB");
  if (bytes.size() > maximum) Resource(component, "control.bytes", maximum);
  if (!Utf8(bytes) || bytes.empty() || bytes.front() != '{')
    Malformed(component, "control.utf8", "a strict UTF-8 JSON object");
  std::vector<std::unordered_set<std::string>> keys;
  std::uint64_t nodes = 0U;
  std::uint64_t strings = 0U;
  auto callback = [&](int depth, Json::parse_event_t event, Json& value) {
    if ((event == Json::parse_event_t::object_start ||
         event == Json::parse_event_t::array_start) &&
        depth >= 16)
      Resource(component, "control.depth", 16U);
    if (event == Json::parse_event_t::object_start ||
        event == Json::parse_event_t::array_start ||
        event == Json::parse_event_t::key ||
        event == Json::parse_event_t::value)
      Add(nodes, 1U, 65536U, component, "control.nodes");
    if (event == Json::parse_event_t::object_start) keys.emplace_back();
    if (event == Json::parse_event_t::key) {
      const auto& key = value.get_ref<const std::string&>();
      if (keys.empty() || !keys.back().insert(key).second)
        Malformed(component, "control.keys", "unique decoded object keys");
    }
    if (event == Json::parse_event_t::object_end) keys.pop_back();
    if ((event == Json::parse_event_t::key ||
         event == Json::parse_event_t::value) &&
        value.is_string()) {
      const auto& text = value.get_ref<const std::string&>();
      if (text.size() > 4096U) Resource(component, "control.string", 4096U);
      Add(strings, text.size(), maximum, component, "control.strings");
    }
    return true;
  };
  return Json::parse(bytes.begin(), bytes.end(), callback, true, false);
}

/** @brief Validates a storage-neutral reference without resolving its bytes. */
inline void Artifact(const ArtifactRef& value, std::string_view component,
                     std::uint64_t maximum_bytes) {
  validate_artifact_digest(value.digest);
  if (value.size_bytes > maximum_bytes)
    Resource(component, "artifact.bytes", maximum_bytes);
  Text(value.logical_name, 4096U, component, "artifact.name", true);
  Text(value.media_type, 4096U, component, "artifact.media_type");
  Text(value.producer, 4096U, component, "artifact.producer", true);
}

/** @brief Converts a digest to explicit official foundation identity fields. */
inline Json DigestJson(const ArtifactDigest& digest) {
  return Json{{"algorithm", artifact_digest_algorithm_name(digest.algorithm)},
              {"hex", digest.hex}};
}

/** @brief Reads a foundation digest without accepting future algorithms. */
inline ArtifactDigest ReadDigest(const Json& value,
                                 std::string_view component) {
  Fields(value, {"algorithm", "hex"}, component);
  if (String(value.at("algorithm"), component) != "sha256")
    throw CompatibilityError("Unsupported interop digest algorithm",
                             Details(component, "digest.algorithm", "sha256"));
  ArtifactDigest result{.algorithm = ArtifactDigestAlgorithm::sha256,
                        .hex = String(value.at("hex"), component)};
  validate_artifact_digest(result);
  return result;
}

/** @brief Encodes a storage-neutral reference, never a local path. */
inline Json ArtifactJson(const ArtifactRef& artifact) {
  return Json{{"digest", DigestJson(artifact.digest)},
              {"size_bytes", artifact.size_bytes},
              {"logical_name", artifact.logical_name},
              {"media_type", artifact.media_type},
              {"producer", artifact.producer}};
}

/** @brief Decodes exactly the supported ArtifactRef metadata fields. */
inline ArtifactRef ReadArtifact(const Json& value, std::string_view component) {
  Fields(value,
         {"digest", "size_bytes", "logical_name", "media_type", "producer"},
         component);
  return {.digest = ReadDigest(value.at("digest"), component),
          .size_bytes = Uint(value.at("size_bytes"), component),
          .logical_name = String(value.at("logical_name"), component),
          .media_type = String(value.at("media_type"), component),
          .producer = String(value.at("producer"), component)};
}

}  // namespace jyppx::liuding::interop::metadata_detail

#endif  // JYPPX_INTEROP_METADATA_VALIDATION_HPP
