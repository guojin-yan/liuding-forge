// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
#ifndef JYPPX_LIUDING_LOCAL_EXECUTION_SUPPORT_HPP
#define JYPPX_LIUDING_LOCAL_EXECUTION_SUPPORT_HPP
#include <jyppx/liuding/foundation/artifact.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <initializer_list>
#include <nlohmann/json.hpp>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif
namespace jyppx::liuding::experiment::local_execution_detail {
using Json = nlohmann::json;
struct Result {
  Json data;
  int exit_code = 0;
};
inline constexpr std::uint64_t kArtifactLimit = 256ULL << 20U;
inline constexpr std::uint64_t kExecutableLimit = 1ULL << 30U;
inline void require_recipe(bool value, std::string_view reason) {
  if (!value) throw InvalidArgument(std::string(reason));
}
inline std::span<const std::byte> bytes(std::string_view text) {
  return std::as_bytes(std::span(text.data(), text.size()));
}
inline std::string path_text(const std::filesystem::path& path) {
  const auto text = path.generic_u8string();
  return {reinterpret_cast<const char*>(text.data()), text.size()};
}
inline std::uint64_t now_ms() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}
inline ArtifactDigest digest(std::string_view text) {
  return {.hex = sha256_hex(bytes(text))};
}

// Preflight identities require a quiescent dataset, not a filesystem snapshot.
inline ArtifactDigest file_digest(const std::filesystem::path& path,
                                  std::uint64_t maximum = kArtifactLimit) {
  auto cursor = std::filesystem::absolute(path).lexically_normal();
  while (!cursor.empty()) {
    require_recipe(
        !std::filesystem::is_symlink(std::filesystem::symlink_status(cursor)),
        "Recipe paths must not traverse symbolic links");
#if defined(_WIN32)
    const auto attributes = GetFileAttributesW(cursor.c_str());
    require_recipe(attributes != INVALID_FILE_ATTRIBUTES &&
                       (attributes & FILE_ATTRIBUTE_REPARSE_POINT) == 0,
                   "Recipe paths must not traverse reparse points");
#endif
    const auto parent = cursor.parent_path();
    if (parent == cursor) break;
    cursor = parent;
  }
  require_recipe(std::filesystem::is_regular_file(path),
                 "Recipe input must be a regular file");
  const auto expected = std::filesystem::file_size(path);
  require_recipe(expected <= maximum,
                 "Recipe file exceeds configured byte limit");
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw IoError("Cannot read recipe input");
  Sha256 hash;
  std::array<char, 65536> buffer{};
  std::uint64_t count = 0;
  while (stream) {
    stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto got = static_cast<std::size_t>(stream.gcount());
    require_recipe(got <= maximum - count,
                   "Recipe file grew beyond its byte limit");
    count += got;
    hash.update(std::as_bytes(std::span(buffer.data(), got)));
  }
  if (!stream.eof() || count != expected ||
      std::filesystem::file_size(path) != expected)
    throw IoError("Recipe input changed while computing its content identity");
  return {.hex = sha256_digest_hex(hash.finish())};
}

inline void fields(const Json& value,
                   std::initializer_list<std::string_view> required,
                   std::initializer_list<std::string_view> optional = {}) {
  if (!value.is_object()) throw SerializationError("Expected a JSON object");
  for (const auto field : required) {
    if (!value.contains(std::string(field)))
      throw SerializationError("Missing required JSON field");
  }
  for (const auto& item : value.items()) {
    if (std::find(required.begin(), required.end(), item.key()) ==
            required.end() &&
        std::find(optional.begin(), optional.end(), item.key()) ==
            optional.end()) {
      throw SerializationError("Unknown JSON field");
    }
  }
}

inline std::uint64_t unsigned_number(const Json& value, std::uint64_t maximum) {
  if (!(value.is_number_unsigned() ||
        (value.is_number_integer() && value.get<std::int64_t>() >= 0))) {
    throw SerializationError("Expected a nonnegative JSON integer");
  }
  const auto result = value.get<std::uint64_t>();
  if (result > maximum)
    throw ResourceExhausted("JSON integer exceeds the supported range");
  return result;
}

inline Json parse_json(std::string_view value) {
  if (value.size() > 8U * 1024U * 1024U)
    throw ResourceExhausted("Command JSON input exceeds 8 MiB");
  std::size_t nodes = 0U;
  std::vector<std::set<std::string>> keys;
  const auto callback = [&](int depth, Json::parse_event_t event,
                            Json& parsed) {
    if (depth > 32 || ++nodes > 262144U)
      throw ResourceExhausted("Command JSON structure limit exceeded");
    if (event == Json::parse_event_t::object_start) keys.emplace_back();
    if (event == Json::parse_event_t::key &&
        (keys.empty() ||
         !keys.back().insert(parsed.get<std::string>()).second)) {
      throw SerializationError("Duplicate JSON field");
    }
    if (event == Json::parse_event_t::object_end && !keys.empty())
      keys.pop_back();
    if (parsed.is_string() &&
        parsed.get_ref<const std::string&>().size() > 65536U) {
      throw ResourceExhausted("Command JSON string exceeds 64 KiB");
    }
    return true;
  };
  try {
    return Json::parse(value.begin(), value.end(), callback, true, false);
  } catch (const Json::exception&) {
    throw SerializationError("Malformed command JSON or UTF-8");
  }
}

}  // namespace jyppx::liuding::experiment::local_execution_detail
#endif
