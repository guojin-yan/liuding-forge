// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_TOOLS_CLI_RECIPE_SUPPORT_HPP
#define JYPPX_TOOLS_CLI_RECIPE_SUPPORT_HPP

#include "cli.hpp"

#include <jyppx/liuding/foundation/artifact.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <array>
#include <chrono>
#include <fstream>

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

namespace jyppx::liuding::cli::recipe_support {
inline constexpr std::uint64_t kArtifactLimit = 256ULL << 20U;
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
inline void write_file(const std::filesystem::path& path,
                       std::span<const std::byte> content) {
  require_recipe(content.size() <= kArtifactLimit,
                 "Recipe artifact exceeds 256 MiB");
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(reinterpret_cast<const char*>(content.data()),
               static_cast<std::streamsize>(content.size()));
  stream.flush();
  if (!stream)
    throw IoError("Cannot write recipe artifact: " + path_text(path));
  stream.close();
  if (!stream)
    throw IoError("Cannot close recipe artifact: " + path_text(path));
}
inline void write_json(const std::filesystem::path& path, const Json& value) {
  write_file(path, bytes(value.dump(2) + "\n"));
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

class RunLock final {
 public:
  explicit RunLock(const std::filesystem::path& root) {
    const auto path = root / ".recipe.lock";
    require_recipe(
        !std::filesystem::is_symlink(std::filesystem::symlink_status(path)),
        "Recipe lock must not be a link");
#if defined(_WIN32)
    handle_ = CreateFileW(
        path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    if (handle_ == INVALID_HANDLE_VALUE)
      throw IoError("Recipe directory is in use or inaccessible");
#else
    handle_ = ::open(path.c_str(), O_CREAT | O_RDWR | O_NOFOLLOW, 0600);
    if (handle_ < 0) throw IoError("Cannot open recipe lock");
    if (::flock(handle_, LOCK_EX | LOCK_NB) != 0) {
      ::close(handle_);
      handle_ = -1;
      throw IoError("Recipe directory is already in use");
    }
#endif
  }
  ~RunLock() {
#if defined(_WIN32)
    if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
#else
    if (handle_ >= 0) ::close(handle_);
#endif
  }
  RunLock(const RunLock&) = delete;
  RunLock& operator=(const RunLock&) = delete;

 private:
#if defined(_WIN32)
  HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
  int handle_ = -1;
#endif
};

}  // namespace jyppx::liuding::cli::recipe_support
#endif  // JYPPX_TOOLS_CLI_RECIPE_SUPPORT_HPP
