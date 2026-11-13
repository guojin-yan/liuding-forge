// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_PRIVATE_ARTIFACT_IO_HPP
#define JYPPX_LIUDING_VISION_TASKS_PRIVATE_ARTIFACT_IO_HPP
#include <jyppx/liuding/foundation/error.hpp>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
namespace jyppx::liuding::vision::tasks::detail {
inline std::vector<std::byte> read_artifact(const std::filesystem::path& path,
                                            std::uint64_t maximum) {
  const auto count = std::filesystem::file_size(path);
  if (count > maximum ||
      count > static_cast<std::uint64_t>(
                  (std::numeric_limits<std::streamsize>::max)()))
    throw ResourceExhausted("Vision artifact exceeds its byte bound");
  std::vector<std::byte> bytes(static_cast<std::size_t>(count));
  std::ifstream input(path, std::ios::binary);
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input || input.peek() != std::char_traits<char>::eof())
    throw IoError("Vision artifact changed or could not be read");
  return bytes;
}
// Same-directory replacement keeps the last complete artifact visible.
// Directory durability and concurrent writers to the same run remain the
// caller's concern.
inline void publish_artifact(const std::filesystem::path& path,
                             std::span<const std::byte> bytes) {
  if (path.empty()) throw InvalidArgument("Vision artifact path is empty");
  static std::atomic<std::uint64_t> sequence{0};
  if (!path.parent_path().empty())
    std::filesystem::create_directories(path.parent_path());
  auto temporary = path;
  temporary += std::filesystem::path(
      ".pending-" +
      std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()) +
      "-" + std::to_string(sequence.fetch_add(1)));
  try {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    output.close();
    if (!output.good()) throw IoError("Cannot stage vision artifact");
#ifdef _WIN32
    if (!MoveFileExW(temporary.c_str(), path.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
      throw IoError("Cannot atomically replace vision artifact");
#else
    std::filesystem::rename(temporary, path);
#endif
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
}
inline void publish_text(const std::filesystem::path& path,
                         std::string_view text) {
  publish_artifact(path, std::as_bytes(std::span(text)));
}
class ArtifactDirectory {
  std::filesystem::path target_, temporary_;
  bool committed_ = false;

 public:
  explicit ArtifactDirectory(std::filesystem::path target)
      : target_(std::move(target)) {
    if (target_.empty() || (std::filesystem::exists(target_) &&
                            !std::filesystem::is_empty(target_)))
      throw InvalidArgument("Artifact directory must be new or empty");
    if (!target_.parent_path().empty())
      std::filesystem::create_directories(target_.parent_path());
    static std::atomic<std::uint64_t> sequence{0};
    temporary_ = target_;
    temporary_ += std::filesystem::path(
        ".pending-" +
        std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) +
        "-" + std::to_string(sequence.fetch_add(1)));
    if (!std::filesystem::create_directory(temporary_))
      throw IoError("Cannot reserve artifact staging directory");
  }
  ~ArtifactDirectory() {
    if (!committed_) {
      std::error_code ignored;
      std::filesystem::remove_all(temporary_, ignored);
    }
  }
  const std::filesystem::path& path() const noexcept { return temporary_; }
  void commit() {
    if (std::filesystem::exists(target_)) {
      if (!std::filesystem::is_empty(target_))
        throw IoError("Artifact destination became nonempty");
      std::filesystem::remove(target_);
    }
    std::filesystem::rename(temporary_, target_);
    committed_ = true;
  }
  ArtifactDirectory(const ArtifactDirectory&) = delete;
  ArtifactDirectory& operator=(const ArtifactDirectory&) = delete;
};
}  // namespace jyppx::liuding::vision::tasks::detail
#endif  // JYPPX_LIUDING_VISION_TASKS_PRIVATE_ARTIFACT_IO_HPP
