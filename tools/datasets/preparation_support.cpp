// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "preparation_support.hpp"

#include <jyppx/liuding/foundation/error.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <system_error>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#ifndef JYPPX_LIUDING_SOURCE_ROOT
#error "Dataset preparation requires JYPPX_LIUDING_SOURCE_ROOT"
#endif
namespace jyppx::liuding::datasets::detail {
namespace fs = std::filesystem;
void reject_links(const fs::path& path) {
  for (auto cursor = path; !cursor.empty();) {
    std::error_code error;
    const auto status = fs::symlink_status(cursor, error);
    if (error && error != std::errc::no_such_file_or_directory)
      throw IoError("Could not inspect dataset path");
    if (fs::is_symlink(status))
      throw InvalidArgument("Dataset paths must not traverse symbolic links");
#if defined(_WIN32)
    const auto attributes = GetFileAttributesW(cursor.c_str());
    if (attributes == INVALID_FILE_ATTRIBUTES) {
      const auto code = GetLastError();
      if (code != ERROR_FILE_NOT_FOUND && code != ERROR_PATH_NOT_FOUND)
        throw IoError("Could not inspect dataset path attributes");
    } else if ((attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0U) {
      throw InvalidArgument("Dataset paths must not traverse reparse points");
    }
#endif
    const auto parent = cursor.parent_path();
    if (parent == cursor) break;
    cursor = parent;
  }
}

bool present(const fs::path& path) {
  std::error_code error;
  const auto status = fs::symlink_status(path, error);
  if (error && error != std::errc::no_such_file_or_directory)
    throw IoError("Could not inspect dataset destination");
  return status.type() != fs::file_type::not_found;
}

void close_checked(std::ofstream& stream) {
  stream.flush();
  if (!stream) throw IoError("Could not write complete dataset artifact");
  stream.close();
  if (!stream) throw IoError("Could not close dataset artifact");
}

fs::path external_root(const fs::path& data_root) {
  if (data_root.empty())
    throw InvalidArgument("Dataset root must not be empty");
  // Check before lexical normalization so link/.. cannot hide a traversal.
  const auto absolute = fs::absolute(data_root);
  reject_links(absolute);
  const auto root = absolute.lexically_normal();
  const auto repository =
      fs::weakly_canonical(fs::u8path(JYPPX_LIUDING_SOURCE_ROOT));
  // weakly_canonical folds existing case variants on supported platforms and
  // removes dot components without creating the output directory.
  const auto resolved = fs::weakly_canonical(root);
  for (auto cursor = resolved; !cursor.empty();) {
    if (cursor == repository || (fs::exists(cursor) && fs::exists(repository) &&
                                 fs::equivalent(cursor, repository)))
      throw InvalidArgument(
          "Prepared datasets must be outside the code repository");
    const auto parent = cursor.parent_path();
    if (parent == cursor) break;
    cursor = parent;
  }

  return root;
}
namespace {
void publish(const fs::path& staging, const fs::path& destination) {
  // No replace flag on Windows; RENAME_NOREPLACE on Linux. A directory which
  // appears after preflight is preserved even when it is empty.
#if defined(_WIN32)
  if (!MoveFileExW(staging.c_str(), destination.c_str(),
                   MOVEFILE_WRITE_THROUGH))
    throw IoError("Could not publish dataset without replacement");
#elif defined(__linux__) && defined(SYS_renameat2)
  constexpr unsigned kNoReplace = 1U;
  if (::syscall(SYS_renameat2, AT_FDCWD, staging.c_str(), AT_FDCWD,
                destination.c_str(), kNoReplace) != 0)
    throw IoError("Could not publish dataset without replacement");
#else
  static_cast<void>(staging);
  static_cast<void>(destination);
  throw NotImplemented(
      "Dataset publication requires Windows or Linux renameat2");
#endif
}
}  // namespace
StagingDirectory::StagingDirectory(const fs::path& root,
                                   std::string_view destination,
                                   std::string_view prefix)
    : destination_(root / std::string(destination)) {
  if (present(destination_))
    throw IoError("Refusing to replace prepared dataset");
  fs::create_directories(root);
  reject_links(root);
  static std::atomic<std::uint64_t> sequence{0U};
  for (unsigned attempt = 0U; attempt < 32U; ++attempt) {
    const auto clock =
        std::chrono::steady_clock::now().time_since_epoch().count();
    const auto candidate =
        root / (std::string(prefix) + std::to_string(clock) + "-" +
                std::to_string(sequence.fetch_add(1U)));
    if (fs::create_directory(candidate)) {
      staging_ = candidate;
      return;
    }
  }
  throw IoError("Could not reserve dataset staging directory");
}
StagingDirectory::~StagingDirectory() {
  if (!staging_.empty()) {
    std::error_code ignored;
    fs::remove_all(staging_, ignored);
  }
}
void StagingDirectory::publish() {
  reject_links(staging_);
  reject_links(destination_.parent_path());
  detail::publish(staging_, destination_);
  staging_.clear();
}
}  // namespace jyppx::liuding::datasets::detail
