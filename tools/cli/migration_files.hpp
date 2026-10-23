// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_TOOLS_MIGRATION_FILES_HPP
#define JYPPX_TOOLS_MIGRATION_FILES_HPP

#include "cli.hpp"

#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <limits>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace jyppx::liuding::cli::migration_files {
inline constexpr std::size_t kInputBytes = 1024U * 1024U;

inline std::filesystem::path path(std::string_view value) {
  return std::filesystem::u8path(value.begin(), value.end());
}

inline std::string sha256(std::string_view value) {
  return sha256_hex(std::as_bytes(std::span(value)));
}

// Inputs and output parents are exclusively owned/quiescent during a command.
// The ancestor check rejects links/junctions; leaf handles additionally detect
// replacement or concurrent writes without trusting a path string as a lock.
inline std::filesystem::path checked_path(const std::filesystem::path& value) {
  std::error_code error;
  const auto absolute = std::filesystem::absolute(value, error);
  if (error) throw IoError("Could not resolve migration path");
  for (const auto& part : value) {
    if (part == "." || part == "..")
      throw InvalidArgument("Migration paths must not contain dot traversal");
  }
#if defined(_WIN32)
  if (absolute.root_name().native().starts_with(L"\\\\"))
    throw InvalidArgument("Migration paths must use a local filesystem drive");
#endif
  auto current = absolute.root_path();
  for (const auto& part : absolute.relative_path()) {
    const auto encoded = part.u8string();
    const auto component = std::string(
        reinterpret_cast<const char*>(encoded.data()), encoded.size());
    if (component.find(':') != std::string::npos ||
        component.find('\\') != std::string::npos || component.ends_with('.') ||
        component.ends_with(' '))
      throw InvalidArgument("Migration path has an unsafe stream/device alias");
    auto base = component.substr(0U, component.find('.'));
    for (auto& character : base)
      if (character >= 'A' && character <= 'Z') character += 'a' - 'A';
    if (base == "con" || base == "prn" || base == "aux" || base == "nul" ||
        (base.size() == 4U &&
         (base.starts_with("com") || base.starts_with("lpt")) &&
         base.back() >= '1' && base.back() <= '9'))
      throw InvalidArgument("Migration path uses a reserved device alias");
    current /= part;
#if defined(_WIN32)
    const auto attributes = GetFileAttributesW(current.c_str());
    if (attributes == INVALID_FILE_ATTRIBUTES)
      throw IoError("Could not inspect migration path");
    if ((attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0U)
      throw InvalidArgument("Migration paths must not contain reparse points");
#else
    const auto status = std::filesystem::symlink_status(current, error);
    if (error) throw IoError("Could not inspect migration path");
    if (std::filesystem::is_symlink(status))
      throw InvalidArgument("Migration paths must not contain links");
#endif
  }
  return absolute;
}

inline std::string read(const std::filesystem::path& value,
                        std::size_t maximum = kInputBytes) {
  const auto absolute = checked_path(value);
  std::string bytes;
#if defined(_WIN32)
  struct Handle {
    HANDLE value;
    ~Handle() {
      if (value != INVALID_HANDLE_VALUE) CloseHandle(value);
    }
  } handle{CreateFileW(absolute.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                       OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT, nullptr)};
  if (handle.value == INVALID_HANDLE_VALUE)
    throw IoError("Could not open immutable migration input");
  BY_HANDLE_FILE_INFORMATION info{};
  if (!GetFileInformationByHandle(handle.value, &info))
    throw IoError("Could not inspect migration input handle");
  const auto length = (static_cast<std::uint64_t>(info.nFileSizeHigh) << 32U) |
                      info.nFileSizeLow;
  if ((info.dwFileAttributes &
       (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) != 0U ||
      info.nNumberOfLinks != 1U || GetFileType(handle.value) != FILE_TYPE_DISK)
    throw InvalidArgument("Migration input must be an unaliased regular file");
  if (length > maximum)
    throw ResourceExhausted("Migration input exceeds the byte limit");
  bytes.resize(static_cast<std::size_t>(length));
  DWORD count = 0U;
  if (!ReadFile(handle.value, bytes.data(), static_cast<DWORD>(bytes.size()),
                &count, nullptr) ||
      count != bytes.size())
    throw IoError("Could not read complete migration input");
  char extra = 0;
  if (!ReadFile(handle.value, &extra, 1U, &count, nullptr) || count != 0U)
    throw IoError("Migration input changed during read");
#else
  struct Handle {
    int value;
    ~Handle() {
      if (value >= 0) ::close(value);
    }
  } handle{::open(absolute.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW)};
  if (handle.value < 0) throw IoError("Could not open migration input");
  struct stat initial {};
  if (::fstat(handle.value, &initial) != 0)
    throw IoError("Could not inspect migration input handle");
  if (!S_ISREG(initial.st_mode) || initial.st_nlink != 1)
    throw InvalidArgument("Migration input must be an unaliased regular file");
  if (initial.st_size < 0 ||
      static_cast<std::uint64_t>(initial.st_size) > maximum)
    throw ResourceExhausted("Migration input exceeds the byte limit");
  bytes.resize(static_cast<std::size_t>(initial.st_size));
  std::size_t position = 0U;
  while (position < bytes.size()) {
    const auto count =
        ::read(handle.value, bytes.data() + position, bytes.size() - position);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) throw IoError("Could not read complete migration input");
    position += static_cast<std::size_t>(count);
  }
  char extra = 0;
  struct stat final_state {};
  struct stat named_state {};
  if (::read(handle.value, &extra, 1U) != 0 ||
      ::fstat(handle.value, &final_state) != 0 ||
      ::lstat(absolute.c_str(), &named_state) != 0 ||
      initial.st_size != final_state.st_size ||
      initial.st_mtime != final_state.st_mtime ||
      initial.st_ctime != final_state.st_ctime ||
      initial.st_dev != named_state.st_dev ||
      initial.st_ino != named_state.st_ino || final_state.st_nlink != 1 ||
      named_state.st_nlink != 1 || !S_ISREG(named_state.st_mode) ||
      final_state.st_size != named_state.st_size)
    throw IoError("Migration input changed during read");
#if defined(__APPLE__)
  if (initial.st_mtimespec.tv_nsec != final_state.st_mtimespec.tv_nsec ||
      initial.st_ctimespec.tv_nsec != final_state.st_ctimespec.tv_nsec)
    throw IoError("Migration input changed during read");
#else
  if (initial.st_mtim.tv_nsec != final_state.st_mtim.tv_nsec ||
      initial.st_ctim.tv_nsec != final_state.st_ctim.tv_nsec)
    throw IoError("Migration input changed during read");
#endif
#endif
  return bytes;
}

inline void publish(
    const std::filesystem::path& destination,
    const std::vector<std::pair<std::string, std::string>>& files) {
  auto parent = destination.parent_path();
  if (parent.empty()) parent = std::filesystem::current_path();
  checked_path(parent);
  if (destination.filename().empty() || destination.filename() == "." ||
      destination.filename() == "..")
    throw InvalidArgument("Use a new migration output directory");
  // Validate the proposed leaf with the same name rules before it exists.
  const auto leaf = destination.filename().u8string();
  if (std::find(leaf.begin(), leaf.end(), u8':') != leaf.end() ||
      std::find(leaf.begin(), leaf.end(), u8'\\') != leaf.end() ||
      leaf.back() == u8'.' || leaf.back() == u8' ')
    throw InvalidArgument("Migration output has an unsafe leaf alias");
  std::error_code error;
  const auto status = std::filesystem::symlink_status(destination, error);
  if (std::filesystem::exists(status) ||
      (error && error != std::errc::no_such_file_or_directory))
    throw IoError("Migration output already exists or cannot be inspected");
  static std::atomic<std::uint64_t> sequence{0U};
  const auto unique =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const auto stage = parent / (".jyppx-migration-" + std::to_string(unique) +
                               "-" + std::to_string(sequence.fetch_add(1U)));
  if (!std::filesystem::create_directory(stage))
    throw IoError("Could not reserve migration staging directory");
  try {
    for (const auto& [name, bytes] : files) {
      std::ofstream output(stage / name, std::ios::binary | std::ios::trunc);
      if (!output) throw IoError("Could not open migration staging file");
      output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
      output.flush();
      if (!output) throw IoError("Could not write complete migration artifact");
      output.close();
      if (!output) throw IoError("Could not close migration artifact");
    }
    if (!std::filesystem::create_directory(destination))
      throw IoError("Could not reserve new migration output directory");
    for (const auto& [name, bytes] : files) {
      static_cast<void>(bytes);
      std::filesystem::create_hard_link(stage / name, destination / name);
      std::filesystem::remove(stage / name);
    }
    std::filesystem::remove(stage);
  } catch (...) {
    std::error_code ignored;
    // Delete only the fixed files this invocation staged, never recurse into
    // user paths or erase a separately existing output directory.
    for (const auto& [name, bytes] : files) {
      static_cast<void>(bytes);
      std::filesystem::remove(stage / name, ignored);
    }
    std::filesystem::remove(stage, ignored);
    throw;
  }
}

}  // namespace jyppx::liuding::cli::migration_files

#endif  // JYPPX_TOOLS_MIGRATION_FILES_HPP
