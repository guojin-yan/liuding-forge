// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_TOOLS_QUANTIZATION_EXCLUSIVE_FILE_HPP
#define JYPPX_LIUDING_TOOLS_QUANTIZATION_EXCLUSIVE_FILE_HPP

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <span>
#include <system_error>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace jyppx::liuding::quantization::tools_detail {

// This private tool helper only creates a new complete file; caller-owned
// parent paths must remain stable. It never follows/replaces an existing file.
inline void WriteExclusive(const std::filesystem::path& path,
                           std::span<const std::byte> bytes) {
#ifdef _WIN32
  const auto file = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr,
                                CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE)
    throw std::system_error(static_cast<int>(GetLastError()),
                            std::system_category(), "Create exclusive file");
  bool open = true;
  try {
    while (!bytes.empty()) {
      const auto count = static_cast<DWORD>(std::min<std::size_t>(
          bytes.size(), std::numeric_limits<DWORD>::max()));
      DWORD written = 0U;
      if (!WriteFile(file, bytes.data(), count, &written, nullptr) ||
          written == 0U)
        throw std::system_error(static_cast<int>(GetLastError()),
                                std::system_category(), "Write exclusive file");
      bytes = bytes.subspan(written);
    }
    if (!FlushFileBuffers(file))
      throw std::system_error(static_cast<int>(GetLastError()),
                              std::system_category(), "Flush exclusive file");
    if (!CloseHandle(file))
      throw std::system_error(static_cast<int>(GetLastError()),
                              std::system_category(), "Close exclusive file");
    open = false;
  } catch (...) {
    if (open) CloseHandle(file);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    throw;
  }
#else
  const auto file =
      ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (file < 0)
    throw std::system_error(errno, std::generic_category(),
                            "Create exclusive file");
  bool open = true;
  try {
    while (!bytes.empty()) {
      const auto count = std::min<std::size_t>(
          bytes.size(),
          static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
      const auto written = ::write(file, bytes.data(), count);
      if (written < 0 && errno == EINTR) continue;
      if (written <= 0)
        throw std::system_error(errno, std::generic_category(),
                                "Write exclusive file");
      bytes = bytes.subspan(static_cast<std::size_t>(written));
    }
    if (::fsync(file) != 0)
      throw std::system_error(errno, std::generic_category(),
                              "Flush exclusive file");
    const auto close_result = ::close(file);
    open = false;
    if (close_result != 0)
      throw std::system_error(errno, std::generic_category(),
                              "Close exclusive file");
  } catch (...) {
    if (open) ::close(file);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    throw;
  }
#endif
}

inline void PublishExclusive(const std::filesystem::path& destination,
                             std::span<const std::byte> bytes) {
  auto temporary = destination;
  temporary += ".liuding-int8-tmp";
  WriteExclusive(temporary, bytes);
  try {
    std::filesystem::create_hard_link(temporary, destination);
    std::filesystem::remove(temporary);
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
}

}  // namespace jyppx::liuding::quantization::tools_detail

#endif  // JYPPX_LIUDING_TOOLS_QUANTIZATION_EXCLUSIVE_FILE_HPP
