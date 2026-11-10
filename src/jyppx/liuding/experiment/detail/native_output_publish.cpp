// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "artifact_publish.hpp"
#include "file_publication.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <cerrno>
#include <unistd.h>
#endif

namespace jyppx::liuding::experiment::detail {
std::error_code native_publish_output_file(
    const std::filesystem::path& staging,
    const std::filesystem::path& destination) noexcept {
#ifdef _WIN32
  try {
    std::error_code error;
    const auto source = windows_artifact_path(staging, error);
    if (error) return error;
    const auto target = windows_artifact_path(destination, error);
    if (error) return error;
    // The original stage retains DELETE_ON_CLOSE. A second hard-link name
    // survives both normal handle closure and process termination.
    if (!CreateHardLinkW(target.c_str(), source.c_str(), nullptr))
      return {static_cast<int>(GetLastError()), std::system_category()};
    return {};
  } catch (...) {
    return std::make_error_code(std::errc::io_error);
  }
#else
  // link is atomic and refuses an existing destination. The caller keeps the
  // staging file open/locked until publication and parent-directory syncing.
  if (::link(staging.c_str(), destination.c_str()) != 0)
    return {errno, std::generic_category()};
  return {};
#endif
}
}  // namespace jyppx::liuding::experiment::detail
