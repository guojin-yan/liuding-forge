// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "artifact_publish.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace jyppx::experiment::detail {

std::error_code native_publish_artifact_file(
    const std::filesystem::path& source,
    const std::filesystem::path& destination) noexcept {
#ifdef _WIN32
  if (!MoveFileExW(source.c_str(), destination.c_str(),
                   MOVEFILE_WRITE_THROUGH | MOVEFILE_REPLACE_EXISTING)) {
    return {static_cast<int>(GetLastError()), std::system_category()};
  }
  return {};
#else
  std::error_code error;
  std::filesystem::rename(source, destination, error);
  return error;
#endif
}

}  // namespace jyppx::experiment::detail
