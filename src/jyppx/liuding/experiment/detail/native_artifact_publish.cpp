// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "artifact_publish.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <new>
#include <stdexcept>
#endif

namespace jyppx::liuding::experiment::detail {

#ifdef _WIN32
std::wstring windows_artifact_path(const std::filesystem::path& path,
                                   std::error_code& error) {
  error.clear();
  const auto qualified = [](const std::wstring& value) {
    return value.starts_with(L"\\\\?\\") || value.starts_with(L"\\\\.\\");
  };
  if (qualified(path.native())) return path.native();

  // GetFullPathNameW resolves drive-relative paths, separators and dot segments
  // with Windows semantics, including the UNC share root. Its size query and
  // retry contract avoid fixed buffers and never truncate the returned path.
  auto capacity = GetFullPathNameW(path.c_str(), 0U, nullptr, nullptr);
  if (capacity == 0U) {
    error = {static_cast<int>(GetLastError()), std::system_category()};
    return {};
  }
  std::wstring absolute;
  for (;;) {
    absolute.resize(capacity);
    const auto count =
        GetFullPathNameW(path.c_str(), capacity, absolute.data(), nullptr);
    if (count == 0U) {
      error = {static_cast<int>(GetLastError()), std::system_category()};
      return {};
    }
    if (count < capacity) {
      absolute.resize(count);
      break;
    }
    capacity = count;
  }
  if (qualified(absolute)) return absolute;
  if (absolute.starts_with(L"\\\\"))
    return L"\\\\?\\UNC\\" + absolute.substr(2U);
  if (absolute.size() >= 3U && absolute[1] == L':' && absolute[2] == L'\\')
    return L"\\\\?\\" + absolute;
  error = {static_cast<int>(ERROR_INVALID_NAME), std::system_category()};
  return {};
}
#endif

std::error_code native_publish_artifact_file(
    const std::filesystem::path& source,
    const std::filesystem::path& destination) noexcept {
#ifdef _WIN32
  try {
    std::error_code error;
    const auto native_source = windows_artifact_path(source, error);
    if (error) return error;
    const auto native_destination = windows_artifact_path(destination, error);
    if (error) return error;
    if (!MoveFileExW(native_source.c_str(), native_destination.c_str(),
                     MOVEFILE_WRITE_THROUGH | MOVEFILE_REPLACE_EXISTING)) {
      return {static_cast<int>(GetLastError()), std::system_category()};
    }
    return {};
  } catch (const std::system_error& error) {
    return error.code();
  } catch (const std::bad_alloc&) {
    return {static_cast<int>(ERROR_NOT_ENOUGH_MEMORY), std::system_category()};
  } catch (const std::length_error&) {
    return {static_cast<int>(ERROR_FILENAME_EXCED_RANGE),
            std::system_category()};
  } catch (...) {
    return {static_cast<int>(ERROR_INVALID_PARAMETER), std::system_category()};
  }
#else
  std::error_code error;
  std::filesystem::rename(source, destination, error);
  return error;
#endif
}

}  // namespace jyppx::liuding::experiment::detail
