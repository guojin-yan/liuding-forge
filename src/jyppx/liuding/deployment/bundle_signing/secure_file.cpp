// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "secure_file.hpp"

#include <jyppx/liuding/foundation/error.hpp>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <openssl/crypto.h>
#include <string_view>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace jyppx::liuding::bundle_signing::detail {
namespace {

[[noreturn]] void IoFailure() {
  // Paths and native error strings may contain credentials or key identifiers.
  throw IoError(
      "Cannot safely read bundle signing input",
      ErrorDetails{.component = "bundle_signing",
                   .operation = "input.read",
                   .expected = "stable bounded ordinary unlinked file",
                   .actual = "safe local read failed",
                   .hint = "Use an explicit immutable local file."});
}

[[noreturn]] void LimitFailure() {
  throw ResourceExhausted(
      "Bundle signing input exceeds its encoded-byte limit");
}

std::filesystem::path Absolute(const std::filesystem::path& path) {
  if (path.empty() || path.native().find(std::filesystem::path::value_type{}) !=
                          std::filesystem::path::string_type::npos)
    IoFailure();
  std::error_code error;
  const auto absolute = std::filesystem::absolute(path, error);
  if (error || !absolute.is_absolute() || absolute.filename().empty())
    IoFailure();
  for (const auto& component : absolute.relative_path()) {
    if (component.empty() || component == "." || component == "..") IoFailure();
#if defined(_WIN32)
    const auto& text = component.native();
    if (text.find(L':') != std::wstring::npos || text.back() == L'.' ||
        text.back() == L' ')
      IoFailure();
#endif
  }
#if defined(_WIN32)
  // Deliberately admit drive-rooted local paths only: no UNC/device/ADS paths.
  const auto root = absolute.root_name().native();
  if (root.size() != 2U || root[1] != L':' ||
      !((root[0] >= L'a' && root[0] <= L'z') ||
        (root[0] >= L'A' && root[0] <= L'Z')))
    IoFailure();
#endif
  return absolute;
}

#if defined(_WIN32)
class Handle final {
 public:
  explicit Handle(HANDLE handle = INVALID_HANDLE_VALUE) : handle_(handle) {
    if (handle == INVALID_HANDLE_VALUE) IoFailure();
  }
  ~Handle() {
    if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
  }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  Handle(Handle&& other) noexcept
      : handle_(std::exchange(other.handle_, INVALID_HANDLE_VALUE)) {}
  HANDLE get() const noexcept { return handle_; }

 private:
  HANDLE handle_;
};

BY_HANDLE_FILE_INFORMATION Info(HANDLE handle, bool directory) {
  BY_HANDLE_FILE_INFORMATION info{};
  if (GetFileType(handle) != FILE_TYPE_DISK ||
      !GetFileInformationByHandle(handle, &info) ||
      (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0U ||
      ((info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0U) != directory ||
      (!directory && info.nNumberOfLinks != 1U))
    IoFailure();
  return info;
}

bool Same(const BY_HANDLE_FILE_INFORMATION& before,
          const BY_HANDLE_FILE_INFORMATION& after) noexcept {
  return before.dwVolumeSerialNumber == after.dwVolumeSerialNumber &&
         before.nFileIndexHigh == after.nFileIndexHigh &&
         before.nFileIndexLow == after.nFileIndexLow &&
         before.nFileSizeHigh == after.nFileSizeHigh &&
         before.nFileSizeLow == after.nFileSizeLow &&
         before.nNumberOfLinks == after.nNumberOfLinks &&
         CompareFileTime(&before.ftLastWriteTime, &after.ftLastWriteTime) ==
             0 &&
         CompareFileTime(&before.ftCreationTime, &after.ftCreationTime) == 0;
}
#else
class Handle final {
 public:
  explicit Handle(int handle) : handle_(handle) {
    if (handle < 0) IoFailure();
  }
  ~Handle() {
    if (handle_ >= 0) close(handle_);
  }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  Handle(Handle&& other) noexcept : handle_(std::exchange(other.handle_, -1)) {}
  int get() const noexcept { return handle_; }

 private:
  int handle_;
};

struct stat Info(int handle, bool directory) {
  struct stat info{};
  if (fstat(handle, &info) != 0 ||
      (directory ? !S_ISDIR(info.st_mode) : !S_ISREG(info.st_mode)) ||
      (!directory && (info.st_nlink != 1 || info.st_size < 0)))
    IoFailure();
  return info;
}

bool Same(const struct stat& before, const struct stat& after) noexcept {
  if (before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
      before.st_mode != after.st_mode || before.st_size != after.st_size ||
      before.st_nlink != after.st_nlink)
    return false;
#if defined(__APPLE__)
  return before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec &&
         before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec &&
         before.st_ctimespec.tv_sec == after.st_ctimespec.tv_sec &&
         before.st_ctimespec.tv_nsec == after.st_ctimespec.tv_nsec;
#else
  return before.st_mtim.tv_sec == after.st_mtim.tv_sec &&
         before.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
         before.st_ctim.tv_sec == after.st_ctim.tv_sec &&
         before.st_ctim.tv_nsec == after.st_ctim.tv_nsec;
#endif
}
#endif

}  // namespace

CleansedBytes::CleansedBytes(std::size_t size) : bytes_(size) {}
CleansedBytes::~CleansedBytes() {
  if (!bytes_.empty()) OPENSSL_cleanse(bytes_.data(), bytes_.size());
}
CleansedBytes::CleansedBytes(CleansedBytes&& other) noexcept
    : bytes_(std::move(other.bytes_)) {}
std::span<std::byte> CleansedBytes::bytes() noexcept { return bytes_; }
std::span<const std::byte> CleansedBytes::bytes() const noexcept {
  return bytes_;
}

CleansedBytes ReadBoundedFile(const std::filesystem::path& path,
                              std::size_t maximum) {
  if (maximum == 0U || maximum > 65536U)
    throw InvalidArgument("Invalid bundle signing input byte limit");
  const auto absolute = Absolute(path);
  std::vector<Handle> ancestors;
  auto current = absolute.root_path();
#if defined(_WIN32)
  auto directory = [&](const std::filesystem::path& value) {
    ancestors.emplace_back(CreateFileW(
        value.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE,
        nullptr, OPEN_EXISTING,
        FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr));
    static_cast<void>(Info(ancestors.back().get(), true));
  };
  directory(current);
  for (const auto& part : absolute.relative_path().parent_path()) {
    current /= part;
    directory(current);
  }
  Handle file(CreateFileW(
      absolute.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
      FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_SEQUENTIAL_SCAN, nullptr));
  const auto before = Info(file.get(), false);
  const auto size = (static_cast<std::uint64_t>(before.nFileSizeHigh) << 32U) |
                    before.nFileSizeLow;
#else
  ancestors.emplace_back(
      open(current.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
  static_cast<void>(Info(ancestors.back().get(), true));
  for (const auto& part : absolute.relative_path().parent_path()) {
    ancestors.emplace_back(
        openat(ancestors.back().get(), part.c_str(),
               O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
    static_cast<void>(Info(ancestors.back().get(), true));
  }
  Handle file(openat(ancestors.back().get(), absolute.filename().c_str(),
                     O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK));
  const auto before = Info(file.get(), false);
  const auto size = static_cast<std::uint64_t>(before.st_size);
#endif
  if (size > maximum) LimitFailure();
  if (size == 0U) IoFailure();
  CleansedBytes result(static_cast<std::size_t>(size));
  std::size_t offset = 0U;
  while (offset < result.bytes().size()) {
    const auto remaining = result.bytes().subspan(offset);
#if defined(_WIN32)
    DWORD received = 0U;
    if (!ReadFile(file.get(), remaining.data(),
                  static_cast<DWORD>(remaining.size()), &received, nullptr) ||
        received == 0U)
      IoFailure();
#else
    const auto received = read(file.get(), remaining.data(), remaining.size());
    if (received < 0 && errno == EINTR) continue;
    if (received <= 0) IoFailure();
#endif
    offset += static_cast<std::size_t>(received);
  }
  if (!Same(before, Info(file.get(), false))) IoFailure();
  return result;
}

}  // namespace jyppx::liuding::bundle_signing::detail
