// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "file_publication.hpp"

#include "artifact_publish.hpp"
#include "jyppx/liuding/foundation/error.hpp"
#include "jyppx/liuding/foundation/sha256.hpp"

#include <algorithm>
#include <array>
#include <fstream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace jyppx::liuding::experiment::detail {
namespace {
[[noreturn]] void io(std::string_view operation,
                     const std::filesystem::path& path) {
  const auto name = path.u8string();
  throw IoError(
      "Could not publish the prepared local file",
      ErrorDetails{.component = "experiment.file_publication",
                   .operation = std::string(operation),
                   .expected = "controlled ordinary local paths",
                   .actual = std::string(
                       reinterpret_cast<const char*>(name.data()), name.size()),
                   .hint = "Retain the intent and retry after fixing "
                           "the local file or directory conflict."});
}

bool verify_existing(const std::filesystem::path& path, ArtifactReader& content,
                     std::stop_token stop, TelemetryDeadline deadline) {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  if (error == std::errc::no_such_file_or_directory ||
      (!error && !std::filesystem::exists(status)))
    return false;
  if (error || !std::filesystem::is_regular_file(status))
    io("destination.type", path);
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input || input.tellg() < 0 ||
      static_cast<std::uint64_t>(input.tellg()) != content.size_bytes())
    io("destination.size", path);
  input.seekg(0);
  Sha256 hash;
  std::array<std::byte, 64U * 1024U> buffer{};
  std::uint64_t total = 0U;
  while (total < content.size_bytes()) {
    check_telemetry_operation(stop, deadline);
    const auto count = static_cast<std::streamsize>(
        std::min<std::uint64_t>(buffer.size(), content.size_bytes() - total));
    input.read(reinterpret_cast<char*>(buffer.data()), count);
    if (input.gcount() != count) io("destination.read", path);
    hash.update(std::span<const std::byte>(buffer.data(),
                                           static_cast<std::size_t>(count)));
    total += static_cast<std::uint64_t>(count);
  }
  if (input.peek() != std::char_traits<char>::eof() || input.bad() ||
      sha256_digest_hex(hash.finish()) != content.digest().hex)
    io("destination.digest", path);
  input.close();
#ifdef _WIN32
  const auto native = windows_artifact_path(path, error);
  if (error) io("destination.path", path);
  const auto handle =
      CreateFileW(native.c_str(), GENERIC_READ | GENERIC_WRITE,
                  FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                  FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
  if (handle == INVALID_HANDLE_VALUE) io("destination.sync.open", path);
  BY_HANDLE_FILE_INFORMATION info{};
  const bool ordinary =
      GetFileInformationByHandle(handle, &info) &&
      !(info.dwFileAttributes &
        (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DIRECTORY));
  const auto synced = ordinary && FlushFileBuffers(handle);
  const auto closed = CloseHandle(handle);
  if (!synced || !closed) io("destination.sync", path);
#else
  const auto descriptor =
      ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (descriptor < 0) io("destination.sync.open", path);
  const auto synced = ::fsync(descriptor);
  const auto closed = ::close(descriptor);
  if (synced != 0 || closed != 0) io("destination.sync", path);
#endif
  return true;
}

void sync_parent(const std::filesystem::path& path) {
#ifndef _WIN32
  const auto descriptor =
      ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (descriptor < 0) io("directory.open", path);
  const auto result = ::fsync(descriptor);
  const auto closed = ::close(descriptor);
  if (result != 0 || closed != 0) io("directory.sync", path);
#else
  // File contents use FlushFileBuffers before no-replace hard-link creation.
  // No general Windows directory/power-loss guarantee is inferred here.
  (void)path;
#endif
}

class Stage final {
 public:
  explicit Stage(std::filesystem::path path) : path_(std::move(path)) {
#ifdef _WIN32
    std::error_code error;
    const auto native = windows_artifact_path(path_, error);
    if (error) io("stage.path", path_);
    handle_ = CreateFileW(native.c_str(), GENERIC_READ | GENERIC_WRITE | DELETE,
                          FILE_SHARE_DELETE, nullptr, OPEN_ALWAYS,
                          FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE |
                              FILE_FLAG_OPEN_REPARSE_POINT,
                          nullptr);
    if (handle_ == INVALID_HANDLE_VALUE) io("stage.open", path_);
    BY_HANDLE_FILE_INFORMATION info{};
    LARGE_INTEGER start{};
    if (!GetFileInformationByHandle(handle_, &info) ||
        (info.dwFileAttributes &
         (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DIRECTORY)) ||
        info.nNumberOfLinks != 1U ||
        !SetFilePointerEx(handle_, start, nullptr, FILE_BEGIN) ||
        !SetEndOfFile(handle_)) {
      CloseHandle(handle_);
      handle_ = INVALID_HANDLE_VALUE;
      io("stage.ordinary_exclusive", path_);
    }
#else
    descriptor_ =
        ::open(path_.c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (descriptor_ < 0) io("stage.open", path_);
    struct stat info{};
    if (::flock(descriptor_, LOCK_EX | LOCK_NB) != 0 ||
        ::fstat(descriptor_, &info) != 0 || !S_ISREG(info.st_mode) ||
        info.st_nlink != 1 || ::ftruncate(descriptor_, 0) != 0) {
      ::close(descriptor_);
      descriptor_ = -1;
      io("stage.ordinary_exclusive", path_);
    }
#endif
  }

  ~Stage() {
#ifdef _WIN32
    if (handle_ != INVALID_HANDLE_VALUE) {
      CloseHandle(handle_);
    }
#else
    if (descriptor_ >= 0) {
      ::unlink(path_.c_str());
      ::close(descriptor_);
    }
#endif
  }

  Stage(const Stage&) = delete;
  Stage& operator=(const Stage&) = delete;

  void write(std::span<const std::byte> bytes) {
    while (!bytes.empty()) {
#ifdef _WIN32
      DWORD written = 0U;
      if (!WriteFile(handle_, bytes.data(), static_cast<DWORD>(bytes.size()),
                     &written, nullptr) ||
          written == 0U)
        io("stage.write", path_);
      const auto count = static_cast<std::size_t>(written);
#else
      const auto written = ::write(descriptor_, bytes.data(), bytes.size());
      if (written < 0 && errno == EINTR) continue;
      if (written <= 0) io("stage.write", path_);
      const auto count = static_cast<std::size_t>(written);
#endif
      bytes = bytes.subspan(count);
    }
  }

  void flush() {
#ifdef _WIN32
    if (!FlushFileBuffers(handle_)) io("stage.sync", path_);
#else
    if (::fsync(descriptor_) != 0) io("stage.sync", path_);
#endif
  }

  void publish(const std::filesystem::path& destination,
               ArtifactReader& content, std::stop_token stop,
               TelemetryDeadline deadline) {
    check_telemetry_operation(stop, deadline);
    const auto error = publish_output_file(path_, destination);
    if (error) {
      // An independent publisher may win the no-replace race. Only full-byte
      // identity permits replay; never replace or silently accept a conflict.
      if (!verify_existing(destination, content, stop, deadline))
        io("destination.publish", destination);
    }
    sync_parent(destination.parent_path());
  }

 private:
  std::filesystem::path path_;
#ifdef _WIN32
  HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
  int descriptor_ = -1;
#endif
};
}  // namespace

void materialize_publication(ArtifactReader& content,
                             const std::filesystem::path& destination,
                             std::string_view intent_digest,
                             std::stop_token stop, TelemetryDeadline deadline) {
  check_telemetry_operation(stop, deadline);
  const auto stage =
      destination.parent_path() /
      (".liuding-publication-" + std::string(intent_digest) + ".stage");
  if (verify_existing(destination, content, stop, deadline)) {
    // A process can die after Linux link() but before unlinking its stage. Do
    // not truncate that hard link on replay: remove only its reserved name.
    std::error_code error;
    const auto status = std::filesystem::symlink_status(stage, error);
    if (!error && std::filesystem::exists(status)) {
      if (!std::filesystem::is_regular_file(status)) io("stage.type", stage);
      std::filesystem::remove(stage, error);
      if (error) io("stage.remove", stage);
    } else if (error && error != std::errc::no_such_file_or_directory) {
      io("stage.inspect", stage);
    }
    sync_parent(destination.parent_path());
    return;
  }
  {
    Stage output(stage);
    std::array<std::byte, 64U * 1024U> buffer{};
    std::uint64_t total = 0U;
    while (total < content.size_bytes()) {
      check_telemetry_operation(stop, deadline);
      const auto wanted = static_cast<std::size_t>(
          std::min<std::uint64_t>(buffer.size(), content.size_bytes() - total));
      const auto count =
          content.read(std::span<std::byte>(buffer).first(wanted));
      if (count == 0U || count > wanted) io("content.read", destination);
      output.write(std::span<const std::byte>(buffer.data(), count));
      total += count;
    }
    if (content.read(buffer) != 0U) io("content.size", destination);
    output.flush();
    output.publish(destination, content, stop, deadline);
  }
  sync_parent(destination.parent_path());
}
}  // namespace jyppx::liuding::experiment::detail
