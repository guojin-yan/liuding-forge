// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "jyppx/liuding/experiment/artifact_store.hpp"

#include "detail/artifact_publish.hpp"
#include "jyppx/liuding/foundation/error.hpp"
#include "jyppx/liuding/foundation/sha256.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <fstream>
#include <limits>
#include <random>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace jyppx::liuding::experiment {
namespace {

constexpr std::size_t kHashIoBufferBytes = 1024U * 1024U;

bool has_control_character(std::string_view value) noexcept {
  return std::any_of(value.begin(), value.end(),
                     [](unsigned char character) { return character < 0x20U; });
}

std::string path_text(const std::filesystem::path& path) {
  return path.string();
}

[[noreturn]] void throw_io(std::string operation,
                           const std::filesystem::path& path,
                           const std::error_code& error) {
  throw IoError(
      "Artifact filesystem operation failed",
      ErrorDetails{
          .component = "experiment.artifact_store",
          .operation = std::move(operation),
          .expected = "accessible filesystem path",
          .actual = path_text(path) + ": " + error.message(),
          .hint = "Check the artifact root, permissions, and free space."});
}

#ifdef _WIN32
// Every filesystem entry point uses the same native spelling, including
// metadata, directory traversal and canonicalization before any file is opened.
std::filesystem::path artifact_filesystem_path(
    const std::filesystem::path& path, std::string_view operation) {
  std::error_code error;
  auto native_path = detail::windows_artifact_path(path, error);
  if (error) throw_io(std::string(operation), path, error);
  return std::filesystem::path(std::move(native_path));
}

std::filesystem::path artifact_logical_root(
    const std::filesystem::path& canonical,
    const std::filesystem::path& supplied) {
  // A caller-supplied namespace keeps its established spelling. Ordinary paths
  // retain ordinary public root()/diagnostic identities after native I/O.
  const auto& original = supplied.native();
  if (original.starts_with(LR"(\\?\)") || original.starts_with(LR"(\\.\)"))
    return canonical;
  const auto& native = canonical.native();
  if (native.starts_with(LR"(\\?\UNC\)"))
    return std::filesystem::path(LR"(\\)" + native.substr(8U));
  if (native.starts_with(LR"(\\?\)") && native.size() >= 7U &&
      native[5U] == L':' && native[6U] == L'\\')
    return std::filesystem::path(native.substr(4U));
  return canonical;
}
#else
const std::filesystem::path& artifact_filesystem_path(
    const std::filesystem::path& path, std::string_view operation) noexcept {
  (void)operation;
  return path;
}
#endif

[[noreturn]] void throw_invalid(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
  throw InvalidArgument("Invalid artifact storage contract",
                        ErrorDetails{.component = "experiment.artifact_store",
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

[[noreturn]] void throw_limit(std::string operation, std::string expected,
                              std::string actual) {
  throw ResourceExhausted(
      "Artifact storage bound exceeded",
      ErrorDetails{
          .component = "experiment.artifact_store",
          .operation = std::move(operation),
          .expected = std::move(expected),
          .actual = std::move(actual),
          .hint = "Raise the configured bound or split the artifact."});
}

void validate_limits(const ArtifactStoreLimits& limits) {
  if (limits.max_artifact_bytes == 0U) {
    throw_invalid("limits.max_artifact_bytes", "a positive byte limit", "0",
                  "Use a nonzero maximum artifact size.");
  }
  if (limits.max_total_bytes < limits.max_artifact_bytes) {
    throw_invalid("limits.max_total_bytes", "at least max_artifact_bytes",
                  std::to_string(limits.max_total_bytes),
                  "The total store bound must contain one maximum artifact.");
  }
  if (limits.max_object_files == 0U) {
    throw_invalid("limits.max_object_files", "a positive object count", "0",
                  "Use a nonzero maximum object count.");
  }
  if (limits.max_root_path_bytes == 0U || limits.max_logical_name_bytes == 0U ||
      limits.max_media_type_bytes == 0U || limits.max_producer_bytes == 0U) {
    throw_invalid("limits", "positive path and metadata bounds", "zero bound",
                  "Use positive bounds for every metadata field.");
  }
  if (limits.stale_temp_age < std::chrono::seconds::zero()) {
    throw_invalid("limits.stale_temp_age", "a nonnegative duration",
                  "negative duration", "Use zero or a positive stale age.");
  }
}

void validate_descriptor(const ArtifactDescriptor& descriptor,
                         const ArtifactStoreLimits& limits) {
  if (descriptor.logical_name.empty()) {
    throw_invalid("descriptor.logical_name", "a nonempty logical name", "empty",
                  "Provide a stable producer-facing artifact name.");
  }
  if (has_control_character(descriptor.logical_name) ||
      has_control_character(descriptor.media_type) ||
      has_control_character(descriptor.producer)) {
    throw_invalid("descriptor", "text without control characters",
                  "control character", "Use display-safe metadata text.");
  }
  if (descriptor.logical_name.size() > limits.max_logical_name_bytes) {
    throw_limit(
        "descriptor.logical_name",
        "at most " + std::to_string(limits.max_logical_name_bytes) + " bytes",
        std::to_string(descriptor.logical_name.size()) + " bytes");
  }
  if (descriptor.media_type.empty()) {
    throw_invalid("descriptor.media_type", "a nonempty media type", "empty",
                  "Use a valid bounded MIME-like media type.");
  }
  if (descriptor.media_type.size() > limits.max_media_type_bytes) {
    throw_limit(
        "descriptor.media_type",
        "at most " + std::to_string(limits.max_media_type_bytes) + " bytes",
        std::to_string(descriptor.media_type.size()) + " bytes");
  }
  if (descriptor.producer.size() > limits.max_producer_bytes) {
    throw_limit(
        "descriptor.producer",
        "at most " + std::to_string(limits.max_producer_bytes) + " bytes",
        std::to_string(descriptor.producer.size()) + " bytes");
  }
  if (descriptor.expected_size_bytes &&
      *descriptor.expected_size_bytes > limits.max_artifact_bytes) {
    throw_limit(
        "descriptor.expected_size_bytes",
        "at most " + std::to_string(limits.max_artifact_bytes) + " bytes",
        std::to_string(*descriptor.expected_size_bytes) + " bytes");
  }
}

std::filesystem::path object_path(const std::filesystem::path& root,
                                  const ArtifactDigest& digest) {
  return root / "objects" / digest.hex.substr(0U, 2U) / digest.hex.substr(2U);
}

bool is_object_relative_path(const std::filesystem::path& relative) {
  std::vector<std::filesystem::path> parts;
  for (const auto& part : relative) parts.push_back(part);
  if (parts.size() != 3U || parts[0] != "objects") return false;
  const auto first = parts[1].string();
  const auto second = parts[2].string();
  if (first.size() != 2U || second.size() != 62U) return false;
  const auto lowercase = [](char value) {
    return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
  };
  return std::all_of(first.begin(), first.end(), lowercase) &&
         std::all_of(second.begin(), second.end(), lowercase);
}

ArtifactDigest digest_from_relative_path(
    const std::filesystem::path& relative) {
  const auto first = relative.begin();
  const auto second = std::next(first);
  const auto third = std::next(second);
  ArtifactDigest digest;
  digest.hex =
      (first->string() == "objects" ? second->string() : "") + third->string();
  return digest;
}

std::uint64_t checked_file_size(const std::filesystem::path& path) {
  std::error_code error;
  const auto size = std::filesystem::file_size(
      artifact_filesystem_path(path, "file_size"), error);
  if (error) throw_io("file_size", path, error);
  return size;
}

void require_unlinked_directory(const std::filesystem::path& path) {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(
      artifact_filesystem_path(path, "inspect_path"), error);
  if (error) throw_io("inspect_directory", path, error);
  if (std::filesystem::is_symlink(status) ||
      !std::filesystem::is_directory(status)) {
    throw_io("inspect_directory", path,
             std::make_error_code(std::errc::not_a_directory));
  }
}

bool is_unlinked_regular_file(const std::filesystem::path& path) {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(
      artifact_filesystem_path(path, "inspect_path"), error);
  if (error) throw_io("inspect_file", path, error);
  return !std::filesystem::is_symlink(status) &&
         std::filesystem::is_regular_file(status);
}

void sync_file(const std::filesystem::path& path) {
#ifdef _WIN32
  std::error_code path_error;
  const auto native_path = detail::windows_artifact_path(path, path_error);
  if (path_error) throw_io("open_for_flush", path, path_error);
  HANDLE handle = CreateFileW(native_path.c_str(), GENERIC_READ | GENERIC_WRITE,
                              FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    const auto code = std::error_code(static_cast<int>(GetLastError()),
                                      std::system_category());
    throw_io("open_for_flush", path, code);
  }
  const BOOL flushed = FlushFileBuffers(handle);
  const auto flush_error = GetLastError();
  const auto close_result = CloseHandle(handle);
  const auto close_error = GetLastError();
  if (!flushed) {
    const auto code =
        std::error_code(static_cast<int>(flush_error), std::system_category());
    throw_io("flush", path, code);
  }
  if (!close_result) {
    const auto code =
        std::error_code(static_cast<int>(close_error), std::system_category());
    throw_io("close_after_flush", path, code);
  }
#else
  const int descriptor = ::open(path.c_str(), O_RDONLY);
  if (descriptor < 0) {
    throw_io("open_for_flush", path,
             std::error_code(errno, std::generic_category()));
  }
  const int result = ::fsync(descriptor);
  const int flush_error = errno;
  const int close_result = ::close(descriptor);
  const int close_error = errno;
  if (result != 0) {
    throw_io("flush", path,
             std::error_code(flush_error, std::generic_category()));
  }
  if (close_result != 0) {
    throw_io("close_after_flush", path,
             std::error_code(close_error, std::generic_category()));
  }
#endif
}

void sync_directory(const std::filesystem::path& path) {
#ifndef _WIN32
  const int descriptor = ::open(path.c_str(), O_RDONLY | O_DIRECTORY);
  if (descriptor < 0) {
    throw_io("open_directory_for_flush", path,
             std::error_code(errno, std::generic_category()));
  }
  const auto result = ::fsync(descriptor);
  const auto flush_error = errno;
  const auto close_result = ::close(descriptor);
  const auto close_error = errno;
  if (result != 0) {
    throw_io("flush_directory", path,
             std::error_code(flush_error, std::generic_category()));
  }
  if (close_result != 0) {
    throw_io("close_directory_after_flush", path,
             std::error_code(close_error, std::generic_category()));
  }
#else
  (void)path;
#endif
}

void create_durable_directories(const std::filesystem::path& leaf) {
  std::error_code error;
  std::vector<std::filesystem::path> missing;
  for (auto path = leaf; !path.empty(); path = path.parent_path()) {
    const bool present = std::filesystem::exists(
        artifact_filesystem_path(path, "exists"), error);
    if (error) throw_io("create_directory.exists", path, error);
    if (present) break;
    missing.push_back(path);
  }
  std::filesystem::create_directories(
      artifact_filesystem_path(leaf, "create_directory"), error);
  if (error) throw_io("create_directory", leaf, error);
  for (const auto& path : missing) {
    require_unlinked_directory(path);
    sync_directory(path);
    if (path.has_parent_path()) sync_directory(path.parent_path());
  }
}

// One native regular-file handle supplies size, verification, and every read.
// Opening the path again after hashing would allow verified A / returned B.
class VerifiedInput final {
 public:
  explicit VerifiedInput(const std::filesystem::path& path) : path_(path) {
#ifdef _WIN32
    std::error_code path_error;
    const auto native_path = detail::windows_artifact_path(path, path_error);
    if (path_error) throw_io("reader.open", path_, path_error);
    handle_ = CreateFileW(
        native_path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE,
        nullptr, OPEN_EXISTING,
        FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (handle_ == INVALID_HANDLE_VALUE)
      throw_io("reader.open", path_,
               {static_cast<int>(GetLastError()), std::system_category()});
    try {
      BY_HANDLE_FILE_INFORMATION information{};
      if (!GetFileInformationByHandle(handle_, &information))
        throw_io("reader.inspect", path_,
                 {static_cast<int>(GetLastError()), std::system_category()});
      if ((information.dwFileAttributes &
           (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DIRECTORY)) != 0U ||
          GetFileType(handle_) != FILE_TYPE_DISK)
        throw_io("reader.inspect", path_,
                 std::make_error_code(std::errc::invalid_argument));
      LARGE_INTEGER length{};
      if (!GetFileSizeEx(handle_, &length) || length.QuadPart < 0)
        throw_io("reader.size", path_,
                 {static_cast<int>(GetLastError()), std::system_category()});
      size_ = static_cast<std::uint64_t>(length.QuadPart);
    } catch (...) {
      close();
      throw;
    }
#else
    descriptor_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (descriptor_ < 0)
      throw_io("reader.open", path_, {errno, std::generic_category()});
    try {
      struct stat information{};
      if (::fstat(descriptor_, &information) != 0)
        throw_io("reader.inspect", path_, {errno, std::generic_category()});
      if (!S_ISREG(information.st_mode) || information.st_size < 0)
        throw_io("reader.inspect", path_,
                 std::make_error_code(std::errc::invalid_argument));
      size_ = static_cast<std::uint64_t>(information.st_size);
    } catch (...) {
      close();
      throw;
    }
#endif
  }
  ~VerifiedInput() { close(); }
  VerifiedInput(const VerifiedInput&) = delete;
  VerifiedInput& operator=(const VerifiedInput&) = delete;
  std::uint64_t size() const noexcept { return size_; }
  std::size_t read(std::span<std::byte> bytes) {
    if (bytes.empty()) return 0U;
    const auto bounded = std::min(bytes.size(), kHashIoBufferBytes);
#ifdef _WIN32
    DWORD count = 0U;
    if (!ReadFile(handle_, bytes.data(), static_cast<DWORD>(bounded), &count,
                  nullptr))
      throw_io("reader.read", path_,
               {static_cast<int>(GetLastError()), std::system_category()});
    return count;
#else
    ssize_t count;
    do {
      count = ::read(descriptor_, bytes.data(), bounded);
    } while (count < 0 && errno == EINTR);
    if (count < 0)
      throw_io("reader.read", path_, {errno, std::generic_category()});
    return static_cast<std::size_t>(count);
#endif
  }
  void rewind() {
#ifdef _WIN32
    LARGE_INTEGER beginning{};
    if (!SetFilePointerEx(handle_, beginning, nullptr, FILE_BEGIN))
      throw_io("reader.rewind", path_,
               {static_cast<int>(GetLastError()), std::system_category()});
#else
    if (::lseek(descriptor_, 0, SEEK_SET) < 0)
      throw_io("reader.rewind", path_, {errno, std::generic_category()});
#endif
  }

 private:
  void close() noexcept {
#ifdef _WIN32
    if (handle_ != INVALID_HANDLE_VALUE) {
      CloseHandle(handle_);
      handle_ = INVALID_HANDLE_VALUE;
    }
#else
    if (descriptor_ >= 0) {
      ::close(descriptor_);
      descriptor_ = -1;
    }
#endif
  }
  std::filesystem::path path_;
  std::uint64_t size_ = 0U;
#ifdef _WIN32
  HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
  int descriptor_ = -1;
#endif
};

std::string sha256_file(const std::filesystem::path& path,
                        std::uint64_t expected_size, std::uint64_t max_size,
                        std::stop_token stop = {},
                        TelemetryDeadline deadline = std::nullopt) {
#ifdef _WIN32
  std::ifstream input(artifact_filesystem_path(path, "hash_file"),
                      std::ios::binary);
#else
  std::ifstream input(path, std::ios::binary);
#endif
  if (!input) {
    throw IoError(
        "Could not open artifact for hashing",
        ErrorDetails{.component = "experiment.artifact_store",
                     .operation = "hash_file",
                     .expected = "readable regular file",
                     .actual = path_text(path),
                     .hint = "Check permissions and filesystem state."});
  }
  Sha256 hash;
  std::vector<std::byte> buffer(kHashIoBufferBytes);
  std::uint64_t total = 0U;
  while (input) {
    check_telemetry_operation(stop, deadline);
    input.read(reinterpret_cast<char*>(buffer.data()),
               static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (count > 0) {
      const auto bytes = static_cast<std::uint64_t>(count);
      if (bytes > max_size - std::min(total, max_size)) {
        throw_limit("artifact",
                    "at most " + std::to_string(max_size) + " bytes",
                    std::to_string(total + bytes) + " bytes");
      }
      hash.update(std::span<const std::byte>(buffer.data(),
                                             static_cast<std::size_t>(count)));
      total += bytes;
    }
  }
  if (!input.eof()) {
    throw IoError(
        "Could not read artifact while hashing",
        ErrorDetails{.component = "experiment.artifact_store",
                     .operation = "hash_file",
                     .expected = "complete readable file",
                     .actual = path_text(path),
                     .hint = "Check for a failed or disconnected filesystem."});
  }
  if (total != expected_size) {
    throw IoError(
        "Artifact changed while being hashed",
        ErrorDetails{.component = "experiment.artifact_store",
                     .operation = "hash_file",
                     .expected = std::to_string(expected_size) + " bytes",
                     .actual = std::to_string(total) + " bytes",
                     .hint = "Retry the upload from a stable source."});
  }
  return sha256_digest_hex(hash.finish());
}

struct ObjectUsage {
  std::uint64_t bytes = 0U;
  std::uint64_t files = 0U;
};

ObjectUsage object_usage(const std::filesystem::path& root,
                         std::stop_token stop = {},
                         TelemetryDeadline deadline = std::nullopt) {
  std::error_code error;
  const auto object_root = root / "objects";
  const bool present = std::filesystem::exists(
      artifact_filesystem_path(object_root, "exists"), error);
  if (error) throw_io("exists", object_root, error);
  if (!present) return {};
  require_unlinked_directory(object_root);
  ObjectUsage usage;
  for (std::filesystem::recursive_directory_iterator iterator(
           artifact_filesystem_path(object_root, "scan_objects"), error),
       end;
       iterator != end; iterator.increment(error)) {
    check_telemetry_operation(stop, deadline);
    if (error) throw_io("scan_objects", object_root, error);
    if (!is_unlinked_regular_file(iterator->path())) {
      continue;
    }
    const auto size = std::filesystem::file_size(
        artifact_filesystem_path(iterator->path(), "file_size"), error);
    if (error) throw_io("file_size", iterator->path(), error);
    if (size > std::numeric_limits<std::uint64_t>::max() - usage.bytes) {
      throw_limit("store", "representable total bytes", "integer overflow");
    }
    usage.bytes += size;
    ++usage.files;
  }
  if (error) throw_io("scan_objects", object_root, error);
  return usage;
}

std::uint64_t file_age_seconds(const std::filesystem::path& path) {
  std::error_code error;
  const auto timestamp = std::filesystem::last_write_time(
      artifact_filesystem_path(path, "last_write_time"), error);
  if (error) throw_io("last_write_time", path, error);
  const auto now = std::filesystem::file_time_type::clock::now();
  if (timestamp >= now) return 0U;
  const auto age = now - timestamp;
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::seconds>(age).count());
}

std::uint64_t duration_seconds(std::chrono::seconds value) {
  return value <= std::chrono::seconds::zero()
             ? 0U
             : static_cast<std::uint64_t>(value.count());
}

}  // namespace

class FileArtifactStore::FileUploadSession final
    : public ArtifactUploadSession {
 public:
  FileUploadSession(FileArtifactStore* owner, ArtifactDescriptor descriptor,
                    std::filesystem::path temporary)
      : owner_(owner),
        descriptor_(std::move(descriptor)),
        temporary_(std::move(temporary)),
#ifdef _WIN32
        native_temporary_(
            artifact_filesystem_path(temporary_, "create_upload")),
        output_(native_temporary_, std::ios::binary | std::ios::trunc) {
#else
        output_(temporary_, std::ios::binary | std::ios::trunc) {
#endif
    if (!output_) {
      std::error_code error(errno, std::generic_category());
      throw_io("create_upload", temporary_, error);
    }
  }

  ~FileUploadSession() override { abort(); }

  void write(std::span<const std::byte> bytes) override {
    if (closed_) {
      throw_invalid("upload.write", "an open upload session", "closed",
                    "Start a new upload for additional bytes.");
    }
    if (bytes.size() >
        owner_->limits_.max_artifact_bytes -
            std::min(bytes_written_, owner_->limits_.max_artifact_bytes)) {
      throw_limit("upload",
                  "at most " +
                      std::to_string(owner_->limits_.max_artifact_bytes) +
                      " bytes",
                  std::to_string(bytes_written_ + bytes.size()) + " bytes");
    }
    if (descriptor_.expected_size_bytes &&
        bytes.size() >
            *descriptor_.expected_size_bytes -
                std::min(bytes_written_, *descriptor_.expected_size_bytes)) {
      throw_limit("upload.expected_size_bytes",
                  "exactly " +
                      std::to_string(*descriptor_.expected_size_bytes) +
                      " bytes",
                  std::to_string(bytes_written_ + bytes.size()) + " bytes");
    }
    if (bytes.empty()) return;
    output_.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
    if (!output_) {
      throw IoError(
          "Could not write artifact upload",
          ErrorDetails{.component = "experiment.artifact_store",
                       .operation = "upload.write",
                       .expected = "writable temporary file",
                       .actual = path_text(temporary_),
                       .hint = "Check free space and filesystem permissions."});
    }
    bytes_written_ += bytes.size();
  }

  const ArtifactDescriptor& descriptor() const noexcept override {
    return descriptor_;
  }

  std::uint64_t bytes_written() const noexcept override {
    return bytes_written_;
  }

  bool closed() const noexcept override { return closed_; }

  void abort() noexcept override {
    if (closed_) return;
    closed_ = true;
    if (output_.is_open()) output_.close();
    std::error_code error;
#ifdef _WIN32
    // Cached while begin() could report errors; abort() remains noexcept.
    std::filesystem::remove(native_temporary_, error);
#else
    std::filesystem::remove(temporary_, error);
#endif
  }

 private:
  friend class FileArtifactStore;

  void close_for_commit() {
    if (closed_) {
      throw_invalid("upload.commit", "an open upload session", "closed",
                    "Commit each upload at most once.");
    }
    output_.flush();
    if (!output_) {
      throw IoError(
          "Could not flush artifact upload",
          ErrorDetails{.component = "experiment.artifact_store",
                       .operation = "upload.flush",
                       .expected = "flushable temporary file",
                       .actual = path_text(temporary_),
                       .hint = "Check free space and filesystem permissions."});
    }
    output_.close();
    if (output_.fail()) {
      throw IoError(
          "Could not close artifact upload",
          ErrorDetails{.component = "experiment.artifact_store",
                       .operation = "upload.close",
                       .expected = "closed temporary file",
                       .actual = path_text(temporary_),
                       .hint = "Check filesystem health and free space."});
    }
  }

  void mark_committed() noexcept { closed_ = true; }

  FileArtifactStore* owner_;
  ArtifactDescriptor descriptor_;
  std::filesystem::path temporary_;
#ifdef _WIN32
  std::filesystem::path native_temporary_;
#endif
  std::ofstream output_;
  std::uint64_t bytes_written_ = 0U;
  bool closed_ = false;
};

class FileArtifactStore::FileArtifactReader final : public ArtifactReader {
 public:
  FileArtifactReader(std::filesystem::path path, ArtifactDigest digest,
                     std::uint64_t maximum, std::stop_token stop,
                     TelemetryDeadline deadline)
      : input_(path), digest_(std::move(digest)), size_(input_.size()) {
    if (size_ > maximum)
      throw_limit("reader.size", "configured artifact byte bound",
                  std::to_string(size_));
    Sha256 hash;
    std::vector<std::byte> buffer(kHashIoBufferBytes);
    std::uint64_t total = 0U;
    for (;;) {
      check_telemetry_operation(stop, deadline);
      const auto count = input_.read(buffer);
      if (count == 0U) break;
      if (count > maximum - std::min(total, maximum))
        throw_limit("reader.hash", "configured artifact byte bound",
                    "growing file");
      total += count;
      hash.update(std::span<const std::byte>(buffer.data(), count));
    }
    if (total != size_ || sha256_digest_hex(hash.finish()) != digest_.hex)
      throw SerializationError(
          "Artifact failed content-address validation",
          ErrorDetails{
              .component = "experiment.artifact_store",
              .operation = "reader.hash",
              .expected = digest_.hex,
              .actual = path_text(path),
              .hint = "Restore the immutable object from a trusted copy."});
    input_.rewind();
  }

  std::size_t read(std::span<std::byte> destination) override {
    return input_.read(destination);
  }

  std::uint64_t size_bytes() const noexcept override { return size_; }

  const ArtifactDigest& digest() const noexcept override { return digest_; }

 private:
  VerifiedInput input_;
  ArtifactDigest digest_;
  std::uint64_t size_;
};

FileArtifactStore::FileArtifactStore(std::filesystem::path root,
                                     ArtifactStoreLimits limits)
    : limits_(limits) {
  validate_limits(limits_);
  if (root.empty()) {
    throw_invalid("root", "a nonempty directory path", "empty",
                  "Choose a dedicated local artifact directory.");
  }
  std::error_code error;
#ifdef _WIN32
  const auto canonical = std::filesystem::weakly_canonical(
      artifact_filesystem_path(root, "absolute"), error);
  if (error) throw_io("canonicalize", root, error);
  root_ = artifact_logical_root(canonical, root);
#else
  root_ = std::filesystem::absolute(root, error);
  if (error) throw_io("absolute", root_, error);
  root_ = std::filesystem::weakly_canonical(root_, error);
  if (error) throw_io("canonicalize", root_, error);
#endif
  if (root_.string().size() > limits_.max_root_path_bytes) {
    throw_limit(
        "root",
        "at most " + std::to_string(limits_.max_root_path_bytes) + " bytes",
        std::to_string(root_.string().size()) + " bytes");
  }
  create_durable_directories(root_ / "objects");
  create_durable_directories(root_ / "tmp");
  require_unlinked_directory(root_ / "objects");
  require_unlinked_directory(root_ / "tmp");
  sync_directory(root_);
  if (!std::filesystem::is_directory(
          artifact_filesystem_path(root_, "inspect_root"), error) ||
      error) {
    if (error) throw_io("inspect_root", root_, error);
    throw_io("inspect_root", root_,
             std::make_error_code(std::errc::not_a_directory));
  }
}

FileArtifactStore::~FileArtifactStore() = default;

std::unique_ptr<ArtifactUploadSession> FileArtifactStore::begin(
    const ArtifactDescriptor& descriptor) {
  validate_descriptor(descriptor, limits_);
  require_unlinked_directory(root_ / "tmp");
  static std::atomic<std::uint64_t> next_upload{0U};
  const auto sequence = ++next_upload;
  const auto nonce = std::random_device{}();
  const auto temporary = root_ / "tmp" /
                         (".upload-" + std::to_string(nonce) + "-" +
                          std::to_string(sequence) + ".part");
  return std::make_unique<FileUploadSession>(this, descriptor, temporary);
}

ArtifactRef FileArtifactStore::commit(
    ArtifactUploadSession& session,
    std::optional<ArtifactDigest> expected_digest) {
  return commit(session, std::move(expected_digest), {}, std::nullopt);
}

ArtifactRef FileArtifactStore::commit(
    ArtifactUploadSession& session,
    std::optional<ArtifactDigest> expected_digest, std::stop_token stop,
    TelemetryDeadline deadline) {
  check_telemetry_operation(stop, deadline);
  auto* file_session = dynamic_cast<FileUploadSession*>(&session);
  if (file_session == nullptr || file_session->owner_ != this) {
    throw_invalid("upload.commit", "a session created by this store",
                  "foreign upload session",
                  "Commit a session returned by this store's begin().");
  }
  if (file_session->closed_) {
    throw_invalid("upload.commit", "an open upload session", "closed",
                  "Commit each upload at most once.");
  }
  const auto bytes_written = file_session->bytes_written_;
  if (file_session->descriptor_.expected_size_bytes &&
      bytes_written != *file_session->descriptor_.expected_size_bytes) {
    file_session->abort();
    throw_invalid("upload.commit", "the declared expected size",
                  std::to_string(bytes_written) + " bytes",
                  "Write exactly the descriptor's expected_size_bytes.");
  }
  if (expected_digest) {
    try {
      validate_artifact_digest(*expected_digest);
    } catch (...) {
      file_session->abort();
      throw;
    }
  }

  try {
    file_session->close_for_commit();
    sync_file(file_session->temporary_);
    const auto hex = sha256_file(file_session->temporary_, bytes_written,
                                 limits_.max_artifact_bytes, stop, deadline);
    ArtifactDigest digest{.algorithm = ArtifactDigestAlgorithm::sha256,
                          .hex = hex};
    if (expected_digest && *expected_digest != digest) {
      file_session->abort();
      throw SerializationError(
          "Artifact digest does not match the producer assertion",
          ErrorDetails{.component = "experiment.artifact_store",
                       .operation = "upload.commit",
                       .expected = expected_digest->hex,
                       .actual = digest.hex,
                       .hint = "Hash the exact bytes supplied to the upload."});
    }

    const auto destination = object_path(root_, digest);
    std::error_code error;
    require_unlinked_directory(root_ / "objects");
    std::filesystem::create_directories(
        artifact_filesystem_path(destination.parent_path(),
                                 "create_object_directory"),
        error);
    if (error)
      throw_io("create_object_directory", destination.parent_path(), error);
    require_unlinked_directory(destination.parent_path());
    sync_directory(root_ / "objects");

    const bool destination_exists = std::filesystem::exists(
        artifact_filesystem_path(destination, "inspect_existing_object"),
        error);
    if (error) throw_io("inspect_existing_object", destination, error);
    if (!destination_exists) {
      const auto usage = object_usage(root_, stop, deadline);
      if (usage.files >= limits_.max_object_files) {
        throw_limit(
            "store.objects",
            "at most " + std::to_string(limits_.max_object_files) + " objects",
            std::to_string(usage.files + 1U) + " objects");
      }
      if (usage.bytes > limits_.max_total_bytes ||
          bytes_written > limits_.max_total_bytes - usage.bytes) {
        throw_limit(
            "store",
            "at most " + std::to_string(limits_.max_total_bytes) + " bytes",
            std::to_string(usage.bytes + bytes_written) + " bytes");
      }
    }

    if (destination_exists) {
      if (error) throw_io("inspect_existing_object", destination, error);
      if (!is_unlinked_regular_file(destination)) {
        throw_io("inspect_existing_object", destination,
                 std::make_error_code(std::errc::file_exists));
      }
      const auto existing_size = checked_file_size(destination);
      if (existing_size != bytes_written ||
          sha256_file(destination, existing_size, limits_.max_artifact_bytes,
                      stop, deadline) != digest.hex) {
        throw SerializationError(
            "Existing artifact object failed content-address validation",
            ErrorDetails{
                .component = "experiment.artifact_store",
                .operation = "upload.commit",
                .expected = digest.hex,
                .actual = path_text(destination),
                .hint =
                    "Remove the corrupt object after preserving evidence."});
      }
      file_session->abort();
    } else {
      check_telemetry_operation(stop, deadline);
      const auto publish_error =
          detail::publish_artifact_file(file_session->temporary_, destination);
      if (publish_error) throw_io("atomic_publish", destination, publish_error);
      sync_directory(destination.parent_path());
      file_session->mark_committed();
    }

    return ArtifactRef{.digest = std::move(digest),
                       .size_bytes = bytes_written,
                       .logical_name = file_session->descriptor_.logical_name,
                       .media_type = file_session->descriptor_.media_type,
                       .producer = file_session->descriptor_.producer};
  } catch (...) {
    if (!file_session->closed_) file_session->abort();
    throw;
  }
}

std::unique_ptr<ArtifactReader> FileArtifactStore::open(
    const ArtifactDigest& digest) const {
  return open(digest, {}, std::nullopt);
}

std::unique_ptr<ArtifactReader> FileArtifactStore::open(
    const ArtifactDigest& digest, std::stop_token stop,
    TelemetryDeadline deadline) const {
  check_telemetry_operation(stop, deadline);
  validate_artifact_digest(digest);
  const auto path = object_path(root_, digest);
  std::error_code error;
  require_unlinked_directory(root_ / "objects");
  if (std::filesystem::exists(
          artifact_filesystem_path(path.parent_path(),
                                   "inspect_object_directory"),
          error)) {
    require_unlinked_directory(path.parent_path());
  }
  if (error) throw_io("inspect_object_directory", path.parent_path(), error);
  if (!std::filesystem::exists(artifact_filesystem_path(path, "exists"),
                               error)) {
    if (error) throw_io("exists", path, error);
    throw IoError(
        "Artifact does not exist",
        ErrorDetails{.component = "experiment.artifact_store",
                     .operation = "open",
                     .expected = digest.hex,
                     .actual = "missing",
                     .hint = "Commit the artifact before opening it."});
  }
  if (error) throw_io("exists", path, error);
  if (!is_unlinked_regular_file(path)) {
    throw_io("inspect_object", path,
             std::make_error_code(std::errc::invalid_argument));
  }
  return std::make_unique<FileArtifactReader>(
      path, digest, limits_.max_artifact_bytes, stop, deadline);
}

bool FileArtifactStore::exists(const ArtifactDigest& digest) const {
  validate_artifact_digest(digest);
  const auto path = object_path(root_, digest);
  std::error_code error;
  if (!std::filesystem::exists(artifact_filesystem_path(path, "exists"),
                               error) ||
      error)
    return false;
  require_unlinked_directory(root_ / "objects");
  require_unlinked_directory(path.parent_path());
  if (!is_unlinked_regular_file(path)) return false;
  const auto size = std::filesystem::file_size(
      artifact_filesystem_path(path, "file_size"), error);
  if (error || size > limits_.max_artifact_bytes) return false;
  try {
    return sha256_file(path, size, limits_.max_artifact_bytes) == digest.hex;
  } catch (const Error&) {
    return false;
  }
}

ArtifactRecoveryReport FileArtifactStore::recover() {
  ArtifactRecoveryReport report;
  const auto temporary_root = root_ / "tmp";
  std::error_code error;
  const bool present = std::filesystem::exists(
      artifact_filesystem_path(temporary_root, "exists"), error);
  if (error) throw_io("exists", temporary_root, error);
  if (!present) return report;
  require_unlinked_directory(temporary_root);
  for (std::filesystem::directory_iterator iterator(
           artifact_filesystem_path(temporary_root, "scan_tmp"), error),
       end;
       iterator != end; iterator.increment(error)) {
    if (error) throw_io("scan_tmp", temporary_root, error);
    if (!is_unlinked_regular_file(iterator->path())) {
      continue;
    }
    const auto name = iterator->path().filename().string();
    if (!name.starts_with(".upload-") || !name.ends_with(".part")) continue;
    const auto age = file_age_seconds(iterator->path());
    if (age >= duration_seconds(limits_.stale_temp_age)) {
      const auto size = checked_file_size(iterator->path());
      std::filesystem::remove(
          artifact_filesystem_path(iterator->path(), "remove"), error);
      if (error) throw_io("remove_stale_tmp", iterator->path(), error);
      ++report.removed_files;
      report.removed_bytes += size;
    } else {
      ++report.retained_files;
    }
  }
  if (error) throw_io("scan_tmp", temporary_root, error);
  return report;
}

ArtifactVerificationReport FileArtifactStore::verify() const {
  ArtifactVerificationReport report;
  const auto object_root = root_ / "objects";
  std::error_code error;
  const bool present = std::filesystem::exists(
      artifact_filesystem_path(object_root, "exists"), error);
  if (error) throw_io("exists", object_root, error);
  if (!present) return report;
  require_unlinked_directory(object_root);
  for (std::filesystem::recursive_directory_iterator iterator(
           artifact_filesystem_path(object_root, "scan_objects"), error),
       end;
       iterator != end; iterator.increment(error)) {
    if (error) throw_io("scan_objects", object_root, error);
    if (!is_unlinked_regular_file(iterator->path())) {
      continue;
    }
    const auto relative = std::filesystem::relative(
        iterator->path(),
        artifact_filesystem_path(root_, "relative_object_path"), error);
    if (error) throw_io("relative_object_path", iterator->path(), error);
#ifdef _WIN32
    const auto report_path = root_ / relative;
#else
    const auto& report_path = iterator->path();
#endif
    if (!is_object_relative_path(relative)) {
      report.unexpected_files.push_back(report_path);
      continue;
    }
    ++report.checked_files;
    const auto size = checked_file_size(iterator->path());
    report.checked_bytes += size;
    const auto digest = digest_from_relative_path(relative);
    try {
      if (size > limits_.max_artifact_bytes ||
          sha256_file(iterator->path(), size, limits_.max_artifact_bytes) !=
              digest.hex) {
        report.corrupt_files.push_back(report_path);
      }
    } catch (const Error&) {
      report.corrupt_files.push_back(report_path);
    }
  }
  if (error) throw_io("scan_objects", object_root, error);
  return report;
}

ArtifactGcReport FileArtifactStore::collect_garbage(
    std::span<const ArtifactDigest> live, std::chrono::seconds grace_period) {
  if (grace_period < std::chrono::seconds::zero()) {
    throw_invalid("gc.grace_period", "a nonnegative duration", "negative",
                  "Use zero or a positive retention grace period.");
  }
  std::unordered_set<std::string> live_hex;
  live_hex.reserve(live.size());
  for (const auto& digest : live) {
    validate_artifact_digest(digest);
    live_hex.insert(digest.hex);
  }

  ArtifactGcReport report;
  const auto object_root = root_ / "objects";
  std::error_code error;
  const bool present = std::filesystem::exists(
      artifact_filesystem_path(object_root, "exists"), error);
  if (error) throw_io("exists", object_root, error);
  if (!present) return report;
  require_unlinked_directory(object_root);
  for (std::filesystem::recursive_directory_iterator iterator(
           artifact_filesystem_path(object_root, "scan_objects"), error),
       end;
       iterator != end; iterator.increment(error)) {
    if (error) throw_io("scan_objects", object_root, error);
    if (!is_unlinked_regular_file(iterator->path())) {
      continue;
    }
    const auto relative = std::filesystem::relative(
        iterator->path(),
        artifact_filesystem_path(root_, "relative_object_path"), error);
    if (error) throw_io("relative_object_path", iterator->path(), error);
    if (!is_object_relative_path(relative)) {
      ++report.retained_files;
      continue;
    }
    const auto digest = digest_from_relative_path(relative);
    if (live_hex.contains(digest.hex) ||
        file_age_seconds(iterator->path()) < duration_seconds(grace_period)) {
      ++report.retained_files;
      continue;
    }
    const auto size = checked_file_size(iterator->path());
    std::filesystem::remove(
        artifact_filesystem_path(iterator->path(), "remove"), error);
    if (error) throw_io("gc_remove", iterator->path(), error);
    ++report.removed_files;
    report.removed_bytes += size;
  }
  if (error) throw_io("scan_objects", object_root, error);
  return report;
}

ArtifactRef FileArtifactStore::put(
    std::span<const std::byte> bytes, const ArtifactDescriptor& descriptor,
    std::optional<ArtifactDigest> expected_digest) {
  auto session = begin(descriptor);
  session->write(bytes);
  return commit(*session, std::move(expected_digest));
}

}  // namespace jyppx::liuding::experiment
