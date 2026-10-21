// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/foundation/sha256.hpp>
#include <jyppx/interop/file_state_archive.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <thread>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace jyppx::interop {
namespace {

constexpr std::array<std::byte, 8> kMagic{
    std::byte{'J'}, std::byte{'Y'}, std::byte{'P'}, std::byte{'P'},
    std::byte{'X'}, std::byte{'C'}, std::byte{'K'}, std::byte{'1'}};
constexpr std::uint32_t kFormatVersion = 1U;
constexpr std::size_t kDigestBytes = 32U;

[[noreturn]] void malformed(std::string_view operation, std::string actual) {
  throw SerializationError(
      "Malformed file state archive",
      ErrorDetails{
          .component = "interop.file_state_archive",
          .operation = std::string(operation),
          .expected = "complete bounded format-v1 checkpoint",
          .actual = std::move(actual),
          .hint = "Verify the artifact or restore a known good backup."});
}

[[noreturn]] void resource(std::string_view operation, std::uint64_t limit,
                           std::uint64_t actual) {
  throw ResourceExhausted(
      "File state archive exceeds a resource limit",
      ErrorDetails{
          .component = "interop.file_state_archive",
          .operation = std::string(operation),
          .expected = "at most " + std::to_string(limit) + " bytes",
          .actual = std::to_string(actual),
          .hint = "Reduce state or raise the explicit storage limit."});
}

[[noreturn]] void io_failure(std::string_view operation,
                             const std::filesystem::path& path,
                             const std::error_code& error) {
  const auto details = ErrorDetails{
      .component = "interop.file_state_archive",
      .operation = std::string(operation),
      .expected = "a readable or writable checkpoint path",
      .actual = path.string() + ": " + error.message(),
      .hint = "Check permissions, free space, and the parent directory."};
  if (error == std::errc::no_space_on_device ||
      error == std::errc::file_too_large) {
    throw ResourceExhausted("Checkpoint storage is exhausted", details);
  }
  throw IoError("File state archive I/O failed", details);
}

std::filesystem::path sibling(const std::filesystem::path& path,
                              const char* suffix) {
  auto result = path;
  result += suffix;
  return result;
}

void remove_temporary(const std::filesystem::path& temporary) noexcept {
  std::error_code ignored;
  std::filesystem::remove(temporary, ignored);
}

class WriterLock final {
 public:
  ~WriterLock() { release(); }

  bool acquire(const std::filesystem::path& path) {
#if defined(_WIN32)
    handle_ = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE,
                          FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                          OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle_ == INVALID_HANDLE_VALUE) {
      io_failure("lock.open", path,
                 std::error_code(static_cast<int>(GetLastError()),
                                 std::system_category()));
    }
    OVERLAPPED overlapped{};
    if (!LockFileEx(handle_,
                    LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1,
                    0, &overlapped)) {
      const auto code = GetLastError();
      release();
      if (code == ERROR_LOCK_VIOLATION || code == ERROR_IO_PENDING)
        return false;
      io_failure(
          "lock.acquire", path,
          std::error_code(static_cast<int>(code), std::system_category()));
    }
#else
    descriptor_ = ::open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (descriptor_ < 0) {
      io_failure("lock.open", path, {errno, std::generic_category()});
    }
    if (::flock(descriptor_, LOCK_EX | LOCK_NB) != 0) {
      const auto code = errno;
      release();
      if (code == EWOULDBLOCK || code == EAGAIN) return false;
      io_failure("lock.acquire", path, {code, std::generic_category()});
    }
#endif
    return true;
  }

  void release() noexcept {
#if defined(_WIN32)
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

 private:
#if defined(_WIN32)
  HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
  int descriptor_ = -1;
#endif
};

class ByteReader final {
 public:
  explicit ByteReader(std::span<const std::byte> bytes) : bytes_(bytes) {}

  std::span<const std::byte> take(std::uint64_t count) {
    if (count > bytes_.size() - offset_) malformed("read.bytes", "truncated");
    const auto result =
        bytes_.subspan(offset_, static_cast<std::size_t>(count));
    offset_ += static_cast<std::size_t>(count);
    return result;
  }

  std::uint64_t integer(unsigned count) {
    const auto bytes = take(count);
    std::uint64_t value = 0;
    for (unsigned index = 0; index < count; ++index) {
      value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8U);
    }
    return value;
  }

  bool empty() const noexcept { return offset_ == bytes_.size(); }

 private:
  std::span<const std::byte> bytes_;
  std::size_t offset_ = 0;
};

struct StoredParticipant {
  StateManifest manifest;
  std::map<std::string, std::vector<std::byte>> blobs;
};

class FieldWriter final : public StateWriter {
 public:
  FieldWriter(StoredParticipant& participant, std::uint64_t& total,
              const FileStateArchiveOptions& options)
      : participant_(participant), total_(total), options_(options) {}

  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    const auto field = std::find_if(
        participant_.manifest.fields.begin(),
        participant_.manifest.fields.end(),
        [name](const auto& candidate) { return candidate.name == name; });
    if (field == participant_.manifest.fields.end()) {
      malformed("write.field", "undeclared field " + std::string(name));
    }
    if (participant_.blobs.contains(std::string(name))) {
      malformed("write.field", "duplicate field " + std::string(name));
    }
    if (data.size() > field->max_bytes) {
      resource("write.field.declared", field->max_bytes, data.size());
    }
    if (data.size() > options_.max_field_bytes) {
      resource("write.field", options_.max_field_bytes, data.size());
    }
    if (data.size() > options_.max_artifact_bytes - total_) {
      resource("write.total", options_.max_artifact_bytes, data.size());
    }
    participant_.blobs.emplace(
        std::string(name), std::vector<std::byte>(data.begin(), data.end()));
    total_ += data.size();
  }

 private:
  StoredParticipant& participant_;
  std::uint64_t& total_;
  const FileStateArchiveOptions& options_;
};

class FieldReader final : public StateReader {
 public:
  explicit FieldReader(const StoredParticipant& participant)
      : participant_(participant) {}
  const StateManifest& manifest() const noexcept override {
    return participant_.manifest;
  }
  bool contains(std::string_view name) const override {
    return participant_.blobs.contains(std::string(name));
  }
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max_bytes) const override {
    const auto found = participant_.blobs.find(std::string(name));
    if (found == participant_.blobs.end())
      malformed("read.field", std::string(name));
    if (found->second.size() > max_bytes) {
      resource("read.field", max_bytes, found->second.size());
    }
    return found->second;
  }

 private:
  const StoredParticipant& participant_;
};

void synchronize_file(const std::filesystem::path& path) {
#if defined(_WIN32)
  const auto handle =
      CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    io_failure("sync.file", path,
               {static_cast<int>(GetLastError()), std::system_category()});
  }
  const auto success = FlushFileBuffers(handle) != 0;
  const auto code = GetLastError();
  CloseHandle(handle);
  if (!success) {
    io_failure("sync.file", path,
               {static_cast<int>(code), std::system_category()});
  }
#else
  const auto descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (descriptor < 0)
    io_failure("sync.file", path, {errno, std::generic_category()});
  const auto result = ::fsync(descriptor);
  const auto code = errno;
  ::close(descriptor);
  if (result != 0)
    io_failure("sync.file", path, {code, std::generic_category()});
#endif
}

void synchronize_directory(const std::filesystem::path& path) {
#if defined(_WIN32)
  // Publication uses MOVEFILE_WRITE_THROUGH on Windows.
  (void)path;
#else
  const auto descriptor =
      ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (descriptor < 0)
    io_failure("sync.directory", path, {errno, std::generic_category()});
  const auto result = ::fsync(descriptor);
  const auto code = errno;
  ::close(descriptor);
  if (result != 0)
    io_failure("sync.directory", path, {code, std::generic_category()});
#endif
}

void create_durable_parent(const std::filesystem::path& parent,
                           bool synchronize) {
  std::vector<std::filesystem::path> missing;
  std::error_code error;
  for (auto path = parent; !path.empty(); path = path.parent_path()) {
    const auto exists = std::filesystem::exists(path, error);
    if (error) io_failure("create.parent.exists", path, error);
    if (exists) break;
    missing.push_back(path);
  }
  std::filesystem::create_directories(parent, error);
  if (error) io_failure("create.parent", parent, error);
  if (synchronize) {
    // Persist every newly created directory entry, from the leaf to the
    // already-existing ancestor, before an archive can be published below it.
    for (const auto& path : missing) {
      synchronize_directory(path);
      synchronize_directory(path.has_parent_path()
                                ? path.parent_path()
                                : std::filesystem::path("."));
    }
  }
}

std::optional<std::vector<std::byte>> read_snapshot(
    const std::filesystem::path& path, std::uint64_t max_bytes) {
#if defined(_WIN32)
  struct SnapshotHandle {
    HANDLE value;
    ~SnapshotHandle() { CloseHandle(value); }
  };
  const auto value =
      CreateFileW(path.c_str(), GENERIC_READ,
                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                  nullptr, OPEN_EXISTING,
                  FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
  if (value == INVALID_HANDLE_VALUE) {
    const auto code = GetLastError();
    if (code == ERROR_FILE_NOT_FOUND || code == ERROR_PATH_NOT_FOUND)
      return std::nullopt;
    io_failure("read.open", path,
               {static_cast<int>(code), std::system_category()});
  }
  const SnapshotHandle handle{value};
  LARGE_INTEGER length{};
  if (!GetFileSizeEx(handle.value, &length) || length.QuadPart < 0) {
    io_failure("read.size", path,
               {static_cast<int>(GetLastError()), std::system_category()});
  }
  const auto size = static_cast<std::uint64_t>(length.QuadPart);
#else
  // Both size and content come from the same opened inode. Atomic path
  // replacement by another writer cannot mix two checkpoint generations.
  errno = 0;
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    if (errno == ENOENT) return std::nullopt;
    io_failure("read.open", path,
               {errno != 0 ? errno : EIO, std::generic_category()});
  }
  const auto length = input.tellg();
  if (length < 0)
    io_failure("read.size", path, std::make_error_code(std::errc::io_error));
  const auto size = static_cast<std::uint64_t>(length);
#endif
  if (size > max_bytes) resource("read.artifact", max_bytes, size);
  if (size < kMagic.size() + 4U + 8U + kDigestBytes)
    malformed("read.header", "truncated");
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
#if defined(_WIN32)
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const auto count = static_cast<DWORD>((std::min)(
        bytes.size() - offset,
        static_cast<std::size_t>((std::numeric_limits<DWORD>::max)())));
    DWORD read = 0;
    if (!ReadFile(handle.value, bytes.data() + offset, count, &read, nullptr)) {
      io_failure("read.bytes", path,
                 {static_cast<int>(GetLastError()), std::system_category()});
    }
    if (read == 0) malformed("read.size", "file changed during read");
    offset += read;
  }
  std::byte extra{};
  DWORD read = 0;
  if (!ReadFile(handle.value, &extra, 1, &read, nullptr)) {
    io_failure("read.bytes", path,
               {static_cast<int>(GetLastError()), std::system_category()});
  }
  if (read != 0) malformed("read.size", "file changed during read");
#else
  input.seekg(0);
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input || input.peek() != std::char_traits<char>::eof())
    malformed("read.size", "file changed during read");
#endif
  return bytes;
}

void publish(const std::filesystem::path& path,
             std::span<const std::byte> bytes,
             const FileStateArchiveOptions& options) {
  const auto parent =
      path.has_parent_path() ? path.parent_path() : std::filesystem::path(".");
  std::error_code error;
  const auto space = std::filesystem::space(parent, error);
  if (error) io_failure("space", parent, error);
  if (bytes.size() > space.available ||
      options.min_free_storage_bytes > space.available - bytes.size()) {
    throw ResourceExhausted(
        "Insufficient space to stage a checkpoint replacement",
        ErrorDetails{
            .component = "interop.file_state_archive",
            .operation = "write.space",
            .expected = "artifact bytes plus configured free-space reserve",
            .actual = std::to_string(space.available),
            .hint = "Free storage or reduce the checkpoint/reserve."});
  }
  const auto temporary = sibling(path, ".tmp");
  remove_temporary(temporary);
  try {
    errno = 0;
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output)
      io_failure("open.temp", temporary,
                 {errno != 0 ? errno : EIO, std::generic_category()});
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    output.flush();
    if (!output)
      io_failure("write.temp", temporary,
                 {errno != 0 ? errno : EIO, std::generic_category()});
    output.close();
    if (!output)
      io_failure("close.temp", temporary,
                 {errno != 0 ? errno : EIO, std::generic_category()});
    if (options.synchronize) {
      synchronize_file(temporary);
      synchronize_directory(parent);
    }
#if defined(_WIN32)
    const auto flags = MOVEFILE_REPLACE_EXISTING |
                       (options.synchronize ? MOVEFILE_WRITE_THROUGH : 0U);
    // Windows metadata probes and external readers may briefly open either
    // path without delete sharing. Keep the old destination intact while
    // retrying only transient sharing/access conflicts, within a fixed bound.
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (!MoveFileExW(temporary.c_str(), path.c_str(), flags)) {
      const auto code = GetLastError();
      if ((code != ERROR_SHARING_VIOLATION && code != ERROR_LOCK_VIOLATION &&
           code != ERROR_ACCESS_DENIED) ||
          std::chrono::steady_clock::now() >= deadline) {
        io_failure("publish", path,
                   {static_cast<int>(code), std::system_category()});
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
#else
    std::filesystem::rename(temporary, path, error);
    if (error) io_failure("publish", path, error);
#endif
    if (options.synchronize) synchronize_directory(parent);
  } catch (...) {
    remove_temporary(temporary);
    throw;
  }
}

}  // namespace

class FileStateArchive::Impl {
 public:
  Impl(std::filesystem::path path, FileStateArchiveOptions options)
      : path_(std::move(path)), options_(options) {
    if (path_.empty() || path_.filename().empty() ||
        options_.max_artifact_bytes == 0U || options_.max_field_bytes == 0U ||
        options_.max_field_bytes > options_.max_artifact_bytes ||
        options_.max_artifact_bytes >
            static_cast<std::uint64_t>(
                (std::numeric_limits<std::streamsize>::max)()) ||
        options_.max_artifact_bytes >
            (std::numeric_limits<std::size_t>::max)()) {
      throw InvalidArgument("Invalid file archive path or byte limits");
    }
    // Validate every parsing limit even for a new destination.
    (void)make_checkpoint_manifest_capability(options_.manifest_limits);
    std::error_code error;
    const auto temporary = sibling(path_, ".tmp");
    const auto stale = std::filesystem::exists(temporary, error);
    if (error) io_failure("exists.temp", temporary, error);
    if (stale) {
      try {
        WriterLock lock;
        if (lock.acquire(sibling(path_, ".lock"))) remove_temporary(temporary);
      } catch (const Error&) {
        // Read-only callers can still open an intact published snapshot.
      }
    }
    load();
  }

  void build_readers() {
    readers_.clear();
    for (const auto& [id, participant] : stored_) {
      readers_.emplace(id, std::make_unique<FieldReader>(participant));
    }
  }

  void load() {
    const auto snapshot = read_snapshot(path_, options_.max_artifact_bytes);
    if (!snapshot) return;
    const auto& bytes = *snapshot;
    const auto payload =
        std::span<const std::byte>(bytes).first(bytes.size() - kDigestBytes);
    const auto expected = sha256(payload);
    if (!std::equal(expected.begin(), expected.end(),
                    bytes.end() - kDigestBytes))
      malformed("read.digest", "artifact digest mismatch");
    ByteReader reader(payload);
    const auto magic = reader.take(kMagic.size());
    if (!std::equal(magic.begin(), magic.end(), kMagic.begin()))
      malformed("read.magic", "mismatch");
    if (reader.integer(4U) != kFormatVersion) {
      throw CompatibilityError("Unsupported file state archive format version");
    }
    const auto metadata_size = reader.integer(8U);
    if (metadata_size > options_.manifest_limits.max_manifest_bytes)
      resource("read.manifest", options_.manifest_limits.max_manifest_bytes,
               metadata_size);
    const auto metadata = reader.take(metadata_size);
    manifest_ = parse_checkpoint_artifact_manifest_json(
        {reinterpret_cast<const char*>(metadata.data()), metadata.size()},
        options_.manifest_limits);
    std::sort(manifest_.participants.begin(), manifest_.participants.end(),
              [](const auto& a, const auto& b) {
                return a.state.participant_id < b.state.participant_id;
              });
    for (const auto& entry : manifest_.participants) {
      StoredParticipant participant{.manifest = entry.state, .blobs = {}};
      for (const auto& field : entry.state.fields) {
        if (field.max_bytes > options_.max_field_bytes)
          resource("read.field.declared", options_.max_field_bytes,
                   field.max_bytes);
      }
      for (const auto& blob : entry.blobs) {
        if (blob.size_bytes > options_.max_field_bytes)
          resource("read.field", options_.max_field_bytes, blob.size_bytes);
        const auto data = reader.take(blob.size_bytes);
        if (sha256_hex(data) != blob.digest.hex)
          malformed("read.blob.digest", blob.field_name);
        participant.blobs.emplace(
            blob.field_name, std::vector<std::byte>(data.begin(), data.end()));
      }
      stored_.emplace(entry.state.participant_id, std::move(participant));
    }
    if (!reader.empty())
      malformed("read.trailing", "bytes after declared blobs");
    build_readers();
  }

  CheckpointArtifactManifest describe_staged() const {
    CheckpointArtifactManifest result{.build = current_build_fingerprint(),
                                      .participants = {}};
    for (const auto& [id, participant] : staged_) {
      CheckpointParticipantEntry entry{.state = participant.manifest,
                                       .blobs = {}};
      for (const auto& [name, bytes] : participant.blobs) {
        entry.blobs.push_back(
            {.field_name = name,
             .relative_path = "blobs/" + id + "/" + name + ".bin",
             .size_bytes = bytes.size(),
             .digest = {.hex = sha256_hex(bytes)}});
      }
      result.participants.push_back(std::move(entry));
    }
    return result;
  }

  std::vector<std::byte> encode(
      const CheckpointArtifactManifest& manifest) const {
    const auto json =
        checkpoint_artifact_manifest_json(manifest, options_.manifest_limits);
    std::vector<std::byte> result;
    const auto append = [&](std::span<const std::byte> bytes) {
      if (bytes.size() > options_.max_artifact_bytes - result.size())
        resource("write.artifact", options_.max_artifact_bytes, bytes.size());
      result.insert(result.end(), bytes.begin(), bytes.end());
    };
    const auto integer = [&](std::uint64_t value, unsigned count) {
      std::array<std::byte, 8> bytes{};
      for (unsigned i = 0; i < count; ++i)
        bytes[i] = static_cast<std::byte>(value >> (i * 8U));
      append(std::span<const std::byte>(bytes).first(count));
    };
    append(kMagic);
    integer(kFormatVersion, 4U);
    integer(json.size(), 8U);
    append({reinterpret_cast<const std::byte*>(json.data()), json.size()});
    for (const auto& entry : manifest.participants) {
      for (const auto& blob : entry.blobs)
        append(
            staged_.at(entry.state.participant_id).blobs.at(blob.field_name));
    }
    const auto digest = sha256(result);
    append(digest);
    return result;
  }

  std::filesystem::path path_;
  FileStateArchiveOptions options_;
  CheckpointArtifactManifest manifest_;
  std::map<std::string, StoredParticipant> stored_;
  std::map<std::string, std::unique_ptr<FieldReader>> readers_;
  std::map<std::string, StoredParticipant> staged_;
  std::map<std::string, std::unique_ptr<FieldWriter>> writers_;
  std::uint64_t staged_bytes_ = 0;
  WriterLock lock_;
  bool open_ = false;
};

FileStateArchive::FileStateArchive(std::filesystem::path path,
                                   FileStateArchiveOptions options)
    : impl_(std::make_unique<Impl>(std::move(path), options)) {}

FileStateArchive::~FileStateArchive() { rollback(); }

const std::filesystem::path& FileStateArchive::path() const noexcept {
  return impl_->path_;
}

const CheckpointArtifactManifest& FileStateArchive::manifest() const noexcept {
  return impl_->manifest_;
}

void FileStateArchive::begin(const std::vector<StateManifest>& participants) {
  if (impl_->open_) throw NotReady("Archive transaction is already open");
  if (participants.empty())
    throw InvalidArgument("Cannot publish an empty participant set");
  if (participants.size() > impl_->options_.manifest_limits.max_participants)
    resource("begin.participants",
             impl_->options_.manifest_limits.max_participants,
             participants.size());
  std::map<std::string, StoredParticipant> staged;
  std::string previous;
  std::size_t fields = 0;
  for (const auto& participant : participants) {
    validate_state_manifest(participant);
    if (!previous.empty() && previous >= participant.participant_id)
      throw InvalidArgument("Participant IDs must be strictly ascending");
    previous = participant.participant_id;
    if (participant.fields.size() >
        impl_->options_.manifest_limits.max_blobs - fields)
      resource("begin.fields", impl_->options_.manifest_limits.max_blobs,
               participant.fields.size());
    fields += participant.fields.size();
    for (const auto& field : participant.fields) {
      if (field.max_bytes > impl_->options_.max_field_bytes)
        resource("begin.field", impl_->options_.max_field_bytes,
                 field.max_bytes);
    }
    staged.emplace(participant.participant_id,
                   StoredParticipant{.manifest = participant, .blobs = {}});
  }
  if (impl_->path_.has_parent_path()) {
    create_durable_parent(impl_->path_.parent_path(),
                          impl_->options_.synchronize);
  }
  if (!impl_->lock_.acquire(sibling(impl_->path_, ".lock")))
    throw NotReady("Another writer owns the checkpoint path");
  try {
    impl_->staged_ = std::move(staged);
    impl_->staged_bytes_ = 0;
    for (auto& [id, participant] : impl_->staged_) {
      impl_->writers_.emplace(
          id, std::make_unique<FieldWriter>(participant, impl_->staged_bytes_,
                                            impl_->options_));
    }
    impl_->open_ = true;
  } catch (...) {
    rollback();
    throw;
  }
}

StateWriter& FileStateArchive::writer_for(std::string_view participant_id) {
  if (!impl_->open_) throw NotReady("Archive transaction is closed");
  const auto found = impl_->writers_.find(std::string(participant_id));
  if (found == impl_->writers_.end())
    throw InvalidArgument("Unknown checkpoint participant");
  return *found->second;
}

void FileStateArchive::commit() {
  if (!impl_->open_) throw NotReady("Archive transaction is closed");
  try {
    auto manifest = impl_->describe_staged();
    // The existing interop validator rejects every absent required field.
    try {
      validate_checkpoint_artifact_manifest(manifest,
                                            impl_->options_.manifest_limits);
    } catch (const InvalidArgument& error) {
      throw SerializationError("Incomplete checkpoint state", error.details());
    }
    auto bytes = impl_->encode(manifest);
    // Prepare readers before the filesystem commit so allocation cannot fail
    // after publication and turn a successful replacement into a false failure.
    std::map<std::string, std::unique_ptr<FieldReader>> readers;
    for (const auto& [id, participant] : impl_->staged_)
      readers.emplace(id, std::make_unique<FieldReader>(participant));
    // A late durability failure may leave the new complete file at the path.
    // Keep this object's old manifest/readers together until a successful
    // commit, just as an independently opened snapshot retains its generation.
    publish(impl_->path_, bytes, impl_->options_);
    impl_->stored_.swap(impl_->staged_);
    impl_->manifest_ = std::move(manifest);
    impl_->readers_.swap(readers);
    rollback();
  } catch (...) {
    rollback();
    throw;
  }
}

void FileStateArchive::rollback() noexcept {
  if (!impl_) return;
  impl_->writers_.clear();
  impl_->staged_.clear();
  impl_->staged_bytes_ = 0;
  impl_->open_ = false;
  impl_->lock_.release();
}

const StateReader& FileStateArchive::reader_for(
    std::string_view participant_id) const {
  const auto found = impl_->readers_.find(std::string(participant_id));
  if (found == impl_->readers_.end())
    malformed("read.participant", std::string(participant_id));
  return *found->second;
}

}  // namespace jyppx::interop
