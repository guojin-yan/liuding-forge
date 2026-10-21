// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_EXPERIMENT_ARTIFACT_STORE_HPP
#define JYPPX_EXPERIMENT_ARTIFACT_STORE_HPP

/**
 * @file artifact_store.hpp
 * @brief Content-addressed artifact contracts and the local file backend.
 *
 * The store deliberately owns bytes only.  Run events, references, and
 * lineage are metadata concerns and can be persisted by a separate RunSink or
 * SQLite backend.  A committed blob is immutable and addressed by its
 * SHA-256 digest.
 */

#include <jyppx/foundation/artifact.hpp>
#include <jyppx/foundation/telemetry.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::experiment {

/**
 * @brief Describes the producer-facing metadata supplied before an upload.
 *
 * These fields are bounded and never interpreted as a path or executable
 * content.  `expected_size_bytes` is optional because streaming producers may
 * not know the final size until the stream is complete.
 */
struct ArtifactDescriptor {
  std::string logical_name;  ///< Stable display/lineage name.
  std::string media_type =
      "application/octet-stream";  ///< Stored content type.
  std::string producer;            ///< Component, recipe, or producer identity.
  std::optional<std::uint64_t>
      expected_size_bytes;  ///< Optional exact byte count.
};

/** @brief Bounds applied to descriptors, uploads, and local storage. */
struct ArtifactStoreLimits {
  std::uint64_t max_artifact_bytes = 1ULL << 40U;  ///< Maximum one-blob size.
  std::uint64_t max_total_bytes = 1ULL << 42U;     ///< Maximum committed bytes.
  std::uint64_t max_object_files =
      1000000U;                             ///< Maximum committed object count.
  std::size_t max_root_path_bytes = 4096U;  ///< Maximum canonical root bytes.
  std::size_t max_logical_name_bytes = 512U;  ///< Maximum logical name bytes.
  std::size_t max_media_type_bytes = 256U;    ///< Maximum media type bytes.
  std::size_t max_producer_bytes = 512U;      ///< Maximum producer name bytes.
  std::chrono::seconds stale_temp_age =
      std::chrono::hours(24);  ///< Temp retention.
};

/** @brief A bounded stream used to write one not-yet-committed artifact. */
class ArtifactUploadSession {
 public:
  /** @brief Releases the upload and abandons uncommitted temporary state. */
  virtual ~ArtifactUploadSession() = default;

  /**
   * @brief Appends bytes to the temporary upload.
   * @param bytes Bytes appended before returning.
   * @throws ResourceExhausted If the artifact bound is exceeded.
   * @throws IoError If the temporary file cannot accept the bytes.
   */
  virtual void write(std::span<const std::byte> bytes) = 0;

  /** @brief Returns the descriptor captured by `begin`.
   * @return Borrowed descriptor valid for the session lifetime.
   */
  virtual const ArtifactDescriptor& descriptor() const noexcept = 0;

  /** @brief Returns bytes written to the temporary stream.
   * @return Total byte count accepted so far.
   */
  virtual std::uint64_t bytes_written() const noexcept = 0;

  /** @brief Returns whether the session has been committed or aborted.
   * @return True after a terminal session operation.
   */
  virtual bool closed() const noexcept = 0;

  /** @brief Aborts the upload and removes its temporary file when possible. */
  virtual void abort() noexcept = 0;
};

/** @brief A read-only stream for one verified immutable artifact. */
class ArtifactReader {
 public:
  /** @brief Releases the verified reader. */
  virtual ~ArtifactReader() = default;

  /**
   * @brief Reads up to `destination.size()` bytes, returning the count read.
   * @param destination Caller-owned writable byte range.
   * @return Zero at end of file.
   */
  virtual std::size_t read(std::span<std::byte> destination) = 0;

  /** @brief Returns the exact immutable blob size.
   * @return Byte count verified when opened.
   */
  virtual std::uint64_t size_bytes() const noexcept = 0;

  /** @brief Returns the digest verified when the reader was opened.
   * @return Borrowed immutable digest identity.
   */
  virtual const ArtifactDigest& digest() const noexcept = 0;
};

/** @brief Result of removing stale temporary upload files. */
struct ArtifactRecoveryReport {
  std::size_t removed_files = 0;    ///< Number of removed stale upload files.
  std::size_t retained_files = 0;   ///< Number of recent upload files retained.
  std::uint64_t removed_bytes = 0;  ///< Total bytes removed.
};

/** @brief Result of checking every immutable blob under the CAS root. */
struct ArtifactVerificationReport {
  std::size_t checked_files = 0;  ///< Objects with canonical CAS names checked.
  std::uint64_t checked_bytes = 0;  ///< Stored byte total observed.
  std::vector<std::filesystem::path> corrupt_files;     ///< Hash/size failures.
  std::vector<std::filesystem::path> unexpected_files;  ///< Invalid CAS paths.

  /** @brief True when no object failed digest/path validation.
   * @return Whether verification found no failures.
   */
  bool ok() const noexcept {
    return corrupt_files.empty() && unexpected_files.empty();
  }
};

/** @brief Result of content-addressed garbage collection. */
struct ArtifactGcReport {
  std::size_t removed_files = 0;    ///< Unreferenced objects removed.
  std::uint64_t removed_bytes = 0;  ///< Total object bytes removed.
  std::size_t retained_files = 0;   ///< Live, recent, or unknown objects kept.
};

/**
 * @brief Narrow storage-neutral contract used by RunSink and adapters.
 *
 * Implementations are not required to be thread-safe.  Callers must serialize
 * access to a store unless a concrete backend documents stronger guarantees.
 */
class ArtifactStore {
 public:
  /** @brief Releases the storage backend. */
  virtual ~ArtifactStore() = default;

  /** @brief Starts a bounded streaming upload in a temporary location.
   * @param descriptor Producer metadata and optional exact byte count.
   * @return Owning temporary upload session.
   */
  virtual std::unique_ptr<ArtifactUploadSession> begin(
      const ArtifactDescriptor& descriptor) = 0;

  /**
   * @brief Flushes, hashes, verifies, and atomically publishes an upload.
   * @param session Session returned by this store's `begin`.
   * @param expected_digest Optional producer assertion checked before publish.
   * @return Reference to the verified immutable bytes.
   * @throws SerializationError On hash mismatch.
   * @throws InvalidArgument On a malformed expected digest.
   */
  virtual ArtifactRef commit(
      ArtifactUploadSession& session,
      std::optional<ArtifactDigest> expected_digest = std::nullopt) = 0;

  /** @brief Opens and verifies an immutable artifact by digest.
   * @param digest Content identity identifying the object.
   * @return Owning verified reader.
   */
  virtual std::unique_ptr<ArtifactReader> open(
      const ArtifactDigest& digest) const = 0;

  /** @brief Checks whether a valid immutable blob is present.
   * @param digest Content identity identifying the object.
   * @return True only if the object exists and its hash is valid.
   */
  virtual bool exists(const ArtifactDigest& digest) const = 0;
};

/**
 * @brief Local content-addressed file backend.
 *
 * Objects are stored under `<root>/objects/aa/<62 hex characters>`.  Uploads
 * first land under `<root>/tmp`, are flushed and hashed, and are then published
 * with an atomic same-filesystem rename.  A crash can leave a temporary file;
 * `recover` removes only files older than `stale_temp_age`.  A metadata backend
 * must serialize writers and retention operations for a root; this backend
 * rejects symbolic links in its managed directories, and does not follow them
 * while scanning.
 */
class FileArtifactStore final : public ArtifactStore {
 public:
  /**
   * @brief Creates a dedicated local content-addressed store handle.
   * @param root Root directory created if it does not exist.
   * @param limits Resource and metadata bounds.
   * @throws IoError If the root cannot be created or is not a directory.
   */
  explicit FileArtifactStore(std::filesystem::path root,
                             ArtifactStoreLimits limits = {});
  /** @brief Releases this backend handle without deleting committed blobs. */
  ~FileArtifactStore() override;

  /** @brief Disallows copying handles borrowed by active upload sessions. */
  FileArtifactStore(const FileArtifactStore&) = delete;
  /** @brief Disallows replacing a handle borrowed by active uploads. */
  FileArtifactStore& operator=(const FileArtifactStore&) = delete;
  /** @brief Disallows moving a handle borrowed by active upload sessions. */
  FileArtifactStore(FileArtifactStore&&) = delete;
  /** @brief Disallows moving over a handle borrowed by active uploads. */
  FileArtifactStore& operator=(FileArtifactStore&&) = delete;

  /** @brief Creates one temporary streaming upload.
   * @param descriptor Producer metadata and optional exact byte count.
   * @return Owning temporary upload session.
   */
  std::unique_ptr<ArtifactUploadSession> begin(
      const ArtifactDescriptor& descriptor) override;
  /** @brief Verifies and atomically publishes one temporary upload.
   * @param session Upload session returned by this handle.
   * @param expected_digest Optional expected content digest.
   * @return Verified immutable artifact reference.
   */
  ArtifactRef commit(
      ArtifactUploadSession& session,
      std::optional<ArtifactDigest> expected_digest = std::nullopt) override;
  /** @brief Publishes with cooperative hash/scan cancellation boundaries.
   * @param session Open upload created by this store.
   * @param expected_digest Optional expected content assertion.
   * @param stop Cooperative cancellation checked per chunk and scan entry.
   * @param deadline Absolute operation deadline, checked before publication.
   * @return Committed content identity; no post-publication stop check.
   */
  ArtifactRef commit(ArtifactUploadSession& session,
                     std::optional<ArtifactDigest> expected_digest,
                     std::stop_token stop,
                     TelemetryDeadline deadline = std::nullopt);
  /** @brief Verifies and opens one immutable CAS object.
   * @param digest Content identity identifying the object.
   * @return Owning verified reader.
   */
  std::unique_ptr<ArtifactReader> open(
      const ArtifactDigest& digest) const override;

  /** @brief Verifies one immutable open handle with cancellation/deadline.
   * @param digest Expected immutable SHA-256 identity.
   * @param stop Cooperative cancellation checked per hashing chunk.
   * @param deadline Absolute steady-clock verification deadline.
   * @return Verified reader retaining the exact hashed file handle.
   */
  std::unique_ptr<ArtifactReader> open(
      const ArtifactDigest& digest, std::stop_token stop,
      TelemetryDeadline deadline = std::nullopt) const;
  /** @brief Reports whether a valid object is present at its content digest.
   * @param digest Content identity identifying the object.
   * @return True only if present and valid.
   */
  bool exists(const ArtifactDigest& digest) const override;

  /** @brief Returns the canonical store root.
   * @return Borrowed canonical directory path.
   */
  const std::filesystem::path& root() const noexcept { return root_; }

  /** @brief Returns the immutable limits captured at construction.
   * @return Borrowed resource and metadata bounds.
   */
  const ArtifactStoreLimits& limits() const noexcept { return limits_; }

  /** @brief Removes stale `.upload-*` files under `<root>/tmp`.
   * @return Removed and retained temporary file counts.
   */
  ArtifactRecoveryReport recover();

  /** @brief Rehashes every CAS object and reports corruption without mutation.
   * @return Per-root verification report.
   */
  ArtifactVerificationReport verify() const;

  /**
   * @brief Removes unreferenced objects older than `grace_period`.
   * @param live Digests retained by the metadata backend.
   * @param grace_period Minimum age before deletion; zero means immediately.
   * @return Counts and bytes removed or retained.
   */
  ArtifactGcReport collect_garbage(
      std::span<const ArtifactDigest> live,
      std::chrono::seconds grace_period = std::chrono::hours(24));

  /** @brief Returns a copy of a blob through the streaming contract.
   * @param bytes Exact object bytes.
   * @param descriptor Producer metadata and optional exact byte count.
   * @param expected_digest Optional expected content digest.
   * @return Verified immutable artifact reference.
   */
  ArtifactRef put(std::span<const std::byte> bytes,
                  const ArtifactDescriptor& descriptor,
                  std::optional<ArtifactDigest> expected_digest = std::nullopt);

 private:
  /** @brief Temporary file implementation of the upload protocol. */
  class FileUploadSession;
  /** @brief Verified immutable file reader implementation. */
  class FileArtifactReader;

  std::filesystem::path root_;  ///< Canonical dedicated CAS directory.
  ArtifactStoreLimits limits_;  ///< Immutable storage and descriptor bounds.
};

}  // namespace jyppx::experiment

#endif  // JYPPX_EXPERIMENT_ARTIFACT_STORE_HPP
