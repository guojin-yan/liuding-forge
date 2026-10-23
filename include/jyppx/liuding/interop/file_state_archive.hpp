// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_INTEROP_FILE_STATE_ARCHIVE_HPP
#define JYPPX_LIUDING_INTEROP_FILE_STATE_ARCHIVE_HPP
/** @file file_state_archive.hpp
 * @brief Persists participant blobs in one bounded, atomically published file.
 */

#include <jyppx/liuding/interop/checkpoint_manifest.hpp>

#include <filesystem>
#include <memory>
#include <string_view>

namespace jyppx::liuding::interop {

/** @brief Explicit storage and parsing limits for a file state archive. */
struct FileStateArchiveOptions {
  std::uint64_t max_artifact_bytes = 1ULL << 40U;  ///< Includes all framing.
  std::uint64_t max_field_bytes = 1ULL << 30U;  ///< Maximum one stored field.
  std::uint64_t min_free_storage_bytes = 0;  ///< Space retained during staging.
  bool synchronize = true;  ///< Synchronize content and publication metadata.
  CheckpointManifestLimits manifest_limits;  ///< Bounds for the embedded JSON.
  bool read_only =
      false;  ///< Require an existing archive; never clean or lock.
};

/**
 * @brief Stores storage-neutral participant states using the interop manifest.
 *
 * Format v1 embeds a canonical checkpoint manifest, participant blobs, and a
 * trailing SHA-256 over the complete payload. Every field is also hashed.
 * Reads validate the whole artifact before exposing any reader. Transactions
 * use a sibling temporary file and atomic replacement on the same filesystem.
 * Writers to the same path are serialized by an operating-system file lock;
 * opening a reader never removes another writer's active temporary file.
 *
 * The object is not thread-safe. Quiesce participant mutation while capturing
 * state. Blobs are held in memory under the explicit byte limits. Readers are
 * valid until the next successful commit or destruction. Failures before
 * publication preserve the previous file; a directory-sync error after rename
 * preserves this object's old snapshot and requires reopening the file to
 * determine which complete version is present at the path.
 */
class FileStateArchive final {
 public:
  /**
   * @brief Opens an existing immutable snapshot or an empty destination.
   * @param path Final file path with an existing or creatable parent directory.
   * @param options Bounds, durability policy, and optional immutable open mode.
   * Read-only mode requires existing valid content and preserves sibling files.
   * @throws Error If options, stored bytes, or filesystem state are invalid.
   */
  explicit FileStateArchive(std::filesystem::path path,
                            FileStateArchiveOptions options = {});
  /** @brief Discards unpublished state and releases any writer lock. */
  ~FileStateArchive();

  /** @brief Snapshot and transaction handles cannot be copied. */
  FileStateArchive(const FileStateArchive&) = delete;
  /** @brief Snapshot and transaction handles cannot be copy assigned. */
  FileStateArchive& operator=(const FileStateArchive&) = delete;
  /** @brief Borrowed readers and writers require a stable archive handle. */
  FileStateArchive(FileStateArchive&&) = delete;
  /** @brief Archive handles cannot be move assigned. */
  FileStateArchive& operator=(FileStateArchive&&) = delete;

  /** @brief Returns the final archive path supplied at construction.
   * @return Borrowed path valid for this object's lifetime.
   */
  const std::filesystem::path& path() const noexcept;
  /** @brief Returns the complete stored manifest, empty before first commit.
   * @return Snapshot metadata valid until the next successful commit.
   */
  const CheckpointArtifactManifest& manifest() const noexcept;
  /**
   * @brief Begins capture of a complete, canonically ordered participant set.
   * @param participants Nonempty manifests in strictly ascending ID order.
   * @throws Error If read-only, active, locked, invalid, or oversized.
   */
  void begin(const std::vector<StateManifest>& participants);
  /**
   * @brief Returns the writer for one declared participant.
   * @param participant_id Stable ID present in the active transaction.
   * @return Writer valid until commit or rollback.
   */
  StateWriter& writer_for(std::string_view participant_id);
  /** @brief Validates and atomically publishes the staged complete archive. */
  void commit();
  /** @brief Discards the current capture transaction without throwing. */
  void rollback() noexcept;
  /**
   * @brief Returns one integrity-checked participant snapshot.
   * @param participant_id Stable ID declared in the stored manifest.
   * @return Reader valid until replacement or destruction.
   * @throws SerializationError If the participant is absent.
   */
  const StateReader& reader_for(std::string_view participant_id) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;  ///< Snapshot, transaction, and codec state.
};

}  // namespace jyppx::liuding::interop

#endif  // JYPPX_LIUDING_INTEROP_FILE_STATE_ARCHIVE_HPP