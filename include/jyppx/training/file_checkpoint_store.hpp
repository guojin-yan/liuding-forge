// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

/**
 * @file file_checkpoint_store.hpp
 * @brief Defines the bounded, atomically published file checkpoint store.
 */

#ifndef JYPPX_TRAINING_FILE_CHECKPOINT_STORE_HPP
#define JYPPX_TRAINING_FILE_CHECKPOINT_STORE_HPP

#include <jyppx/foundation/build_config.hpp>
#include <jyppx/training/checkpoint_coordinator.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>

#if JYPPX_COMPONENT_INTEROP_ENABLED

namespace jyppx::training {

/**
 * @brief Bounds one file checkpoint artifact and its temporary publication.
 */
struct FileCheckpointStoreOptions {
  std::uint64_t max_artifact_bytes = 1ULL << 40U;  ///< Maximum final file size.
  std::uint64_t max_field_bytes = 1ULL << 30U;     ///< Maximum one field size.
  std::uint64_t min_free_storage_bytes =
      0;  ///< Minimum free bytes to retain after staging the replacement.
  bool synchronize =
      true;  ///< Flush file and directory metadata before publish.
};

/**
 * @brief Stores a complete checkpoint in one atomically published file.
 *
 * A store can be constructed before a checkpoint exists, used as a sink, and
 * reopened by another process as an immutable source. The current file format
 * is owned by interop::FileStateArchive; this adapter only translates the
 * training coordinator descriptor. It is available when the interop component
 * is enabled. Failures before atomic publication preserve the previous file;
 * after a directory-sync error, the in-memory descriptor and readers retain
 * the previous snapshot. Reopen to inspect the complete published file.
 * Readers remain valid until the next successful commit. The class is not
 * thread-safe.
 */
class FileCheckpointStore final : public CheckpointStateSink,
                                  public CheckpointStateSource {
 public:
  /**
   * @brief Opens a checkpoint path for writing or reading.
   * @param path Final artifact path, which must have a non-empty filename.
   * @param options Positive resource and synchronization options.
   * @throws InvalidArgument If the path or limits are invalid.
   * @throws SerializationError If an existing artifact is malformed.
   * @throws ResourceExhausted If an existing artifact exceeds the limits.
   */
  explicit FileCheckpointStore(std::filesystem::path path,
                               FileCheckpointStoreOptions options = {});

  /** @brief Destroys the store and discards any unpublished transaction. */
  ~FileCheckpointStore() override;

  /** @brief Transaction handles cannot be copied. */
  FileCheckpointStore(const FileCheckpointStore&) = delete;
  /** @brief Transaction handles cannot be copy assigned. */
  FileCheckpointStore& operator=(const FileCheckpointStore&) = delete;
  /** @brief Borrowed readers and writers require a stable store handle. */
  FileCheckpointStore(FileCheckpointStore&&) = delete;
  /** @brief Store handles cannot be move assigned. */
  FileCheckpointStore& operator=(FileCheckpointStore&&) = delete;

  /** @brief Returns the final artifact path supplied at construction.
   * @return Borrowed path valid for this object's lifetime.
   */
  const std::filesystem::path& path() const noexcept;

  /** @brief Returns the immutable stored descriptor after a successful commit.
   * @return Snapshot descriptor valid until the next successful commit.
   */
  const CheckpointDescriptor& descriptor() const noexcept override;

  /** @brief Begins a replacement transaction for one complete descriptor.
   * @param descriptor Validated complete checkpoint descriptor copied by the
   * store.
   */
  void begin(const CheckpointDescriptor& descriptor) override;

  /** @brief Returns the writer for one declared participant field set.
   * @param participant_id Stable identity present in the active transaction.
   * @return Writer valid until commit or rollback.
   */
  StateWriter& writer_for(std::string_view participant_id) override;

  /** @brief Publishes the staged checkpoint with atomic replacement. */
  void commit() override;

  /** @brief Discards staged state without throwing. */
  void rollback() noexcept override;

  /** @brief Returns one validated immutable reader from the published file.
   * @param participant_id Stable identity present in the stored descriptor.
   * @return Reader valid until the next successful commit or destruction.
   */
  const StateReader& reader_for(std::string_view participant_id) const override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;  ///< Private implementation and staged storage.
};

}  // namespace jyppx::training

#endif  // JYPPX_COMPONENT_INTEROP_ENABLED

#endif  // JYPPX_TRAINING_FILE_CHECKPOINT_STORE_HPP
