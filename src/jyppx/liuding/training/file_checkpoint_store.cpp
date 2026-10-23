// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/interop/file_state_archive.hpp>
#include <jyppx/liuding/training/file_checkpoint_store.hpp>

#include <utility>

namespace jyppx::liuding::training {

class FileCheckpointStore::Impl {
 public:
  Impl(std::filesystem::path path, FileCheckpointStoreOptions options)
      : archive_(std::move(path),
                 {.max_artifact_bytes = options.max_artifact_bytes,
                  .max_field_bytes = options.max_field_bytes,
                  .min_free_storage_bytes = options.min_free_storage_bytes,
                  .synchronize = options.synchronize,
                  .manifest_limits = {}}) {
    for (const auto& entry : archive_.manifest().participants) {
      descriptor_.participants.push_back(entry.state);
      if (static_cast<std::uint8_t>(entry.state.restore_level) >
          static_cast<std::uint8_t>(descriptor_.restore_level)) {
        descriptor_.restore_level = entry.state.restore_level;
      }
    }
    if (!descriptor_.participants.empty())
      validate_checkpoint_descriptor(descriptor_);
  }

  interop::FileStateArchive archive_;
  CheckpointDescriptor descriptor_;
  CheckpointDescriptor staged_;
};

FileCheckpointStore::FileCheckpointStore(std::filesystem::path path,
                                         FileCheckpointStoreOptions options)
    : impl_(std::make_unique<Impl>(std::move(path), options)) {}
FileCheckpointStore::~FileCheckpointStore() = default;

const std::filesystem::path& FileCheckpointStore::path() const noexcept {
  return impl_->archive_.path();
}
const CheckpointDescriptor& FileCheckpointStore::descriptor() const noexcept {
  return impl_->descriptor_;
}
void FileCheckpointStore::begin(const CheckpointDescriptor& descriptor) {
  validate_checkpoint_descriptor(descriptor);
  auto staged = descriptor;
  impl_->archive_.begin(descriptor.participants);
  impl_->staged_ = std::move(staged);
}
StateWriter& FileCheckpointStore::writer_for(std::string_view participant_id) {
  return impl_->archive_.writer_for(participant_id);
}
void FileCheckpointStore::commit() {
  impl_->archive_.commit();
  impl_->descriptor_ = std::move(impl_->staged_);
}
void FileCheckpointStore::rollback() noexcept {
  impl_->archive_.rollback();
  impl_->staged_ = {};
}
const StateReader& FileCheckpointStore::reader_for(
    std::string_view participant_id) const {
  return impl_->archive_.reader_for(participant_id);
}
}  // namespace jyppx::liuding::training
