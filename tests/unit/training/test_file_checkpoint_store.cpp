// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/training/checkpoint_coordinator.hpp>
#include <jyppx/liuding/training/file_checkpoint_store.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

class CounterParticipant final : public jyppx::liuding::CheckpointParticipant {
 public:
  explicit CounterParticipant(std::int32_t value) : value_(value) {}

  std::string participant_id() const override { return "counter"; }

  jyppx::liuding::StateManifest describe_state() const override {
    return jyppx::liuding::StateManifest{
        .participant_id = "counter",
        .schema_version = 1,
        .restore_level = jyppx::liuding::RestoreLevel::exact,
        .fields = {{.name = "value",
                    .media_type = "application/vnd.jyppx.counter",
                    .max_bytes = sizeof(value_),
                    .required = true}},
    };
  }

  void save_state(jyppx::liuding::StateWriter& writer) const override {
    std::array<std::byte, sizeof(value_)> bytes{};
    std::memcpy(bytes.data(), &value_, sizeof(value_));
    writer.write_blob("value", bytes);
  }

  void load_state(const jyppx::liuding::StateReader& reader,
                  jyppx::liuding::RestorePolicy policy) override {
    if (!jyppx::liuding::restore_policy_accepts(
            policy, reader.manifest().restore_level)) {
      throw jyppx::liuding::CompatibilityError(
          "counter restore policy rejected");
    }
    const auto bytes = reader.read_blob("value", sizeof(value_));
    if (bytes.size() != sizeof(value_)) {
      throw jyppx::liuding::SerializationError("counter byte count mismatch");
    }
    std::int32_t candidate = 0;
    std::memcpy(&candidate, bytes.data(), sizeof(candidate));
    value_ = candidate;
  }

  std::int32_t value() const noexcept { return value_; }

  void set_value(std::int32_t value) noexcept { value_ = value; }

 private:
  std::int32_t value_;
};

class FileCheckpointStoreTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    path_ = std::filesystem::temp_directory_path() /
            (std::string("jyppx-file-checkpoint-") + info->name() + ".bin");
    std::error_code error;
    std::filesystem::remove(path_, error);
    std::filesystem::remove(path_.string() + ".tmp", error);
  }

  void TearDown() override {
    std::error_code error;
    std::filesystem::remove(path_, error);
    std::filesystem::remove(path_.string() + ".tmp", error);
  }

  std::filesystem::path path_;
};

std::shared_ptr<CounterParticipant> register_counter(
    jyppx::liuding::training::CheckpointParticipantRegistry& registry,
    std::int32_t value) {
  auto participant = std::make_shared<CounterParticipant>(value);
  registry.register_participant(participant);
  return participant;
}

jyppx::liuding::training::CheckpointDescriptor single_blob_descriptor(
    std::uint32_t schema_version = 1, std::uint64_t max_bytes = 128) {
  return jyppx::liuding::training::CheckpointDescriptor{
      .restore_level = jyppx::liuding::RestoreLevel::exact,
      .participants = {jyppx::liuding::StateManifest{
          .participant_id = "blob",
          .schema_version = schema_version,
          .restore_level = jyppx::liuding::RestoreLevel::exact,
          .fields = {{.name = "payload",
                      .media_type = "application/octet-stream",
                      .max_bytes = max_bytes,
                      .required = true}}}}};
}

TEST_F(FileCheckpointStoreTest, SavesReopensAndRestoresExactState) {
  jyppx::liuding::training::CheckpointParticipantRegistry registry;
  const auto participant = register_counter(registry, 42);
  jyppx::liuding::training::CheckpointCoordinator coordinator(registry);
  jyppx::liuding::training::FileCheckpointStore store(path_);

  const auto descriptor = coordinator.save(store);
  EXPECT_EQ(descriptor.restore_level, jyppx::liuding::RestoreLevel::exact);
  EXPECT_TRUE(std::filesystem::exists(path_));

  jyppx::liuding::training::FileCheckpointStore reopened(path_);
  jyppx::liuding::training::CheckpointParticipantRegistry restored_registry;
  const auto restored = register_counter(restored_registry, 0);
  jyppx::liuding::training::CheckpointCoordinator restored_coordinator(
      restored_registry);
  const auto restored_descriptor = restored_coordinator.restore(reopened);

  EXPECT_EQ(restored_descriptor.participants.size(), 1U);
  EXPECT_EQ(restored->value(), 42);
  EXPECT_EQ(participant->value(), 42);
}

TEST_F(FileCheckpointStoreTest, FailedCommitPreservesPreviousPublishedState) {
  jyppx::liuding::training::CheckpointParticipantRegistry registry;
  const auto participant = register_counter(registry, 7);
  jyppx::liuding::training::CheckpointCoordinator coordinator(registry);
  jyppx::liuding::training::FileCheckpointStore store(path_);
  coordinator.save(store);

  participant->set_value(99);
  store.begin(coordinator.describe());
  EXPECT_THROW(store.commit(), jyppx::liuding::SerializationError);
  EXPECT_FALSE(std::filesystem::exists(path_.string() + ".tmp"));

  jyppx::liuding::training::FileCheckpointStore reopened(path_);
  jyppx::liuding::training::CheckpointParticipantRegistry restored_registry;
  const auto restored = register_counter(restored_registry, 0);
  jyppx::liuding::training::CheckpointCoordinator restored_coordinator(
      restored_registry);
  restored_coordinator.restore(reopened);
  EXPECT_EQ(restored->value(), 7);
}

TEST_F(FileCheckpointStoreTest, RejectsCorruptionAndCleansStaleTemporaryFile) {
  jyppx::liuding::training::CheckpointParticipantRegistry registry;
  register_counter(registry, 13);
  jyppx::liuding::training::CheckpointCoordinator coordinator(registry);
  jyppx::liuding::training::FileCheckpointStore store(path_);
  coordinator.save(store);

  std::ifstream input(path_, std::ios::binary);
  const std::vector<char> raw((std::istreambuf_iterator<char>(input)), {});
  std::vector<std::byte> bytes(raw.size());
  std::memcpy(bytes.data(), raw.data(), raw.size());
  ASSERT_GT(bytes.size(), 1U);
  bytes.back() ^= std::byte{0x01};
  input.close();
  std::ofstream output(path_, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  output.close();
  EXPECT_THROW(
      { jyppx::liuding::training::FileCheckpointStore invalid_store(path_); },
      jyppx::liuding::SerializationError);

  std::ofstream stale(path_.string() + ".tmp", std::ios::binary);
  stale << "interrupted";
  stale.close();
  std::error_code error;
  std::filesystem::remove(path_, error);
  jyppx::liuding::training::FileCheckpointStore fresh(path_);
  EXPECT_FALSE(std::filesystem::exists(path_.string() + ".tmp"));
  EXPECT_TRUE(fresh.descriptor().participants.empty());
}

TEST_F(FileCheckpointStoreTest, RejectsArtifactExceedingConfiguredLimit) {
  jyppx::liuding::training::CheckpointParticipantRegistry registry;
  register_counter(registry, 21);
  jyppx::liuding::training::CheckpointCoordinator coordinator(registry);
  jyppx::liuding::training::FileCheckpointStoreOptions options;
  options.max_artifact_bytes = 64;
  options.max_field_bytes = sizeof(std::int32_t);
  jyppx::liuding::training::FileCheckpointStore store(path_, options);

  EXPECT_THROW(coordinator.save(store), jyppx::liuding::ResourceExhausted);
  EXPECT_FALSE(std::filesystem::exists(path_));
  EXPECT_FALSE(std::filesystem::exists(path_.string() + ".tmp"));
}

TEST_F(FileCheckpointStoreTest,
       RejectsDeclaredFieldAboveStoreLimitBeforeWriting) {
  jyppx::liuding::training::FileCheckpointStoreOptions options;
  options.max_artifact_bytes = 4096;
  options.max_field_bytes = 4;
  jyppx::liuding::training::FileCheckpointStore store(path_, options);

  EXPECT_THROW(store.begin(single_blob_descriptor(1, 8)),
               jyppx::liuding::ResourceExhausted);
  EXPECT_FALSE(std::filesystem::exists(path_));
  EXPECT_FALSE(std::filesystem::exists(path_.string() + ".tmp"));
}

TEST_F(FileCheckpointStoreTest,
       FailedReplacementLeavesPreviouslyPublishedBytesUsable) {
  jyppx::liuding::training::FileCheckpointStoreOptions options;
  options.max_artifact_bytes = 4096;
  options.max_field_bytes = 4096;
  jyppx::liuding::training::FileCheckpointStore store(path_, options);
  const auto descriptor = single_blob_descriptor(1, 4096);

  store.begin(descriptor);
  const std::array<std::byte, 1> first{std::byte{0x2a}};
  store.writer_for("blob").write_blob("payload", first);
  store.commit();
  const auto published_size = std::filesystem::file_size(path_);

  store.begin(descriptor);
  const std::vector<std::byte> oversized(4000, std::byte{0x5a});
  store.writer_for("blob").write_blob("payload", oversized);
  EXPECT_THROW(store.commit(), jyppx::liuding::ResourceExhausted);

  EXPECT_EQ(std::filesystem::file_size(path_), published_size);
  jyppx::liuding::training::FileCheckpointStore reopened(path_, options);
  const auto payload = reopened.reader_for("blob").read_blob("payload", 1);
  ASSERT_EQ(payload.size(), 1U);
  EXPECT_EQ(payload.front(), std::byte{0x2a});
  EXPECT_FALSE(std::filesystem::exists(path_.string() + ".tmp"));
}

}  // namespace
