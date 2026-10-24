// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/training/checkpoint_coordinator.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

enum class SnapshotViolation {
  none,
  undeclared_field,
  duplicate_field,
  oversized_field,
  missing_required_field,
};

class MemoryStateStore final : public jyppx::liuding::StateWriter,
                               public jyppx::liuding::StateReader {
 public:
  explicit MemoryStateStore(jyppx::liuding::StateManifest manifest)
      : manifest_(std::move(manifest)) {
    jyppx::liuding::validate_state_manifest(manifest_);
  }

  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    const auto field = std::find_if(
        manifest_.fields.begin(), manifest_.fields.end(),
        [name](const jyppx::liuding::StateFieldDescriptor& candidate) {
          return candidate.name == name;
        });
    if (field == manifest_.fields.end()) {
      throw jyppx::liuding::SerializationError("unknown test field");
    }
    if (data.size() > field->max_bytes) {
      throw jyppx::liuding::ResourceExhausted("oversized test field");
    }
    if (!blobs_
             .emplace(std::string(name),
                      std::vector<std::byte>(data.begin(), data.end()))
             .second) {
      throw jyppx::liuding::SerializationError("duplicate test field");
    }
  }

  const jyppx::liuding::StateManifest& manifest() const noexcept override {
    return manifest_;
  }

  bool contains(std::string_view name) const override {
    return blobs_.contains(std::string(name));
  }

  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max_bytes) const override {
    const auto blob = blobs_.find(std::string(name));
    if (blob == blobs_.end()) {
      throw jyppx::liuding::SerializationError("missing test field");
    }
    if (blob->second.size() > max_bytes) {
      throw jyppx::liuding::ResourceExhausted("test read limit exceeded");
    }
    return blob->second;
  }

  void finalize() const {
    for (const auto& field : manifest_.fields) {
      if (field.required && !blobs_.contains(field.name)) {
        throw jyppx::liuding::SerializationError("required test field missing");
      }
    }
  }

 private:
  jyppx::liuding::StateManifest manifest_;
  std::map<std::string, std::vector<std::byte>> blobs_;
};

class MemoryCheckpointStore final
    : public jyppx::liuding::training::CheckpointStateSink,
      public jyppx::liuding::training::CheckpointStateSource {
 public:
  void begin(const jyppx::liuding::training::CheckpointDescriptor& descriptor)
      override {
    ++begin_count_;
    if (fail_begin_) {
      throw jyppx::liuding::SerializationError("injected begin failure");
    }
    jyppx::liuding::training::validate_checkpoint_descriptor(descriptor);
    staging_descriptor_ = descriptor;
    staging_.clear();
    for (const auto& manifest : descriptor.participants) {
      staging_.emplace(manifest.participant_id,
                       std::make_shared<MemoryStateStore>(manifest));
    }
  }

  jyppx::liuding::StateWriter& writer_for(
      std::string_view participant_id) override {
    write_order_.emplace_back(participant_id);
    const auto writer = staging_.find(std::string(participant_id));
    if (writer == staging_.end()) {
      throw jyppx::liuding::SerializationError("unknown staged participant");
    }
    return *writer->second;
  }

  void commit() override {
    if (fail_commit_) {
      throw jyppx::liuding::SerializationError("injected commit failure");
    }
    for (const auto& [identifier, store] : staging_) {
      (void)identifier;
      store->finalize();
    }
    committed_descriptor_ = staging_descriptor_;
    committed_ = std::move(staging_);
    ++commit_count_;
  }

  void rollback() noexcept override {
    staging_.clear();
    ++rollback_count_;
  }

  const jyppx::liuding::training::CheckpointDescriptor& descriptor()
      const noexcept override {
    return committed_descriptor_;
  }

  const jyppx::liuding::StateReader& reader_for(
      std::string_view participant_id) const override {
    const auto reader = committed_.find(std::string(participant_id));
    if (reader == committed_.end()) {
      throw jyppx::liuding::SerializationError("missing committed participant");
    }
    return *reader->second;
  }

  jyppx::liuding::training::CheckpointDescriptor& mutable_descriptor() {
    return committed_descriptor_;
  }

  void set_fail_commit(bool fail_commit) { fail_commit_ = fail_commit; }
  void set_fail_begin(bool fail_begin) { fail_begin_ = fail_begin; }

  const std::vector<std::string>& write_order() const { return write_order_; }
  int begin_count() const { return begin_count_; }
  int commit_count() const { return commit_count_; }
  int rollback_count() const { return rollback_count_; }

 private:
  jyppx::liuding::training::CheckpointDescriptor staging_descriptor_;
  jyppx::liuding::training::CheckpointDescriptor committed_descriptor_;
  std::map<std::string, std::shared_ptr<MemoryStateStore>> staging_;
  std::map<std::string, std::shared_ptr<MemoryStateStore>> committed_;
  std::vector<std::string> write_order_;
  bool fail_commit_ = false;
  bool fail_begin_ = false;
  int begin_count_ = 0;
  int commit_count_ = 0;
  int rollback_count_ = 0;
};

class CounterParticipant final : public jyppx::liuding::CheckpointParticipant {
 public:
  CounterParticipant(std::string identifier, std::int32_t value,
                     jyppx::liuding::RestoreLevel restore_level =
                         jyppx::liuding::RestoreLevel::exact)
      : identifier_(std::move(identifier)),
        value_(value),
        restore_level_(restore_level) {}

  std::string participant_id() const override { return identifier_; }

  jyppx::liuding::StateManifest describe_state() const override {
    auto manifest = jyppx::liuding::StateManifest{
        .participant_id = mismatch_manifest_id_.value_or(identifier_),
        .schema_version = 1,
        .restore_level = restore_level_,
        .fields = {{.name = "value",
                    .media_type = "application/vnd.jyppx.counter",
                    .max_bytes = sizeof(value_)}},
    };
    if (restore_level_ != jyppx::liuding::RestoreLevel::exact) {
      manifest.degradation_reasons = {"test participant is not exact"};
    }
    return manifest;
  }

  void save_state(jyppx::liuding::StateWriter& writer) const override {
    if (events_ != nullptr) {
      events_->push_back("save:" + identifier_);
    }
    if (fail_save_) {
      throw jyppx::liuding::SerializationError(
          "injected participant save failure");
    }
    std::array<std::byte, sizeof(value_)> bytes{};
    std::memcpy(bytes.data(), &value_, sizeof(value_));
    if (snapshot_violation_ == SnapshotViolation::undeclared_field) {
      writer.write_blob("undeclared", bytes);
      return;
    }
    if (snapshot_violation_ == SnapshotViolation::oversized_field) {
      std::array<std::byte, sizeof(value_) + 1> oversized{};
      writer.write_blob("value", oversized);
      return;
    }
    if (snapshot_violation_ == SnapshotViolation::missing_required_field) {
      return;
    }
    writer.write_blob("value", bytes);
    if (snapshot_violation_ == SnapshotViolation::duplicate_field) {
      writer.write_blob("value", bytes);
    }
  }

  void load_state(const jyppx::liuding::StateReader& reader,
                  jyppx::liuding::RestorePolicy policy) override {
    const auto& stored_manifest = reader.manifest();
    jyppx::liuding::validate_state_manifest(stored_manifest);
    if (stored_manifest.participant_id != identifier_ ||
        stored_manifest.schema_version != 1) {
      throw jyppx::liuding::CompatibilityError("incompatible counter state");
    }
    if (!jyppx::liuding::restore_policy_accepts(
            policy, stored_manifest.restore_level)) {
      throw jyppx::liuding::CompatibilityError(
          "counter restore policy rejected");
    }
    const auto bytes = reader.read_blob("value", sizeof(value_));
    if (bytes.size() != sizeof(value_)) {
      throw jyppx::liuding::SerializationError("invalid counter byte count");
    }
    std::int32_t candidate = 0;
    std::memcpy(&candidate, bytes.data(), sizeof(candidate));
    if (std::find(rejected_values_.begin(), rejected_values_.end(),
                  candidate) != rejected_values_.end()) {
      throw jyppx::liuding::SerializationError(
          "injected participant load failure");
    }
    if (events_ != nullptr) {
      events_->push_back("load:" + identifier_ + ":" +
                         std::to_string(candidate));
    }
    value_ = candidate;
  }

  std::int32_t value() const { return value_; }
  void set_value(std::int32_t value) { value_ = value; }
  void set_fail_save(bool fail_save) { fail_save_ = fail_save; }
  void set_snapshot_violation(SnapshotViolation violation) {
    snapshot_violation_ = violation;
  }
  void reject_value(std::optional<std::int32_t> value) {
    rejected_values_.clear();
    if (value.has_value()) {
      rejected_values_.push_back(*value);
    }
  }
  void reject_values(std::vector<std::int32_t> values) {
    rejected_values_ = std::move(values);
  }
  void set_manifest_id(std::optional<std::string> identifier) {
    mismatch_manifest_id_ = std::move(identifier);
  }
  void record_events(std::vector<std::string>* events) { events_ = events; }

 private:
  std::string identifier_;
  std::int32_t value_;
  jyppx::liuding::RestoreLevel restore_level_;
  bool fail_save_ = false;
  SnapshotViolation snapshot_violation_ = SnapshotViolation::none;
  std::vector<std::int32_t> rejected_values_;
  std::optional<std::string> mismatch_manifest_id_;
  std::vector<std::string>* events_ = nullptr;
};

TEST(CheckpointCoordinatorTest, ValidatesCanonicalAggregateDescriptor) {
  const auto exact = CounterParticipant("model", 1).describe_state();
  const auto bounded =
      CounterParticipant("optimizer", 2, jyppx::liuding::RestoreLevel::bounded)
          .describe_state();
  const jyppx::liuding::training::CheckpointDescriptor descriptor{
      .restore_level = jyppx::liuding::RestoreLevel::bounded,
      .participants = {exact, bounded},
  };
  EXPECT_NO_THROW(
      jyppx::liuding::training::validate_checkpoint_descriptor(descriptor));

  auto invalid = descriptor;
  invalid.restore_level = jyppx::liuding::RestoreLevel::exact;
  EXPECT_THROW(
      jyppx::liuding::training::validate_checkpoint_descriptor(invalid),
      jyppx::liuding::InvalidArgument);
  invalid = descriptor;
  std::swap(invalid.participants[0], invalid.participants[1]);
  EXPECT_THROW(
      jyppx::liuding::training::validate_checkpoint_descriptor(invalid),
      jyppx::liuding::InvalidArgument);
}

TEST(CheckpointCoordinatorTest,
     RegistryRejectsNullDuplicateAndMismatchedParticipants) {
  jyppx::liuding::training::CheckpointParticipantRegistry registry;
  EXPECT_THROW(registry.register_participant(nullptr),
               jyppx::liuding::InvalidArgument);

  registry.register_participant(
      std::make_shared<CounterParticipant>("model", 1));
  EXPECT_TRUE(registry.contains("model"));
  EXPECT_EQ(registry.size(), 1U);
  EXPECT_THROW(registry.register_participant(
                   std::make_shared<CounterParticipant>("model", 2)),
               jyppx::liuding::InvalidArgument);

  auto mismatched = std::make_shared<CounterParticipant>("optimizer", 3);
  mismatched->set_manifest_id("different");
  EXPECT_THROW(registry.register_participant(mismatched),
               jyppx::liuding::InvalidArgument);
}

TEST(CheckpointCoordinatorTest, GroupRegistrationHasStrongGuarantee) {
  jyppx::liuding::training::CheckpointParticipantRegistry registry;
  registry.register_participant(
      std::make_shared<CounterParticipant>("existing", 1));
  std::vector<std::shared_ptr<jyppx::liuding::CheckpointParticipant>> group;
  group.push_back(std::make_shared<CounterParticipant>("new.first", 2));
  group.push_back(std::make_shared<CounterParticipant>("existing", 3));

  EXPECT_THROW(registry.register_participants(std::move(group)),
               jyppx::liuding::InvalidArgument);
  EXPECT_EQ(registry.participant_ids(), std::vector<std::string>{"existing"});
  EXPECT_THROW(registry.register_participants({}),
               jyppx::liuding::InvalidArgument);
}

TEST(CheckpointCoordinatorTest, SavesInCanonicalOrderAndCommitsOnce) {
  std::vector<std::string> events;
  auto later = std::make_shared<CounterParticipant>("z.model", 7);
  auto earlier = std::make_shared<CounterParticipant>("a.rng", 3);
  later->record_events(&events);
  earlier->record_events(&events);

  jyppx::liuding::training::CheckpointParticipantRegistry registry;
  registry.register_participant(later);
  registry.register_participant(earlier);
  jyppx::liuding::training::CheckpointCoordinator coordinator(registry);
  MemoryCheckpointStore store;

  const auto descriptor = coordinator.save(store);
  EXPECT_EQ(registry.participant_ids(),
            std::vector<std::string>({"a.rng", "z.model"}));
  EXPECT_EQ(store.write_order(),
            std::vector<std::string>({"a.rng", "z.model"}));
  EXPECT_EQ(events, std::vector<std::string>({"save:a.rng", "save:z.model"}));
  EXPECT_EQ(descriptor.participants.front().participant_id, "a.rng");
  EXPECT_EQ(store.begin_count(), 1);
  EXPECT_EQ(store.commit_count(), 1);
  EXPECT_EQ(store.rollback_count(), 0);
}

TEST(CheckpointCoordinatorTest, SaveFailureRollsBackSinkTransaction) {
  auto first = std::make_shared<CounterParticipant>("a.first", 1);
  auto second = std::make_shared<CounterParticipant>("b.second", 2);
  second->set_fail_save(true);
  jyppx::liuding::training::CheckpointParticipantRegistry registry;
  registry.register_participant(first);
  registry.register_participant(second);
  jyppx::liuding::training::CheckpointCoordinator coordinator(registry);
  MemoryCheckpointStore store;

  EXPECT_THROW(coordinator.save(store), jyppx::liuding::SerializationError);
  EXPECT_EQ(store.begin_count(), 1);
  EXPECT_EQ(store.commit_count(), 0);
  EXPECT_EQ(store.rollback_count(), 1);
}

TEST(CheckpointCoordinatorTest, BeginFailureRollsBackSinkTransaction) {
  jyppx::liuding::training::CheckpointParticipantRegistry registry;
  registry.register_participant(
      std::make_shared<CounterParticipant>("model", 1));
  jyppx::liuding::training::CheckpointCoordinator coordinator(registry);
  MemoryCheckpointStore store;
  store.set_fail_begin(true);

  EXPECT_THROW(coordinator.save(store), jyppx::liuding::SerializationError);
  EXPECT_EQ(store.begin_count(), 1);
  EXPECT_EQ(store.commit_count(), 0);
  EXPECT_EQ(store.rollback_count(), 1);
}

TEST(CheckpointCoordinatorTest, CommitFailureRollsBackSinkTransaction) {
  jyppx::liuding::training::CheckpointParticipantRegistry registry;
  registry.register_participant(
      std::make_shared<CounterParticipant>("model", 1));
  jyppx::liuding::training::CheckpointCoordinator coordinator(registry);
  MemoryCheckpointStore store;
  store.set_fail_commit(true);

  EXPECT_THROW(coordinator.save(store), jyppx::liuding::SerializationError);
  EXPECT_EQ(store.commit_count(), 0);
  EXPECT_EQ(store.rollback_count(), 1);
}

TEST(CheckpointCoordinatorTest, RestoresCompleteParticipantSet) {
  auto model = std::make_shared<CounterParticipant>("model", 11);
  auto optimizer = std::make_shared<CounterParticipant>("optimizer", 22);
  jyppx::liuding::training::CheckpointParticipantRegistry registry;
  registry.register_participant(model);
  registry.register_participant(optimizer);
  jyppx::liuding::training::CheckpointCoordinator coordinator(registry);
  MemoryCheckpointStore store;
  coordinator.save(store);

  model->set_value(101);
  optimizer->set_value(202);
  const auto restored = coordinator.restore(store);

  EXPECT_EQ(model->value(), 11);
  EXPECT_EQ(optimizer->value(), 22);
  EXPECT_EQ(restored.restore_level, jyppx::liuding::RestoreLevel::exact);
}

TEST(CheckpointCoordinatorTest, RejectsPolicyBeforeAnyParticipantMutation) {
  auto participant = std::make_shared<CounterParticipant>(
      "sampler", 9, jyppx::liuding::RestoreLevel::bounded);
  jyppx::liuding::training::CheckpointParticipantRegistry registry;
  registry.register_participant(participant);
  jyppx::liuding::training::CheckpointCoordinator coordinator(registry);
  MemoryCheckpointStore store;
  coordinator.save(store);
  participant->set_value(44);

  EXPECT_THROW(
      coordinator.restore(store, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::CompatibilityError);
  EXPECT_EQ(participant->value(), 44);
}

TEST(CheckpointCoordinatorTest,
     RejectsTopologyAndReaderMismatchBeforeMutation) {
  auto first = std::make_shared<CounterParticipant>("a.first", 1);
  auto second = std::make_shared<CounterParticipant>("b.second", 2);
  jyppx::liuding::training::CheckpointParticipantRegistry registry;
  registry.register_participant(first);
  registry.register_participant(second);
  jyppx::liuding::training::CheckpointCoordinator coordinator(registry);
  MemoryCheckpointStore store;
  coordinator.save(store);
  first->set_value(10);
  second->set_value(20);

  store.mutable_descriptor().participants.pop_back();
  EXPECT_THROW(coordinator.restore(store), jyppx::liuding::CompatibilityError);
  EXPECT_EQ(first->value(), 10);
  EXPECT_EQ(second->value(), 20);

  coordinator.save(store);
  store.mutable_descriptor().participants.front().schema_version = 2;
  EXPECT_THROW(coordinator.restore(store), jyppx::liuding::CompatibilityError);
  EXPECT_EQ(first->value(), 10);
  EXPECT_EQ(second->value(), 20);
}

TEST(CheckpointCoordinatorTest, LaterLoadFailureRollsBackEarlierParticipant) {
  std::vector<std::string> events;
  auto first = std::make_shared<CounterParticipant>("a.first", 1);
  auto second = std::make_shared<CounterParticipant>("b.second", 2);
  first->record_events(&events);
  second->record_events(&events);
  jyppx::liuding::training::CheckpointParticipantRegistry registry;
  registry.register_participant(first);
  registry.register_participant(second);
  jyppx::liuding::training::CheckpointCoordinator coordinator(registry);
  MemoryCheckpointStore store;
  coordinator.save(store);

  events.clear();
  first->set_value(10);
  second->set_value(20);
  second->reject_value(2);
  EXPECT_THROW(coordinator.restore(store), jyppx::liuding::SerializationError);

  EXPECT_EQ(first->value(), 10);
  EXPECT_EQ(second->value(), 20);
  EXPECT_EQ(events, std::vector<std::string>(
                        {"save:a.first", "save:b.second", "load:a.first:1",
                         "load:b.second:20", "load:a.first:10"}));
}

TEST(CheckpointCoordinatorTest,
     RollbackFailureReportsInternalErrorWithOriginalCause) {
  auto first = std::make_shared<CounterParticipant>("a.first", 1);
  auto second = std::make_shared<CounterParticipant>("b.second", 2);
  jyppx::liuding::training::CheckpointParticipantRegistry registry;
  registry.register_participant(first);
  registry.register_participant(second);
  jyppx::liuding::training::CheckpointCoordinator coordinator(registry);
  MemoryCheckpointStore store;
  coordinator.save(store);

  first->set_value(10);
  second->set_value(20);
  second->reject_values({2, 20});
  try {
    (void)coordinator.restore(store);
    FAIL() << "restore should report incomplete rollback";
  } catch (const jyppx::liuding::InternalError& error) {
    EXPECT_EQ(error.details().actual, "b.second");
    try {
      std::rethrow_if_nested(error);
      FAIL() << "restore failure should remain nested";
    } catch (const jyppx::liuding::SerializationError&) {
    }
  }
  EXPECT_EQ(first->value(), 10);
  EXPECT_EQ(second->value(), 20);
}

TEST(CheckpointCoordinatorTest, EnforcesRollbackBudgetBeforeMutation) {
  auto first = std::make_shared<CounterParticipant>("a.first", 1);
  auto second = std::make_shared<CounterParticipant>("b.second", 2);
  jyppx::liuding::training::CheckpointParticipantRegistry registry;
  registry.register_participant(first);
  registry.register_participant(second);
  jyppx::liuding::training::CheckpointCoordinator coordinator(
      registry,
      {.max_participants = 2, .max_rollback_bytes = sizeof(std::int32_t)});
  MemoryCheckpointStore store;
  coordinator.save(store);
  first->set_value(10);
  second->set_value(20);

  EXPECT_THROW(coordinator.restore(store), jyppx::liuding::ResourceExhausted);
  EXPECT_EQ(first->value(), 10);
  EXPECT_EQ(second->value(), 20);
}

TEST(CheckpointCoordinatorTest,
     RejectsMalformedRollbackSnapshotsBeforeParticipantMutation) {
  const auto expect_rejected = [](SnapshotViolation violation) {
    auto participant = std::make_shared<CounterParticipant>("model", 1);
    jyppx::liuding::training::CheckpointParticipantRegistry registry;
    registry.register_participant(participant);
    jyppx::liuding::training::CheckpointCoordinator coordinator(registry);
    MemoryCheckpointStore store;
    coordinator.save(store);

    participant->set_value(10);
    participant->set_snapshot_violation(violation);
    EXPECT_THROW(coordinator.restore(store), jyppx::liuding::Error);
    EXPECT_EQ(participant->value(), 10);
  };

  expect_rejected(SnapshotViolation::undeclared_field);
  expect_rejected(SnapshotViolation::duplicate_field);
  expect_rejected(SnapshotViolation::oversized_field);
  expect_rejected(SnapshotViolation::missing_required_field);
}

TEST(CheckpointCoordinatorTest, RejectsEmptyRegistryAndZeroLimits) {
  jyppx::liuding::training::CheckpointParticipantRegistry registry;
  jyppx::liuding::training::CheckpointCoordinator coordinator(registry);
  EXPECT_THROW(coordinator.describe(), jyppx::liuding::InvalidArgument);
  EXPECT_THROW(jyppx::liuding::training::CheckpointCoordinator(
                   registry, {.max_participants = 0, .max_rollback_bytes = 1}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(jyppx::liuding::training::CheckpointCoordinator(
                   registry, {.max_participants = 1, .max_rollback_bytes = 0}),
               jyppx::liuding::InvalidArgument);
}

}  // namespace
