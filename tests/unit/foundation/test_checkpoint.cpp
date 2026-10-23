// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/checkpoint.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

class MemoryStateStore final : public jyppx::liuding::StateWriter,
                               public jyppx::liuding::StateReader {
 public:
  explicit MemoryStateStore(jyppx::liuding::StateManifest manifest)
      : manifest_(std::move(manifest)) {
    jyppx::liuding::validate_state_manifest(manifest_);
  }

  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    const auto* descriptor = find_descriptor(name);
    if (descriptor == nullptr) {
      throw jyppx::liuding::SerializationError("Undeclared state field");
    }
    if (data.size() > descriptor->max_bytes) {
      throw jyppx::liuding::ResourceExhausted("State field exceeds its size limit");
    }
    const std::string owned_name(name);
    if (blobs_.contains(owned_name)) {
      throw jyppx::liuding::SerializationError("Duplicate state field");
    }
    blobs_.emplace(owned_name,
                   std::vector<std::byte>(data.begin(), data.end()));
  }

  bool contains(std::string_view name) const override {
    return blobs_.contains(std::string(name));
  }

  const jyppx::liuding::StateManifest& manifest() const noexcept override {
    return manifest_;
  }

  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max_bytes) const override {
    const auto found = blobs_.find(std::string(name));
    if (found == blobs_.end()) {
      throw jyppx::liuding::SerializationError("Missing state field");
    }
    if (found->second.size() > max_bytes) {
      throw jyppx::liuding::ResourceExhausted("State field exceeds the read limit");
    }
    return found->second;
  }

 private:
  const jyppx::liuding::StateFieldDescriptor* find_descriptor(
      std::string_view name) const {
    for (const auto& descriptor : manifest_.fields) {
      if (descriptor.name == name) {
        return &descriptor;
      }
    }
    return nullptr;
  }

  jyppx::liuding::StateManifest manifest_;
  std::unordered_map<std::string, std::vector<std::byte>> blobs_;
};

class CounterParticipant final : public jyppx::liuding::CheckpointParticipant {
 public:
  explicit CounterParticipant(
      std::int64_t value,
      jyppx::liuding::RestoreLevel restore_level = jyppx::liuding::RestoreLevel::exact)
      : value_(value), restore_level_(restore_level) {}

  std::string participant_id() const override { return "test.counter"; }

  jyppx::liuding::StateManifest describe_state() const override {
    jyppx::liuding::StateManifest manifest{
        .participant_id = participant_id(),
        .schema_version = 1,
        .restore_level = restore_level_,
        .fields = {{.name = "value",
                    .media_type = "application/vnd.jyppx.test-counter",
                    .max_bytes = sizeof(value_)}},
    };
    if (restore_level_ != jyppx::liuding::RestoreLevel::exact) {
      manifest.degradation_reasons.push_back(
          "Synthetic test participant does not promise exact continuation.");
    }
    return manifest;
  }

  void save_state(jyppx::liuding::StateWriter& writer) const override {
    const auto bytes = std::as_bytes(std::span(&value_, 1));
    writer.write_blob("value", bytes);
  }

  void load_state(const jyppx::liuding::StateReader& reader,
                  jyppx::liuding::RestorePolicy policy) override {
    const auto& stored_manifest = reader.manifest();
    jyppx::liuding::validate_state_manifest(stored_manifest);
    if (stored_manifest.participant_id != participant_id() ||
        stored_manifest.schema_version != 1) {
      throw jyppx::liuding::CompatibilityError("Unsupported counter state schema");
    }
    if (!jyppx::liuding::restore_policy_accepts(policy, stored_manifest.restore_level)) {
      throw jyppx::liuding::CompatibilityError("Restore policy rejects participant");
    }
    const auto bytes = reader.read_blob("value", sizeof(value_));
    if (bytes.size() != sizeof(value_)) {
      throw jyppx::liuding::SerializationError("Invalid counter state size");
    }
    std::int64_t candidate = 0;
    std::memcpy(&candidate, bytes.data(), sizeof(candidate));
    if (candidate < 0) {
      throw jyppx::liuding::SerializationError("Counter state must be non-negative");
    }
    value_ = candidate;
  }

  std::int64_t value() const noexcept { return value_; }

 private:
  std::int64_t value_;
  jyppx::liuding::RestoreLevel restore_level_;
};

TEST(CheckpointContractTest, KeepsRestoreEnumsStableAndNamed) {
  EXPECT_EQ(static_cast<std::uint8_t>(jyppx::liuding::RestoreLevel::exact), 1);
  EXPECT_EQ(static_cast<std::uint8_t>(jyppx::liuding::RestoreLevel::bounded), 2);
  EXPECT_EQ(static_cast<std::uint8_t>(jyppx::liuding::RestoreLevel::best_effort), 3);
  EXPECT_EQ(jyppx::liuding::restore_level_name(jyppx::liuding::RestoreLevel::exact), "exact");
  EXPECT_EQ(jyppx::liuding::restore_level_name(jyppx::liuding::RestoreLevel::bounded), "bounded");
  EXPECT_EQ(jyppx::liuding::restore_level_name(jyppx::liuding::RestoreLevel::best_effort),
            "best_effort");
  EXPECT_EQ(jyppx::liuding::restore_level_name(static_cast<jyppx::liuding::RestoreLevel>(0)),
            "unknown");

  EXPECT_EQ(static_cast<std::uint8_t>(jyppx::liuding::RestorePolicy::exact_only), 1);
  EXPECT_EQ(static_cast<std::uint8_t>(jyppx::liuding::RestorePolicy::allow_bounded), 2);
  EXPECT_EQ(static_cast<std::uint8_t>(jyppx::liuding::RestorePolicy::allow_best_effort),
            3);
  EXPECT_EQ(jyppx::liuding::restore_policy_name(jyppx::liuding::RestorePolicy::allow_best_effort),
            "allow_best_effort");
  EXPECT_EQ(jyppx::liuding::restore_policy_name(static_cast<jyppx::liuding::RestorePolicy>(0)),
            "unknown");
}

TEST(CheckpointContractTest, RestorePolicyFormsAnExplicitGuaranteeLattice) {
  using jyppx::liuding::RestoreLevel;
  using jyppx::liuding::RestorePolicy;
  EXPECT_TRUE(jyppx::liuding::restore_policy_accepts(RestorePolicy::exact_only,
                                            RestoreLevel::exact));
  EXPECT_FALSE(jyppx::liuding::restore_policy_accepts(RestorePolicy::exact_only,
                                             RestoreLevel::bounded));
  EXPECT_TRUE(jyppx::liuding::restore_policy_accepts(RestorePolicy::allow_bounded,
                                            RestoreLevel::exact));
  EXPECT_TRUE(jyppx::liuding::restore_policy_accepts(RestorePolicy::allow_bounded,
                                            RestoreLevel::bounded));
  EXPECT_FALSE(jyppx::liuding::restore_policy_accepts(RestorePolicy::allow_bounded,
                                             RestoreLevel::best_effort));
  EXPECT_TRUE(jyppx::liuding::restore_policy_accepts(RestorePolicy::allow_best_effort,
                                            RestoreLevel::best_effort));
  EXPECT_FALSE(jyppx::liuding::restore_policy_accepts(static_cast<RestorePolicy>(0),
                                             RestoreLevel::exact));
  EXPECT_FALSE(jyppx::liuding::restore_policy_accepts(RestorePolicy::allow_best_effort,
                                             static_cast<RestoreLevel>(0)));
}

TEST(CheckpointContractTest, ValidatesManifestIdentityFieldsAndDegradation) {
  CounterParticipant exact(7);
  EXPECT_NO_THROW(jyppx::liuding::validate_checkpoint_participant(exact));

  CounterParticipant bounded(7, jyppx::liuding::RestoreLevel::bounded);
  EXPECT_NO_THROW(jyppx::liuding::validate_checkpoint_participant(bounded));

  auto invalid = exact.describe_state();
  invalid.schema_version = 0;
  EXPECT_THROW(jyppx::liuding::validate_state_manifest(invalid), jyppx::liuding::InvalidArgument);
  invalid = exact.describe_state();
  invalid.fields.front().max_bytes = 0;
  EXPECT_THROW(jyppx::liuding::validate_state_manifest(invalid), jyppx::liuding::InvalidArgument);
  invalid = exact.describe_state();
  invalid.fields.push_back(invalid.fields.front());
  EXPECT_THROW(jyppx::liuding::validate_state_manifest(invalid), jyppx::liuding::InvalidArgument);
  invalid = exact.describe_state();
  invalid.participant_id = "../unsafe";
  EXPECT_THROW(jyppx::liuding::validate_state_manifest(invalid), jyppx::liuding::InvalidArgument);
  invalid = exact.describe_state();
  invalid.restore_level = jyppx::liuding::RestoreLevel::bounded;
  EXPECT_THROW(jyppx::liuding::validate_state_manifest(invalid), jyppx::liuding::InvalidArgument);
}

TEST(CheckpointContractTest,
     MatchesFieldContractsByIdentityWithoutLosingBounds) {
  const std::vector<jyppx::liuding::StateFieldDescriptor> fields{
      {.name = "topology", .max_bytes = 128},
      {.name = "state", .max_bytes = 8},
      {.name = "optional", .max_bytes = 16, .required = false}};
  auto reordered = fields;
  std::sort(reordered.begin(), reordered.end(),
            [](const auto& left, const auto& right) {
              return left.name < right.name;
            });
  do {
    EXPECT_TRUE(jyppx::liuding::state_fields_equal(fields, reordered));
    EXPECT_TRUE(jyppx::liuding::state_fields_equal(reordered, fields));
  } while (std::next_permutation(reordered.begin(), reordered.end(),
                                 [](const auto& left, const auto& right) {
                                   return left.name < right.name;
                                 }));

  auto changed = fields;
  changed.pop_back();
  EXPECT_FALSE(jyppx::liuding::state_fields_equal(fields, changed));
  changed = fields;
  changed.push_back({.name = "unknown", .max_bytes = 1});
  EXPECT_FALSE(jyppx::liuding::state_fields_equal(fields, changed));
  changed = fields;
  changed.back().name = "unknown";
  EXPECT_FALSE(jyppx::liuding::state_fields_equal(fields, changed));
  changed = fields;
  changed.back() = changed.front();
  EXPECT_FALSE(jyppx::liuding::state_fields_equal(fields, changed));
  EXPECT_FALSE(jyppx::liuding::state_fields_equal(changed, fields));
  EXPECT_FALSE(jyppx::liuding::state_fields_equal(changed, changed));
  changed = fields;
  changed.front().media_type = "application/x-other";
  EXPECT_FALSE(jyppx::liuding::state_fields_equal(fields, changed));
  changed = fields;
  --changed.front().max_bytes;
  EXPECT_FALSE(jyppx::liuding::state_fields_equal(fields, changed));
  changed = fields;
  changed.back().required = true;
  EXPECT_FALSE(jyppx::liuding::state_fields_equal(fields, changed));
}

TEST(CheckpointContractTest, ParticipantRoundTripsThroughStorageNeutralBlobs) {
  CounterParticipant source(42);
  MemoryStateStore store(source.describe_state());
  source.save_state(store);

  CounterParticipant restored(0);
  restored.load_state(store, jyppx::liuding::RestorePolicy::exact_only);
  EXPECT_EQ(restored.value(), 42);
}

TEST(CheckpointContractTest, FailedRestoreDoesNotMutateParticipant) {
  CounterParticipant invalid_source(-1);
  MemoryStateStore store(invalid_source.describe_state());
  invalid_source.save_state(store);

  CounterParticipant target(17);
  EXPECT_THROW(
      target.load_state(store, jyppx::liuding::RestorePolicy::allow_best_effort),
      jyppx::liuding::SerializationError);
  EXPECT_EQ(target.value(), 17);
}

TEST(CheckpointContractTest, ReaderWriterEnforceNamesDuplicatesAndLimits) {
  CounterParticipant source(9);
  MemoryStateStore store(source.describe_state());
  source.save_state(store);

  EXPECT_THROW(source.save_state(store), jyppx::liuding::SerializationError);
  const std::byte value{0};
  EXPECT_THROW(store.write_blob("unknown", std::span(&value, 1)),
               jyppx::liuding::SerializationError);
  EXPECT_THROW(store.read_blob("value", sizeof(std::int64_t) - 1),
               jyppx::liuding::ResourceExhausted);
  EXPECT_THROW(store.read_blob("missing", 1), jyppx::liuding::SerializationError);
}

TEST(CheckpointContractTest, PolicyRejectionPrecedesStateMutation) {
  CounterParticipant source(33, jyppx::liuding::RestoreLevel::bounded);
  MemoryStateStore store(source.describe_state());
  source.save_state(store);

  CounterParticipant target(21);
  EXPECT_THROW(target.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::CompatibilityError);
  EXPECT_EQ(target.value(), 21);
}

TEST(CheckpointContractTest, SchemaRejectionPrecedesStateMutation) {
  CounterParticipant source(33);
  auto future_manifest = source.describe_state();
  future_manifest.schema_version = 2;
  MemoryStateStore store(std::move(future_manifest));
  source.save_state(store);

  CounterParticipant target(21);
  EXPECT_THROW(
      target.load_state(store, jyppx::liuding::RestorePolicy::allow_best_effort),
      jyppx::liuding::CompatibilityError);
  EXPECT_EQ(target.value(), 21);
}

}  // namespace
