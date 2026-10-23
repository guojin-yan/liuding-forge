// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/random_checkpoint_participant.hpp>

#include <torch/torch.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <map>
#include <span>
#include <string>
#include <string_view>
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
    const auto field =
        std::find_if(manifest_.fields.begin(), manifest_.fields.end(),
                     [name](const jyppx::liuding::StateFieldDescriptor& candidate) {
                       return candidate.name == name;
                     });
    if (field == manifest_.fields.end()) {
      throw jyppx::liuding::SerializationError("unknown RNG state field");
    }
    if (data.size() > field->max_bytes) {
      throw jyppx::liuding::ResourceExhausted("oversized RNG state field");
    }
    if (!blobs_
             .emplace(std::string(name),
                      std::vector<std::byte>(data.begin(), data.end()))
             .second) {
      throw jyppx::liuding::SerializationError("duplicate RNG state field");
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
      throw jyppx::liuding::SerializationError("missing RNG state field");
    }
    if (blob->second.size() > max_bytes) {
      throw jyppx::liuding::ResourceExhausted("RNG test read limit exceeded");
    }
    return blob->second;
  }

  std::vector<std::byte>& mutable_blob(std::string_view name) {
    return blobs_.at(std::string(name));
  }

  void erase_blob(std::string_view name) { blobs_.erase(std::string(name)); }

  jyppx::liuding::StateManifest& mutable_manifest() noexcept { return manifest_; }

 private:
  jyppx::liuding::StateManifest manifest_;  ///< Descriptor enforced by this test store.
  std::map<std::string, std::vector<std::byte>>
      blobs_;  ///< Complete fields indexed by stable field name.
};

MemoryStateStore save_participant(
    const jyppx::liuding::core::RngCheckpointParticipant& participant) {
  MemoryStateStore store(participant.describe_state());
  participant.save_state(store);
  return store;
}

TEST(RngCheckpointParticipantTest, RestoresExactCpuContinuation) {
  jyppx::liuding::core::RngGuard guard(20261016, false);
  jyppx::liuding::core::RngCheckpointParticipant participant(
      {.participant_id = "rng.training", .include_cuda = false});
  auto store = save_participant(participant);

  const auto expected = torch::rand({16}, torch::kFloat64);
  jyppx::liuding::core::manual_seed(17, false);
  static_cast<void>(torch::rand({7}, torch::kFloat64));
  participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only);
  const auto actual = torch::rand({16}, torch::kFloat64);

  EXPECT_TRUE(torch::equal(actual, expected));
  EXPECT_EQ(participant.describe_state().fields.size(), 2U);
}

TEST(RngCheckpointParticipantTest, RejectsCorruptionWithoutMutation) {
  jyppx::liuding::core::RngGuard guard(41, false);
  jyppx::liuding::core::RngCheckpointParticipant participant({.include_cuda = false});
  auto store = save_participant(participant);
  store.mutable_blob("topology").front() ^= std::byte{0x01};

  jyppx::liuding::core::manual_seed(99, false);
  const auto before = jyppx::liuding::core::get_rng_state(false);
  EXPECT_THROW(participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::CompatibilityError);
  const auto after = jyppx::liuding::core::get_rng_state(false);
  EXPECT_TRUE(torch::equal(after.cpu, before.cpu));
}

TEST(RngCheckpointParticipantTest, RejectsManifestAndResourceMismatch) {
  jyppx::liuding::core::RngGuard guard(73, false);
  jyppx::liuding::core::RngCheckpointParticipant participant({.include_cuda = false});
  auto store = save_participant(participant);
  store.mutable_manifest().participant_id = "rng.other";
  EXPECT_THROW(participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::CompatibilityError);

  EXPECT_THROW(jyppx::liuding::core::RngCheckpointParticipant(
                   {.include_cuda = false, .max_total_bytes = 1}),
               jyppx::liuding::ResourceExhausted);
}

TEST(RngCheckpointParticipantTest,
     RejectsUnknownPolicyAndMissingCpuStateWithoutMutation) {
  jyppx::liuding::core::RngGuard guard(83, false);
  jyppx::liuding::core::RngCheckpointParticipant participant({.include_cuda = false});
  auto store = save_participant(participant);

  jyppx::liuding::core::manual_seed(101, false);
  const auto before = jyppx::liuding::core::get_rng_state(false);
  EXPECT_THROW(
      participant.load_state(store, static_cast<jyppx::liuding::RestorePolicy>(255)),
      jyppx::liuding::CompatibilityError);
  EXPECT_TRUE(torch::equal(jyppx::liuding::core::get_rng_state(false).cpu, before.cpu));

  store.erase_blob("cpu");
  EXPECT_THROW(participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::SerializationError);
  EXPECT_TRUE(torch::equal(jyppx::liuding::core::get_rng_state(false).cpu, before.cpu));
}

TEST(RngCheckpointParticipantTest, RejectsTruncatedCpuStateWithoutMutation) {
  jyppx::liuding::core::RngGuard guard(89, false);
  jyppx::liuding::core::RngCheckpointParticipant participant({.include_cuda = false});
  auto store = save_participant(participant);
  store.mutable_blob("cpu").pop_back();

  jyppx::liuding::core::manual_seed(103, false);
  const auto before = jyppx::liuding::core::get_rng_state(false);
  EXPECT_THROW(participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::SerializationError);
  EXPECT_TRUE(torch::equal(jyppx::liuding::core::get_rng_state(false).cpu, before.cpu));
}

TEST(RngCheckpointParticipantCudaTest, RestoresEveryVisibleGenerator) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available";
  }
  jyppx::liuding::core::RngGuard guard(20261017, true);
  jyppx::liuding::core::RngCheckpointParticipant participant;
  auto store = save_participant(participant);

  std::vector<torch::Tensor> expected;
  const auto device_count = at::getNumGPUs();
  expected.reserve(static_cast<std::size_t>(device_count));
  for (std::int64_t index = 0; index < device_count; ++index) {
    expected.push_back(torch::rand(
        {8}, torch::TensorOptions().device(torch::Device(
                 torch::kCUDA, static_cast<c10::DeviceIndex>(index)))));
  }
  jyppx::liuding::core::manual_seed(31, true);
  participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only);
  for (std::int64_t index = 0; index < device_count; ++index) {
    const auto actual = torch::rand(
        {8}, torch::TensorOptions().device(torch::Device(
                 torch::kCUDA, static_cast<c10::DeviceIndex>(index))));
    EXPECT_TRUE(torch::equal(actual.cpu(),
                             expected[static_cast<std::size_t>(index)].cpu()));
  }
}

}  // namespace
