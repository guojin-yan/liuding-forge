// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/optim/sgd_adapter.hpp>
#include <jyppx/training/amp/grad_scaler_checkpoint_participant.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

class MemoryStateStore final : public jyppx::StateWriter,
                               public jyppx::StateReader {
 public:
  explicit MemoryStateStore(jyppx::StateManifest manifest)
      : manifest_(std::move(manifest)) {
    jyppx::validate_state_manifest(manifest_);
  }

  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    const auto field =
        std::find_if(manifest_.fields.begin(), manifest_.fields.end(),
                     [name](const jyppx::StateFieldDescriptor& candidate) {
                       return candidate.name == name;
                     });
    if (field == manifest_.fields.end()) {
      throw jyppx::SerializationError("unknown GradScaler state field");
    }
    if (data.size() > field->max_bytes) {
      throw jyppx::ResourceExhausted("oversized GradScaler state field");
    }
    if (!blobs_
             .emplace(std::string(name),
                      std::vector<std::byte>(data.begin(), data.end()))
             .second) {
      throw jyppx::SerializationError("duplicate GradScaler state field");
    }
  }

  const jyppx::StateManifest& manifest() const noexcept override {
    return manifest_;
  }

  bool contains(std::string_view name) const override {
    return blobs_.contains(std::string(name));
  }

  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max_bytes) const override {
    const auto blob = blobs_.find(std::string(name));
    if (blob == blobs_.end()) {
      throw jyppx::SerializationError("missing GradScaler state field");
    }
    if (blob->second.size() > max_bytes) {
      throw jyppx::ResourceExhausted("GradScaler test read limit exceeded");
    }
    return blob->second;
  }

  std::vector<std::byte>& mutable_blob(std::string_view name) {
    return blobs_.at(std::string(name));
  }

  void erase_blob(std::string_view name) { blobs_.erase(std::string(name)); }

 private:
  jyppx::StateManifest manifest_;  ///< Descriptor enforced by this test store.
  std::map<std::string, std::vector<std::byte>>
      blobs_;  ///< Complete fields indexed by stable field name.
};

using jyppx::training::amp::GradScaler;
using jyppx::training::amp::GradScalerCheckpointParticipant;

std::shared_ptr<jyppx::optim::SgdOptimizerAdapter> make_optimizer(
    const jyppx::Tensor& parameter) {
  return std::make_shared<jyppx::optim::SgdOptimizerAdapter>(
      "optimizer.scaler.checkpoint",
      std::vector<jyppx::optim::SgdParameterGroup>{
          {.group_id = "group.main",
           .parameters = {{.parameter_id = "parameter.main",
                           .parameter = parameter}},
           .options = {.learning_rate = 0.1}}});
}

void run_cycle(GradScaler& scaler,
               const std::shared_ptr<jyppx::optim::OptimizerAdapter>& optimizer,
               const jyppx::Tensor& parameter, double gradient) {
  parameter.mutable_grad() = torch::full_like(parameter, gradient);
  scaler.unscale(optimizer);
  const bool applied = scaler.step_allowed(optimizer);
  if (applied) {
    optimizer->step();
  }
  scaler.mark_step(optimizer, applied);
  scaler.update();
}

MemoryStateStore save_participant(
    const GradScalerCheckpointParticipant& participant) {
  MemoryStateStore store(participant.describe_state());
  participant.save_state(store);
  return store;
}

TEST(GradScalerCheckpointParticipantTest, RestoresExactContinuationState) {
  auto parameter = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto optimizer = make_optimizer(parameter);
  auto scaler =
      std::make_shared<GradScaler>(jyppx::training::amp::GradScalerOptions{
          .init_scale = 8.0,
          .growth_factor = 2.0,
          .growth_interval = 3,
          .scaler_id = "grad_scaler.training"});
  GradScalerCheckpointParticipant participant(scaler);

  run_cycle(*scaler, optimizer, parameter, 8.0);
  const auto saved = scaler->capture_state();
  auto store = save_participant(participant);
  run_cycle(*scaler, optimizer, parameter,
            std::numeric_limits<double>::infinity());
  ASSERT_NE(scaler->get_scale(), saved.scale);

  participant.load_state(store, jyppx::RestorePolicy::exact_only);
  const auto restored = scaler->capture_state();
  EXPECT_DOUBLE_EQ(restored.scale, saved.scale);
  EXPECT_EQ(restored.growth_tracker, saved.growth_tracker);

  run_cycle(*scaler, optimizer, parameter, restored.scale);
  run_cycle(*scaler, optimizer, parameter, restored.scale);
  EXPECT_DOUBLE_EQ(scaler->get_scale(), 16.0);
}

TEST(GradScalerCheckpointParticipantTest, RejectsActiveOptimizerCycle) {
  auto parameter = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto optimizer = make_optimizer(parameter);
  auto scaler = std::make_shared<GradScaler>(
      jyppx::training::amp::GradScalerOptions{.init_scale = 2.0});
  GradScalerCheckpointParticipant participant(scaler);
  MemoryStateStore store(participant.describe_state());

  parameter.mutable_grad() = torch::full_like(parameter, 2.0);
  scaler->unscale(optimizer);
  EXPECT_THROW(participant.save_state(store), jyppx::NotReady);
  const bool applied = scaler->step_allowed(optimizer);
  if (applied) {
    optimizer->step();
  }
  scaler->mark_step(optimizer, applied);
  scaler->update();
}

TEST(GradScalerCheckpointParticipantTest,
     RejectsMalformedControlWithoutMutation) {
  auto scaler = std::make_shared<GradScaler>(
      jyppx::training::amp::GradScalerOptions{.init_scale = 4.0});
  GradScalerCheckpointParticipant participant(scaler);
  auto store = save_participant(participant);
  auto& control = store.mutable_blob("control");
  std::fill(control.begin() + 8, control.begin() + 16, std::byte{0});
  const auto before = scaler->capture_state();

  EXPECT_THROW(participant.load_state(store, jyppx::RestorePolicy::exact_only),
               jyppx::SerializationError);
  const auto after = scaler->capture_state();
  EXPECT_DOUBLE_EQ(after.scale, before.scale);
  EXPECT_EQ(after.growth_tracker, before.growth_tracker);
}

TEST(GradScalerCheckpointParticipantTest,
     RejectsUnknownPolicyAndMissingRequiredFieldWithoutMutation) {
  auto scaler = std::make_shared<GradScaler>(
      jyppx::training::amp::GradScalerOptions{.init_scale = 4.0});
  GradScalerCheckpointParticipant participant(scaler);
  auto store = save_participant(participant);
  const auto before = scaler->capture_state();

  EXPECT_THROW(
      participant.load_state(store, static_cast<jyppx::RestorePolicy>(255)),
      jyppx::CompatibilityError);
  EXPECT_DOUBLE_EQ(scaler->capture_state().scale, before.scale);

  store.erase_blob("control");
  EXPECT_THROW(participant.load_state(store, jyppx::RestorePolicy::exact_only),
               jyppx::SerializationError);
  const auto after = scaler->capture_state();
  EXPECT_DOUBLE_EQ(after.scale, before.scale);
  EXPECT_EQ(after.growth_tracker, before.growth_tracker);
}

TEST(GradScalerCheckpointParticipantTest,
     RejectsCorruptTopologyAndInvalidControlHeader) {
  auto scaler = std::make_shared<GradScaler>(
      jyppx::training::amp::GradScalerOptions{.init_scale = 4.0});
  GradScalerCheckpointParticipant participant(scaler);
  auto topology_store = save_participant(participant);
  topology_store.mutable_blob("topology").front() ^= std::byte{0x01};
  EXPECT_THROW(
      participant.load_state(topology_store, jyppx::RestorePolicy::exact_only),
      jyppx::CompatibilityError);

  auto control_store = save_participant(participant);
  control_store.mutable_blob("control").front() ^= std::byte{0x01};
  EXPECT_THROW(
      participant.load_state(control_store, jyppx::RestorePolicy::exact_only),
      jyppx::SerializationError);
}

TEST(GradScalerCheckpointParticipantTest, RejectsInvalidResourceLimits) {
  EXPECT_THROW(GradScalerCheckpointParticipant(
                   std::make_shared<GradScaler>(),
                   jyppx::training::amp::GradScalerCheckpointOptions{
                       .max_control_bytes = 23}),
               jyppx::InvalidArgument);
  EXPECT_THROW(GradScalerCheckpointParticipant(
                   std::make_shared<GradScaler>(),
                   jyppx::training::amp::GradScalerCheckpointOptions{
                       .max_total_bytes = 1}),
               jyppx::ResourceExhausted);
  EXPECT_THROW(
      GradScalerCheckpointParticipant(
          std::make_shared<GradScaler>(
              jyppx::training::amp::GradScalerOptions{.scaler_id = "long"}),
          jyppx::training::amp::GradScalerCheckpointOptions{
              .max_identifier_bytes = 3}),
      jyppx::ResourceExhausted);
}

TEST(GradScalerCheckpointParticipantTest, RejectsDifferentFrozenOptions) {
  auto source = std::make_shared<GradScaler>(
      jyppx::training::amp::GradScalerOptions{.init_scale = 2.0});
  GradScalerCheckpointParticipant source_participant(source);
  auto store = save_participant(source_participant);

  auto target = std::make_shared<GradScaler>(
      jyppx::training::amp::GradScalerOptions{.init_scale = 4.0});
  GradScalerCheckpointParticipant target_participant(target);
  EXPECT_THROW(
      target_participant.load_state(store, jyppx::RestorePolicy::exact_only),
      jyppx::CompatibilityError);
  EXPECT_DOUBLE_EQ(target->get_scale(), 4.0);
}

}  // namespace
