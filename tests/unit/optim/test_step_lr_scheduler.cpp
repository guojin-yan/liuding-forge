// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/optim/sgd_adapter.hpp>
#include <jyppx/optim/step_lr_scheduler.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
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
      throw jyppx::SerializationError("unknown StepLR test field");
    }
    if (data.size() > field->max_bytes) {
      throw jyppx::ResourceExhausted("oversized StepLR test field");
    }
    if (!blobs_
             .emplace(std::string(name),
                      std::vector<std::byte>(data.begin(), data.end()))
             .second) {
      throw jyppx::SerializationError("duplicate StepLR test field");
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
      throw jyppx::SerializationError("missing StepLR test field");
    }
    if (blob->second.size() > max_bytes) {
      throw jyppx::ResourceExhausted("StepLR test read limit exceeded");
    }
    return blob->second;
  }

  std::vector<std::byte>& mutable_blob(std::string_view name) {
    return blobs_.at(std::string(name));
  }

  const std::map<std::string, std::vector<std::byte>>& blobs() const noexcept {
    return blobs_;
  }

 private:
  jyppx::StateManifest manifest_;  ///< Descriptor enforced by this test store.
  std::map<std::string, std::vector<std::byte>>
      blobs_;  ///< Complete fields indexed by their stable names.
};

std::shared_ptr<jyppx::optim::SgdOptimizerAdapter> make_optimizer(
    std::string optimizer_id = "optimizer.main",
    std::string second_group_id = "head") {
  auto backbone = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto head = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  return std::make_shared<jyppx::optim::SgdOptimizerAdapter>(
      std::move(optimizer_id),
      std::vector<jyppx::optim::SgdParameterGroup>{
          {.group_id = "backbone",
           .parameters = {{.parameter_id = "backbone.weight",
                           .parameter = std::move(backbone)}},
           .options = {.learning_rate = 0.2}},
          {.group_id = std::move(second_group_id),
           .parameters = {{.parameter_id = "head.weight",
                           .parameter = std::move(head)}},
           .options = {.learning_rate = 0.05}}});
}

MemoryStateStore save_participant(
    const jyppx::optim::StepLrSchedulerCheckpointParticipant& participant) {
  MemoryStateStore store(participant.describe_state());
  participant.save_state(store);
  return store;
}

TEST(StepLrSchedulerTest, PublishesTypedStableDescriptor) {
  auto optimizer = make_optimizer();
  jyppx::optim::StepLrScheduler scheduler(
      optimizer,
      {.scheduler_id = "scheduler.epochs", .step_size = 3, .gamma = 0.25});

  const auto& descriptor = scheduler.descriptor();
  EXPECT_EQ(descriptor.schema_version, 1U);
  EXPECT_EQ(descriptor.type_id, "jyppx.scheduler.step_lr/v1");
  EXPECT_EQ(descriptor.step_mode, jyppx::optim::SchedulerStepMode::epoch);
  EXPECT_EQ(descriptor.optimizer_id, "optimizer.main");
  EXPECT_EQ(descriptor.optimizer_group_ids,
            (std::vector<std::string>{"backbone", "head"}));
  EXPECT_EQ(std::get<std::int64_t>(descriptor.config.at("step_size")), 3);
  EXPECT_DOUBLE_EQ(std::get<double>(descriptor.config.at("gamma")), 0.25);
  EXPECT_EQ(std::get<std::vector<double>>(
                descriptor.config.at("base_learning_rates")),
            (std::vector<double>{0.2, 0.05}));
  EXPECT_FALSE(descriptor.optimizer_group_signature.empty());
  EXPECT_EQ(descriptor.fingerprint.size(), 16U);
  EXPECT_EQ(jyppx::optim::scheduler_step_mode_name(descriptor.step_mode),
            "epoch");
}

TEST(StepLrSchedulerTest, DecaysEveryConfiguredCompletedStepInterval) {
  auto optimizer = make_optimizer();
  jyppx::optim::StepLrScheduler scheduler(optimizer,
                                          {.step_size = 2, .gamma = 0.5});

  scheduler.step();
  EXPECT_EQ(scheduler.completed_steps(), 1U);
  EXPECT_EQ(scheduler.learning_rates(), (std::vector<double>{0.2, 0.05}));
  scheduler.step();
  EXPECT_EQ(scheduler.completed_steps(), 2U);
  EXPECT_EQ(scheduler.learning_rates(), (std::vector<double>{0.1, 0.025}));
  scheduler.step();
  scheduler.step();
  EXPECT_EQ(scheduler.completed_steps(), 4U);
  EXPECT_EQ(scheduler.learning_rates(), (std::vector<double>{0.05, 0.0125}));
}

TEST(StepLrSchedulerTest, RejectsInvalidConstructionBeforeMutation) {
  auto optimizer = make_optimizer();
  const auto before = optimizer->learning_rates();
  EXPECT_THROW(jyppx::optim::StepLrScheduler(nullptr), jyppx::InvalidArgument);
  EXPECT_THROW(jyppx::optim::StepLrScheduler(
                   optimizer, {.scheduler_id = "", .step_size = 1}),
               jyppx::InvalidArgument);
  EXPECT_THROW(
      jyppx::optim::StepLrScheduler(optimizer, {.step_size = 0, .gamma = 0.1}),
      jyppx::InvalidArgument);
  EXPECT_THROW(
      jyppx::optim::StepLrScheduler(optimizer, {.step_size = 1, .gamma = -0.1}),
      jyppx::InvalidArgument);
  EXPECT_EQ(optimizer->learning_rates(), before);
}

TEST(StepLrSchedulerCheckpointTest,
     RestoresCounterAndRatesForExactContinuation) {
  auto source_optimizer = make_optimizer();
  auto source = std::make_shared<jyppx::optim::StepLrScheduler>(
      source_optimizer,
      jyppx::optim::StepLrSchedulerOptions{.step_size = 2, .gamma = 0.5});
  source->step();
  source->step();
  source->step();
  jyppx::optim::StepLrSchedulerCheckpointParticipant source_participant(source);
  const auto store = save_participant(source_participant);

  auto resumed_optimizer = make_optimizer();
  auto resumed = std::make_shared<jyppx::optim::StepLrScheduler>(
      resumed_optimizer,
      jyppx::optim::StepLrSchedulerOptions{.step_size = 2, .gamma = 0.5});
  resumed_optimizer->set_learning_rates({0.9, 0.8});
  resumed->step();
  jyppx::optim::StepLrSchedulerCheckpointParticipant resumed_participant(
      resumed);
  resumed_participant.load_state(store, jyppx::RestorePolicy::exact_only);

  EXPECT_EQ(resumed->completed_steps(), 3U);
  EXPECT_EQ(resumed->learning_rates(), source->learning_rates());
  source->step();
  resumed->step();
  EXPECT_EQ(resumed->completed_steps(), source->completed_steps());
  EXPECT_EQ(resumed->learning_rates(), source->learning_rates());
}

TEST(StepLrSchedulerCheckpointTest,
     RejectsTopologyAndMalformedControlWithoutMutation) {
  auto source_optimizer = make_optimizer();
  auto source = std::make_shared<jyppx::optim::StepLrScheduler>(
      source_optimizer,
      jyppx::optim::StepLrSchedulerOptions{.step_size = 2, .gamma = 0.5});
  source->step();
  jyppx::optim::StepLrSchedulerCheckpointParticipant source_participant(source);
  const auto store = save_participant(source_participant);

  auto different_optimizer = make_optimizer("optimizer.main", "classifier");
  auto different = std::make_shared<jyppx::optim::StepLrScheduler>(
      different_optimizer,
      jyppx::optim::StepLrSchedulerOptions{.step_size = 2, .gamma = 0.5});
  jyppx::optim::StepLrSchedulerCheckpointParticipant different_participant(
      different);
  const auto before_mismatch = save_participant(different_participant);
  EXPECT_THROW(
      different_participant.load_state(store, jyppx::RestorePolicy::exact_only),
      jyppx::CompatibilityError);
  EXPECT_EQ(save_participant(different_participant).blobs(),
            before_mismatch.blobs());

  auto matching_optimizer = make_optimizer();
  auto matching = std::make_shared<jyppx::optim::StepLrScheduler>(
      matching_optimizer,
      jyppx::optim::StepLrSchedulerOptions{.step_size = 2, .gamma = 0.5});
  matching_optimizer->set_learning_rates({0.7, 0.6});
  jyppx::optim::StepLrSchedulerCheckpointParticipant matching_participant(
      matching);
  const auto before_malformed = save_participant(matching_participant);
  auto malformed = store;
  malformed.mutable_blob("control").front() = std::byte{'X'};
  EXPECT_THROW(matching_participant.load_state(
                   malformed, jyppx::RestorePolicy::exact_only),
               jyppx::SerializationError);
  EXPECT_EQ(save_participant(matching_participant).blobs(),
            before_malformed.blobs());
}

TEST(StepLrSchedulerCheckpointTest, UsesBoundedCurrentFormatOnly) {
  auto scheduler =
      std::make_shared<jyppx::optim::StepLrScheduler>(make_optimizer());
  EXPECT_THROW(jyppx::optim::StepLrSchedulerCheckpointParticipant(
                   scheduler, {.max_groups = 0}),
               jyppx::InvalidArgument);
  EXPECT_THROW(jyppx::optim::StepLrSchedulerCheckpointParticipant(
                   scheduler, {.max_identifier_bytes = 3}),
               jyppx::InvalidArgument);
  EXPECT_THROW(jyppx::optim::StepLrSchedulerCheckpointParticipant(
                   scheduler, {.max_topology_bytes = 8}),
               jyppx::ResourceExhausted);

  jyppx::optim::StepLrSchedulerCheckpointParticipant participant(scheduler);
  const auto manifest = participant.describe_state();
  ASSERT_EQ(manifest.fields.size(), 2U);
  EXPECT_EQ(manifest.fields[0].name, "topology");
  EXPECT_EQ(manifest.fields[1].name, "control");
  const auto store = save_participant(participant);
  const auto topology = store.read_blob("topology", 1024 * 1024);
  ASSERT_GE(topology.size(), 8U);
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(topology.data()), 8),
            "JYPPXSLD");
}

}  // namespace
