// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/optim/composite_lr_scheduler.hpp>
#include <jyppx/liuding/optim/sgd_adapter.hpp>
#include <jyppx/liuding/optim/step_lr_scheduler.hpp>

#include <algorithm>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
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
      throw jyppx::liuding::SerializationError("unknown composite scheduler field");
    }
    if (data.size() > field->max_bytes) {
      throw jyppx::liuding::ResourceExhausted("oversized composite scheduler field");
    }
    if (!blobs_
             .emplace(std::string(name),
                      std::vector<std::byte>(data.begin(), data.end()))
             .second) {
      throw jyppx::liuding::SerializationError("duplicate composite scheduler field");
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
      throw jyppx::liuding::SerializationError("missing composite scheduler field");
    }
    if (blob->second.size() > max_bytes) {
      throw jyppx::liuding::ResourceExhausted(
          "composite scheduler test read limit exceeded");
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
  jyppx::liuding::StateManifest manifest_;  ///< Descriptor enforced by this test store.
  std::map<std::string, std::vector<std::byte>>
      blobs_;  ///< Complete fields indexed by stable field name.
};

std::shared_ptr<jyppx::liuding::optim::SgdOptimizerAdapter> make_optimizer(
    double learning_rate = 0.2, std::string optimizer_id = "optimizer.main") {
  auto parameter = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  return std::make_shared<jyppx::liuding::optim::SgdOptimizerAdapter>(
      std::move(optimizer_id),
      std::vector<jyppx::liuding::optim::SgdParameterGroup>{
          {.group_id = "main",
           .parameters = {{.parameter_id = "weight",
                           .parameter = std::move(parameter)}},
           .options = {.learning_rate = learning_rate}}});
}

std::vector<std::unique_ptr<jyppx::liuding::optim::LrScheduler>> make_step_children(
    const std::shared_ptr<jyppx::liuding::optim::SgdOptimizerAdapter>& optimizer,
    double first_gamma = 0.5, double second_gamma = 0.1) {
  std::vector<std::unique_ptr<jyppx::liuding::optim::LrScheduler>> children;
  children.push_back(std::make_unique<jyppx::liuding::optim::StepLrScheduler>(
      optimizer, jyppx::liuding::optim::StepLrSchedulerOptions{
                     .scheduler_id = "scheduler.child.first",
                     .step_size = 1,
                     .gamma = first_gamma}));
  children.push_back(std::make_unique<jyppx::liuding::optim::StepLrScheduler>(
      optimizer, jyppx::liuding::optim::StepLrSchedulerOptions{
                     .scheduler_id = "scheduler.child.second",
                     .step_size = 1,
                     .gamma = second_gamma}));
  return children;
}

MemoryStateStore save_participant(
    const jyppx::liuding::optim::LrSchedulerCheckpointParticipant& participant) {
  MemoryStateStore store(participant.describe_state());
  participant.save_state(store);
  return store;
}

TEST(SequentialLrSchedulerTest, PublishesOrderedTypedDescriptor) {
  auto optimizer = make_optimizer();
  jyppx::liuding::optim::SequentialLrScheduler scheduler(
      optimizer, make_step_children(optimizer),
      {.scheduler_id = "scheduler.root", .milestones = {2}});

  const auto& descriptor = scheduler.descriptor();
  EXPECT_EQ(descriptor.type_id, "jyppx.scheduler.sequential_lr/v1");
  EXPECT_EQ(descriptor.step_mode, jyppx::liuding::optim::SchedulerStepMode::epoch);
  EXPECT_EQ(
      std::get<std::vector<std::int64_t>>(descriptor.config.at("milestones")),
      (std::vector<std::int64_t>{2}));
  EXPECT_EQ(std::get<std::vector<std::string>>(
                descriptor.config.at("child_scheduler_ids")),
            (std::vector<std::string>{"scheduler.child.first",
                                      "scheduler.child.second"}));
  ASSERT_EQ(descriptor.children.size(), 2U);
  EXPECT_EQ(descriptor.children[0].type_id, "jyppx.scheduler.step_lr/v1");
  EXPECT_EQ(descriptor.children[1].type_id, "jyppx.scheduler.step_lr/v1");
  EXPECT_EQ(scheduler.next_child_index(), 0U);
}

TEST(SequentialLrSchedulerTest, AssignsCompleteEventsToEachChild) {
  auto optimizer = make_optimizer();
  jyppx::liuding::optim::SequentialLrScheduler scheduler(
      optimizer, make_step_children(optimizer), {.milestones = {2}});

  scheduler.step();
  EXPECT_EQ(scheduler.learning_rates(), (std::vector<double>{0.1}));
  EXPECT_EQ(scheduler.next_child_index(), 0U);
  scheduler.step();
  EXPECT_EQ(scheduler.learning_rates(), (std::vector<double>{0.05}));
  EXPECT_EQ(scheduler.next_child_index(), 1U);
  scheduler.step();
  EXPECT_NEAR(scheduler.learning_rates().front(), 0.005, 1e-15);
  EXPECT_EQ(scheduler.completed_steps(), 3U);
}

TEST(ChainedLrSchedulerTest, AppliesEveryChildWithOneOptimizerCommit) {
  auto optimizer = make_optimizer();
  jyppx::liuding::optim::ChainedLrScheduler scheduler(optimizer,
                                             make_step_children(optimizer));

  scheduler.step();
  EXPECT_NEAR(scheduler.learning_rates().front(), 0.01, 1e-15);
  scheduler.step();
  EXPECT_NEAR(scheduler.learning_rates().front(), 0.0005, 1e-15);
  EXPECT_EQ(scheduler.completed_steps(), 2U);
}

TEST(CompositeLrSchedulerTest, RestoresEveryChildForExactContinuation) {
  auto source_optimizer = make_optimizer();
  auto source = std::make_shared<jyppx::liuding::optim::SequentialLrScheduler>(
      source_optimizer, make_step_children(source_optimizer),
      jyppx::liuding::optim::SequentialLrSchedulerOptions{
          .scheduler_id = "scheduler.root", .milestones = {2}});
  source->step();
  source->step();
  source->step();
  jyppx::liuding::optim::LrSchedulerCheckpointParticipant source_participant(source);
  const auto store = save_participant(source_participant);

  auto resumed_optimizer = make_optimizer();
  auto resumed = std::make_shared<jyppx::liuding::optim::SequentialLrScheduler>(
      resumed_optimizer, make_step_children(resumed_optimizer),
      jyppx::liuding::optim::SequentialLrSchedulerOptions{
          .scheduler_id = "scheduler.root", .milestones = {2}});
  resumed->step();
  jyppx::liuding::optim::LrSchedulerCheckpointParticipant resumed_participant(resumed);
  resumed_participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only);

  EXPECT_EQ(resumed->completed_steps(), source->completed_steps());
  EXPECT_EQ(resumed->learning_rates(), source->learning_rates());
  source->step();
  resumed->step();
  EXPECT_EQ(resumed->learning_rates(), source->learning_rates());
}

TEST(CompositeLrSchedulerTest, RejectsTopologyAndControlBeforeMutation) {
  auto source_optimizer = make_optimizer();
  auto source = std::make_shared<jyppx::liuding::optim::SequentialLrScheduler>(
      source_optimizer, make_step_children(source_optimizer),
      jyppx::liuding::optim::SequentialLrSchedulerOptions{
          .scheduler_id = "scheduler.root", .milestones = {2}});
  source->step();
  jyppx::liuding::optim::LrSchedulerCheckpointParticipant source_participant(source);
  const auto store = save_participant(source_participant);

  auto target_optimizer = make_optimizer();
  auto target = std::make_shared<jyppx::liuding::optim::SequentialLrScheduler>(
      target_optimizer, make_step_children(target_optimizer, 0.25, 0.1),
      jyppx::liuding::optim::SequentialLrSchedulerOptions{
          .scheduler_id = "scheduler.root", .milestones = {2}});
  jyppx::liuding::optim::LrSchedulerCheckpointParticipant target_participant(target);
  const auto topology_before = save_participant(target_participant);
  EXPECT_THROW(
      target_participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::CompatibilityError);
  EXPECT_EQ(save_participant(target_participant).blobs(),
            topology_before.blobs());

  auto matching_optimizer = make_optimizer();
  auto matching = std::make_shared<jyppx::liuding::optim::SequentialLrScheduler>(
      matching_optimizer, make_step_children(matching_optimizer),
      jyppx::liuding::optim::SequentialLrSchedulerOptions{
          .scheduler_id = "scheduler.root", .milestones = {2}});
  jyppx::liuding::optim::LrSchedulerCheckpointParticipant matching_participant(matching);
  const auto control_before = save_participant(matching_participant);
  auto malformed = store;
  malformed.mutable_blob("control").front() = std::byte{'X'};
  EXPECT_THROW(matching_participant.load_state(
                   malformed, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::SerializationError);
  EXPECT_EQ(save_participant(matching_participant).blobs(),
            control_before.blobs());
}

TEST(ChainedLrSchedulerTest, FailedChildPreviewLeavesWholeTreeUnchanged) {
  auto optimizer = make_optimizer(2.0);
  auto scheduler = std::make_shared<jyppx::liuding::optim::ChainedLrScheduler>(
      optimizer,
      make_step_children(optimizer, 0.5, std::numeric_limits<double>::max()));
  scheduler->step();
  jyppx::liuding::optim::LrSchedulerCheckpointParticipant participant(scheduler);
  const auto before = save_participant(participant);

  EXPECT_THROW(scheduler->step(), jyppx::liuding::NumericalError);
  EXPECT_EQ(save_participant(participant).blobs(), before.blobs());
}

TEST(CompositeLrSchedulerTest, RejectsInvalidOwnershipAndStructure) {
  auto optimizer = make_optimizer();
  EXPECT_THROW(jyppx::liuding::optim::ChainedLrScheduler(optimizer, {}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(
      jyppx::liuding::optim::SequentialLrScheduler(
          optimizer, make_step_children(optimizer), {.milestones = {}}),
      jyppx::liuding::InvalidArgument);
  EXPECT_THROW(
      jyppx::liuding::optim::SequentialLrScheduler(
          optimizer, make_step_children(optimizer), {.milestones = {0}}),
      jyppx::liuding::InvalidArgument);

  auto other_optimizer = make_optimizer(0.2, "optimizer.other");
  EXPECT_THROW(jyppx::liuding::optim::ChainedLrScheduler(
                   optimizer, make_step_children(other_optimizer)),
               jyppx::liuding::InvalidArgument);

  auto duplicate_ids = make_step_children(optimizer);
  duplicate_ids[1] = std::make_unique<jyppx::liuding::optim::StepLrScheduler>(
      optimizer, jyppx::liuding::optim::StepLrSchedulerOptions{
                     .scheduler_id = "scheduler.child.first"});
  EXPECT_THROW(
      jyppx::liuding::optim::ChainedLrScheduler(optimizer, std::move(duplicate_ids)),
      jyppx::liuding::InvalidArgument);
}

TEST(CompositeLrSchedulerTest, SupportsNestedTransactionalTrees) {
  auto optimizer = make_optimizer();
  auto sequential = std::make_unique<jyppx::liuding::optim::SequentialLrScheduler>(
      optimizer, make_step_children(optimizer),
      jyppx::liuding::optim::SequentialLrSchedulerOptions{
          .scheduler_id = "scheduler.nested", .milestones = {1}});
  std::vector<std::unique_ptr<jyppx::liuding::optim::LrScheduler>> root_children;
  root_children.push_back(std::move(sequential));
  root_children.push_back(std::make_unique<jyppx::liuding::optim::StepLrScheduler>(
      optimizer,
      jyppx::liuding::optim::StepLrSchedulerOptions{
          .scheduler_id = "scheduler.tail", .step_size = 1, .gamma = 0.5}));
  jyppx::liuding::optim::ChainedLrScheduler scheduler(
      optimizer, std::move(root_children), {.scheduler_id = "scheduler.root"});

  scheduler.step();
  EXPECT_NEAR(scheduler.learning_rates().front(), 0.05, 1e-15);
  ASSERT_EQ(scheduler.descriptor().children.size(), 2U);
  ASSERT_EQ(scheduler.descriptor().children[0].children.size(), 2U);
}

}  // namespace
