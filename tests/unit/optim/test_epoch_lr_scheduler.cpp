// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/optim/composite_lr_scheduler.hpp>
#include <jyppx/liuding/optim/epoch_lr_scheduler.hpp>
#include <jyppx/liuding/optim/sgd_adapter.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
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
      throw jyppx::liuding::SerializationError("unknown epoch scheduler test field");
    }
    if (data.size() > field->max_bytes) {
      throw jyppx::liuding::ResourceExhausted("oversized epoch scheduler test field");
    }
    if (!blobs_
             .emplace(std::string(name),
                      std::vector<std::byte>(data.begin(), data.end()))
             .second) {
      throw jyppx::liuding::SerializationError("duplicate epoch scheduler test field");
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
      throw jyppx::liuding::SerializationError("missing epoch scheduler test field");
    }
    if (blob->second.size() > max_bytes) {
      throw jyppx::liuding::ResourceExhausted(
          "epoch scheduler test read limit exceeded");
    }
    return blob->second;
  }

  const std::map<std::string, std::vector<std::byte>>& blobs() const noexcept {
    return blobs_;
  }

 private:
  jyppx::liuding::StateManifest manifest_;  ///< Manifest enforced by this test store.
  std::map<std::string, std::vector<std::byte>>
      blobs_;  ///< Complete fields indexed by stable field name.
};

std::shared_ptr<jyppx::liuding::optim::SgdOptimizerAdapter> make_optimizer(
    std::string optimizer_id = "optimizer.main") {
  auto first = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto second = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  return std::make_shared<jyppx::liuding::optim::SgdOptimizerAdapter>(
      std::move(optimizer_id),
      std::vector<jyppx::liuding::optim::SgdParameterGroup>{
          {.group_id = "backbone",
           .parameters = {{.parameter_id = "backbone.weight",
                           .parameter = std::move(first)}},
           .options = {.learning_rate = 0.2}},
          {.group_id = "head",
           .parameters = {{.parameter_id = "head.weight",
                           .parameter = std::move(second)}},
           .options = {.learning_rate = 0.05}}});
}

MemoryStateStore save_participant(
    const jyppx::liuding::optim::LrSchedulerCheckpointParticipant& participant) {
  MemoryStateStore store(participant.describe_state());
  participant.save_state(store);
  return store;
}

using SchedulerFactory =
    std::function<std::shared_ptr<jyppx::liuding::optim::LrScheduler>(
        const std::shared_ptr<jyppx::liuding::optim::SgdOptimizerAdapter>&)>;

TEST(EpochLrSchedulerTest, PublishesCompleteTypedDescriptors) {
  auto optimizer = make_optimizer();
  jyppx::liuding::optim::ExponentialLrScheduler exponential(
      optimizer, {.scheduler_id = "scheduler.exp", .gamma = 0.5});
  EXPECT_EQ(exponential.descriptor().type_id,
            "jyppx.scheduler.exponential_lr/v1");
  EXPECT_EQ(std::get<double>(exponential.descriptor().config.at("gamma")), 0.5);

  jyppx::liuding::optim::MultiStepLrScheduler multi_step(
      optimizer,
      {.scheduler_id = "scheduler.multi", .milestones = {2, 4}, .gamma = 0.25});
  EXPECT_EQ(multi_step.descriptor().type_id,
            "jyppx.scheduler.multi_step_lr/v1");
  EXPECT_EQ(std::get<std::vector<std::int64_t>>(
                multi_step.descriptor().config.at("milestones")),
            (std::vector<std::int64_t>{2, 4}));

  jyppx::liuding::optim::CosineAnnealingLrScheduler cosine(
      optimizer,
      {.scheduler_id = "scheduler.cosine", .t_max = 4, .eta_min = 0.01});
  EXPECT_EQ(cosine.descriptor().type_id,
            "jyppx.scheduler.cosine_annealing_lr/v1");
  EXPECT_EQ(std::get<std::int64_t>(cosine.descriptor().config.at("t_max")), 4);
  EXPECT_DOUBLE_EQ(std::get<double>(cosine.descriptor().config.at("eta_min")),
                   0.01);
  EXPECT_EQ(cosine.descriptor().step_mode,
            jyppx::liuding::optim::SchedulerStepMode::epoch);

  jyppx::liuding::optim::CosineAnnealingWarmRestartsLrScheduler warm_restarts(
      optimizer, {.scheduler_id = "scheduler.warm_restarts",
                  .t_0 = 3,
                  .t_mult = 2,
                  .eta_min = 0.01});
  EXPECT_EQ(warm_restarts.descriptor().type_id,
            "jyppx.scheduler.cosine_annealing_warm_restarts_lr/v2");
  EXPECT_EQ(std::get<std::int64_t>(warm_restarts.descriptor().config.at("t_0")),
            3);
  EXPECT_EQ(
      std::get<std::int64_t>(warm_restarts.descriptor().config.at("t_mult")),
      2);
}

TEST(EpochLrSchedulerTest, AppliesExponentialAndMilestoneDecayToAllGroups) {
  auto exponential_optimizer = make_optimizer("optimizer.exponential");
  jyppx::liuding::optim::ExponentialLrScheduler exponential(exponential_optimizer,
                                                   {.gamma = 0.5});
  exponential.step();
  exponential.step();
  EXPECT_EQ(exponential.learning_rates(), (std::vector<double>{0.05, 0.0125}));

  auto multi_optimizer = make_optimizer("optimizer.multi");
  jyppx::liuding::optim::MultiStepLrScheduler multi_step(
      multi_optimizer, {.milestones = {2, 4}, .gamma = 0.1});
  multi_step.step();
  EXPECT_EQ(multi_step.learning_rates(), (std::vector<double>{0.2, 0.05}));
  multi_step.step();
  EXPECT_NEAR(multi_step.learning_rates()[0], 0.02, 1e-15);
  EXPECT_NEAR(multi_step.learning_rates()[1], 0.005, 1e-15);
  multi_step.step();
  multi_step.step();
  EXPECT_NEAR(multi_step.learning_rates()[0], 0.002, 1e-15);
  EXPECT_NEAR(multi_step.learning_rates()[1], 0.0005, 1e-15);
}

TEST(EpochLrSchedulerTest, CosineTraversesMinimumBoundaryWithoutRestart) {
  auto optimizer = make_optimizer();
  jyppx::liuding::optim::CosineAnnealingLrScheduler scheduler(
      optimizer, {.t_max = 4, .eta_min = 0.0});

  for (std::uint64_t step = 1; step <= 5; ++step) {
    scheduler.step();
    const auto expected = 0.2 *
                          (1.0 + std::cos(3.14159265358979323846 *
                                          static_cast<double>(step) / 4.0)) /
                          2.0;
    EXPECT_NEAR(scheduler.learning_rates()[0], expected, 1e-14)
        << "step " << step;
  }
  EXPECT_EQ(scheduler.completed_steps(), 5U);
}

TEST(EpochLrSchedulerTest, WarmRestartsGrowCyclesAndResetEveryGroup) {
  auto optimizer = make_optimizer();
  jyppx::liuding::optim::CosineAnnealingWarmRestartsLrScheduler scheduler(
      optimizer, {.t_0 = 3, .t_mult = 2, .eta_min = 0.0});

  scheduler.step();
  EXPECT_NEAR(scheduler.learning_rates()[0],
              0.2 * (1.0 + std::cos(3.14159265358979323846 / 3.0)) / 2.0,
              1e-14);
  scheduler.step();
  scheduler.step();
  EXPECT_EQ(scheduler.learning_rates(), (std::vector<double>{0.2, 0.05}));
  EXPECT_EQ(scheduler.cycle_index(), 1U);
  EXPECT_EQ(scheduler.cycle_length(), 6U);
  EXPECT_EQ(scheduler.cycle_position(), 0U);
  scheduler.step();
  EXPECT_EQ(scheduler.cycle_position(), 1U);
  EXPECT_NEAR(scheduler.learning_rates()[1],
              0.05 * (1.0 + std::cos(3.14159265358979323846 / 6.0)) / 2.0,
              1e-14);
}

TEST(EpochLrSchedulerTest, RejectsInvalidOptionsBeforeRateMutation) {
  auto optimizer = make_optimizer();
  const auto before = optimizer->learning_rates();
  EXPECT_THROW(jyppx::liuding::optim::ExponentialLrScheduler(nullptr),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(jyppx::liuding::optim::ExponentialLrScheduler(
                   optimizer, {.scheduler_id = "", .gamma = 0.5}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(jyppx::liuding::optim::ExponentialLrScheduler(optimizer, {.gamma = -0.1}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(jyppx::liuding::optim::MultiStepLrScheduler(
                   optimizer, {.milestones = {}, .gamma = 0.1}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(jyppx::liuding::optim::MultiStepLrScheduler(
                   optimizer, {.milestones = {2, 2}, .gamma = 0.1}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(jyppx::liuding::optim::MultiStepLrScheduler(
                   optimizer, {.milestones = {0, 2}, .gamma = 0.1}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(jyppx::liuding::optim::CosineAnnealingLrScheduler(
                   optimizer, {.t_max = 0, .eta_min = 0.0}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(jyppx::liuding::optim::CosineAnnealingLrScheduler(
                   optimizer, {.t_max = 4, .eta_min = 0.3}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(
      jyppx::liuding::optim::CosineAnnealingLrScheduler(
          optimizer,
          {.t_max = 4, .eta_min = std::numeric_limits<double>::infinity()}),
      jyppx::liuding::InvalidArgument);
  EXPECT_THROW(jyppx::liuding::optim::CosineAnnealingWarmRestartsLrScheduler(optimizer,
                                                                    {.t_0 = 0}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(jyppx::liuding::optim::CosineAnnealingWarmRestartsLrScheduler(
                   optimizer, {.t_0 = 2, .t_mult = 0}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(jyppx::liuding::optim::CosineAnnealingWarmRestartsLrScheduler(
                   optimizer, {.t_0 = 2, .eta_min = 0.3}),
               jyppx::liuding::InvalidArgument);
  EXPECT_EQ(optimizer->learning_rates(), before);
}

TEST(EpochLrSchedulerTest, OverflowingPreviewLeavesAllStateUnchanged) {
  auto optimizer = make_optimizer();
  jyppx::liuding::optim::ExponentialLrScheduler scheduler(
      optimizer, {.gamma = std::numeric_limits<double>::max()});
  scheduler.step();
  const auto before_rates = scheduler.learning_rates();
  const auto before_steps = scheduler.completed_steps();
  EXPECT_THROW(scheduler.step(), jyppx::liuding::NumericalError);
  EXPECT_EQ(scheduler.learning_rates(), before_rates);
  EXPECT_EQ(scheduler.completed_steps(), before_steps);
}

TEST(EpochLrSchedulerCheckpointTest, EveryAlgorithmRestoresExactContinuation) {
  const std::vector<std::pair<std::string, SchedulerFactory>> cases = {
      {"exponential",
       [](const auto& optimizer) {
         return std::make_shared<jyppx::liuding::optim::ExponentialLrScheduler>(
             optimizer,
             jyppx::liuding::optim::ExponentialLrSchedulerOptions{
                 .scheduler_id = "scheduler.exponential", .gamma = 0.5});
       }},
      {"multi_step",
       [](const auto& optimizer) {
         return std::make_shared<jyppx::liuding::optim::MultiStepLrScheduler>(
             optimizer, jyppx::liuding::optim::MultiStepLrSchedulerOptions{
                            .scheduler_id = "scheduler.multi_step",
                            .milestones = {2, 4},
                            .gamma = 0.1});
       }},
      {"cosine",
       [](const auto& optimizer) {
         return std::make_shared<jyppx::liuding::optim::CosineAnnealingLrScheduler>(
             optimizer, jyppx::liuding::optim::CosineAnnealingLrSchedulerOptions{
                            .scheduler_id = "scheduler.cosine",
                            .t_max = 4,
                            .eta_min = 0.01});
       }},
      {"cosine_warm_restarts",
       [](const auto& optimizer) {
         return std::make_shared<
             jyppx::liuding::optim::CosineAnnealingWarmRestartsLrScheduler>(
             optimizer,
             jyppx::liuding::optim::CosineAnnealingWarmRestartsLrSchedulerOptions{
                 .scheduler_id = "scheduler.cosine_warm_restarts",
                 .t_0 = 2,
                 .t_mult = 2,
                 .eta_min = 0.01});
       }},
  };

  for (const auto& [name, factory] : cases) {
    SCOPED_TRACE(name);
    auto source_optimizer = make_optimizer();
    auto source = factory(source_optimizer);
    source->step();
    source->step();
    source->step();
    jyppx::liuding::optim::LrSchedulerCheckpointParticipant source_participant(source);
    const auto store = save_participant(source_participant);

    auto resumed_optimizer = make_optimizer();
    auto resumed = factory(resumed_optimizer);
    resumed_optimizer->set_learning_rates({0.9, 0.8});
    jyppx::liuding::optim::LrSchedulerCheckpointParticipant resumed_participant(resumed);
    resumed_participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only);
    EXPECT_EQ(resumed->completed_steps(), source->completed_steps());
    EXPECT_EQ(resumed->learning_rates(), source->learning_rates());

    source->step();
    resumed->step();
    EXPECT_EQ(resumed->completed_steps(), source->completed_steps());
    EXPECT_EQ(resumed->learning_rates(), source->learning_rates());
  }
}

TEST(EpochLrSchedulerCheckpointTest,
     WarmRestartRejectsCounterInconsistentCycleState) {
  auto scheduler =
      std::make_shared<jyppx::liuding::optim::CosineAnnealingWarmRestartsLrScheduler>(
          make_optimizer(),
          jyppx::liuding::optim::CosineAnnealingWarmRestartsLrSchedulerOptions{
              .t_0 = 3, .t_mult = 2});
  scheduler->step();
  auto state =
      jyppx::liuding::optim::detail::SchedulerStateAccess::capture_state(*scheduler);
  state.control["cycle_position"] = std::int64_t{2};
  EXPECT_THROW(jyppx::liuding::optim::detail::SchedulerStateAccess::validate_state(
                   *scheduler, state),
               jyppx::liuding::CompatibilityError);
  EXPECT_EQ(scheduler->completed_steps(), 1U);
  EXPECT_EQ(scheduler->cycle_position(), 1U);
}

TEST(EpochLrSchedulerCheckpointTest,
     DifferentTypedConfigurationFailsWithoutMutation) {
  auto source = std::make_shared<jyppx::liuding::optim::MultiStepLrScheduler>(
      make_optimizer(), jyppx::liuding::optim::MultiStepLrSchedulerOptions{
                            .milestones = {2, 4}, .gamma = 0.1});
  source->step();
  jyppx::liuding::optim::LrSchedulerCheckpointParticipant source_participant(source);
  const auto store = save_participant(source_participant);

  auto target = std::make_shared<jyppx::liuding::optim::MultiStepLrScheduler>(
      make_optimizer(), jyppx::liuding::optim::MultiStepLrSchedulerOptions{
                            .milestones = {2, 5}, .gamma = 0.1});
  jyppx::liuding::optim::LrSchedulerCheckpointParticipant target_participant(target);
  const auto before = save_participant(target_participant);
  EXPECT_THROW(
      target_participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::CompatibilityError);
  const auto after = save_participant(target_participant);
  EXPECT_EQ(after.blobs(), before.blobs());
  EXPECT_EQ(target->completed_steps(), 0U);
  EXPECT_EQ(target->learning_rates(), (std::vector<double>{0.2, 0.05}));
}

TEST(EpochLrSchedulerTest, ComposesAtomicallyInsideChainedScheduler) {
  auto optimizer = make_optimizer();
  std::vector<std::unique_ptr<jyppx::liuding::optim::LrScheduler>> children;
  children.push_back(std::make_unique<jyppx::liuding::optim::ExponentialLrScheduler>(
      optimizer,
      jyppx::liuding::optim::ExponentialLrSchedulerOptions{
          .scheduler_id = "scheduler.child.exponential", .gamma = 0.5}));
  children.push_back(std::make_unique<jyppx::liuding::optim::MultiStepLrScheduler>(
      optimizer, jyppx::liuding::optim::MultiStepLrSchedulerOptions{
                     .scheduler_id = "scheduler.child.multi_step",
                     .milestones = {1},
                     .gamma = 0.1}));
  jyppx::liuding::optim::ChainedLrScheduler chained(
      optimizer, std::move(children),
      {.scheduler_id = "scheduler.chained.epoch"});

  chained.step();
  EXPECT_NEAR(chained.learning_rates()[0], 0.01, 1e-15);
  EXPECT_NEAR(chained.learning_rates()[1], 0.0025, 1e-15);
  ASSERT_EQ(chained.descriptor().children.size(), 2U);
  EXPECT_EQ(chained.descriptor().children[0].type_id,
            "jyppx.scheduler.exponential_lr/v1");
  EXPECT_EQ(chained.descriptor().children[1].type_id,
            "jyppx.scheduler.multi_step_lr/v1");
}

}  // namespace
