// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/optim/event_lr_scheduler.hpp>
#include <jyppx/liuding/optim/sgd_adapter.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
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
    const auto field = std::find_if(
        manifest_.fields.begin(), manifest_.fields.end(),
        [name](const jyppx::liuding::StateFieldDescriptor& candidate) {
          return candidate.name == name;
        });
    if (field == manifest_.fields.end()) {
      throw jyppx::liuding::SerializationError(
          "unknown event-scheduler test field");
    }
    if (data.size() > field->max_bytes) {
      throw jyppx::liuding::ResourceExhausted(
          "oversized event-scheduler test field");
    }
    if (!blobs_
             .emplace(std::string(name),
                      std::vector<std::byte>(data.begin(), data.end()))
             .second) {
      throw jyppx::liuding::SerializationError(
          "duplicate event-scheduler test field");
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
      throw jyppx::liuding::SerializationError(
          "missing event-scheduler test field");
    }
    if (blob->second.size() > max_bytes) {
      throw jyppx::liuding::ResourceExhausted(
          "event-scheduler read limit exceeded");
    }
    return blob->second;
  }

  std::vector<std::byte>& mutable_blob(std::string_view name) {
    return blobs_.at(std::string(name));
  }

 private:
  jyppx::liuding::StateManifest
      manifest_;  ///< Manifest enforced by this memory store.
  std::map<std::string, std::vector<std::byte>>
      blobs_;  ///< Complete fields indexed by stable field name.
};

std::shared_ptr<jyppx::liuding::optim::SgdOptimizerAdapter> make_optimizer(
    std::string optimizer_id = "optimizer.events") {
  auto backbone = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto head = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  return std::make_shared<jyppx::liuding::optim::SgdOptimizerAdapter>(
      std::move(optimizer_id),
      std::vector<jyppx::liuding::optim::SgdParameterGroup>{
          {.group_id = "backbone",
           .parameters = {{.parameter_id = "backbone.weight",
                           .parameter = std::move(backbone)}},
           .options = {.learning_rate = 0.2}},
          {.group_id = "head",
           .parameters = {{.parameter_id = "head.weight",
                           .parameter = std::move(head)}},
           .options = {.learning_rate = 0.05}}});
}

MemoryStateStore save_participant(
    const jyppx::liuding::optim::LrSchedulerCheckpointParticipant&
        participant) {
  MemoryStateStore store(participant.describe_state());
  participant.save_state(store);
  return store;
}

TEST(EventLrSchedulerTest, PlateauPublishesMetricDescriptorAndCooldownState) {
  auto optimizer = make_optimizer();
  jyppx::liuding::optim::ReduceLrOnPlateauScheduler scheduler(
      optimizer, {.scheduler_id = "scheduler.validation_loss",
                  .factor = 0.5,
                  .patience = 1,
                  .cooldown = 1,
                  .min_learning_rates = {0.06, 0.0},
                  .eps = 0.0});

  EXPECT_EQ(scheduler.descriptor().step_mode,
            jyppx::liuding::optim::SchedulerStepMode::metric);
  EXPECT_EQ(scheduler.descriptor().type_id,
            "jyppx.scheduler.reduce_lr_on_plateau/v2");
  EXPECT_THROW(scheduler.step(), jyppx::liuding::InvalidArgument);
  scheduler.step(1.0);
  scheduler.step(1.1);
  EXPECT_EQ(scheduler.bad_observations(), 1U);
  scheduler.step(1.2);
  EXPECT_EQ(scheduler.learning_rates(), (std::vector<double>{0.1, 0.025}));
  EXPECT_EQ(scheduler.cooldown_remaining(), 1U);
  scheduler.step(0.9);
  EXPECT_DOUBLE_EQ(scheduler.best(), 0.9);
  EXPECT_EQ(scheduler.bad_observations(), 0U);
  EXPECT_EQ(scheduler.cooldown_remaining(), 0U);
  scheduler.step(1.4);
  scheduler.step(1.5);
  EXPECT_EQ(scheduler.learning_rates(), (std::vector<double>{0.06, 0.0125}));
  EXPECT_TRUE(scheduler.has_best());
  EXPECT_DOUBLE_EQ(scheduler.best(), 0.9);
}

TEST(EventLrSchedulerTest, PlateauSupportsMaximizeAbsoluteThreshold) {
  auto optimizer = make_optimizer();
  jyppx::liuding::optim::ReduceLrOnPlateauScheduler scheduler(
      optimizer,
      {.direction = jyppx::liuding::optim::MetricDirection::maximize,
       .factor = 0.5,
       .patience = 0,
       .threshold = 0.1,
       .threshold_mode = jyppx::liuding::optim::MetricThresholdMode::absolute,
       .eps = 0.0});

  scheduler.step(-2.0);
  scheduler.step(-1.95);
  EXPECT_EQ(scheduler.learning_rates(), (std::vector<double>{0.1, 0.025}));
  scheduler.step(-1.0);
  EXPECT_DOUBLE_EQ(scheduler.best(), -1.0);
}

TEST(EventLrSchedulerTest, PlateauCheckpointRestoresCompleteMetricHistory) {
  auto source_optimizer = make_optimizer();
  auto source =
      std::make_shared<jyppx::liuding::optim::ReduceLrOnPlateauScheduler>(
          source_optimizer,
          jyppx::liuding::optim::ReduceLrOnPlateauSchedulerOptions{
              .factor = 0.5, .patience = 1, .cooldown = 2, .eps = 0.0});
  source->step(3.0);
  source->step(3.1);
  source->step(3.2);
  source->step(3.3);
  jyppx::liuding::optim::LrSchedulerCheckpointParticipant source_participant(
      source);
  const auto store = save_participant(source_participant);
  EXPECT_EQ(store.manifest().schema_version, 3U);
  EXPECT_EQ(store.manifest().fields[1].media_type,
            "application/vnd.jyppx.scheduler-state-tree-v2");

  auto restored_optimizer = make_optimizer();
  auto restored =
      std::make_shared<jyppx::liuding::optim::ReduceLrOnPlateauScheduler>(
          restored_optimizer,
          jyppx::liuding::optim::ReduceLrOnPlateauSchedulerOptions{
              .factor = 0.5, .patience = 1, .cooldown = 2, .eps = 0.0});
  jyppx::liuding::optim::LrSchedulerCheckpointParticipant restored_participant(
      restored);
  restored_participant.load_state(store,
                                  jyppx::liuding::RestorePolicy::exact_only);

  EXPECT_EQ(restored->completed_steps(), source->completed_steps());
  EXPECT_EQ(restored->learning_rates(), source->learning_rates());
  EXPECT_EQ(restored->bad_observations(), source->bad_observations());
  EXPECT_EQ(restored->cooldown_remaining(), source->cooldown_remaining());
  EXPECT_DOUBLE_EQ(restored->best(), source->best());
  source->step(3.4);
  restored->step(3.4);
  EXPECT_EQ(restored->learning_rates(), source->learning_rates());
  EXPECT_EQ(restored->cooldown_remaining(), source->cooldown_remaining());
}

TEST(EventLrSchedulerTest, PlateauRejectsInvalidInputWithoutMutation) {
  auto optimizer = make_optimizer();
  const auto before = optimizer->learning_rates();
  EXPECT_THROW(jyppx::liuding::optim::ReduceLrOnPlateauScheduler(
                   optimizer, {.factor = 1.0}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(jyppx::liuding::optim::ReduceLrOnPlateauScheduler(
                   optimizer, {.min_learning_rates = {0.3}}),
               jyppx::liuding::InvalidArgument);
  EXPECT_EQ(optimizer->learning_rates(), before);

  jyppx::liuding::optim::ReduceLrOnPlateauScheduler scheduler(optimizer);
  EXPECT_THROW(scheduler.step(std::numeric_limits<double>::quiet_NaN()),
               jyppx::liuding::InvalidArgument);
  EXPECT_EQ(scheduler.completed_steps(), 0U);
  EXPECT_EQ(optimizer->learning_rates(), before);
}

TEST(EventLrSchedulerTest, OneCycleHasBoundedNonNegativeEndpoints) {
  auto optimizer = make_optimizer();
  jyppx::liuding::optim::OneCycleLrScheduler scheduler(
      optimizer,
      {.max_learning_rates = {0.4, 0.1},
       .total_steps = 10,
       .pct_start = 0.3,
       .anneal_strategy = jyppx::liuding::optim::AnnealStrategy::linear,
       .div_factor = 4.0,
       .final_div_factor = 10.0});

  EXPECT_EQ(scheduler.descriptor().step_mode,
            jyppx::liuding::optim::SchedulerStepMode::batch);
  EXPECT_EQ(scheduler.learning_rates(), (std::vector<double>{0.1, 0.025}));
  scheduler.step();
  scheduler.step();
  scheduler.step();
  EXPECT_EQ(scheduler.learning_rates(), (std::vector<double>{0.4, 0.1}));
  for (int step = 3; step < 10; ++step) {
    scheduler.step();
  }
  EXPECT_NEAR(scheduler.learning_rates()[0], 0.01, 1e-14);
  EXPECT_NEAR(scheduler.learning_rates()[1], 0.0025, 1e-14);
  EXPECT_THROW(scheduler.step(), jyppx::liuding::ResourceExhausted);
  EXPECT_EQ(scheduler.completed_steps(), 10U);
}

TEST(EventLrSchedulerTest, OneCycleThreePhaseReturnsThroughInitialRate) {
  auto optimizer = make_optimizer();
  jyppx::liuding::optim::OneCycleLrScheduler scheduler(
      optimizer,
      {.max_learning_rates = {0.4, 0.1},
       .total_steps = 10,
       .pct_start = 0.3,
       .anneal_strategy = jyppx::liuding::optim::AnnealStrategy::linear,
       .div_factor = 4.0,
       .final_div_factor = 10.0,
       .three_phase = true});
  for (int step = 0; step < 3; ++step) {
    scheduler.step();
  }
  EXPECT_EQ(scheduler.learning_rates(), (std::vector<double>{0.4, 0.1}));
  for (int step = 0; step < 3; ++step) {
    scheduler.step();
  }
  EXPECT_NEAR(scheduler.learning_rates()[0], 0.1, 1e-14);
  EXPECT_NEAR(scheduler.learning_rates()[1], 0.025, 1e-14);
  for (int step = 0; step < 4; ++step) {
    scheduler.step();
  }
  EXPECT_NEAR(scheduler.learning_rates()[0], 0.01, 1e-14);
  EXPECT_NEAR(scheduler.learning_rates()[1], 0.0025, 1e-14);
}

TEST(EventLrSchedulerTest, BatchSchedulersRejectNonAtomicMomentumCycling) {
  auto optimizer = make_optimizer();
  const auto before = optimizer->learning_rates();
  EXPECT_THROW(jyppx::liuding::optim::OneCycleLrScheduler(
                   optimizer, {.max_learning_rates = {0.4},
                               .total_steps = 10,
                               .cycle_momentum = true}),
               jyppx::liuding::NotImplemented);
  EXPECT_THROW(jyppx::liuding::optim::CyclicLrScheduler(
                   optimizer, {.base_learning_rates = {0.1},
                               .max_learning_rates = {0.4},
                               .step_size_up = 2,
                               .cycle_momentum = true}),
               jyppx::liuding::NotImplemented);
  EXPECT_EQ(optimizer->learning_rates(), before);
}

TEST(EventLrSchedulerTest, CyclicProfilesAndCallableIdentityAreStable) {
  auto optimizer = make_optimizer();
  jyppx::liuding::optim::CyclicLrScheduler scheduler(
      optimizer, {.scheduler_id = "scheduler.cyclic.validation",
                  .base_learning_rates = {0.02, 0.01},
                  .max_learning_rates = {0.2, 0.05},
                  .step_size_up = 2,
                  .step_size_down = 2,
                  .mode = jyppx::liuding::optim::CyclicLrMode::triangular2});
  EXPECT_EQ(scheduler.learning_rates(), (std::vector<double>{0.02, 0.01}));
  scheduler.step();
  EXPECT_NEAR(scheduler.learning_rates()[0], 0.11, 1e-14);
  scheduler.step();
  EXPECT_NEAR(scheduler.learning_rates()[0], 0.2, 1e-14);
  scheduler.step();
  scheduler.step();
  scheduler.step();
  EXPECT_NEAR(scheduler.learning_rates()[0], 0.065, 1e-14);

  auto custom_optimizer = make_optimizer("optimizer.custom_cycle");
  jyppx::liuding::optim::CyclicLrScheduler custom(
      custom_optimizer,
      {.base_learning_rates = {0.02, 0.01},
       .max_learning_rates = {0.2, 0.05},
       .step_size_up = 1,
       .scale_function = [](double iteration) { return 1.0 / iteration; },
       .scale_function_id = "scale.inverse_iteration/v1",
       .scale_mode = jyppx::liuding::optim::CyclicScaleMode::iterations});
  ASSERT_TRUE(custom.descriptor().callable_id.has_value());
  EXPECT_EQ(*custom.descriptor().callable_id, "scale.inverse_iteration/v1");
  custom.step();
  EXPECT_NEAR(custom.learning_rates()[0], 0.2, 1e-14);
}

TEST(EventLrSchedulerTest, LambdaFailureLeavesCounterAndRatesUnchanged) {
  auto optimizer = make_optimizer();
  auto fail = std::make_shared<bool>(false);
  jyppx::liuding::optim::LambdaLrScheduler scheduler(
      optimizer, {.scheduler_id = "scheduler.lambda.decay",
                  .callable_id = "schedule.inverse_epoch/v1",
                  .functions = {[fail](std::uint64_t step) {
                    if (*fail) {
                      throw std::runtime_error("injected callback failure");
                    }
                    return 1.0 / static_cast<double>(step + 1);
                  }}});
  EXPECT_EQ(scheduler.descriptor().step_mode,
            jyppx::liuding::optim::SchedulerStepMode::epoch);
  EXPECT_EQ(scheduler.descriptor().callable_id,
            std::optional<std::string>("schedule.inverse_epoch/v1"));
  scheduler.step();
  EXPECT_EQ(scheduler.learning_rates(), (std::vector<double>{0.1, 0.025}));
  const auto before_rates = scheduler.learning_rates();
  const auto before_steps = scheduler.completed_steps();
  *fail = true;
  EXPECT_THROW(scheduler.step(), jyppx::liuding::BackendError);
  EXPECT_EQ(scheduler.learning_rates(), before_rates);
  EXPECT_EQ(scheduler.completed_steps(), before_steps);
  EXPECT_THROW(scheduler.step(1.0), jyppx::liuding::InvalidArgument);
}

TEST(EventLrSchedulerTest, MalformedCurrentControlDoesNotMutateTarget) {
  auto source_optimizer = make_optimizer();
  auto source = std::make_shared<jyppx::liuding::optim::OneCycleLrScheduler>(
      source_optimizer,
      jyppx::liuding::optim::OneCycleLrSchedulerOptions{
          .max_learning_rates = {0.4, 0.1}, .total_steps = 5});
  source->step();
  jyppx::liuding::optim::LrSchedulerCheckpointParticipant source_participant(
      source);
  auto malformed = save_participant(source_participant);
  malformed.mutable_blob("control").front() = std::byte{'X'};

  auto target_optimizer = make_optimizer();
  auto target = std::make_shared<jyppx::liuding::optim::OneCycleLrScheduler>(
      target_optimizer,
      jyppx::liuding::optim::OneCycleLrSchedulerOptions{
          .max_learning_rates = {0.4, 0.1}, .total_steps = 5});
  const auto before_rates = target->learning_rates();
  jyppx::liuding::optim::LrSchedulerCheckpointParticipant target_participant(
      target);
  EXPECT_THROW(target_participant.load_state(
                   malformed, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::SerializationError);
  EXPECT_EQ(target->learning_rates(), before_rates);
  EXPECT_EQ(target->completed_steps(), 0U);
}

}  // namespace
