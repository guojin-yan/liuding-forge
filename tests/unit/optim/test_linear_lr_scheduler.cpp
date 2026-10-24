// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/optim/linear_lr_scheduler.hpp>
#include <jyppx/liuding/optim/sgd_adapter.hpp>

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
    if (field == manifest_.fields.end() || data.size() > field->max_bytes) {
      throw jyppx::liuding::SerializationError("invalid linear scheduler test field");
    }
    if (!blobs_
             .emplace(std::string(name),
                      std::vector<std::byte>(data.begin(), data.end()))
             .second) {
      throw jyppx::liuding::SerializationError("duplicate linear scheduler test field");
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
      throw jyppx::liuding::SerializationError("missing linear scheduler test field");
    }
    if (blob->second.size() > max_bytes) {
      throw jyppx::liuding::ResourceExhausted(
          "linear scheduler test field exceeds read limit");
    }
    return blob->second;
  }

  const std::map<std::string, std::vector<std::byte>>& blobs() const noexcept {
    return blobs_;
  }

 private:
  jyppx::liuding::StateManifest manifest_;  ///< Manifest enforced by this test store.
  std::map<std::string, std::vector<std::byte>>
      blobs_;  ///< Stored fields indexed by stable name.
};

std::shared_ptr<jyppx::liuding::optim::SgdOptimizerAdapter> make_optimizer(
    std::string optimizer_id = "optimizer.linear_family") {
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

MemoryStateStore save_state(
    const jyppx::liuding::optim::LrSchedulerCheckpointParticipant& participant) {
  MemoryStateStore store(participant.describe_state());
  participant.save_state(store);
  return store;
}

TEST(LinearFamilySchedulerTest, PublishesTypedDescriptorsAndConstructionRates) {
  auto optimizer = make_optimizer();
  jyppx::liuding::optim::LinearLrScheduler linear(optimizer,
                                         {.scheduler_id = "scheduler.linear",
                                          .start_factor = 0.2,
                                          .end_factor = 1.0,
                                          .total_iters = 4});
  EXPECT_EQ(linear.descriptor().type_id, "jyppx.scheduler.linear_lr/v1");
  EXPECT_EQ(linear.descriptor().step_mode,
            jyppx::liuding::optim::SchedulerStepMode::epoch);
  EXPECT_NEAR(linear.learning_rates()[0], 0.04, 1e-14);
  EXPECT_NEAR(linear.learning_rates()[1], 0.01, 1e-14);
  EXPECT_EQ(
      std::get<std::int64_t>(linear.descriptor().config.at("total_iters")), 4);

  auto warmup_optimizer = make_optimizer("optimizer.warmup");
  jyppx::liuding::optim::WarmupLrScheduler warmup(warmup_optimizer,
                                         {.scheduler_id = "scheduler.warmup",
                                          .warmup_iters = 4,
                                          .start_factor = 0.2});
  EXPECT_EQ(warmup.descriptor().type_id, "jyppx.scheduler.warmup_lr/v1");
  EXPECT_NEAR(warmup.learning_rates()[0], 0.04, 1e-14);
  EXPECT_NEAR(warmup.learning_rates()[1], 0.01, 1e-14);
}

TEST(LinearFamilySchedulerTest, LinearReachesEndFactorAndClamps) {
  auto optimizer = make_optimizer();
  jyppx::liuding::optim::LinearLrScheduler scheduler(
      optimizer, {.start_factor = 0.2, .end_factor = 1.0, .total_iters = 4});
  const std::vector<double> expected{0.08, 0.12, 0.16, 0.2, 0.2};
  for (const double expected_first_rate : expected) {
    scheduler.step();
    EXPECT_NEAR(scheduler.learning_rates().front(), expected_first_rate, 1e-14);
  }
}

TEST(LinearFamilySchedulerTest, WarmupReachesBaseAndClamps) {
  auto optimizer = make_optimizer();
  jyppx::liuding::optim::WarmupLrScheduler scheduler(
      optimizer, {.warmup_iters = 4, .start_factor = 0.2});
  const std::vector<double> expected{0.08, 0.12, 0.16, 0.2, 0.2};
  for (const double expected_first_rate : expected) {
    scheduler.step();
    EXPECT_NEAR(scheduler.learning_rates().front(), expected_first_rate, 1e-14);
  }
}

TEST(LinearFamilySchedulerTest, ConstantUsesPrefixAndThenBase) {
  auto optimizer = make_optimizer();
  jyppx::liuding::optim::ConstantLrScheduler scheduler(
      optimizer, {.factor = 0.2, .total_iters = 2});
  EXPECT_NEAR(scheduler.learning_rates().front(), 0.04, 1e-14);
  scheduler.step();
  EXPECT_NEAR(scheduler.learning_rates().front(), 0.04, 1e-14);
  scheduler.step();
  EXPECT_NEAR(scheduler.learning_rates().front(), 0.04, 1e-14);
  scheduler.step();
  EXPECT_NEAR(scheduler.learning_rates().front(), 0.2, 1e-14);
}

TEST(LinearFamilySchedulerTest, PolynomialReachesZeroAndClamps) {
  auto optimizer = make_optimizer();
  jyppx::liuding::optim::PolynomialLrScheduler scheduler(
      optimizer, {.total_iters = 4, .power = 2.0});
  scheduler.step();
  EXPECT_NEAR(scheduler.learning_rates().front(), 0.1125, 1e-14);
  scheduler.step();
  EXPECT_NEAR(scheduler.learning_rates().front(), 0.05, 1e-14);
  scheduler.step();
  EXPECT_NEAR(scheduler.learning_rates().front(), 0.0125, 1e-14);
  scheduler.step();
  EXPECT_DOUBLE_EQ(scheduler.learning_rates().front(), 0.0);
  scheduler.step();
  EXPECT_DOUBLE_EQ(scheduler.learning_rates().front(), 0.0);
}

TEST(LinearFamilySchedulerTest, ZeroTotalIterationsHaveExplicitBoundaries) {
  auto linear_optimizer = make_optimizer("optimizer.linear.zero");
  jyppx::liuding::optim::LinearLrScheduler linear(
      linear_optimizer,
      {.start_factor = 0.25, .end_factor = 0.75, .total_iters = 0});
  EXPECT_NEAR(linear.learning_rates().front(), 0.05, 1e-14);
  linear.step();
  EXPECT_NEAR(linear.learning_rates().front(), 0.15, 1e-14);

  auto constant_optimizer = make_optimizer("optimizer.constant.zero");
  jyppx::liuding::optim::ConstantLrScheduler constant(
      constant_optimizer, {.factor = 0.25, .total_iters = 0});
  constant.step();
  EXPECT_NEAR(constant.learning_rates().front(), 0.2, 1e-14);
}

TEST(LinearFamilySchedulerTest, InvalidOptionsDoNotMutateOptimizer) {
  auto optimizer = make_optimizer();
  const auto before = optimizer->learning_rates();
  EXPECT_THROW(jyppx::liuding::optim::LinearLrScheduler(
                   optimizer, {.start_factor = 0.0, .end_factor = 1.0}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(jyppx::liuding::optim::ConstantLrScheduler(optimizer, {.factor = -0.1}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(
      jyppx::liuding::optim::PolynomialLrScheduler(optimizer, {.total_iters = 0}),
      jyppx::liuding::InvalidArgument);
  EXPECT_THROW(jyppx::liuding::optim::WarmupLrScheduler(optimizer, {.warmup_iters = 0}),
               jyppx::liuding::InvalidArgument);
  EXPECT_EQ(optimizer->learning_rates(), before);
}

TEST(LinearFamilySchedulerTest,
     RejectsNonFiniteFactorsAndUnrepresentableIntervalsWithoutMutation) {
  auto optimizer = make_optimizer();
  const auto before = optimizer->learning_rates();
  for (const double factor : {std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(jyppx::liuding::optim::LinearLrScheduler(
                     optimizer, {.start_factor = factor, .end_factor = 1.0}),
                 jyppx::liuding::InvalidArgument);
    EXPECT_THROW(jyppx::liuding::optim::PolynomialLrScheduler(
                     optimizer, {.total_iters = 4, .power = factor}),
                 jyppx::liuding::InvalidArgument);
    EXPECT_EQ(optimizer->learning_rates(), before);
  }
  EXPECT_THROW(jyppx::liuding::optim::PolynomialLrScheduler(
                   optimizer, {.total_iters = 4, .power = -1.0}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(jyppx::liuding::optim::LinearLrScheduler(
                   optimizer,
                   {.total_iters = std::numeric_limits<std::uint64_t>::max()}),
               jyppx::liuding::InvalidArgument);
  EXPECT_EQ(optimizer->learning_rates(), before);
}

TEST(LinearFamilySchedulerTest, OverflowingTransitionDoesNotCommitRatesOrStep) {
  auto optimizer = make_optimizer();
  jyppx::liuding::optim::LinearLrScheduler scheduler(
      optimizer, {.start_factor = std::numeric_limits<double>::min(),
                  .end_factor = std::numeric_limits<double>::max() / 2.0,
                  .total_iters = 1});
  const auto before = optimizer->learning_rates();
  EXPECT_THROW(scheduler.step(), jyppx::liuding::NumericalError);
  EXPECT_EQ(scheduler.completed_steps(), 0U);
  EXPECT_EQ(optimizer->learning_rates(), before);
  EXPECT_EQ(scheduler.learning_rates(), before);
}

TEST(LinearFamilySchedulerCheckpointTest, RestoresExactContinuation) {
  auto source_optimizer = make_optimizer("optimizer.source");
  auto source = std::make_shared<jyppx::liuding::optim::LinearLrScheduler>(
      source_optimizer,
      jyppx::liuding::optim::LinearLrSchedulerOptions{.scheduler_id = "scheduler.linear",
                                             .start_factor = 0.2,
                                             .end_factor = 1.0,
                                             .total_iters = 4});
  source->step();
  source->step();
  jyppx::liuding::optim::LrSchedulerCheckpointParticipant source_participant(source);
  const auto store = save_state(source_participant);

  auto target_optimizer = make_optimizer("optimizer.source");
  auto target = std::make_shared<jyppx::liuding::optim::LinearLrScheduler>(
      target_optimizer,
      jyppx::liuding::optim::LinearLrSchedulerOptions{.scheduler_id = "scheduler.linear",
                                             .start_factor = 0.2,
                                             .end_factor = 1.0,
                                             .total_iters = 4});
  target_optimizer->set_learning_rates({0.9, 0.8});
  jyppx::liuding::optim::LrSchedulerCheckpointParticipant target_participant(target);
  target_participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only);
  EXPECT_EQ(target->completed_steps(), source->completed_steps());
  EXPECT_EQ(target->learning_rates(), source->learning_rates());

  source->step();
  target->step();
  EXPECT_EQ(target->completed_steps(), source->completed_steps());
  EXPECT_EQ(target->learning_rates(), source->learning_rates());
}

TEST(LinearFamilySchedulerCheckpointTest, DescriptorMismatchDoesNotMutate) {
  auto source_optimizer = make_optimizer("optimizer.mismatch");
  auto source = std::make_shared<jyppx::liuding::optim::PolynomialLrScheduler>(
      source_optimizer,
      jyppx::liuding::optim::PolynomialLrSchedulerOptions{
          .scheduler_id = "scheduler.poly", .total_iters = 4, .power = 2.0});
  source->step();
  jyppx::liuding::optim::LrSchedulerCheckpointParticipant source_participant(source);
  const auto store = save_state(source_participant);

  auto target_optimizer = make_optimizer("optimizer.mismatch");
  auto target = std::make_shared<jyppx::liuding::optim::PolynomialLrScheduler>(
      target_optimizer,
      jyppx::liuding::optim::PolynomialLrSchedulerOptions{
          .scheduler_id = "scheduler.poly", .total_iters = 5, .power = 2.0});
  jyppx::liuding::optim::LrSchedulerCheckpointParticipant target_participant(target);
  const auto before = save_state(target_participant).blobs();
  EXPECT_THROW(
      target_participant.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
      jyppx::liuding::CompatibilityError);
  EXPECT_EQ(save_state(target_participant).blobs(), before);
  EXPECT_EQ(target->completed_steps(), 0U);
}

TEST(LinearFamilySchedulerCheckpointTest,
     RestoresWarmupAndConstantExactContinuation) {
  auto warm_source_optimizer = make_optimizer("optimizer.warmup.restore");
  auto warm_source = std::make_shared<jyppx::liuding::optim::WarmupLrScheduler>(
      warm_source_optimizer, jyppx::liuding::optim::WarmupLrSchedulerOptions{
                                 .scheduler_id = "scheduler.warmup.restore",
                                 .warmup_iters = 4,
                                 .start_factor = 0.2});
  warm_source->step();
  warm_source->step();
  jyppx::liuding::optim::LrSchedulerCheckpointParticipant warm_source_participant(
      warm_source);
  const auto warm_store = save_state(warm_source_participant);

  auto warm_target_optimizer = make_optimizer("optimizer.warmup.restore");
  auto warm_target = std::make_shared<jyppx::liuding::optim::WarmupLrScheduler>(
      warm_target_optimizer, jyppx::liuding::optim::WarmupLrSchedulerOptions{
                                 .scheduler_id = "scheduler.warmup.restore",
                                 .warmup_iters = 4,
                                 .start_factor = 0.2});
  warm_target_optimizer->set_learning_rates({0.91, 0.82});
  jyppx::liuding::optim::LrSchedulerCheckpointParticipant warm_target_participant(
      warm_target);
  warm_target_participant.load_state(warm_store,
                                     jyppx::liuding::RestorePolicy::exact_only);
  EXPECT_EQ(warm_target->completed_steps(), warm_source->completed_steps());
  EXPECT_EQ(warm_target->learning_rates(), warm_source->learning_rates());
  EXPECT_EQ(save_state(warm_target_participant).blobs(), warm_store.blobs());
  for (const double expected_rate : {0.16, 0.2, 0.2}) {
    warm_source->step();
    warm_target->step();
    EXPECT_EQ(warm_target->completed_steps(), warm_source->completed_steps());
    EXPECT_EQ(warm_target->learning_rates(), warm_source->learning_rates());
    EXPECT_NEAR(warm_target->learning_rates().front(), expected_rate, 1e-14);
    EXPECT_NEAR(warm_target->learning_rates().back(), expected_rate / 4.0,
                1e-14);
  }

  auto constant_source_optimizer = make_optimizer("optimizer.constant.restore");
  auto constant_source = std::make_shared<jyppx::liuding::optim::ConstantLrScheduler>(
      constant_source_optimizer,
      jyppx::liuding::optim::ConstantLrSchedulerOptions{
          .scheduler_id = "scheduler.constant.restore",
          .factor = 0.2,
          .total_iters = 4});
  constant_source->step();
  constant_source->step();
  jyppx::liuding::optim::LrSchedulerCheckpointParticipant constant_source_participant(
      constant_source);
  const auto constant_store = save_state(constant_source_participant);

  auto constant_target_optimizer = make_optimizer("optimizer.constant.restore");
  auto constant_target = std::make_shared<jyppx::liuding::optim::ConstantLrScheduler>(
      constant_target_optimizer,
      jyppx::liuding::optim::ConstantLrSchedulerOptions{
          .scheduler_id = "scheduler.constant.restore",
          .factor = 0.2,
          .total_iters = 4});
  constant_target_optimizer->set_learning_rates({0.91, 0.82});
  jyppx::liuding::optim::LrSchedulerCheckpointParticipant constant_target_participant(
      constant_target);
  constant_target_participant.load_state(constant_store,
                                         jyppx::liuding::RestorePolicy::exact_only);
  EXPECT_EQ(constant_target->completed_steps(),
            constant_source->completed_steps());
  EXPECT_EQ(constant_target->learning_rates(),
            constant_source->learning_rates());
  EXPECT_EQ(save_state(constant_target_participant).blobs(),
            constant_store.blobs());
  for (const double expected_rate : {0.04, 0.04, 0.2}) {
    constant_source->step();
    constant_target->step();
    EXPECT_EQ(constant_target->completed_steps(),
              constant_source->completed_steps());
    EXPECT_EQ(constant_target->learning_rates(),
              constant_source->learning_rates());
    EXPECT_NEAR(constant_target->learning_rates().front(), expected_rate,
                1e-14);
    EXPECT_NEAR(constant_target->learning_rates().back(), expected_rate / 4.0,
                1e-14);
  }
}

}  // namespace
