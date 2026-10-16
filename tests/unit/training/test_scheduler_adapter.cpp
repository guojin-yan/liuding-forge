// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/optim/event_lr_scheduler.hpp>
#include <jyppx/optim/sgd_adapter.hpp>
#include <jyppx/optim/step_lr_scheduler.hpp>
#include <jyppx/training/scheduler_adapter.hpp>

#include <algorithm>
#include <cmath>
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
  /**
   * @brief Creates a bounded in-memory participant store.
   * @param manifest Exact manifest enforced by reads and writes.
   */
  explicit MemoryStateStore(jyppx::StateManifest manifest)
      : manifest_(std::move(manifest)) {
    jyppx::validate_state_manifest(manifest_);
  }

  /** @brief Stores one declared field exactly once. */
  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    const auto field =
        std::find_if(manifest_.fields.begin(), manifest_.fields.end(),
                     [name](const jyppx::StateFieldDescriptor& candidate) {
                       return candidate.name == name;
                     });
    if (field == manifest_.fields.end() || data.size() > field->max_bytes ||
        !blobs_
             .emplace(std::string(name),
                      std::vector<std::byte>(data.begin(), data.end()))
             .second) {
      throw jyppx::SerializationError("invalid scheduler adapter test write");
    }
  }

  /** @brief Returns the exact manifest supplied at construction. */
  const jyppx::StateManifest& manifest() const noexcept override {
    return manifest_;
  }

  /** @brief Reports whether a named field has been written. */
  bool contains(std::string_view name) const override {
    return blobs_.contains(std::string(name));
  }

  /** @brief Reads one stored field under the caller's byte limit. */
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max_bytes) const override {
    const auto field = blobs_.find(std::string(name));
    if (field == blobs_.end() || field->second.size() > max_bytes) {
      throw jyppx::SerializationError("invalid scheduler adapter test read");
    }
    return field->second;
  }

 private:
  jyppx::StateManifest manifest_;  ///< Exact delegated participant manifest.
  std::map<std::string, std::vector<std::byte>>
      blobs_;  ///< Encoded fields indexed by stable field name.
};

std::shared_ptr<jyppx::optim::SgdOptimizerAdapter> make_optimizer(
    std::string identifier = "optimizer.main") {
  auto parameter = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  return std::make_shared<jyppx::optim::SgdOptimizerAdapter>(
      std::move(identifier),
      std::vector<jyppx::optim::SgdParameterGroup>{
          {.group_id = "main",
           .parameters = {{.parameter_id = "weight",
                           .parameter = std::move(parameter)}},
           .options = {.learning_rate = 0.2}}});
}

TEST(SchedulerAdapterTest, RejectsWrongDuplicateAndSkippedEpochEvents) {
  auto optimizer = make_optimizer();
  auto scheduler = std::make_shared<jyppx::optim::StepLrScheduler>(
      optimizer,
      jyppx::optim::StepLrSchedulerOptions{.step_size = 1, .gamma = 0.5});
  jyppx::training::SchedulerAdapter adapter(scheduler);
  const auto initial_rates = scheduler->learning_rates();

  EXPECT_EQ(adapter.mode(), jyppx::optim::SchedulerStepMode::epoch);
  EXPECT_THROW(adapter.step({.trigger = jyppx::optim::SchedulerStepMode::batch,
                             .event_index = 1}),
               jyppx::InvalidArgument);
  EXPECT_THROW(adapter.step({.trigger = jyppx::optim::SchedulerStepMode::epoch,
                             .event_index = 2}),
               jyppx::InvalidArgument);
  EXPECT_THROW(adapter.step({.trigger = jyppx::optim::SchedulerStepMode::epoch,
                             .event_index = 1,
                             .metric = 1.0}),
               jyppx::InvalidArgument);
  EXPECT_EQ(scheduler->completed_steps(), 0U);
  EXPECT_EQ(scheduler->learning_rates(), initial_rates);

  adapter.step(
      {.trigger = jyppx::optim::SchedulerStepMode::epoch, .event_index = 1});
  EXPECT_EQ(scheduler->completed_steps(), 1U);
  EXPECT_EQ(scheduler->learning_rates(), (std::vector<double>{0.1}));
  EXPECT_THROW(adapter.step({.trigger = jyppx::optim::SchedulerStepMode::epoch,
                             .event_index = 1}),
               jyppx::InvalidArgument);
  EXPECT_THROW(adapter.step({.trigger = jyppx::optim::SchedulerStepMode::epoch,
                             .event_index = 3}),
               jyppx::InvalidArgument);
  EXPECT_EQ(scheduler->completed_steps(), 1U);
}

TEST(SchedulerAdapterTest, DeliversStrictBatchEventsToFiniteScheduler) {
  auto scheduler = std::make_shared<jyppx::optim::OneCycleLrScheduler>(
      make_optimizer(), jyppx::optim::OneCycleLrSchedulerOptions{
                            .max_learning_rates = {0.4}, .total_steps = 2});
  jyppx::training::SchedulerAdapter adapter(scheduler);

  EXPECT_EQ(adapter.mode(), jyppx::optim::SchedulerStepMode::batch);
  adapter.step(
      {.trigger = jyppx::optim::SchedulerStepMode::batch, .event_index = 1});
  adapter.step(
      {.trigger = jyppx::optim::SchedulerStepMode::batch, .event_index = 2});
  EXPECT_EQ(scheduler->completed_steps(), 2U);
  EXPECT_THROW(adapter.step({.trigger = jyppx::optim::SchedulerStepMode::batch,
                             .event_index = 3}),
               jyppx::ResourceExhausted);
}

TEST(SchedulerAdapterTest, RequiresOneFiniteMetricPerMetricEvent) {
  auto scheduler = std::make_shared<jyppx::optim::ReduceLrOnPlateauScheduler>(
      make_optimizer(), jyppx::optim::ReduceLrOnPlateauSchedulerOptions{
                            .factor = 0.5, .patience = 0});
  jyppx::training::SchedulerAdapter adapter(scheduler);

  EXPECT_THROW(adapter.step({.trigger = jyppx::optim::SchedulerStepMode::metric,
                             .event_index = 1}),
               jyppx::InvalidArgument);
  EXPECT_THROW(
      adapter.step({.trigger = jyppx::optim::SchedulerStepMode::metric,
                    .event_index = 1,
                    .metric = std::numeric_limits<double>::quiet_NaN()}),
      jyppx::InvalidArgument);
  EXPECT_EQ(scheduler->completed_steps(), 0U);

  adapter.step({.trigger = jyppx::optim::SchedulerStepMode::metric,
                .event_index = 1,
                .metric = 1.0});
  adapter.step({.trigger = jyppx::optim::SchedulerStepMode::metric,
                .event_index = 2,
                .metric = 2.0});
  EXPECT_EQ(scheduler->completed_steps(), 2U);
  EXPECT_EQ(scheduler->learning_rates(), (std::vector<double>{0.1}));
}

TEST(SchedulerAdapterTest, DelegatesExactCheckpointRoundTrip) {
  auto source_scheduler = std::make_shared<jyppx::optim::StepLrScheduler>(
      make_optimizer(),
      jyppx::optim::StepLrSchedulerOptions{
          .scheduler_id = "scheduler.main", .step_size = 1, .gamma = 0.5});
  jyppx::training::SchedulerAdapter source(source_scheduler);
  source.step(
      {.trigger = jyppx::optim::SchedulerStepMode::epoch, .event_index = 1});
  MemoryStateStore store(source.describe_state());
  source.save_state(store);

  auto restored_scheduler = std::make_shared<jyppx::optim::StepLrScheduler>(
      make_optimizer(),
      jyppx::optim::StepLrSchedulerOptions{
          .scheduler_id = "scheduler.main", .step_size = 1, .gamma = 0.5});
  jyppx::training::SchedulerAdapter restored(restored_scheduler);
  EXPECT_EQ(restored.participant_id(), "scheduler.main");
  restored.load_state(store, jyppx::RestorePolicy::exact_only);
  EXPECT_EQ(restored_scheduler->completed_steps(), 1U);
  EXPECT_EQ(restored_scheduler->learning_rates(), (std::vector<double>{0.1}));

  restored.step(
      {.trigger = jyppx::optim::SchedulerStepMode::epoch, .event_index = 2});
  EXPECT_EQ(restored_scheduler->learning_rates(), (std::vector<double>{0.05}));
}

}  // namespace
