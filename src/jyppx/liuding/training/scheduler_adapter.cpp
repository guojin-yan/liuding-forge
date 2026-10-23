// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/training/scheduler_adapter.hpp>

#include <cmath>
#include <limits>
#include <string>
#include <utility>

namespace jyppx::liuding::training {
namespace {

constexpr char kComponent[] = "training.scheduler_adapter";

[[noreturn]] void throw_invalid(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
  throw InvalidArgument("Invalid training scheduler event",
                        ErrorDetails{.component = kComponent,
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

[[noreturn]] void throw_event_exhausted() {
  throw ResourceExhausted(
      "Training scheduler event sequence exhausted",
      ErrorDetails{.component = kComponent,
                   .operation = "step.event_index",
                   .expected = "a completed-step count below uint64 maximum",
                   .actual = "uint64 maximum",
                   .hint = "Start a new training run identity."});
}

}  // namespace

SchedulerAdapter::SchedulerAdapter(
    std::shared_ptr<optim::LrScheduler> scheduler,
    optim::LrSchedulerCheckpointOptions checkpoint_options)
    : scheduler_(scheduler),
      participant_(std::move(scheduler), checkpoint_options) {}

optim::SchedulerStepMode SchedulerAdapter::mode() const noexcept {
  return scheduler_->descriptor().step_mode;
}

void SchedulerAdapter::step(const SchedulerInput& input) {
  const auto expected_mode = mode();
  if (input.trigger != expected_mode) {
    throw_invalid(
        "step.trigger",
        std::string(optim::scheduler_step_mode_name(expected_mode)),
        std::string(optim::scheduler_step_mode_name(input.trigger)),
        "Deliver the scheduler only from its declared lifecycle hook.");
  }
  const auto completed = scheduler_->completed_steps();
  if (completed == std::numeric_limits<std::uint64_t>::max()) {
    throw_event_exhausted();
  }
  const auto expected_index = completed + 1;
  if (input.event_index != expected_index) {
    throw_invalid("step.event_index", std::to_string(expected_index),
                  std::to_string(input.event_index),
                  "Reject duplicate or skipped lifecycle events.");
  }

  if (expected_mode == optim::SchedulerStepMode::metric) {
    if (!input.metric.has_value() || !std::isfinite(*input.metric)) {
      throw_invalid(
          "step.metric", "one finite monitored metric",
          input.metric.has_value() ? std::to_string(*input.metric) : "missing",
          "Supply the validated metric from the evaluation event.");
    }
    scheduler_->step(*input.metric);
    return;
  }
  if (input.metric.has_value()) {
    throw_invalid("step.metric", "no metric for a non-metric scheduler",
                  std::to_string(*input.metric),
                  "Remove metric data from batch, epoch, or manual events.");
  }
  scheduler_->step();
}

optim::LrScheduler& SchedulerAdapter::scheduler() noexcept {
  return *scheduler_;
}

const optim::LrScheduler& SchedulerAdapter::scheduler() const noexcept {
  return *scheduler_;
}

std::string SchedulerAdapter::participant_id() const {
  return participant_.participant_id();
}

StateManifest SchedulerAdapter::describe_state() const {
  return participant_.describe_state();
}

void SchedulerAdapter::save_state(StateWriter& writer) const {
  participant_.save_state(writer);
}

void SchedulerAdapter::load_state(const StateReader& reader,
                                  RestorePolicy policy) {
  participant_.load_state(reader, policy);
}

}  // namespace jyppx::liuding::training
