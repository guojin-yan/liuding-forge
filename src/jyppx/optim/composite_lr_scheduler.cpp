// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/optim/composite_lr_scheduler.hpp>

#include <algorithm>
#include <limits>
#include <set>
#include <string_view>
#include <utility>

namespace jyppx::optim {
namespace {

constexpr char kComponent[] = "optim.composite_lr_scheduler";
constexpr char kSequentialTypeId[] = "jyppx.scheduler.sequential_lr/v1";
constexpr char kChainedTypeId[] = "jyppx.scheduler.chained_lr/v1";

[[noreturn]] void throw_invalid(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
  throw InvalidArgument("Invalid composite scheduler structure",
                        ErrorDetails{.component = kComponent,
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

[[noreturn]] void throw_compatibility(std::string operation,
                                      std::string expected, std::string actual,
                                      std::string hint) {
  throw CompatibilityError("Composite scheduler state is incompatible",
                           ErrorDetails{.component = kComponent,
                                        .operation = std::move(operation),
                                        .expected = std::move(expected),
                                        .actual = std::move(actual),
                                        .hint = std::move(hint)});
}

[[noreturn]] void throw_counter_exhausted(std::string_view type_id) {
  throw ResourceExhausted(
      "Composite scheduler counter exhausted",
      ErrorDetails{.component = kComponent,
                   .operation = std::string(type_id) + ".completed_steps",
                   .expected = "a count below uint64 maximum",
                   .actual = "uint64 maximum",
                   .hint = "Start a new training run identity."});
}

struct ChildMetadata {
  SchedulerStepMode step_mode;  ///< Common event mode for every child.
  std::vector<std::string> scheduler_ids;  ///< Ordered stable child identities.
  std::vector<SchedulerDescriptor> descriptors;  ///< Ordered child topology.
};

ChildMetadata validate_children(
    const LrScheduler& parent,
    const std::vector<std::unique_ptr<LrScheduler>>& children) {
  if (children.empty()) {
    throw_invalid("children", "at least one uniquely owned scheduler", "empty",
                  "Provide an ordered child scheduler list.");
  }
  const auto& expected_optimizer =
      detail::SchedulerStateAccess::optimizer(parent);
  std::set<std::string> scheduler_ids;
  scheduler_ids.insert(parent.scheduler_id());
  ChildMetadata metadata{
      .step_mode = SchedulerStepMode::manual,
      .scheduler_ids = {},
      .descriptors = {},
  };
  metadata.scheduler_ids.reserve(children.size());
  metadata.descriptors.reserve(children.size());
  for (std::size_t index = 0; index < children.size(); ++index) {
    const auto& child = children[index];
    if (!child) {
      throw_invalid("children", "non-null child schedulers", "null",
                    "Remove null children before construction.");
    }
    if (detail::SchedulerStateAccess::optimizer(*child).get() !=
        expected_optimizer.get()) {
      throw_invalid("children.optimizer", "the exact parent optimizer object",
                    child->descriptor().optimizer_id,
                    "Construct every child from the parent optimizer.");
    }
    if (!scheduler_ids.insert(child->scheduler_id()).second) {
      throw_invalid("children.scheduler_id", "unique child and parent IDs",
                    child->scheduler_id(),
                    "Assign every scheduler node a stable unique ID.");
    }
    if (index == 0) {
      metadata.step_mode = child->descriptor().step_mode;
      if (metadata.step_mode == SchedulerStepMode::metric) {
        throw_invalid("children.step_mode", "batch, epoch, or manual", "metric",
                      "Use a metric-aware composite implementation.");
      }
    } else if (child->descriptor().step_mode != metadata.step_mode) {
      throw_invalid(
          "children.step_mode", "one common child step mode",
          std::string(scheduler_step_mode_name(child->descriptor().step_mode)),
          "Do not mix batch, epoch, metric, or manual events.");
    }
    metadata.scheduler_ids.push_back(child->scheduler_id());
    metadata.descriptors.push_back(child->descriptor());
  }
  return metadata;
}

detail::SchedulerState capture_children(
    std::uint64_t completed_steps,
    const std::vector<std::unique_ptr<LrScheduler>>& children) {
  detail::SchedulerState state{
      .completed_steps = completed_steps,
      .children = {},
  };
  state.children.reserve(children.size());
  for (const auto& child : children) {
    state.children.push_back(
        detail::SchedulerStateAccess::capture_state(*child));
  }
  return state;
}

void validate_child_state(
    const std::vector<std::unique_ptr<LrScheduler>>& children,
    const detail::SchedulerState& state) {
  if (state.children.size() != children.size()) {
    throw_compatibility("state.children", std::to_string(children.size()),
                        std::to_string(state.children.size()),
                        "Restore the exact ordered child tree.");
  }
  for (std::size_t index = 0; index < children.size(); ++index) {
    detail::SchedulerStateAccess::validate_state(*children[index],
                                                 state.children[index]);
  }
}

void apply_child_state(std::vector<std::unique_ptr<LrScheduler>>& children,
                       const detail::SchedulerState& state) noexcept {
  for (std::size_t index = 0; index < children.size(); ++index) {
    detail::SchedulerStateAccess::apply_state(*children[index],
                                              state.children[index]);
  }
}

}  // namespace

SequentialLrScheduler::SequentialLrScheduler(
    std::shared_ptr<OptimizerRateControl> optimizer,
    std::vector<std::unique_ptr<LrScheduler>> children,
    SequentialLrSchedulerOptions options)
    : LrScheduler(std::move(optimizer), options.scheduler_id),
      children_(std::move(children)),
      milestones_(std::move(options.milestones)) {
  const auto metadata = validate_children(*this, children_);
  if (milestones_.size() + 1 != children_.size()) {
    throw_invalid("milestones", "exactly one fewer milestone than children",
                  std::to_string(milestones_.size()),
                  "Define one boundary between each adjacent child.");
  }
  for (std::size_t index = 0; index < milestones_.size(); ++index) {
    if (milestones_[index] <= 0 ||
        (index > 0 && milestones_[index] <= milestones_[index - 1])) {
      throw_invalid("milestones", "positive strictly increasing values",
                    std::to_string(milestones_[index]),
                    "Use total committed parent-event boundaries.");
    }
  }
  configure_descriptor(kSequentialTypeId, metadata.step_mode,
                       {{"base_learning_rates", base_learning_rates()},
                        {"child_scheduler_ids", metadata.scheduler_ids},
                        {"milestones", milestones_}},
                       metadata.descriptors);
}

SequentialLrScheduler::~SequentialLrScheduler() = default;

std::size_t SequentialLrScheduler::child_count() const noexcept {
  return children_.size();
}

std::size_t SequentialLrScheduler::next_child_index() const noexcept {
  std::size_t index = 0;
  while (index < milestones_.size() &&
         completed_steps_ >= static_cast<std::uint64_t>(milestones_[index])) {
    ++index;
  }
  return index;
}

detail::SchedulerTransition SequentialLrScheduler::preview_step_impl(
    std::span<const double> learning_rates) const {
  if (completed_steps_ == std::numeric_limits<std::uint64_t>::max()) {
    throw_counter_exhausted(kSequentialTypeId);
  }
  auto state = capture_state_impl();
  const auto active = next_child_index();
  auto child_transition = detail::SchedulerStateAccess::preview_step(
      *children_[active], learning_rates);
  state.completed_steps = completed_steps_ + 1;
  state.children[active] = std::move(child_transition.state);
  return {.state = std::move(state),
          .learning_rates = std::move(child_transition.learning_rates)};
}

detail::SchedulerState SequentialLrScheduler::capture_state_impl() const {
  return capture_children(completed_steps_, children_);
}

std::uint64_t SequentialLrScheduler::completed_steps_impl() const noexcept {
  return completed_steps_;
}

void SequentialLrScheduler::validate_state_impl(
    const detail::SchedulerState& state) const {
  validate_child_state(children_, state);
}

void SequentialLrScheduler::apply_state_impl(
    const detail::SchedulerState& state) noexcept {
  apply_child_state(children_, state);
  completed_steps_ = state.completed_steps;
}

ChainedLrScheduler::ChainedLrScheduler(
    std::shared_ptr<OptimizerRateControl> optimizer,
    std::vector<std::unique_ptr<LrScheduler>> children,
    ChainedLrSchedulerOptions options)
    : LrScheduler(std::move(optimizer), std::move(options.scheduler_id)),
      children_(std::move(children)) {
  const auto metadata = validate_children(*this, children_);
  configure_descriptor(kChainedTypeId, metadata.step_mode,
                       {{"base_learning_rates", base_learning_rates()},
                        {"child_scheduler_ids", metadata.scheduler_ids}},
                       metadata.descriptors);
}

ChainedLrScheduler::~ChainedLrScheduler() = default;

std::size_t ChainedLrScheduler::child_count() const noexcept {
  return children_.size();
}

detail::SchedulerTransition ChainedLrScheduler::preview_step_impl(
    std::span<const double> learning_rates) const {
  if (completed_steps_ == std::numeric_limits<std::uint64_t>::max()) {
    throw_counter_exhausted(kChainedTypeId);
  }
  auto state = capture_state_impl();
  std::vector<double> rates(learning_rates.begin(), learning_rates.end());
  for (std::size_t index = 0; index < children_.size(); ++index) {
    auto child_transition =
        detail::SchedulerStateAccess::preview_step(*children_[index], rates);
    state.children[index] = std::move(child_transition.state);
    rates = std::move(child_transition.learning_rates);
  }
  state.completed_steps = completed_steps_ + 1;
  return {.state = std::move(state), .learning_rates = std::move(rates)};
}

detail::SchedulerState ChainedLrScheduler::capture_state_impl() const {
  return capture_children(completed_steps_, children_);
}

std::uint64_t ChainedLrScheduler::completed_steps_impl() const noexcept {
  return completed_steps_;
}

void ChainedLrScheduler::validate_state_impl(
    const detail::SchedulerState& state) const {
  validate_child_state(children_, state);
}

void ChainedLrScheduler::apply_state_impl(
    const detail::SchedulerState& state) noexcept {
  apply_child_state(children_, state);
  completed_steps_ = state.completed_steps;
}

}  // namespace jyppx::optim
