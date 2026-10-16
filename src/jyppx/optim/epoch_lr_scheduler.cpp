// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/optim/epoch_lr_scheduler.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace jyppx::optim {
namespace {

constexpr char kComponent[] = "optim.epoch_lr_scheduler";
constexpr char kExponentialTypeId[] = "jyppx.scheduler.exponential_lr/v1";
constexpr char kMultiStepTypeId[] = "jyppx.scheduler.multi_step_lr/v1";
constexpr char kCosineTypeId[] = "jyppx.scheduler.cosine_annealing_lr/v1";
constexpr double kPi = 3.141592653589793238462643383279502884;

[[noreturn]] void throw_invalid(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
  throw InvalidArgument("Invalid epoch learning-rate scheduler configuration",
                        ErrorDetails{.component = kComponent,
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

[[noreturn]] void throw_numerical(std::string scheduler_type, double value) {
  throw NumericalError(
      "Epoch scheduler produced an invalid learning rate",
      ErrorDetails{.component = kComponent,
                   .operation = std::move(scheduler_type) + ".learning_rate",
                   .expected = "a finite non-negative rate",
                   .actual = std::to_string(value),
                   .hint = "Reduce the multiplier or current learning rate."});
}

[[noreturn]] void throw_counter_exhausted(std::string scheduler_type) {
  throw ResourceExhausted(
      "Epoch scheduler counter exhausted",
      ErrorDetails{.component = kComponent,
                   .operation = std::move(scheduler_type) + ".completed_steps",
                   .expected = "a count below uint64 maximum",
                   .actual = "uint64 maximum",
                   .hint = "Start a new training run identity."});
}

[[noreturn]] void throw_non_leaf(std::string scheduler_type) {
  throw CompatibilityError(
      "Scheduler checkpoint state tree is incompatible",
      ErrorDetails{.component = kComponent,
                   .operation = std::move(scheduler_type) + ".state.children",
                   .expected = "a leaf scheduler state",
                   .actual = "children present",
                   .hint = "Restore state from the same scheduler type."});
}

std::uint64_t next_step(std::uint64_t completed_steps,
                        const char* scheduler_type) {
  if (completed_steps == std::numeric_limits<std::uint64_t>::max()) {
    throw_counter_exhausted(scheduler_type);
  }
  return completed_steps + 1;
}

void validate_rate(double value, const char* scheduler_type) {
  if (!std::isfinite(value) || value < 0.0) {
    throw_numerical(scheduler_type, value);
  }
}

void validate_leaf(const detail::SchedulerState& state,
                   const char* scheduler_type) {
  if (!state.children.empty()) {
    throw_non_leaf(scheduler_type);
  }
}

std::vector<std::int64_t> encode_milestones(
    const std::vector<std::uint64_t>& milestones) {
  std::vector<std::int64_t> encoded;
  encoded.reserve(milestones.size());
  for (const auto milestone : milestones) {
    encoded.push_back(static_cast<std::int64_t>(milestone));
  }
  return encoded;
}

double cosine_weight(std::uint64_t step, std::uint64_t t_max) {
  const auto quotient = step / t_max;
  const auto remainder = step % t_max;
  double phase = static_cast<double>(remainder) / static_cast<double>(t_max);
  if ((quotient & 1U) != 0U) {
    phase = 1.0 + phase;
  }
  const auto half_cosine = std::cos(0.5 * kPi * phase);
  return half_cosine * half_cosine;
}

bool follows_cosine_minimum(std::uint64_t next, std::uint64_t t_max) {
  if (next <= t_max) {
    return false;
  }
  const auto previous = next - 1;
  return previous % t_max == 0 && ((previous / t_max) & 1U) != 0U;
}

}  // namespace

ExponentialLrScheduler::ExponentialLrScheduler(
    std::shared_ptr<OptimizerRateControl> optimizer,
    ExponentialLrSchedulerOptions options)
    : LrScheduler(std::move(optimizer), options.scheduler_id),
      options_(std::move(options)) {
  if (!std::isfinite(options_.gamma) || options_.gamma < 0.0) {
    throw_invalid("exponential.gamma", "a finite non-negative multiplier",
                  std::to_string(options_.gamma),
                  "Choose a valid exponential decay multiplier.");
  }
  configure_descriptor(kExponentialTypeId, SchedulerStepMode::epoch,
                       {{"base_learning_rates", base_learning_rates()},
                        {"gamma", options_.gamma}});
}

ExponentialLrScheduler::~ExponentialLrScheduler() = default;

detail::SchedulerTransition ExponentialLrScheduler::preview_step_impl(
    std::span<const double> learning_rates) const {
  const auto next = next_step(completed_steps_, "exponential");
  std::vector<double> candidate(learning_rates.begin(), learning_rates.end());
  for (auto& rate : candidate) {
    rate *= options_.gamma;
    validate_rate(rate, "exponential");
  }
  return {.state = {.completed_steps = next, .children = {}},
          .learning_rates = std::move(candidate)};
}

detail::SchedulerState ExponentialLrScheduler::capture_state_impl() const {
  return {.completed_steps = completed_steps_, .children = {}};
}

std::uint64_t ExponentialLrScheduler::completed_steps_impl() const noexcept {
  return completed_steps_;
}

void ExponentialLrScheduler::validate_state_impl(
    const detail::SchedulerState& state) const {
  validate_leaf(state, "exponential");
}

void ExponentialLrScheduler::apply_state_impl(
    const detail::SchedulerState& state) noexcept {
  completed_steps_ = state.completed_steps;
}

MultiStepLrScheduler::MultiStepLrScheduler(
    std::shared_ptr<OptimizerRateControl> optimizer,
    MultiStepLrSchedulerOptions options)
    : LrScheduler(std::move(optimizer), options.scheduler_id),
      options_(std::move(options)) {
  if (options_.milestones.empty()) {
    throw_invalid("multi_step.milestones",
                  "a nonempty strictly increasing list", "empty",
                  "Declare every completed-epoch decay boundary.");
  }
  std::uint64_t previous = 0;
  for (const auto milestone : options_.milestones) {
    if (milestone == 0 || milestone <= previous ||
        milestone > static_cast<std::uint64_t>(
                        std::numeric_limits<std::int64_t>::max())) {
      throw_invalid("multi_step.milestones",
                    "positive strictly increasing int64-compatible values",
                    std::to_string(milestone),
                    "Sort, deduplicate, and bound the milestone list.");
    }
    previous = milestone;
  }
  if (!std::isfinite(options_.gamma) || options_.gamma < 0.0) {
    throw_invalid("multi_step.gamma", "a finite non-negative multiplier",
                  std::to_string(options_.gamma),
                  "Choose a valid milestone decay multiplier.");
  }
  configure_descriptor(
      kMultiStepTypeId, SchedulerStepMode::epoch,
      {{"base_learning_rates", base_learning_rates()},
       {"gamma", options_.gamma},
       {"milestones", encode_milestones(options_.milestones)}});
}

MultiStepLrScheduler::~MultiStepLrScheduler() = default;

detail::SchedulerTransition MultiStepLrScheduler::preview_step_impl(
    std::span<const double> learning_rates) const {
  const auto next = next_step(completed_steps_, "multi_step");
  std::vector<double> candidate(learning_rates.begin(), learning_rates.end());
  if (std::binary_search(options_.milestones.begin(), options_.milestones.end(),
                         next)) {
    for (auto& rate : candidate) {
      rate *= options_.gamma;
      validate_rate(rate, "multi_step");
    }
  }
  return {.state = {.completed_steps = next, .children = {}},
          .learning_rates = std::move(candidate)};
}

detail::SchedulerState MultiStepLrScheduler::capture_state_impl() const {
  return {.completed_steps = completed_steps_, .children = {}};
}

std::uint64_t MultiStepLrScheduler::completed_steps_impl() const noexcept {
  return completed_steps_;
}

void MultiStepLrScheduler::validate_state_impl(
    const detail::SchedulerState& state) const {
  validate_leaf(state, "multi_step");
}

void MultiStepLrScheduler::apply_state_impl(
    const detail::SchedulerState& state) noexcept {
  completed_steps_ = state.completed_steps;
}

CosineAnnealingLrScheduler::CosineAnnealingLrScheduler(
    std::shared_ptr<OptimizerRateControl> optimizer,
    CosineAnnealingLrSchedulerOptions options)
    : LrScheduler(std::move(optimizer), options.scheduler_id),
      options_(std::move(options)) {
  if (options_.t_max == 0 ||
      options_.t_max > static_cast<std::uint64_t>(
                           std::numeric_limits<std::int64_t>::max())) {
    throw_invalid("cosine.t_max", "a positive int64-compatible value",
                  std::to_string(options_.t_max),
                  "Choose a bounded positive cosine half-period.");
  }
  if (!std::isfinite(options_.eta_min) || options_.eta_min < 0.0) {
    throw_invalid("cosine.eta_min", "a finite non-negative rate",
                  std::to_string(options_.eta_min),
                  "Choose a valid minimum learning rate.");
  }
  for (const auto base_rate : base_learning_rates()) {
    if (options_.eta_min > base_rate) {
      throw_invalid("cosine.eta_min",
                    "no greater than every construction-time group rate",
                    std::to_string(options_.eta_min),
                    "Lower eta_min or raise the affected base rate.");
    }
  }
  configure_descriptor(kCosineTypeId, SchedulerStepMode::epoch,
                       {{"base_learning_rates", base_learning_rates()},
                        {"eta_min", options_.eta_min},
                        {"t_max", static_cast<std::int64_t>(options_.t_max)}});
}

CosineAnnealingLrScheduler::~CosineAnnealingLrScheduler() = default;

detail::SchedulerTransition CosineAnnealingLrScheduler::preview_step_impl(
    std::span<const double> learning_rates) const {
  const auto next = next_step(completed_steps_, "cosine_annealing");
  std::vector<double> candidate;
  candidate.reserve(learning_rates.size());
  if (follows_cosine_minimum(next, options_.t_max)) {
    const auto angle = 0.5 * kPi / static_cast<double>(options_.t_max);
    const auto sine = std::sin(angle);
    const auto increment_weight = sine * sine;
    for (std::size_t index = 0; index < learning_rates.size(); ++index) {
      const auto rate =
          learning_rates[index] +
          (base_learning_rates()[index] - options_.eta_min) * increment_weight;
      validate_rate(rate, "cosine_annealing");
      candidate.push_back(rate);
    }
  } else {
    const auto previous_weight =
        cosine_weight(completed_steps_, options_.t_max);
    const auto next_weight = cosine_weight(next, options_.t_max);
    if (!(previous_weight > 0.0) || !std::isfinite(previous_weight)) {
      throw_numerical("cosine_annealing.weight", previous_weight);
    }
    const auto ratio = next_weight / previous_weight;
    for (const auto current_rate : learning_rates) {
      const auto rate =
          options_.eta_min + (current_rate - options_.eta_min) * ratio;
      validate_rate(rate, "cosine_annealing");
      candidate.push_back(rate);
    }
  }
  return {.state = {.completed_steps = next, .children = {}},
          .learning_rates = std::move(candidate)};
}

detail::SchedulerState CosineAnnealingLrScheduler::capture_state_impl() const {
  return {.completed_steps = completed_steps_, .children = {}};
}

std::uint64_t CosineAnnealingLrScheduler::completed_steps_impl()
    const noexcept {
  return completed_steps_;
}

void CosineAnnealingLrScheduler::validate_state_impl(
    const detail::SchedulerState& state) const {
  validate_leaf(state, "cosine_annealing");
}

void CosineAnnealingLrScheduler::apply_state_impl(
    const detail::SchedulerState& state) noexcept {
  completed_steps_ = state.completed_steps;
}

}  // namespace jyppx::optim
