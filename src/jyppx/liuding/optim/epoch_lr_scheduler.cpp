// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/optim/epoch_lr_scheduler.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace jyppx::liuding::optim {
namespace {

constexpr char kComponent[] = "optim.epoch_lr_scheduler";
constexpr char kExponentialTypeId[] = "jyppx.scheduler.exponential_lr/v1";
constexpr char kMultiStepTypeId[] = "jyppx.scheduler.multi_step_lr/v1";
constexpr char kCosineTypeId[] = "jyppx.scheduler.cosine_annealing_lr/v1";
constexpr char kWarmRestartsTypeId[] =
    "jyppx.scheduler.cosine_annealing_warm_restarts_lr/v2";
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
                   .operation = std::move(scheduler_type) + ".state.shape",
                   .expected = "a counter-only leaf scheduler state",
                   .actual = "control values or children present",
                   .hint = "Restore state from the same scheduler type."});
}

[[noreturn]] void throw_incompatible_state(std::string operation,
                                           std::string expected,
                                           std::string actual,
                                           std::string hint) {
  throw CompatibilityError("Epoch scheduler checkpoint state is incompatible",
                           ErrorDetails{.component = kComponent,
                                        .operation = std::move(operation),
                                        .expected = std::move(expected),
                                        .actual = std::move(actual),
                                        .hint = std::move(hint)});
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
  if (!state.control.empty() || !state.children.empty()) {
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

struct WarmRestartCycleState {
  std::uint64_t index;     ///< Zero-based active cycle index.
  std::uint64_t length;    ///< Positive active cycle length.
  std::uint64_t position;  ///< Completed events within the active cycle.
};

WarmRestartCycleState derive_warm_restart_cycle(
    std::uint64_t completed_steps,
    const CosineAnnealingWarmRestartsLrSchedulerOptions& options) {
  const auto max_control =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  if (options.t_mult == 1) {
    const auto index = completed_steps / options.t_0;
    if (index > max_control) {
      throw_counter_exhausted("cosine_annealing_warm_restarts.cycle_index");
    }
    return {.index = index,
            .length = options.t_0,
            .position = completed_steps % options.t_0};
  }

  auto remaining = completed_steps;
  std::uint64_t index = 0;
  auto length = options.t_0;
  while (remaining >= length) {
    remaining -= length;
    if (index == max_control || length > max_control / options.t_mult) {
      throw_counter_exhausted("cosine_annealing_warm_restarts.cycle");
    }
    ++index;
    length *= options.t_mult;
  }
  return {.index = index, .length = length, .position = remaining};
}

std::int64_t control_count(std::uint64_t value) noexcept {
  return static_cast<std::int64_t>(value);
}

std::uint64_t read_warm_restart_control(const detail::SchedulerState& state,
                                        std::string_view key) {
  const auto found = state.control.find(std::string(key));
  if (found == state.control.end() ||
      !std::holds_alternative<std::int64_t>(found->second)) {
    throw_incompatible_state(
        "cosine_annealing_warm_restarts.state." + std::string(key),
        "an int64 control value",
        found == state.control.end() ? "missing" : "wrong type",
        "Restore state from the same scheduler schema.");
  }
  const auto value = std::get<std::int64_t>(found->second);
  if (value < 0) {
    throw_incompatible_state(
        "cosine_annealing_warm_restarts.state." + std::string(key),
        "a non-negative control value", std::to_string(value),
        "Reject malformed scheduler state before mutation.");
  }
  return static_cast<std::uint64_t>(value);
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
  return {.state = {.completed_steps = next, .control = {}, .children = {}},
          .learning_rates = std::move(candidate)};
}

detail::SchedulerState ExponentialLrScheduler::capture_state_impl() const {
  return {.completed_steps = completed_steps_, .control = {}, .children = {}};
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
  return {.state = {.completed_steps = next, .control = {}, .children = {}},
          .learning_rates = std::move(candidate)};
}

detail::SchedulerState MultiStepLrScheduler::capture_state_impl() const {
  return {.completed_steps = completed_steps_, .control = {}, .children = {}};
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
  return {.state = {.completed_steps = next, .control = {}, .children = {}},
          .learning_rates = std::move(candidate)};
}

detail::SchedulerState CosineAnnealingLrScheduler::capture_state_impl() const {
  return {.completed_steps = completed_steps_, .control = {}, .children = {}};
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

CosineAnnealingWarmRestartsLrScheduler::CosineAnnealingWarmRestartsLrScheduler(
    std::shared_ptr<OptimizerRateControl> optimizer,
    CosineAnnealingWarmRestartsLrSchedulerOptions options)
    : LrScheduler(std::move(optimizer), options.scheduler_id),
      options_(std::move(options)),
      cycle_length_(options_.t_0) {
  const auto max_control =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  if (options_.t_0 == 0 || options_.t_0 > max_control) {
    throw_invalid("cosine_warm_restarts.t_0",
                  "a positive int64-compatible cycle length",
                  std::to_string(options_.t_0),
                  "Choose a bounded positive first-cycle length.");
  }
  if (options_.t_mult == 0 || options_.t_mult > max_control) {
    throw_invalid("cosine_warm_restarts.t_mult",
                  "a positive int64-compatible multiplier",
                  std::to_string(options_.t_mult),
                  "Choose a bounded positive cycle multiplier.");
  }
  if (!std::isfinite(options_.eta_min) || options_.eta_min < 0.0) {
    throw_invalid("cosine_warm_restarts.eta_min", "a finite non-negative rate",
                  std::to_string(options_.eta_min),
                  "Choose a valid minimum learning rate.");
  }
  for (const auto base_rate : base_learning_rates()) {
    if (options_.eta_min > base_rate) {
      throw_invalid("cosine_warm_restarts.eta_min",
                    "no greater than every construction-time group rate",
                    std::to_string(options_.eta_min),
                    "Lower eta_min or raise the affected base rate.");
    }
  }
  configure_descriptor(
      kWarmRestartsTypeId, SchedulerStepMode::epoch,
      {{"base_learning_rates", base_learning_rates()},
       {"eta_min", options_.eta_min},
       {"t_0", static_cast<std::int64_t>(options_.t_0)},
       {"t_mult", static_cast<std::int64_t>(options_.t_mult)}});
}

CosineAnnealingWarmRestartsLrScheduler::
    ~CosineAnnealingWarmRestartsLrScheduler() = default;

std::uint64_t CosineAnnealingWarmRestartsLrScheduler::cycle_index()
    const noexcept {
  return cycle_index_;
}

std::uint64_t CosineAnnealingWarmRestartsLrScheduler::cycle_length()
    const noexcept {
  return cycle_length_;
}

std::uint64_t CosineAnnealingWarmRestartsLrScheduler::cycle_position()
    const noexcept {
  return cycle_position_;
}

detail::SchedulerTransition
CosineAnnealingWarmRestartsLrScheduler::preview_step_impl(
    std::span<const double> learning_rates) const {
  static_cast<void>(learning_rates);
  const auto next =
      next_step(completed_steps_, "cosine_annealing_warm_restarts");
  const auto cycle = derive_warm_restart_cycle(next, options_);
  const auto phase =
      static_cast<double>(cycle.position) / static_cast<double>(cycle.length);
  const auto weight = (1.0 + std::cos(kPi * phase)) / 2.0;
  std::vector<double> candidate;
  candidate.reserve(base_learning_rates().size());
  for (const auto base_rate : base_learning_rates()) {
    const auto rate =
        options_.eta_min + (base_rate - options_.eta_min) * weight;
    validate_rate(rate, "cosine_annealing_warm_restarts");
    candidate.push_back(rate);
  }
  return {
      .state = {.completed_steps = next,
                .control = {{"cycle_index", control_count(cycle.index)},
                            {"cycle_length", control_count(cycle.length)},
                            {"cycle_position", control_count(cycle.position)}},
                .children = {}},
      .learning_rates = std::move(candidate)};
}

detail::SchedulerState
CosineAnnealingWarmRestartsLrScheduler::capture_state_impl() const {
  return {.completed_steps = completed_steps_,
          .control = {{"cycle_index", control_count(cycle_index_)},
                      {"cycle_length", control_count(cycle_length_)},
                      {"cycle_position", control_count(cycle_position_)}},
          .children = {}};
}

std::uint64_t CosineAnnealingWarmRestartsLrScheduler::completed_steps_impl()
    const noexcept {
  return completed_steps_;
}

void CosineAnnealingWarmRestartsLrScheduler::validate_state_impl(
    const detail::SchedulerState& state) const {
  if (!state.children.empty() || state.control.size() != 3) {
    throw_incompatible_state(
        "cosine_annealing_warm_restarts.state.shape",
        "three cycle control values and no children",
        std::to_string(state.control.size()) + " control values and " +
            std::to_string(state.children.size()) + " children",
        "Restore state from the same scheduler type.");
  }
  const auto index = read_warm_restart_control(state, "cycle_index");
  const auto length = read_warm_restart_control(state, "cycle_length");
  const auto position = read_warm_restart_control(state, "cycle_position");
  const auto expected =
      derive_warm_restart_cycle(state.completed_steps, options_);
  if (index != expected.index || length != expected.length ||
      position != expected.position) {
    throw_incompatible_state(
        "cosine_annealing_warm_restarts.state.consistency",
        std::to_string(expected.index) + ":" + std::to_string(expected.length) +
            ":" + std::to_string(expected.position),
        std::to_string(index) + ":" + std::to_string(length) + ":" +
            std::to_string(position),
        "Reject cycle state that disagrees with completed_steps.");
  }
}

void CosineAnnealingWarmRestartsLrScheduler::apply_state_impl(
    const detail::SchedulerState& state) noexcept {
  const auto index = state.control.find("cycle_index");
  const auto length = state.control.find("cycle_length");
  const auto position = state.control.find("cycle_position");
  completed_steps_ = state.completed_steps;
  cycle_index_ =
      static_cast<std::uint64_t>(*std::get_if<std::int64_t>(&index->second));
  cycle_length_ =
      static_cast<std::uint64_t>(*std::get_if<std::int64_t>(&length->second));
  cycle_position_ =
      static_cast<std::uint64_t>(*std::get_if<std::int64_t>(&position->second));
}

}  // namespace jyppx::liuding::optim
