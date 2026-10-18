// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/optim/linear_lr_scheduler.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

namespace jyppx::optim {
namespace {

constexpr char kComponent[] = "optim.linear_lr_scheduler";
constexpr char kLinearTypeId[] = "jyppx.scheduler.linear_lr/v1";
constexpr char kConstantTypeId[] = "jyppx.scheduler.constant_lr/v1";
constexpr char kPolynomialTypeId[] = "jyppx.scheduler.polynomial_lr/v1";
constexpr char kWarmupTypeId[] = "jyppx.scheduler.warmup_lr/v1";

[[noreturn]] void throw_invalid(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
  throw InvalidArgument("Invalid linear-family scheduler configuration",
                        ErrorDetails{.component = kComponent,
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

[[noreturn]] void throw_compatibility(std::string operation,
                                      std::string expected, std::string actual,
                                      std::string hint) {
  throw CompatibilityError("Linear-family scheduler state is incompatible",
                           ErrorDetails{.component = kComponent,
                                        .operation = std::move(operation),
                                        .expected = std::move(expected),
                                        .actual = std::move(actual),
                                        .hint = std::move(hint)});
}

[[noreturn]] void throw_numerical(std::string operation, std::string actual,
                                  std::string hint) {
  throw NumericalError(
      "Linear-family scheduler produced an invalid rate",
      ErrorDetails{.component = kComponent,
                   .operation = std::move(operation),
                   .expected = "finite non-negative learning rates",
                   .actual = std::move(actual),
                   .hint = std::move(hint)});
}

void validate_positive_factor(std::string_view name, double value) {
  if (!std::isfinite(value) || value <= 0.0) {
    throw_invalid(
        std::string(name), "a finite positive factor", std::to_string(value),
        "Use a factor greater than zero so closed-form ratios are defined.");
  }
}

void validate_non_negative_power(double value) {
  if (!std::isfinite(value) || value < 0.0) {
    throw_invalid("power", "a finite non-negative exponent",
                  std::to_string(value),
                  "Use a finite exponent at or above zero.");
  }
}

std::int64_t descriptor_iters(std::uint64_t value, std::string_view name) {
  if (value >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    throw_invalid(
        std::string(name), "an int64-compatible positive value",
        std::to_string(value),
        "Use a smaller event interval for the current descriptor schema.");
  }
  return static_cast<std::int64_t>(value);
}

std::uint64_t next_step(std::uint64_t completed_steps,
                        std::string_view scheduler_type) {
  if (completed_steps == std::numeric_limits<std::uint64_t>::max()) {
    throw ResourceExhausted(
        "Linear-family scheduler counter exhausted",
        ErrorDetails{
            .component = kComponent,
            .operation = std::string(scheduler_type) + ".completed_steps",
            .expected = "a count below uint64 maximum",
            .actual = "uint64 maximum",
            .hint = "Start a new training run identity."});
  }
  return completed_steps + 1;
}

double linear_factor(double start_factor, double end_factor,
                     std::uint64_t total_iters, std::uint64_t step) {
  if (total_iters == 0) {
    return step == 0 ? start_factor : end_factor;
  }
  if (step >= total_iters) {
    return end_factor;
  }
  const double progress = std::min(
      1.0, static_cast<double>(step) / static_cast<double>(total_iters));
  return start_factor + (end_factor - start_factor) * progress;
}

double polynomial_factor(std::uint64_t total_iters, double power,
                         std::uint64_t step) {
  const double progress = std::min(
      1.0, static_cast<double>(step) / static_cast<double>(total_iters));
  return std::pow(std::max(0.0, 1.0 - progress), power);
}

std::vector<double> scale_rates(std::span<const double> learning_rates,
                                double previous_factor, double next_factor,
                                std::string_view scheduler_type) {
  if (!std::isfinite(previous_factor) || previous_factor < 0.0 ||
      !std::isfinite(next_factor) || next_factor < 0.0) {
    throw_numerical(std::string(scheduler_type) + ".factor",
                    std::to_string(next_factor),
                    "Reduce the factor range or use finite scheduler options.");
  }

  std::vector<double> candidate;
  candidate.reserve(learning_rates.size());
  if (previous_factor == 0.0) {
    if (next_factor != 0.0) {
      throw_numerical(
          std::string(scheduler_type) + ".factor_transition",
          std::to_string(next_factor),
          "A zero polynomial factor cannot recover without a new scheduler.");
    }
    candidate.assign(learning_rates.size(), 0.0);
    return candidate;
  }

  const double ratio = next_factor / previous_factor;
  if (!std::isfinite(ratio) || ratio < 0.0) {
    throw_numerical(std::string(scheduler_type) + ".factor_ratio",
                    std::to_string(ratio),
                    "Choose factors whose transition is finite.");
  }
  for (const double rate : learning_rates) {
    const double next_rate = rate * ratio;
    if (!std::isfinite(next_rate) || next_rate < 0.0) {
      throw_numerical(std::string(scheduler_type) + ".learning_rate",
                      std::to_string(next_rate),
                      "Reduce the factor or current optimizer rate.");
    }
    candidate.push_back(next_rate);
  }
  return candidate;
}

void validate_leaf(const detail::SchedulerState& state,
                   std::string_view scheduler_type) {
  if (!state.control.empty() || !state.children.empty()) {
    throw_compatibility("state.shape", "a counter-only leaf state",
                        "control values or children present",
                        "Restore state produced by the same scheduler type.");
  }
  static_cast<void>(scheduler_type);
}

}  // namespace

LinearLrScheduler::LinearLrScheduler(
    std::shared_ptr<OptimizerRateControl> optimizer,
    LinearLrSchedulerOptions options)
    : LrScheduler(optimizer, options.scheduler_id),
      options_(std::move(options)) {
  validate_positive_factor("start_factor", options_.start_factor);
  validate_positive_factor("end_factor", options_.end_factor);
  const auto total_iters =
      descriptor_iters(options_.total_iters, "total_iters");
  configure_descriptor(kLinearTypeId, SchedulerStepMode::epoch,
                       {{"base_learning_rates", base_learning_rates()},
                        {"end_factor", options_.end_factor},
                        {"start_factor", options_.start_factor},
                        {"total_iters", total_iters}});

  std::vector<double> initial;
  initial.reserve(base_learning_rates().size());
  for (const double base_rate : base_learning_rates()) {
    initial.push_back(base_rate * options_.start_factor);
  }
  initialize_learning_rates(initial);
}

LinearLrScheduler::~LinearLrScheduler() = default;

detail::SchedulerTransition LinearLrScheduler::preview_step_impl(
    std::span<const double> learning_rates) const {
  const auto next = next_step(completed_steps_, kLinearTypeId);
  const double previous_factor =
      linear_factor(options_.start_factor, options_.end_factor,
                    options_.total_iters, completed_steps_);
  const double next_factor = linear_factor(
      options_.start_factor, options_.end_factor, options_.total_iters, next);
  return {.state = {.completed_steps = next, .control = {}, .children = {}},
          .learning_rates = scale_rates(learning_rates, previous_factor,
                                        next_factor, kLinearTypeId)};
}

detail::SchedulerState LinearLrScheduler::capture_state_impl() const {
  return {.completed_steps = completed_steps_, .control = {}, .children = {}};
}

std::uint64_t LinearLrScheduler::completed_steps_impl() const noexcept {
  return completed_steps_;
}

void LinearLrScheduler::validate_state_impl(
    const detail::SchedulerState& state) const {
  validate_leaf(state, kLinearTypeId);
}

void LinearLrScheduler::apply_state_impl(
    const detail::SchedulerState& state) noexcept {
  completed_steps_ = state.completed_steps;
}

ConstantLrScheduler::ConstantLrScheduler(
    std::shared_ptr<OptimizerRateControl> optimizer,
    ConstantLrSchedulerOptions options)
    : LrScheduler(optimizer, options.scheduler_id),
      options_(std::move(options)) {
  validate_positive_factor("factor", options_.factor);
  const auto total_iters =
      descriptor_iters(options_.total_iters, "total_iters");
  configure_descriptor(kConstantTypeId, SchedulerStepMode::epoch,
                       {{"base_learning_rates", base_learning_rates()},
                        {"factor", options_.factor},
                        {"total_iters", total_iters}});

  std::vector<double> initial;
  initial.reserve(base_learning_rates().size());
  for (const double base_rate : base_learning_rates()) {
    initial.push_back(base_rate * options_.factor);
  }
  initialize_learning_rates(initial);
}

ConstantLrScheduler::~ConstantLrScheduler() = default;

detail::SchedulerTransition ConstantLrScheduler::preview_step_impl(
    std::span<const double> learning_rates) const {
  const auto next = next_step(completed_steps_, kConstantTypeId);
  const double previous_factor =
      completed_steps_ <= options_.total_iters ? options_.factor : 1.0;
  const double next_factor =
      next <= options_.total_iters ? options_.factor : 1.0;
  return {.state = {.completed_steps = next, .control = {}, .children = {}},
          .learning_rates = scale_rates(learning_rates, previous_factor,
                                        next_factor, kConstantTypeId)};
}

detail::SchedulerState ConstantLrScheduler::capture_state_impl() const {
  return {.completed_steps = completed_steps_, .control = {}, .children = {}};
}

std::uint64_t ConstantLrScheduler::completed_steps_impl() const noexcept {
  return completed_steps_;
}

void ConstantLrScheduler::validate_state_impl(
    const detail::SchedulerState& state) const {
  validate_leaf(state, kConstantTypeId);
}

void ConstantLrScheduler::apply_state_impl(
    const detail::SchedulerState& state) noexcept {
  completed_steps_ = state.completed_steps;
}

PolynomialLrScheduler::PolynomialLrScheduler(
    std::shared_ptr<OptimizerRateControl> optimizer,
    PolynomialLrSchedulerOptions options)
    : LrScheduler(optimizer, options.scheduler_id),
      options_(std::move(options)) {
  if (options_.total_iters == 0) {
    throw_invalid("total_iters", "a positive interval", "0",
                  "Use at least one event for polynomial decay.");
  }
  const auto total_iters =
      descriptor_iters(options_.total_iters, "total_iters");
  validate_non_negative_power(options_.power);
  configure_descriptor(kPolynomialTypeId, SchedulerStepMode::epoch,
                       {{"base_learning_rates", base_learning_rates()},
                        {"power", options_.power},
                        {"total_iters", total_iters}});
}

PolynomialLrScheduler::~PolynomialLrScheduler() = default;

detail::SchedulerTransition PolynomialLrScheduler::preview_step_impl(
    std::span<const double> learning_rates) const {
  const auto next = next_step(completed_steps_, kPolynomialTypeId);
  const double previous_factor =
      polynomial_factor(options_.total_iters, options_.power, completed_steps_);
  const double next_factor =
      polynomial_factor(options_.total_iters, options_.power, next);
  return {.state = {.completed_steps = next, .control = {}, .children = {}},
          .learning_rates = scale_rates(learning_rates, previous_factor,
                                        next_factor, kPolynomialTypeId)};
}

detail::SchedulerState PolynomialLrScheduler::capture_state_impl() const {
  return {.completed_steps = completed_steps_, .control = {}, .children = {}};
}

std::uint64_t PolynomialLrScheduler::completed_steps_impl() const noexcept {
  return completed_steps_;
}

void PolynomialLrScheduler::validate_state_impl(
    const detail::SchedulerState& state) const {
  validate_leaf(state, kPolynomialTypeId);
}

void PolynomialLrScheduler::apply_state_impl(
    const detail::SchedulerState& state) noexcept {
  completed_steps_ = state.completed_steps;
}

WarmupLrScheduler::WarmupLrScheduler(
    std::shared_ptr<OptimizerRateControl> optimizer,
    WarmupLrSchedulerOptions options)
    : LrScheduler(optimizer, options.scheduler_id),
      options_(std::move(options)) {
  if (options_.warmup_iters == 0) {
    throw_invalid("warmup_iters", "a positive interval", "0",
                  "Use at least one event to reach the base rates.");
  }
  const auto warmup_iters =
      descriptor_iters(options_.warmup_iters, "warmup_iters");
  validate_positive_factor("start_factor", options_.start_factor);
  configure_descriptor(kWarmupTypeId, SchedulerStepMode::epoch,
                       {{"base_learning_rates", base_learning_rates()},
                        {"start_factor", options_.start_factor},
                        {"warmup_iters", warmup_iters}});

  std::vector<double> initial;
  initial.reserve(base_learning_rates().size());
  for (const double base_rate : base_learning_rates()) {
    initial.push_back(base_rate * options_.start_factor);
  }
  initialize_learning_rates(initial);
}

WarmupLrScheduler::~WarmupLrScheduler() = default;

detail::SchedulerTransition WarmupLrScheduler::preview_step_impl(
    std::span<const double> learning_rates) const {
  const auto next = next_step(completed_steps_, kWarmupTypeId);
  const double previous_factor = linear_factor(
      options_.start_factor, 1.0, options_.warmup_iters, completed_steps_);
  const double next_factor =
      linear_factor(options_.start_factor, 1.0, options_.warmup_iters, next);
  return {.state = {.completed_steps = next, .control = {}, .children = {}},
          .learning_rates = scale_rates(learning_rates, previous_factor,
                                        next_factor, kWarmupTypeId)};
}

detail::SchedulerState WarmupLrScheduler::capture_state_impl() const {
  return {.completed_steps = completed_steps_, .control = {}, .children = {}};
}

std::uint64_t WarmupLrScheduler::completed_steps_impl() const noexcept {
  return completed_steps_;
}

void WarmupLrScheduler::validate_state_impl(
    const detail::SchedulerState& state) const {
  validate_leaf(state, kWarmupTypeId);
}

void WarmupLrScheduler::apply_state_impl(
    const detail::SchedulerState& state) noexcept {
  completed_steps_ = state.completed_steps;
}

}  // namespace jyppx::optim
