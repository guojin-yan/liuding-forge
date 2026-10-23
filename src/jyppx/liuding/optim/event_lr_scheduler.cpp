// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/optim/event_lr_scheduler.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace jyppx::liuding::optim {
namespace {

constexpr char kComponent[] = "optim.event_lr_scheduler";
constexpr char kPlateauTypeId[] = "jyppx.scheduler.reduce_lr_on_plateau/v2";
constexpr char kOneCycleTypeId[] = "jyppx.scheduler.one_cycle_lr/v2";
constexpr char kCyclicTypeId[] = "jyppx.scheduler.cyclic_lr/v2";
constexpr char kLambdaTypeId[] = "jyppx.scheduler.lambda_lr/v2";
constexpr double kPi = 3.141592653589793238462643383279502884;

[[noreturn]] void throw_invalid(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
  throw InvalidArgument("Invalid event-driven scheduler configuration",
                        ErrorDetails{.component = kComponent,
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

[[noreturn]] void throw_compatibility(std::string operation,
                                      std::string expected, std::string actual,
                                      std::string hint) {
  throw CompatibilityError("Event-driven scheduler state is incompatible",
                           ErrorDetails{.component = kComponent,
                                        .operation = std::move(operation),
                                        .expected = std::move(expected),
                                        .actual = std::move(actual),
                                        .hint = std::move(hint)});
}

[[noreturn]] void throw_numerical(std::string operation, double actual,
                                  std::string hint) {
  throw NumericalError(
      "Event-driven scheduler produced an invalid learning rate",
      ErrorDetails{.component = kComponent,
                   .operation = std::move(operation),
                   .expected = "a finite non-negative learning rate",
                   .actual = std::to_string(actual),
                   .hint = std::move(hint)});
}

[[noreturn]] void throw_counter_exhausted(std::string_view type_id) {
  throw ResourceExhausted(
      "Event-driven scheduler counter exhausted",
      ErrorDetails{.component = kComponent,
                   .operation = std::string(type_id) + ".completed_steps",
                   .expected = "another event within the declared schedule",
                   .actual = "event limit reached",
                   .hint = "Start a new scheduler run identity."});
}

[[noreturn]] void throw_momentum_not_implemented(std::string_view type_id) {
  throw NotImplemented(
      "Atomic momentum cycling is not implemented",
      ErrorDetails{
          .component = kComponent,
          .operation = std::string(type_id) + ".cycle_momentum",
          .expected = "false until optimizer hyperparameter transactions exist",
          .actual = "true",
          .hint = "Disable momentum cycling; learning-rate cycling remains "
                  "exact."});
}

[[noreturn]] void throw_callable_failure(std::string_view type_id,
                                         std::string actual) {
  throw BackendError(
      "Scheduler callable failed",
      ErrorDetails{.component = kComponent,
                   .operation = std::string(type_id) + ".callable",
                   .expected = "a successful finite non-negative result",
                   .actual = std::move(actual),
                   .hint = "Fix the registered callable before retrying."});
}

std::int64_t descriptor_count(std::uint64_t value, std::string_view name,
                              bool allow_zero = false) {
  if ((!allow_zero && value == 0) ||
      value > static_cast<std::uint64_t>(
                  std::numeric_limits<std::int64_t>::max())) {
    throw_invalid(std::string(name),
                  allow_zero ? "an int64-compatible value"
                             : "a positive int64-compatible value",
                  std::to_string(value),
                  "Choose a bounded scheduler event count.");
  }
  return static_cast<std::int64_t>(value);
}

std::vector<double> resolve_group_rates(std::span<const double> configured,
                                        std::size_t group_count,
                                        std::string_view name,
                                        bool allow_empty = false) {
  if (configured.empty() && allow_empty) {
    return std::vector<double>(group_count, 0.0);
  }
  if (configured.size() != 1 && configured.size() != group_count) {
    throw_invalid(std::string(name), "one broadcast value or one per group",
                  std::to_string(configured.size()),
                  "Match the stable optimizer group topology.");
  }
  std::vector<double> resolved;
  resolved.reserve(group_count);
  for (std::size_t index = 0; index < group_count; ++index) {
    const auto value =
        configured.size() == 1 ? configured.front() : configured[index];
    if (!std::isfinite(value) || value < 0.0) {
      throw_invalid(std::string(name), "finite non-negative rates",
                    std::to_string(value),
                    "Provide valid learning rates for every group.");
    }
    resolved.push_back(value);
  }
  return resolved;
}

std::uint64_t next_unbounded_step(std::uint64_t completed_steps,
                                  std::string_view type_id) {
  if (completed_steps == std::numeric_limits<std::uint64_t>::max()) {
    throw_counter_exhausted(type_id);
  }
  return completed_steps + 1;
}

void validate_counter_only_leaf(
    const detail::SchedulerState& state, std::string_view type_id,
    std::optional<std::uint64_t> maximum = std::nullopt) {
  if (!state.control.empty() || !state.children.empty()) {
    throw_compatibility(std::string(type_id) + ".state.shape",
                        "a counter-only leaf state",
                        "control values or children present",
                        "Restore state from the same scheduler type.");
  }
  if (maximum.has_value() && state.completed_steps > *maximum) {
    throw_compatibility(std::string(type_id) + ".completed_steps",
                        "at most " + std::to_string(*maximum),
                        std::to_string(state.completed_steps),
                        "Reject state beyond the finite schedule.");
  }
}

std::string_view metric_direction_name(MetricDirection direction) {
  switch (direction) {
    case MetricDirection::minimize:
      return "minimize";
    case MetricDirection::maximize:
      return "maximize";
  }
  return "unknown";
}

std::string_view threshold_mode_name(MetricThresholdMode mode) {
  switch (mode) {
    case MetricThresholdMode::relative:
      return "relative";
    case MetricThresholdMode::absolute:
      return "absolute";
  }
  return "unknown";
}

std::string_view anneal_strategy_name(AnnealStrategy strategy) {
  switch (strategy) {
    case AnnealStrategy::cosine:
      return "cosine";
    case AnnealStrategy::linear:
      return "linear";
  }
  return "unknown";
}

std::string_view cyclic_mode_name(CyclicLrMode mode) {
  switch (mode) {
    case CyclicLrMode::triangular:
      return "triangular";
    case CyclicLrMode::triangular2:
      return "triangular2";
    case CyclicLrMode::exp_range:
      return "exp_range";
  }
  return "unknown";
}

std::string_view scale_mode_name(CyclicScaleMode mode) {
  switch (mode) {
    case CyclicScaleMode::cycle:
      return "cycle";
    case CyclicScaleMode::iterations:
      return "iterations";
  }
  return "unknown";
}

double interpolate(double start, double end, double progress,
                   AnnealStrategy strategy) {
  const auto bounded = std::clamp(progress, 0.0, 1.0);
  const auto weight = strategy == AnnealStrategy::cosine
                          ? 0.5 * (1.0 - std::cos(kPi * bounded))
                          : bounded;
  return start + (end - start) * weight;
}

template <typename Value>
const Value& control_value(const detail::SchedulerState& state,
                           std::string_view key) {
  const auto found = state.control.find(std::string(key));
  if (found == state.control.end()) {
    throw_compatibility("state.control", "all plateau control keys",
                        std::string(key) + " missing",
                        "Restore complete current-format scheduler state.");
  }
  const auto* typed = std::get_if<Value>(&found->second);
  if (typed == nullptr) {
    throw_compatibility("state.control." + std::string(key),
                        "the declared typed value", "different type",
                        "Reject state from a different scheduler schema.");
  }
  return *typed;
}

double invoke_callable(const std::function<double(double)>& function,
                       double argument, std::string_view type_id) {
  try {
    return function(argument);
  } catch (const Error&) {
    throw;
  } catch (const std::exception& error) {
    throw_callable_failure(type_id, error.what());
  } catch (...) {
    throw_callable_failure(type_id, "non-standard exception");
  }
}

double invoke_callable(const std::function<double(std::uint64_t)>& function,
                       std::uint64_t argument, std::string_view type_id) {
  try {
    return function(argument);
  } catch (const Error&) {
    throw;
  } catch (const std::exception& error) {
    throw_callable_failure(type_id, error.what());
  } catch (...) {
    throw_callable_failure(type_id, "non-standard exception");
  }
}

}  // namespace

ReduceLrOnPlateauScheduler::ReduceLrOnPlateauScheduler(
    std::shared_ptr<OptimizerRateControl> optimizer,
    ReduceLrOnPlateauSchedulerOptions options)
    : LrScheduler(std::move(optimizer), options.scheduler_id),
      options_(std::move(options)) {
  if (metric_direction_name(options_.direction) == "unknown") {
    throw_invalid("direction", "minimize or maximize", "unknown",
                  "Use a declared MetricDirection value.");
  }
  if (!std::isfinite(options_.factor) || options_.factor <= 0.0 ||
      options_.factor >= 1.0) {
    throw_invalid("factor", "a finite value in (0, 1)",
                  std::to_string(options_.factor),
                  "Choose a strict decay multiplier.");
  }
  static_cast<void>(descriptor_count(options_.patience, "patience", true));
  if (!std::isfinite(options_.threshold) || options_.threshold < 0.0) {
    throw_invalid("threshold", "a finite non-negative value",
                  std::to_string(options_.threshold),
                  "Choose a valid improvement threshold.");
  }
  if (threshold_mode_name(options_.threshold_mode) == "unknown") {
    throw_invalid("threshold_mode", "relative or absolute", "unknown",
                  "Use a declared MetricThresholdMode value.");
  }
  static_cast<void>(descriptor_count(options_.cooldown, "cooldown", true));
  if (!std::isfinite(options_.eps) || options_.eps < 0.0) {
    throw_invalid("eps", "a finite non-negative value",
                  std::to_string(options_.eps),
                  "Choose a valid minimum rate delta.");
  }
  min_learning_rates_ = resolve_group_rates(options_.min_learning_rates,
                                            base_learning_rates().size(),
                                            "min_learning_rates", true);
  for (std::size_t index = 0; index < min_learning_rates_.size(); ++index) {
    if (min_learning_rates_[index] > base_learning_rates()[index]) {
      throw_invalid("min_learning_rates", "values at or below current rates",
                    std::to_string(min_learning_rates_[index]),
                    "Lower the minimum for the corresponding group.");
    }
  }
  configure_descriptor(
      kPlateauTypeId, SchedulerStepMode::metric,
      {{"base_learning_rates", base_learning_rates()},
       {"cooldown", descriptor_count(options_.cooldown, "cooldown", true)},
       {"direction", std::string(metric_direction_name(options_.direction))},
       {"eps", options_.eps},
       {"factor", options_.factor},
       {"min_learning_rates", min_learning_rates_},
       {"patience", descriptor_count(options_.patience, "patience", true)},
       {"threshold", options_.threshold},
       {"threshold_mode",
        std::string(threshold_mode_name(options_.threshold_mode))}});
}

ReduceLrOnPlateauScheduler::~ReduceLrOnPlateauScheduler() = default;

bool ReduceLrOnPlateauScheduler::has_best() const noexcept { return has_best_; }

double ReduceLrOnPlateauScheduler::best() const noexcept { return best_; }

std::uint64_t ReduceLrOnPlateauScheduler::bad_observations() const noexcept {
  return bad_observations_;
}

std::uint64_t ReduceLrOnPlateauScheduler::cooldown_remaining() const noexcept {
  return cooldown_remaining_;
}

detail::SchedulerTransition ReduceLrOnPlateauScheduler::preview_step_impl(
    std::span<const double> learning_rates) const {
  static_cast<void>(learning_rates);
  throw_invalid("step.event", "a finite metric observation",
                "metric-less event", "Call step(metric) instead.");
}

detail::SchedulerTransition
ReduceLrOnPlateauScheduler::preview_metric_step_impl(
    std::span<const double> learning_rates, double metric) const {
  const auto next_step = next_unbounded_step(completed_steps_, kPlateauTypeId);
  auto candidate_rates =
      std::vector<double>(learning_rates.begin(), learning_rates.end());
  auto next_has_best = has_best_;
  auto next_best = best_;
  auto next_bad = bad_observations_;
  auto next_cooldown = cooldown_remaining_;

  bool improved = !has_best_;
  if (has_best_) {
    const auto delta = options_.threshold_mode == MetricThresholdMode::absolute
                           ? options_.threshold
                           : std::abs(best_) * options_.threshold;
    improved = options_.direction == MetricDirection::minimize
                   ? metric < best_ - delta
                   : metric > best_ + delta;
  }
  if (improved) {
    next_has_best = true;
    next_best = metric;
    next_bad = 0;
  }
  if (next_cooldown > 0) {
    --next_cooldown;
    next_bad = 0;
  } else if (!improved) {
    if (next_bad == std::numeric_limits<std::uint64_t>::max()) {
      throw_counter_exhausted(kPlateauTypeId);
    }
    ++next_bad;
    if (next_bad > options_.patience) {
      for (std::size_t index = 0; index < candidate_rates.size(); ++index) {
        const auto reduced = std::max(candidate_rates[index] * options_.factor,
                                      min_learning_rates_[index]);
        if (!std::isfinite(reduced) || reduced < 0.0) {
          throw_numerical("plateau.learning_rate", reduced,
                          "Reduce factor or the current optimizer rate.");
        }
        if (candidate_rates[index] - reduced > options_.eps) {
          candidate_rates[index] = reduced;
        }
      }
      next_bad = 0;
      next_cooldown = options_.cooldown;
    }
  }

  return {
      .state = {.completed_steps = next_step,
                .control = {{"bad_observations",
                             static_cast<std::int64_t>(next_bad)},
                            {"best", next_best},
                            {"cooldown_remaining",
                             static_cast<std::int64_t>(next_cooldown)},
                            {"has_best", next_has_best}},
                .children = {}},
      .learning_rates = std::move(candidate_rates),
  };
}

detail::SchedulerState ReduceLrOnPlateauScheduler::capture_state_impl() const {
  return {.completed_steps = completed_steps_,
          .control = {{"bad_observations",
                       static_cast<std::int64_t>(bad_observations_)},
                      {"best", best_},
                      {"cooldown_remaining",
                       static_cast<std::int64_t>(cooldown_remaining_)},
                      {"has_best", has_best_}},
          .children = {}};
}

std::uint64_t ReduceLrOnPlateauScheduler::completed_steps_impl()
    const noexcept {
  return completed_steps_;
}

void ReduceLrOnPlateauScheduler::validate_state_impl(
    const detail::SchedulerState& state) const {
  if (!state.children.empty() || state.control.size() != 4) {
    throw_compatibility(
        "plateau.state.shape", "four control values and no children",
        "different state shape", "Restore current plateau scheduler state.");
  }
  const auto has_best = control_value<bool>(state, "has_best");
  const auto best = control_value<double>(state, "best");
  const auto bad = control_value<std::int64_t>(state, "bad_observations");
  const auto cooldown =
      control_value<std::int64_t>(state, "cooldown_remaining");
  if (!std::isfinite(best) || (!has_best && best != 0.0)) {
    throw_compatibility("plateau.state.best",
                        "finite, and zero before the first observation",
                        std::to_string(best), "Reject malformed metric state.");
  }
  if (bad < 0 || static_cast<std::uint64_t>(bad) > options_.patience) {
    throw_compatibility("plateau.state.bad_observations",
                        "zero through configured patience", std::to_string(bad),
                        "Reject impossible patience state.");
  }
  if (cooldown < 0 ||
      static_cast<std::uint64_t>(cooldown) > options_.cooldown) {
    throw_compatibility(
        "plateau.state.cooldown_remaining", "zero through configured cooldown",
        std::to_string(cooldown), "Reject impossible cooldown state.");
  }
}

void ReduceLrOnPlateauScheduler::apply_state_impl(
    const detail::SchedulerState& state) noexcept {
  const auto has_best = state.control.find("has_best");
  const auto best = state.control.find("best");
  const auto bad_observations = state.control.find("bad_observations");
  const auto cooldown_remaining = state.control.find("cooldown_remaining");
  completed_steps_ = state.completed_steps;
  has_best_ = *std::get_if<bool>(&has_best->second);
  best_ = *std::get_if<double>(&best->second);
  bad_observations_ = static_cast<std::uint64_t>(
      *std::get_if<std::int64_t>(&bad_observations->second));
  cooldown_remaining_ = static_cast<std::uint64_t>(
      *std::get_if<std::int64_t>(&cooldown_remaining->second));
}

OneCycleLrScheduler::OneCycleLrScheduler(
    std::shared_ptr<OptimizerRateControl> optimizer,
    OneCycleLrSchedulerOptions options)
    : LrScheduler(std::move(optimizer), options.scheduler_id),
      options_(std::move(options)) {
  static_cast<void>(descriptor_count(options_.total_steps, "total_steps"));
  if (!std::isfinite(options_.pct_start) || options_.pct_start <= 0.0 ||
      options_.pct_start >= 1.0) {
    throw_invalid("pct_start", "a finite value in (0, 1)",
                  std::to_string(options_.pct_start),
                  "Reserve events for both increasing and decreasing phases.");
  }
  if (anneal_strategy_name(options_.anneal_strategy) == "unknown") {
    throw_invalid("anneal_strategy", "cosine or linear", "unknown",
                  "Use a declared AnnealStrategy value.");
  }
  if (!std::isfinite(options_.div_factor) || options_.div_factor <= 0.0 ||
      !std::isfinite(options_.final_div_factor) ||
      options_.final_div_factor <= 0.0) {
    throw_invalid("divisors", "finite positive values", "invalid",
                  "Choose positive initial and final divisors.");
  }
  if (options_.cycle_momentum) {
    throw_momentum_not_implemented(kOneCycleTypeId);
  }
  max_learning_rates_ =
      resolve_group_rates(options_.max_learning_rates,
                          base_learning_rates().size(), "max_learning_rates");
  initial_learning_rates_.reserve(max_learning_rates_.size());
  final_learning_rates_.reserve(max_learning_rates_.size());
  for (const auto maximum : max_learning_rates_) {
    if (maximum <= 0.0) {
      throw_invalid("max_learning_rates", "finite positive values",
                    std::to_string(maximum),
                    "Use a positive one-cycle maximum.");
    }
    const auto initial = maximum / options_.div_factor;
    const auto final_rate = initial / options_.final_div_factor;
    if (!std::isfinite(initial) || initial < 0.0 ||
        !std::isfinite(final_rate) || final_rate < 0.0) {
      throw_invalid("max_learning_rates", "divisors yielding valid rates",
                    std::to_string(maximum),
                    "Reduce maximum rates or divisors.");
    }
    initial_learning_rates_.push_back(initial);
    final_learning_rates_.push_back(final_rate);
  }
  warmup_steps_ = std::max<std::uint64_t>(
      1, static_cast<std::uint64_t>(std::floor(
             static_cast<double>(options_.total_steps) * options_.pct_start)));
  if (options_.total_steps > 1) {
    warmup_steps_ = std::min(warmup_steps_, options_.total_steps - 1);
  }
  if (options_.three_phase &&
      (options_.total_steps < 3 ||
       warmup_steps_ >= options_.total_steps - warmup_steps_)) {
    throw_invalid("three_phase", "two warmup-length phases plus final decay",
                  std::to_string(options_.total_steps),
                  "Increase total_steps or reduce pct_start.");
  }
  configure_descriptor(
      kOneCycleTypeId, SchedulerStepMode::batch,
      {{"anneal_strategy",
        std::string(anneal_strategy_name(options_.anneal_strategy))},
       {"cycle_momentum", options_.cycle_momentum},
       {"div_factor", options_.div_factor},
       {"final_div_factor", options_.final_div_factor},
       {"final_learning_rates", final_learning_rates_},
       {"initial_learning_rates", initial_learning_rates_},
       {"max_learning_rates", max_learning_rates_},
       {"pct_start", options_.pct_start},
       {"three_phase", options_.three_phase},
       {"total_steps", descriptor_count(options_.total_steps, "total_steps")},
       {"warmup_steps", descriptor_count(warmup_steps_, "warmup_steps")}});
  initialize_learning_rates(initial_learning_rates_);
}

OneCycleLrScheduler::~OneCycleLrScheduler() = default;

detail::SchedulerTransition OneCycleLrScheduler::preview_step_impl(
    std::span<const double> learning_rates) const {
  static_cast<void>(learning_rates);
  if (completed_steps_ >= options_.total_steps) {
    throw_counter_exhausted(kOneCycleTypeId);
  }
  const auto next_step = completed_steps_ + 1;
  std::vector<double> candidate;
  candidate.reserve(max_learning_rates_.size());
  for (std::size_t index = 0; index < max_learning_rates_.size(); ++index) {
    double rate = 0.0;
    if (options_.total_steps == 1) {
      rate = final_learning_rates_[index];
    } else if (next_step <= warmup_steps_) {
      rate = interpolate(
          initial_learning_rates_[index], max_learning_rates_[index],
          static_cast<double>(next_step) / static_cast<double>(warmup_steps_),
          options_.anneal_strategy);
    } else if (options_.three_phase && next_step <= 2 * warmup_steps_) {
      rate = interpolate(max_learning_rates_[index],
                         initial_learning_rates_[index],
                         static_cast<double>(next_step - warmup_steps_) /
                             static_cast<double>(warmup_steps_),
                         options_.anneal_strategy);
    } else {
      const auto phase_start =
          options_.three_phase ? 2 * warmup_steps_ : warmup_steps_;
      const auto start_rate = options_.three_phase
                                  ? initial_learning_rates_[index]
                                  : max_learning_rates_[index];
      rate = interpolate(
          start_rate, final_learning_rates_[index],
          static_cast<double>(next_step - phase_start) /
              static_cast<double>(options_.total_steps - phase_start),
          options_.anneal_strategy);
    }
    if (!std::isfinite(rate) || rate < 0.0) {
      throw_numerical("one_cycle.learning_rate", rate,
                      "Reduce maximum rates or schedule divisors.");
    }
    candidate.push_back(rate);
  }
  return {
      .state = {.completed_steps = next_step, .control = {}, .children = {}},
      .learning_rates = std::move(candidate)};
}

detail::SchedulerState OneCycleLrScheduler::capture_state_impl() const {
  return {.completed_steps = completed_steps_, .control = {}, .children = {}};
}

std::uint64_t OneCycleLrScheduler::completed_steps_impl() const noexcept {
  return completed_steps_;
}

void OneCycleLrScheduler::validate_state_impl(
    const detail::SchedulerState& state) const {
  validate_counter_only_leaf(state, kOneCycleTypeId, options_.total_steps);
}

void OneCycleLrScheduler::apply_state_impl(
    const detail::SchedulerState& state) noexcept {
  completed_steps_ = state.completed_steps;
}

CyclicLrScheduler::CyclicLrScheduler(
    std::shared_ptr<OptimizerRateControl> optimizer,
    CyclicLrSchedulerOptions options)
    : LrScheduler(std::move(optimizer), options.scheduler_id),
      options_(std::move(options)) {
  static_cast<void>(descriptor_count(options_.step_size_up, "step_size_up"));
  step_size_down_ = options_.step_size_down == 0 ? options_.step_size_up
                                                 : options_.step_size_down;
  static_cast<void>(descriptor_count(step_size_down_, "step_size_down"));
  if (options_.step_size_up >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) -
          step_size_down_) {
    throw_invalid("cycle_length", "an int64-compatible positive sum",
                  "overflow", "Reduce cyclic half-cycle lengths.");
  }
  if (cyclic_mode_name(options_.mode) == "unknown") {
    throw_invalid("mode", "triangular, triangular2, or exp_range", "unknown",
                  "Use a declared CyclicLrMode value.");
  }
  if (!std::isfinite(options_.gamma) || options_.gamma <= 0.0) {
    throw_invalid("gamma", "a finite positive value",
                  std::to_string(options_.gamma),
                  "Choose a valid exp-range base.");
  }
  if (scale_mode_name(options_.scale_mode) == "unknown") {
    throw_invalid("scale_mode", "cycle or iterations", "unknown",
                  "Use a declared CyclicScaleMode value.");
  }
  if (options_.cycle_momentum) {
    throw_momentum_not_implemented(kCyclicTypeId);
  }
  if (static_cast<bool>(options_.scale_function) !=
      !options_.scale_function_id.empty()) {
    throw_invalid("scale_function_id",
                  "a stable ID exactly when a callback is provided",
                  options_.scale_function_id.empty() ? "missing" : "orphaned",
                  "Register custom scale semantics with a stable identity.");
  }
  base_learning_rates_ =
      resolve_group_rates(options_.base_learning_rates,
                          base_learning_rates().size(), "base_learning_rates");
  max_learning_rates_ =
      resolve_group_rates(options_.max_learning_rates,
                          base_learning_rates().size(), "max_learning_rates");
  for (std::size_t index = 0; index < base_learning_rates_.size(); ++index) {
    if (max_learning_rates_[index] <= base_learning_rates_[index]) {
      throw_invalid("max_learning_rates", "values above every base rate",
                    std::to_string(max_learning_rates_[index]),
                    "Increase the corresponding cyclic maximum.");
    }
  }
  configure_descriptor(
      kCyclicTypeId, SchedulerStepMode::batch,
      {{"base_learning_rates", base_learning_rates_},
       {"cycle_momentum", options_.cycle_momentum},
       {"gamma", options_.gamma},
       {"max_learning_rates", max_learning_rates_},
       {"mode", std::string(cyclic_mode_name(options_.mode))},
       {"scale_mode", std::string(scale_mode_name(options_.scale_mode))},
       {"step_size_down", descriptor_count(step_size_down_, "step_size_down")},
       {"step_size_up",
        descriptor_count(options_.step_size_up, "step_size_up")}},
      {},
      options_.scale_function ? std::optional(options_.scale_function_id)
                              : std::nullopt);
  initialize_learning_rates(base_learning_rates_);
}

CyclicLrScheduler::~CyclicLrScheduler() = default;

detail::SchedulerTransition CyclicLrScheduler::preview_step_impl(
    std::span<const double> learning_rates) const {
  static_cast<void>(learning_rates);
  const auto next_step = next_unbounded_step(completed_steps_, kCyclicTypeId);
  const auto cycle_length = options_.step_size_up + step_size_down_;
  const auto position = next_step % cycle_length;
  const auto cycle = (next_step - 1) / cycle_length + 1;
  const auto height = position == 0 ? 0.0
                      : position <= options_.step_size_up
                          ? static_cast<double>(position) /
                                static_cast<double>(options_.step_size_up)
                          : static_cast<double>(cycle_length - position) /
                                static_cast<double>(step_size_down_);
  double scale = 1.0;
  if (options_.scale_function) {
    const auto argument = options_.scale_mode == CyclicScaleMode::cycle
                              ? static_cast<double>(cycle)
                              : static_cast<double>(next_step);
    scale = invoke_callable(options_.scale_function, argument, kCyclicTypeId);
  } else if (options_.mode == CyclicLrMode::triangular2) {
    scale = std::pow(0.5, static_cast<double>(cycle - 1));
  } else if (options_.mode == CyclicLrMode::exp_range) {
    scale = std::pow(options_.gamma, static_cast<double>(next_step));
  }
  if (!std::isfinite(scale) || scale < 0.0) {
    throw_numerical("cyclic.scale", scale,
                    "Return a finite non-negative amplitude scale.");
  }
  std::vector<double> candidate;
  candidate.reserve(base_learning_rates_.size());
  for (std::size_t index = 0; index < base_learning_rates_.size(); ++index) {
    const auto rate =
        base_learning_rates_[index] +
        (max_learning_rates_[index] - base_learning_rates_[index]) * height *
            scale;
    if (!std::isfinite(rate) || rate < 0.0) {
      throw_numerical("cyclic.learning_rate", rate,
                      "Reduce the scale function or cyclic maximum.");
    }
    candidate.push_back(rate);
  }
  return {
      .state = {.completed_steps = next_step, .control = {}, .children = {}},
      .learning_rates = std::move(candidate)};
}

detail::SchedulerState CyclicLrScheduler::capture_state_impl() const {
  return {.completed_steps = completed_steps_, .control = {}, .children = {}};
}

std::uint64_t CyclicLrScheduler::completed_steps_impl() const noexcept {
  return completed_steps_;
}

void CyclicLrScheduler::validate_state_impl(
    const detail::SchedulerState& state) const {
  validate_counter_only_leaf(state, kCyclicTypeId);
}

void CyclicLrScheduler::apply_state_impl(
    const detail::SchedulerState& state) noexcept {
  completed_steps_ = state.completed_steps;
}

LambdaLrScheduler::LambdaLrScheduler(
    std::shared_ptr<OptimizerRateControl> optimizer,
    LambdaLrSchedulerOptions options)
    : LrScheduler(std::move(optimizer), options.scheduler_id),
      options_(std::move(options)) {
  if (options_.callable_id.empty()) {
    throw_invalid("callable_id", "a nonempty stable callable-bundle ID",
                  "empty", "Register callable semantics before construction.");
  }
  if (options_.functions.size() != 1 &&
      options_.functions.size() != base_learning_rates().size()) {
    throw_invalid("functions", "one callback or one per optimizer group",
                  std::to_string(options_.functions.size()),
                  "Match the stable optimizer group topology.");
  }
  for (const auto& function : options_.functions) {
    if (!function) {
      throw_invalid("functions", "nonempty callbacks", "empty callback",
                    "Provide every declared schedule function.");
    }
  }
  configure_descriptor(kLambdaTypeId, SchedulerStepMode::epoch,
                       {{"base_learning_rates", base_learning_rates()},
                        {"function_count",
                         static_cast<std::int64_t>(options_.functions.size())}},
                       {}, options_.callable_id);

  std::vector<double> initial;
  initial.reserve(base_learning_rates().size());
  for (std::size_t index = 0; index < base_learning_rates().size(); ++index) {
    const auto& function = options_.functions.size() == 1
                               ? options_.functions.front()
                               : options_.functions[index];
    const auto factor = invoke_callable(function, 0, kLambdaTypeId);
    const auto rate = base_learning_rates()[index] * factor;
    if (!std::isfinite(factor) || factor < 0.0 || !std::isfinite(rate) ||
        rate < 0.0) {
      throw_numerical("lambda.initial_rate", rate,
                      "Return a finite non-negative event-zero factor.");
    }
    initial.push_back(rate);
  }
  initialize_learning_rates(initial);
}

LambdaLrScheduler::~LambdaLrScheduler() = default;

detail::SchedulerTransition LambdaLrScheduler::preview_step_impl(
    std::span<const double> learning_rates) const {
  static_cast<void>(learning_rates);
  const auto next_step = next_unbounded_step(completed_steps_, kLambdaTypeId);
  std::vector<double> candidate;
  candidate.reserve(base_learning_rates().size());
  for (std::size_t index = 0; index < base_learning_rates().size(); ++index) {
    const auto& function = options_.functions.size() == 1
                               ? options_.functions.front()
                               : options_.functions[index];
    const auto factor = invoke_callable(function, next_step, kLambdaTypeId);
    const auto rate = base_learning_rates()[index] * factor;
    if (!std::isfinite(factor) || factor < 0.0 || !std::isfinite(rate) ||
        rate < 0.0) {
      throw_numerical("lambda.learning_rate", rate,
                      "Return finite non-negative callable factors.");
    }
    candidate.push_back(rate);
  }
  return {
      .state = {.completed_steps = next_step, .control = {}, .children = {}},
      .learning_rates = std::move(candidate)};
}

detail::SchedulerState LambdaLrScheduler::capture_state_impl() const {
  return {.completed_steps = completed_steps_, .control = {}, .children = {}};
}

std::uint64_t LambdaLrScheduler::completed_steps_impl() const noexcept {
  return completed_steps_;
}

void LambdaLrScheduler::validate_state_impl(
    const detail::SchedulerState& state) const {
  validate_counter_only_leaf(state, kLambdaTypeId);
}

void LambdaLrScheduler::apply_state_impl(
    const detail::SchedulerState& state) noexcept {
  completed_steps_ = state.completed_steps;
}

}  // namespace jyppx::liuding::optim
