// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/training/amp/grad_scaler.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string_view>
#include <utility>

namespace jyppx::training::amp {
namespace {

constexpr std::string_view kComponent = "training.amp.grad_scaler";

[[noreturn]] void throw_invalid(std::string_view operation,
                                std::string_view expected,
                                std::string_view actual,
                                std::string_view hint) {
  throw InvalidArgument(
      "GradScaler rejected " + std::string(operation),
      ErrorDetails{.component = std::string(kComponent),
                   .operation = std::string(operation),
                   .expected = std::string(expected),
                   .actual = std::string(actual),
                   .hint = std::string(hint)});
}

[[noreturn]] void throw_not_ready(std::string_view operation,
                                  std::string_view expected,
                                  std::string_view actual,
                                  std::string_view hint) {
  throw NotReady(
      "GradScaler " + std::string(operation) + " is not ready",
      ErrorDetails{.component = std::string(kComponent),
                   .operation = std::string(operation),
                   .expected = std::string(expected),
                   .actual = std::string(actual),
                   .hint = std::string(hint)});
}

[[noreturn]] void throw_incompatible(std::string_view operation,
                                     std::string_view expected,
                                     std::string_view actual,
                                     std::string_view hint) {
  throw CompatibilityError(
      "GradScaler state is incompatible for " + std::string(operation),
      ErrorDetails{.component = std::string(kComponent),
                   .operation = std::string(operation),
                   .expected = std::string(expected),
                   .actual = std::string(actual),
                   .hint = std::string(hint)});
}

[[noreturn]] void throw_malformed(std::string_view operation,
                                  std::string_view expected,
                                  std::string_view actual,
                                  std::string_view hint) {
  throw SerializationError(
      "GradScaler state is malformed for " + std::string(operation),
      ErrorDetails{.component = std::string(kComponent),
                   .operation = std::string(operation),
                   .expected = std::string(expected),
                   .actual = std::string(actual),
                   .hint = std::string(hint)});
}

void validate_options(const GradScalerOptions& options) {
  if (!std::isfinite(options.init_scale) || options.init_scale <= 0.0) {
    throw_invalid("init_scale", "a finite value greater than zero",
                  "non-positive or non-finite", "Choose a finite positive scale");
  }
  if (!std::isfinite(options.growth_factor) || options.growth_factor <= 1.0) {
    throw_invalid("growth_factor", "a finite value greater than one",
                  "not greater than one", "Choose a growth factor above 1.0");
  }
  if (!std::isfinite(options.backoff_factor) || options.backoff_factor <= 0.0 ||
      options.backoff_factor >= 1.0) {
    throw_invalid("backoff_factor", "a finite value in (0, 1)",
                  "outside the open interval", "Choose a backoff factor in (0, 1)");
  }
  if (options.growth_interval <= 0) {
    throw_invalid("growth_interval", "a positive integer", "non-positive",
                  "Choose the number of finite cycles before growth");
  }
  if (options.scaler_id.empty() ||
      options.scaler_id.find('\0') != std::string::npos) {
    throw_invalid("scaler_id", "a nonempty identifier without NUL bytes",
                  "empty or contains NUL", "Choose a stable checkpoint identity");
  }
}

void validate_state_values(const GradScalerState& state,
                           const GradScalerOptions& options) {
  if (state.format_version != kGradScalerStateFormatVersion) {
    throw_incompatible("format_version",
                       std::to_string(kGradScalerStateFormatVersion),
                       std::to_string(state.format_version),
                       "Migrate the checkpoint to the supported state schema");
  }
  if (state.scaler_id != options.scaler_id ||
      state.init_scale != options.init_scale ||
      state.growth_factor != options.growth_factor ||
      state.backoff_factor != options.backoff_factor ||
      state.growth_interval != options.growth_interval ||
      state.enabled != options.enabled) {
    throw_incompatible("configuration", "the active scaler options",
                       "different immutable options or scaler_id",
                       "Restore into a scaler with matching frozen options");
  }
  if (!std::isfinite(state.scale) || state.scale <= 0.0 ||
      (!state.enabled && state.scale != 1.0) ||
      state.growth_tracker >=
          static_cast<std::uint64_t>(state.growth_interval)) {
    throw_malformed("mutable_state", "finite positive scale and valid tracker",
                    "invalid scale or growth_tracker", "Repair or discard the checkpoint");
  }
}

}  // namespace

GradScaler::GradScaler(GradScalerOptions options)
    : options_(std::move(options)),
      scale_(options_.enabled ? options_.init_scale : 1.0),
      growth_tracker_(0),
      found_inf_(false) {
  validate_options(options_);
}

Tensor GradScaler::scale(const Tensor& loss) const {
  return options_.enabled ? loss * scale_ : loss;
}

void GradScaler::unscale(
    const std::shared_ptr<optim::OptimizerAdapter>& optimizer) {
  if (!optimizer) {
    throw_invalid("optimizer", "a non-null optimizer", "null",
                  "Pass the optimizer participating in this cycle");
  }
  if (optimizer_states_.empty()) {
    found_inf_ = false;
  }
  const auto existing = optimizer_states_.find(optimizer.get());
  if (existing != optimizer_states_.end()) {
    if (existing->second.stage == OptimizerStage::stepped) {
      throw_invalid("unscale", "a new optimizer cycle",
                    "optimizer already marked in the active cycle",
                    "Call update() before starting the next cycle");
    }
    return;
  }

  OptimizerState state;
  const torch::NoGradGuard no_grad;
  for (const auto& binding : optimizer->parameter_bindings()) {
    const auto& parameter = binding.parameter;
    if (!parameter.defined() || !parameter.grad().defined()) {
      continue;
    }
    auto& gradient = parameter.mutable_grad();
    if (options_.enabled) {
      gradient.div_(scale_);
      if (!torch::isfinite(gradient).all().item<bool>()) {
        state.found_inf = true;
      }
    }
  }
  found_inf_ = found_inf_ || state.found_inf;
  optimizer_states_.emplace(optimizer.get(), state);
}

bool GradScaler::step_allowed(
    const std::shared_ptr<optim::OptimizerAdapter>& optimizer) const {
  if (!optimizer) {
    throw_invalid("optimizer", "a non-null optimizer", "null",
                  "Pass the optimizer participating in this cycle");
  }
  const auto existing = optimizer_states_.find(optimizer.get());
  if (existing == optimizer_states_.end()) {
    throw_not_ready("step_allowed", "optimizer already unscaled",
                    "optimizer is absent from the active cycle",
                    "Call unscale() before requesting the step decision");
  }
  if (existing->second.stage == OptimizerStage::stepped) {
    throw_invalid("step_allowed", "an unscaled optimizer not yet marked",
                  "optimizer already marked in the active cycle",
                  "Call update() before requesting another decision");
  }
  return !existing->second.found_inf;
}

bool GradScaler::cycle_allowed() const {
  if (optimizer_states_.empty()) {
    throw_not_ready("cycle_allowed", "at least one unscaled optimizer",
                    "the active cycle is empty",
                    "Call unscale() for every optimizer in the cycle");
  }
  if (std::any_of(optimizer_states_.begin(), optimizer_states_.end(),
                  [](const auto& entry) {
                    return entry.second.stage == OptimizerStage::stepped;
                  })) {
    throw_invalid("cycle_allowed", "no optimizer marked yet",
                  "an optimizer outcome is already committed",
                  "Request the aggregate decision before mark_step()");
  }
  return !found_inf_;
}

void GradScaler::mark_step(
    const std::shared_ptr<optim::OptimizerAdapter>& optimizer, bool applied) {
  if (!optimizer) {
    throw_invalid("optimizer", "a non-null optimizer", "null",
                  "Pass the optimizer participating in this cycle");
  }
  auto existing = optimizer_states_.find(optimizer.get());
  if (existing == optimizer_states_.end()) {
    throw_not_ready("mark_step", "optimizer already unscaled",
                    "optimizer is absent from the active cycle",
                    "Call unscale() before reporting the step result");
  }
  if (existing->second.stage == OptimizerStage::stepped) {
    throw_invalid("mark_step", "an optimizer not yet marked",
                  "optimizer already marked in the active cycle",
                  "Call update() before reporting another result");
  }
  if (applied == found_inf_) {
    throw_invalid("mark_step", "applied=false when the cycle overflowed and "
                              "applied=true otherwise",
                  "result conflicts with the aggregate overflow decision",
                  "Use the value returned by step_allowed()");
  }
  existing->second.stage = OptimizerStage::stepped;
  existing->second.applied = applied;
}

void GradScaler::update() {
  if (optimizer_states_.empty()) {
    throw_not_ready("update", "a populated optimizer cycle",
                    "the active cycle is empty",
                    "Unscale and mark every optimizer before update()");
  }
  for (const auto& [optimizer, state] : optimizer_states_) {
    static_cast<void>(optimizer);
    if (state.stage != OptimizerStage::stepped) {
      throw_not_ready("update", "mark_step() for every optimizer",
                      "at least one optimizer remains open",
                      "Report every optimizer result before update()");
    }
    if (state.applied == found_inf_) {
      throw_invalid("update", "consistent optimizer outcomes",
                    "an outcome conflicts with the aggregate overflow decision",
                    "Use the value returned by step_allowed()");
    }
  }

  if (!options_.enabled) {
    optimizer_states_.clear();
    found_inf_ = false;
    return;
  }
  if (found_inf_) {
    scale_ = std::max(scale_ * options_.backoff_factor,
                      std::numeric_limits<double>::min());
    growth_tracker_ = 0;
  } else {
    ++growth_tracker_;
    if (growth_tracker_ >=
        static_cast<std::uint64_t>(options_.growth_interval)) {
      const auto maximum = std::numeric_limits<double>::max();
      scale_ = scale_ > maximum / options_.growth_factor
                   ? maximum
                   : scale_ * options_.growth_factor;
      growth_tracker_ = 0;
    }
  }
  optimizer_states_.clear();
}

double GradScaler::get_scale() const noexcept { return scale_; }

bool GradScaler::found_inf() const noexcept { return found_inf_; }

bool GradScaler::checkpoint_ready() const noexcept {
  return optimizer_states_.empty();
}

GradScalerState GradScaler::capture_state() const {
  if (!checkpoint_ready()) {
    throw_not_ready("capture_state", "a completed step/update cycle",
                    "an optimizer cycle is in progress",
                    "Finish the active cycle before checkpointing");
  }
  return {.format_version = kGradScalerStateFormatVersion,
          .scaler_id = options_.scaler_id,
          .init_scale = options_.init_scale,
          .growth_factor = options_.growth_factor,
          .backoff_factor = options_.backoff_factor,
          .growth_interval = options_.growth_interval,
          .enabled = options_.enabled,
          .scale = scale_,
          .growth_tracker = growth_tracker_};
}

void GradScaler::restore_state(const GradScalerState& state) {
  validate_state(state);
  scale_ = state.scale;
  growth_tracker_ = state.growth_tracker;
  found_inf_ = false;
}

void GradScaler::validate_state(const GradScalerState& state) const {
  if (!checkpoint_ready()) {
    throw_not_ready("validate_state", "a completed step/update cycle",
                    "an optimizer cycle is in progress",
                    "Finish the active cycle before restoring state");
  }
  validate_state_values(state, options_);
}

const GradScalerOptions& GradScaler::options() const noexcept {
  return options_;
}

}  // namespace jyppx::training::amp
