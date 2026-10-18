// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/training/amp/grad_scaler.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace jyppx::training::amp {
namespace {

void validate_options(const GradScalerOptions& options) {
  if (!std::isfinite(options.init_scale) || options.init_scale <= 0.0) {
    throw std::invalid_argument("GradScaler init_scale must be positive");
  }
  if (!std::isfinite(options.growth_factor) || options.growth_factor <= 1.0) {
    throw std::invalid_argument("GradScaler growth_factor must exceed 1.0");
  }
  if (!std::isfinite(options.backoff_factor) || options.backoff_factor <= 0.0 ||
      options.backoff_factor >= 1.0) {
    throw std::invalid_argument("GradScaler backoff_factor must be in (0, 1)");
  }
  if (options.growth_interval <= 0) {
    throw std::invalid_argument("GradScaler growth_interval must be positive");
  }
  if (options.scaler_id.empty() ||
      options.scaler_id.find('\0') != std::string::npos) {
    throw std::invalid_argument(
        "GradScaler scaler_id must be a nonempty non-NUL identifier");
  }
}

void validate_state_values(const GradScalerState& state,
                           const GradScalerOptions& options) {
  if (state.format_version != kGradScalerStateFormatVersion) {
    throw std::invalid_argument("Unsupported GradScaler state format version");
  }
  if (state.scaler_id != options.scaler_id ||
      state.init_scale != options.init_scale ||
      state.growth_factor != options.growth_factor ||
      state.backoff_factor != options.backoff_factor ||
      state.growth_interval != options.growth_interval ||
      state.enabled != options.enabled) {
    throw std::invalid_argument(
        "GradScaler state configuration is incompatible");
  }
  if (!std::isfinite(state.scale) || state.scale <= 0.0 ||
      (!state.enabled && state.scale != 1.0) ||
      state.growth_tracker >=
          static_cast<std::uint64_t>(state.growth_interval)) {
    throw std::invalid_argument("Invalid GradScaler mutable state");
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
    throw std::invalid_argument("GradScaler optimizer cannot be null");
  }
  if (optimizer_states_.empty()) {
    found_inf_ = false;
  }
  const auto existing = optimizer_states_.find(optimizer.get());
  if (existing != optimizer_states_.end()) {
    if (existing->second.stage == OptimizerStage::stepped) {
      throw std::logic_error(
          "GradScaler cannot unscale an optimizer after mark_step() before "
          "update()");
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
    throw std::invalid_argument("GradScaler optimizer cannot be null");
  }
  const auto existing = optimizer_states_.find(optimizer.get());
  if (existing == optimizer_states_.end()) {
    throw std::logic_error(
        "GradScaler.step_allowed() requires preceding unscale()");
  }
  if (existing->second.stage == OptimizerStage::stepped) {
    throw std::logic_error(
        "GradScaler step decision must be followed by update()");
  }
  return !existing->second.found_inf;
}

bool GradScaler::cycle_allowed() const {
  if (optimizer_states_.empty()) {
    throw std::logic_error(
        "GradScaler.cycle_allowed() requires preceding unscale()");
  }
  if (std::any_of(optimizer_states_.begin(), optimizer_states_.end(),
                  [](const auto& entry) {
                    return entry.second.stage == OptimizerStage::stepped;
                  })) {
    throw std::logic_error(
        "GradScaler cycle decision must precede every mark_step()");
  }
  return !found_inf_;
}

void GradScaler::mark_step(
    const std::shared_ptr<optim::OptimizerAdapter>& optimizer, bool applied) {
  if (!optimizer) {
    throw std::invalid_argument("GradScaler optimizer cannot be null");
  }
  auto existing = optimizer_states_.find(optimizer.get());
  if (existing == optimizer_states_.end()) {
    throw std::logic_error("GradScaler.mark_step() requires unscale()");
  }
  if (existing->second.stage == OptimizerStage::stepped) {
    throw std::logic_error(
        "GradScaler.mark_step() for this optimizer must be followed by "
        "update()");
  }
  if (applied == found_inf_) {
    throw std::logic_error(
        "GradScaler applied result conflicts with the cycle overflow decision");
  }
  existing->second.stage = OptimizerStage::stepped;
  existing->second.applied = applied;
}

void GradScaler::update() {
  if (optimizer_states_.empty()) {
    throw std::logic_error(
        "GradScaler.update() requires a preceding optimizer cycle");
  }
  for (const auto& [optimizer, state] : optimizer_states_) {
    static_cast<void>(optimizer);
    if (state.stage != OptimizerStage::stepped) {
      throw std::logic_error(
          "GradScaler.update() requires mark_step() for every optimizer");
    }
    if (state.applied == found_inf_) {
      throw std::logic_error(
          "GradScaler optimizer outcomes disagree with the cycle decision");
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
    throw std::logic_error(
        "GradScaler state requires a completed step/update cycle");
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
    throw std::logic_error(
        "GradScaler restore requires a completed step/update cycle");
  }
  validate_state_values(state, options_);
}

const GradScalerOptions& GradScaler::options() const noexcept {
  return options_;
}

}  // namespace jyppx::training::amp
