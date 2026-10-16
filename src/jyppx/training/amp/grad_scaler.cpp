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

void GradScaler::unscale(const std::shared_ptr<optim::Optimizer>& optimizer) {
  if (!optimizer) {
    throw std::invalid_argument("GradScaler optimizer cannot be null");
  }
  if (!options_.enabled) {
    return;
  }

  if (optimizer_states_.empty()) {
    found_inf_ = false;
  }
  const auto existing = optimizer_states_.find(optimizer.get());
  if (existing != optimizer_states_.end()) {
    if (existing->second.stage == OptimizerStage::stepped) {
      throw std::logic_error(
          "GradScaler cannot unscale an optimizer after step() before "
          "update()");
    }
    return;
  }

  OptimizerState state;
  const torch::NoGradGuard no_grad;
  for (const auto& group : optimizer->param_groups()) {
    for (const auto& parameter : group.params) {
      if (!parameter.defined() || !parameter.grad().defined()) {
        continue;
      }
      auto& gradient = parameter.mutable_grad();
      gradient.div_(scale_);
      if (torch::isinf(gradient).any().item<bool>() ||
          torch::isnan(gradient).any().item<bool>()) {
        state.found_inf = true;
      }
    }
  }
  found_inf_ = found_inf_ || state.found_inf;
  optimizer_states_.emplace(optimizer.get(), state);
}

void GradScaler::step(const std::shared_ptr<optim::Optimizer>& optimizer) {
  if (!optimizer) {
    throw std::invalid_argument("GradScaler optimizer cannot be null");
  }
  if (!options_.enabled) {
    optimizer->step();
    return;
  }

  const auto existing = optimizer_states_.find(optimizer.get());
  if (existing != optimizer_states_.end() &&
      existing->second.stage == OptimizerStage::stepped) {
    throw std::logic_error(
        "GradScaler.step() for this optimizer must be followed by update()");
  }
  unscale(optimizer);
  auto& state = optimizer_states_.at(optimizer.get());
  if (!state.found_inf) {
    optimizer->step();
  }
  state.stage = OptimizerStage::stepped;
}

void GradScaler::update() {
  if (!options_.enabled) {
    return;
  }
  if (optimizer_states_.empty()) {
    throw std::logic_error("GradScaler.update() requires a preceding step()");
  }
  for (const auto& [optimizer, state] : optimizer_states_) {
    static_cast<void>(optimizer);
    if (state.stage != OptimizerStage::stepped) {
      throw std::logic_error(
          "GradScaler.update() requires step() for every unscaled optimizer");
    }
  }

  if (found_inf_) {
    scale_ = std::max(scale_ * options_.backoff_factor,
                      std::numeric_limits<double>::min());
    growth_tracker_ = 0;
  } else {
    ++growth_tracker_;
    if (growth_tracker_ >=
        static_cast<std::uint64_t>(options_.growth_interval)) {
      const double grown = scale_ * options_.growth_factor;
      if (!std::isfinite(grown)) {
        throw std::overflow_error("GradScaler scale overflow");
      }
      scale_ = grown;
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
