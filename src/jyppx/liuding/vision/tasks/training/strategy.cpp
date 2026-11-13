// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/vision/tasks/training/strategy.hpp>

#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/instancenorm.h>
#include <torch/nn/modules/normalization.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <utility>

namespace jyppx::liuding::vision::tasks {
namespace {
bool Normalization(const torch::nn::Module& module) {
  return dynamic_cast<const torch::nn::BatchNorm1dImpl*>(&module) ||
         dynamic_cast<const torch::nn::BatchNorm2dImpl*>(&module) ||
         dynamic_cast<const torch::nn::BatchNorm3dImpl*>(&module) ||
         dynamic_cast<const torch::nn::InstanceNorm1dImpl*>(&module) ||
         dynamic_cast<const torch::nn::InstanceNorm2dImpl*>(&module) ||
         dynamic_cast<const torch::nn::InstanceNorm3dImpl*>(&module) ||
         dynamic_cast<const torch::nn::GroupNormImpl*>(&module) ||
         dynamic_cast<const torch::nn::LayerNormImpl*>(&module);
}

void Validate(const WarmupCosineOptions& options) {
  if (options.total_updates == 0 ||
      options.total_updates > static_cast<std::uint64_t>(
                                  (std::numeric_limits<std::int64_t>::max)()) ||
      options.warmup_updates >= options.total_updates ||
      !std::isfinite(options.warmup_start_factor) ||
      options.warmup_start_factor <= 0 || options.warmup_start_factor > 1 ||
      !std::isfinite(options.final_factor) || options.final_factor < 0 ||
      options.final_factor > 1)
    throw InvalidArgument("Invalid successful-update warmup/cosine recipe");
}
}  // namespace

DetectionSgdGroups make_detection_sgd_groups(
    torch::nn::Module& model, const DetectionSgdOptions& options) {
  if (!std::isfinite(options.learning_rate) || options.learning_rate <= 0 ||
      !std::isfinite(options.momentum) || options.momentum < 0 ||
      options.momentum >= 1 || !std::isfinite(options.weight_decay) ||
      options.weight_decay < 0 || options.effective_batch == 0 ||
      options.nominal_batch == 0 || (options.nesterov && options.momentum == 0))
    throw InvalidArgument("Invalid detection SGD parameter-group recipe");
  DetectionSgdGroups result;
  result.effective_weight_decay = options.weight_decay;
  if (options.scale_weight_decay)
    result.effective_weight_decay *=
        static_cast<double>(options.effective_batch) /
        static_cast<double>(options.nominal_batch);
  if (!std::isfinite(result.effective_weight_decay))
    throw InvalidArgument("Scaled detection weight decay is not finite");
  std::set<std::string> normalization;
  const auto collect = [&](const torch::nn::Module& owner,
                           const std::string& prefix) {
    if (Normalization(owner))
      for (const auto& parameter : owner.named_parameters(false))
        normalization.insert(prefix + parameter.key());
  };
  collect(model, "");
  for (const auto& module : model.named_modules("", false))
    collect(*module.value(), module.key() + ".");
  std::array<optim::SgdParameterGroup, 3> groups;
  groups[0].group_id = "decay";
  groups[1].group_id = "no_decay";
  groups[2].group_id = "bias";
  for (auto& group : groups) {
    group.options.learning_rate = options.learning_rate;
    group.options.momentum = options.momentum;
    group.options.nesterov = options.nesterov;
  }
  groups[0].options.weight_decay = result.effective_weight_decay;
  for (const auto& parameter : model.named_parameters()) {
    if (!parameter.value().requires_grad()) continue;
    const auto count = parameter.value().numel();
    if (count >
        (std::numeric_limits<std::int64_t>::max)() - result.parameter_elements)
      throw ResourceExhausted(
          "Detection trainable parameter count exceeds int64");
    result.parameter_elements += count;
    const auto& name = parameter.key();
    const auto index =
        name == "bias" || name.ends_with(".bias")
            ? 2U
            : (normalization.contains(name) || parameter.value().dim() <= 1
                   ? 1U
                   : 0U);
    groups[index].parameters.push_back(
        {.parameter_id = name, .parameter = parameter.value()});
  }
  for (auto& group : groups)
    if (!group.parameters.empty()) result.groups.push_back(std::move(group));
  if (result.groups.empty())
    throw InvalidArgument("Detection model has no trainable parameters");
  return result;
}

double warmup_cosine_factor(const WarmupCosineOptions& options,
                            std::uint64_t completed_updates) {
  Validate(options);
  if (completed_updates < options.warmup_updates)
    return options.warmup_start_factor +
           (1 - options.warmup_start_factor) *
               (static_cast<double>(completed_updates) /
                static_cast<double>(options.warmup_updates));
  const auto elapsed = (std::min)(completed_updates, options.total_updates) -
                       options.warmup_updates;
  const auto progress =
      static_cast<double>(elapsed) /
      static_cast<double>(options.total_updates - options.warmup_updates);
  return options.final_factor + (1 - options.final_factor) *
                                    (1 + std::cos(std::acos(-1.0) * progress)) *
                                    0.5;
}

WarmupCosineLrScheduler::WarmupCosineLrScheduler(
    std::shared_ptr<optim::OptimizerRateControl> optimizer,
    WarmupCosineOptions options)
    : LrScheduler(std::move(optimizer), options.scheduler_id),
      options_(std::move(options)) {
  Validate(options_);
  configure_descriptor(
      "jyppx.vision.scheduler.warmup-cosine/v1",
      optim::SchedulerStepMode::batch,
      {{"base_learning_rates", base_learning_rates()},
       {"total_updates", static_cast<std::int64_t>(options_.total_updates)},
       {"warmup_updates", static_cast<std::int64_t>(options_.warmup_updates)},
       {"warmup_start_factor", options_.warmup_start_factor},
       {"final_factor", options_.final_factor}});
  auto initial = base_learning_rates();
  const auto factor = warmup_cosine_factor(options_, 0);
  for (auto& rate : initial) rate *= factor;
  initialize_learning_rates(initial);
}

WarmupCosineLrScheduler::~WarmupCosineLrScheduler() = default;

optim::detail::SchedulerTransition WarmupCosineLrScheduler::preview_step_impl(
    std::span<const double> learning_rates) const {
  static_cast<void>(learning_rates);
  if (completed_ == (std::numeric_limits<std::uint64_t>::max)())
    throw ResourceExhausted("Vision scheduler update counter exhausted");
  const auto next = completed_ + 1;
  auto rates = base_learning_rates();
  const auto factor = warmup_cosine_factor(options_, next);
  for (auto& rate : rates) rate *= factor;
  return {.state = {.completed_steps = next},
          .learning_rates = std::move(rates)};
}

optim::detail::SchedulerState WarmupCosineLrScheduler::capture_state_impl()
    const {
  return {.completed_steps = completed_};
}

std::uint64_t WarmupCosineLrScheduler::completed_steps_impl() const noexcept {
  return completed_;
}

void WarmupCosineLrScheduler::validate_state_impl(
    const optim::detail::SchedulerState& state) const {
  if (!state.control.empty() || !state.children.empty())
    throw CompatibilityError(
        "Vision warmup/cosine scheduler requires a counter-only leaf");
}

void WarmupCosineLrScheduler::apply_state_impl(
    const optim::detail::SchedulerState& state) noexcept {
  completed_ = state.completed_steps;
}
}  // namespace jyppx::liuding::vision::tasks
