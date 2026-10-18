// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/optim/lamb_adapter.hpp>

#include <torch/optim/optimizer.h>

#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace jyppx::optim {
namespace {

/** @brief Frozen upper bound applied to the layer parameter norm. */
constexpr double kMaximumWeightNorm = 10.0;

/** @brief Private option object used by the common stateful adapter engine. */
class NativeLambOptions final : public torch::optim::OptimizerOptions {
 public:
  /** @brief Creates options with the supplied learning rate. */
  explicit NativeLambOptions(double learning_rate = 1e-3)
      : learning_rate_(learning_rate) {}

  /** @brief Returns the mutable learning rate. */
  double lr() const noexcept { return learning_rate_; }

  /** @brief Replaces the mutable learning rate and supports fluent setup. */
  NativeLambOptions& lr(double value) noexcept {
    learning_rate_ = value;
    return *this;
  }

  /** @brief Returns first- and second-moment decay factors. */
  std::pair<double, double> betas() const noexcept { return betas_; }

  /** @brief Replaces decay factors during private construction. */
  NativeLambOptions& betas(std::pair<double, double> value) noexcept {
    betas_ = value;
    return *this;
  }

  /** @brief Returns the numerical stability constant. */
  double eps() const noexcept { return epsilon_; }

  /** @brief Replaces the numerical stability constant. */
  NativeLambOptions& eps(double value) noexcept {
    epsilon_ = value;
    return *this;
  }

  /** @brief Returns the update-direction regularization coefficient. */
  double weight_decay() const noexcept { return weight_decay_; }

  /** @brief Replaces the regularization coefficient. */
  NativeLambOptions& weight_decay(double value) noexcept {
    weight_decay_ = value;
    return *this;
  }

  /** @brief Reports whether gradients are negated to maximize the objective. */
  bool maximize() const noexcept { return maximize_; }

  /** @brief Selects objective minimization or maximization. */
  NativeLambOptions& maximize(bool value) noexcept {
    maximize_ = value;
    return *this;
  }

  /** @brief Implements the generic optimizer learning-rate query. */
  double get_lr() const override { return learning_rate_; }

  /** @brief Implements the generic optimizer learning-rate mutation. */
  void set_lr(double value) override { learning_rate_ = value; }

 private:
  /** @brief Clones options when LibTorch creates a parameter group. */
  std::unique_ptr<torch::optim::OptimizerOptions> clone() const override {
    return std::make_unique<NativeLambOptions>(*this);
  }

  double learning_rate_ = 1e-3;  ///< Current group learning rate.
  std::pair<double, double> betas_ = {0.9, 0.999};  ///< Moment decays.
  double epsilon_ = 1e-6;      ///< Immutable stability constant.
  double weight_decay_ = 0.0;  ///< Immutable update-direction L2 coefficient.
  bool maximize_ = false;      ///< Whether the update maximizes the objective.
};

/** @brief Private per-parameter LAMB state owned by the native optimizer. */
class NativeLambParamState final
    : public torch::optim::OptimizerCloneableParamState<NativeLambParamState> {
 public:
  /** @brief Returns the number of updates applied to this parameter. */
  std::int64_t step() const noexcept { return step_; }

  /** @brief Replaces the per-parameter update count. */
  NativeLambParamState& step(std::int64_t value) noexcept {
    step_ = value;
    return *this;
  }

  /** @brief Returns the first-moment accumulator. */
  const Tensor& exp_avg() const noexcept { return exp_avg_; }

  /** @brief Returns mutable first-moment accumulator storage. */
  Tensor& exp_avg() noexcept { return exp_avg_; }

  /** @brief Replaces first-moment storage during restore. */
  NativeLambParamState& exp_avg(Tensor value) {
    exp_avg_ = std::move(value);
    return *this;
  }

  /** @brief Returns the second-moment accumulator. */
  const Tensor& exp_avg_sq() const noexcept { return exp_avg_sq_; }

  /** @brief Returns mutable second-moment accumulator storage. */
  Tensor& exp_avg_sq() noexcept { return exp_avg_sq_; }

  /** @brief Replaces second-moment storage during restore. */
  NativeLambParamState& exp_avg_sq(Tensor value) {
    exp_avg_sq_ = std::move(value);
    return *this;
  }

 private:
  std::int64_t step_ = 0;  ///< Successful updates for this parameter.
  Tensor exp_avg_;         ///< Exponential average of gradients.
  Tensor exp_avg_sq_;      ///< Exponential average of squared gradients.
};

/** @brief Private tensor kernel implementing the frozen JYPPX LAMB contract. */
class NativeLambOptimizer final : public torch::optim::Optimizer {
 public:
  /** @brief Creates an optimizer over explicit LibTorch parameter groups. */
  NativeLambOptimizer(
      const std::vector<torch::optim::OptimizerParamGroup>& groups,
      NativeLambOptions defaults)
      : Optimizer(groups,
                  std::make_unique<NativeLambOptions>(std::move(defaults))) {}

  /** @brief Executes one LAMB update over parameters with gradients. */
  Tensor step(LossClosure closure = nullptr) override {
    Tensor loss;
    if (closure) {
      at::AutoGradMode enable_grad(true);
      loss = closure();
    }

    at::NoGradGuard no_grad;
    for (auto& group : param_groups()) {
      auto* options =
          dynamic_cast<NativeLambOptions*>(std::addressof(group.options()));
      TORCH_CHECK(options != nullptr, "Invalid native LAMB option type");
      const auto [beta1, beta2] = options->betas();

      for (auto& parameter : group.params()) {
        auto gradient = parameter.grad();
        if (!gradient.defined()) {
          continue;
        }

        const auto key = parameter.unsafeGetTensorImpl();
        auto iterator = state().find(key);
        if (iterator == state().end()) {
          auto created = std::make_unique<NativeLambParamState>();
          created->exp_avg(
              torch::zeros_like(parameter, at::MemoryFormat::Preserve));
          created->exp_avg_sq(
              torch::zeros_like(parameter, at::MemoryFormat::Preserve));
          iterator = state().emplace(key, std::move(created)).first;
        }
        auto* parameter_state =
            dynamic_cast<NativeLambParamState*>(iterator->second.get());
        TORCH_CHECK(parameter_state != nullptr,
                    "Invalid native LAMB parameter state type");
        TORCH_CHECK(
            parameter_state->step() < std::numeric_limits<std::int64_t>::max(),
            "Native LAMB parameter step counter exhausted");
        parameter_state->step(parameter_state->step() + 1);

        if (options->maximize()) {
          gradient = gradient.neg();
        }
        auto& first_moment = parameter_state->exp_avg();
        auto& second_moment = parameter_state->exp_avg_sq();
        first_moment.mul_(beta1).add_(gradient, 1.0 - beta1);
        second_moment.mul_(beta2).addcmul_(gradient, gradient, 1.0 - beta2);

        const auto step = static_cast<double>(parameter_state->step());
        const auto bias_correction1 = 1.0 - std::pow(beta1, step);
        const auto bias_correction2 = 1.0 - std::pow(beta2, step);
        auto direction = first_moment.div(bias_correction1)
                             .div_(second_moment.div(bias_correction2)
                                       .sqrt()
                                       .add_(options->eps()));
        if (options->weight_decay() != 0.0) {
          direction.add_(parameter, options->weight_decay());
        }

        const auto weight_norm =
            std::min(parameter.norm().item<double>(), kMaximumWeightNorm);
        const auto direction_norm = direction.norm().item<double>();
        const auto trust_ratio = weight_norm == 0.0 || direction_norm == 0.0
                                     ? 1.0
                                     : weight_norm / direction_norm;
        parameter.add_(direction, -options->lr() * trust_ratio);
      }
    }
    return loss;
  }
};

}  // namespace
}  // namespace jyppx::optim

#define JYPPX_STATEFUL_OPTIMIZER_LAMB
#define JYPPX_STATEFUL_OPTIMIZER_ADAPTER LambOptimizerAdapter
#define JYPPX_STATEFUL_OPTIMIZER_OPTIONS LambAdapterOptions
#define JYPPX_STATEFUL_OPTIMIZER_GROUP LambParameterGroup
#define JYPPX_STATEFUL_OPTIMIZER_CHECKPOINT_OPTIONS \
  LambOptimizerCheckpointOptions
#define JYPPX_STATEFUL_OPTIMIZER_PARTICIPANT LambOptimizerCheckpointParticipant
#define JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIMIZER NativeLambOptimizer
#define JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIONS NativeLambOptions
#define JYPPX_STATEFUL_OPTIMIZER_NATIVE_STATE NativeLambParamState
#define JYPPX_STATEFUL_OPTIMIZER_NAME "LAMB"
#define JYPPX_STATEFUL_OPTIMIZER_COMPONENT "optim.lamb_adapter"
#define JYPPX_STATEFUL_OPTIMIZER_MAGIC_TAG 'L'
#define JYPPX_STATEFUL_OPTIMIZER_TOPOLOGY_MAGIC "JYPPXALM"
#define JYPPX_STATEFUL_OPTIMIZER_CONTROL_MAGIC "JYPPXALC"
#define JYPPX_STATEFUL_OPTIMIZER_MEDIA_STEM "lamb"
#define JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIONS_NAME "NativeLambOptions"
#define JYPPX_STATEFUL_OPTIMIZER_NATIVE_STATE_NAME "NativeLambParamState"
#define JYPPX_STATEFUL_OPTIMIZER_ADAPTER_DESCRIPTION \
  "a non-null shared LambOptimizerAdapter"

#include "detail/stateful_optimizer_adapter.inc"

#undef JYPPX_STATEFUL_OPTIMIZER_ADAPTER_DESCRIPTION
#undef JYPPX_STATEFUL_OPTIMIZER_NATIVE_STATE_NAME
#undef JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIONS_NAME
#undef JYPPX_STATEFUL_OPTIMIZER_MEDIA_STEM
#undef JYPPX_STATEFUL_OPTIMIZER_CONTROL_MAGIC
#undef JYPPX_STATEFUL_OPTIMIZER_TOPOLOGY_MAGIC
#undef JYPPX_STATEFUL_OPTIMIZER_MAGIC_TAG
#undef JYPPX_STATEFUL_OPTIMIZER_COMPONENT
#undef JYPPX_STATEFUL_OPTIMIZER_NAME
#undef JYPPX_STATEFUL_OPTIMIZER_NATIVE_STATE
#undef JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIONS
#undef JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIMIZER
#undef JYPPX_STATEFUL_OPTIMIZER_PARTICIPANT
#undef JYPPX_STATEFUL_OPTIMIZER_CHECKPOINT_OPTIONS
#undef JYPPX_STATEFUL_OPTIMIZER_GROUP
#undef JYPPX_STATEFUL_OPTIMIZER_OPTIONS
#undef JYPPX_STATEFUL_OPTIMIZER_ADAPTER
#undef JYPPX_STATEFUL_OPTIMIZER_LAMB
