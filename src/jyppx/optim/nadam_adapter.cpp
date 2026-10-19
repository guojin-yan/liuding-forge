// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/optim/nadam_adapter.hpp>

#include <torch/optim/optimizer.h>

#include <ATen/core/grad_mode.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

namespace jyppx::optim {
namespace {

/** @brief Private option object used by the common stateful adapter engine. */
class NativeNAdamOptions final : public torch::optim::OptimizerOptions {
 public:
  /** @brief Creates options with the supplied learning rate. */
  explicit NativeNAdamOptions(double learning_rate = 0.002)
      : learning_rate_(learning_rate) {}

  /** @brief Returns the mutable learning rate. */
  double lr() const noexcept { return learning_rate_; }

  /** @brief Replaces the mutable learning rate and supports fluent setup. */
  NativeNAdamOptions& lr(double value) noexcept {
    learning_rate_ = value;
    return *this;
  }

  /** @brief Returns the immutable first- and second-moment decay factors. */
  std::tuple<double, double> betas() const noexcept { return betas_; }

  /** @brief Replaces decay factors during private construction. */
  NativeNAdamOptions& betas(std::tuple<double, double> value) noexcept {
    betas_ = value;
    return *this;
  }

  /** @brief Returns the denominator stability constant. */
  double eps() const noexcept { return epsilon_; }

  /** @brief Replaces the denominator stability constant. */
  NativeNAdamOptions& eps(double value) noexcept {
    epsilon_ = value;
    return *this;
  }

  /** @brief Returns the regularization coefficient. */
  double weight_decay() const noexcept { return weight_decay_; }

  /** @brief Replaces the regularization coefficient. */
  NativeNAdamOptions& weight_decay(double value) noexcept {
    weight_decay_ = value;
    return *this;
  }

  /** @brief Returns the momentum-schedule decay exponent. */
  double momentum_decay() const noexcept { return momentum_decay_; }

  /** @brief Replaces the momentum-schedule decay exponent. */
  NativeNAdamOptions& momentum_decay(double value) noexcept {
    momentum_decay_ = value;
    return *this;
  }

  /** @brief Reports whether weight decay is applied outside the moments. */
  bool decoupled_weight_decay() const noexcept {
    return decoupled_weight_decay_;
  }

  /** @brief Selects coupled NAdam or decoupled NAdamW regularization. */
  NativeNAdamOptions& decoupled_weight_decay(bool value) noexcept {
    decoupled_weight_decay_ = value;
    return *this;
  }

  /** @brief Reports whether gradients are negated to maximize the objective. */
  bool maximize() const noexcept { return maximize_; }

  /** @brief Selects objective minimization or maximization. */
  NativeNAdamOptions& maximize(bool value) noexcept {
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
    return std::make_unique<NativeNAdamOptions>(*this);
  }

  double learning_rate_ = 0.002;  ///< Current group learning rate.
  std::tuple<double, double> betas_ = {
      0.9, 0.999};             ///< Immutable moment decay factors.
  double epsilon_ = 1e-8;      ///< Immutable denominator constant.
  double weight_decay_ = 0.0;  ///< Immutable regularization coefficient.
  double momentum_decay_ =
      0.004;  ///< Immutable momentum-schedule decay exponent.
  bool decoupled_weight_decay_ =
      false;  ///< Whether parameter decay bypasses moment estimates.
  bool maximize_ = false;  ///< Whether the update maximizes the objective.
};

/** @brief Private per-parameter NAdam state owned by the native optimizer. */
class NativeNAdamParamState final
    : public torch::optim::OptimizerCloneableParamState<NativeNAdamParamState> {
 public:
  /** @brief Returns the number of updates applied to this parameter. */
  std::int64_t step() const noexcept { return step_; }

  /** @brief Replaces the per-parameter update count. */
  NativeNAdamParamState& step(std::int64_t value) noexcept {
    step_ = value;
    return *this;
  }

  /** @brief Returns the first-moment tensor. */
  const Tensor& exp_avg() const noexcept { return exp_avg_; }

  /** @brief Returns mutable first-moment storage. */
  Tensor& exp_avg() noexcept { return exp_avg_; }

  /** @brief Replaces first-moment storage during restore. */
  NativeNAdamParamState& exp_avg(Tensor value) {
    exp_avg_ = std::move(value);
    return *this;
  }

  /** @brief Returns the second-moment tensor. */
  const Tensor& exp_avg_sq() const noexcept { return exp_avg_sq_; }

  /** @brief Returns mutable second-moment storage. */
  Tensor& exp_avg_sq() noexcept { return exp_avg_sq_; }

  /** @brief Replaces second-moment storage during restore. */
  NativeNAdamParamState& exp_avg_sq(Tensor value) {
    exp_avg_sq_ = std::move(value);
    return *this;
  }

  /** @brief Returns the float32-compatible cumulative momentum product. */
  float mu_product() const noexcept { return mu_product_; }

  /** @brief Replaces the cumulative momentum product during update/restore. */
  NativeNAdamParamState& mu_product(float value) noexcept {
    mu_product_ = value;
    return *this;
  }

 private:
  std::int64_t step_ = 0;  ///< Successful updates for this parameter.
  Tensor exp_avg_;         ///< First exponential moving average.
  Tensor exp_avg_sq_;      ///< Second exponential moving average.
  float mu_product_ =
      1.0F;  ///< Momentum product using PyTorch's scalar-state precision.
};

/**
 * @brief Private tensor kernel aligned with torch.optim.NAdam single-tensor.
 */
class NativeNAdamOptimizer final : public torch::optim::Optimizer {
 public:
  /** @brief Creates an optimizer over explicit LibTorch parameter groups. */
  NativeNAdamOptimizer(
      const std::vector<torch::optim::OptimizerParamGroup>& groups,
      NativeNAdamOptions defaults)
      : Optimizer(groups,
                  std::make_unique<NativeNAdamOptions>(std::move(defaults))) {}

  /** @brief Executes one NAdam/NAdamW update over parameters with gradients. */
  Tensor step(LossClosure closure = nullptr) override {
    Tensor loss;
    if (closure) {
      at::AutoGradMode enable_grad(true);
      loss = closure();
    }

    at::NoGradGuard no_grad;
    for (auto& group : param_groups()) {
      auto* options =
          dynamic_cast<NativeNAdamOptions*>(std::addressof(group.options()));
      TORCH_CHECK(options != nullptr, "Invalid native NAdam option type");
      const auto [beta1, beta2] = options->betas();

      for (auto& parameter : group.params()) {
        auto gradient = parameter.grad();
        if (!gradient.defined()) {
          continue;
        }

        const auto key = parameter.unsafeGetTensorImpl();
        auto iterator = state().find(key);
        if (iterator == state().end()) {
          auto created = std::make_unique<NativeNAdamParamState>();
          created->exp_avg(
              torch::zeros_like(parameter, at::MemoryFormat::Preserve));
          created->exp_avg_sq(
              torch::zeros_like(parameter, at::MemoryFormat::Preserve));
          iterator = state().emplace(key, std::move(created)).first;
        }
        auto* parameter_state =
            dynamic_cast<NativeNAdamParamState*>(iterator->second.get());
        TORCH_CHECK(parameter_state != nullptr,
                    "Invalid native NAdam parameter state type");
        TORCH_CHECK(
            parameter_state->step() < std::numeric_limits<std::int64_t>::max(),
            "Native NAdam parameter step counter exhausted");

        const auto step = parameter_state->step() + 1;
        parameter_state->step(step);
        if (options->maximize()) {
          gradient = gradient.neg();
        }
        if (options->weight_decay() != 0.0) {
          if (options->decoupled_weight_decay()) {
            parameter.mul_(1.0 - options->lr() * options->weight_decay());
          } else {
            gradient = gradient.add(parameter, options->weight_decay());
          }
        }

        const auto numeric_step = static_cast<double>(step);
        const auto mu =
            beta1 * (1.0 - 0.5 * std::pow(0.96, numeric_step *
                                                    options->momentum_decay()));
        const auto mu_next =
            beta1 * (1.0 - 0.5 * std::pow(0.96, (numeric_step + 1.0) *
                                                    options->momentum_decay()));
        const auto mu_product = static_cast<float>(
            parameter_state->mu_product() * static_cast<float>(mu));
        parameter_state->mu_product(mu_product);

        auto& first = parameter_state->exp_avg();
        auto& second = parameter_state->exp_avg_sq();
        first.lerp_(gradient, 1.0 - beta1);
        second.mul_(beta2).addcmul_(gradient, gradient, 1.0 - beta2);

        const auto bias_correction2 = 1.0 - std::pow(beta2, numeric_step);
        auto denominator =
            second.div(bias_correction2).sqrt().add_(options->eps());
        const auto mu_product_next = static_cast<double>(mu_product) * mu_next;
        parameter.addcdiv_(gradient, denominator,
                           -options->lr() * (1.0 - mu) /
                               (1.0 - static_cast<double>(mu_product)));
        parameter.addcdiv_(first, denominator,
                           -options->lr() * mu_next / (1.0 - mu_product_next));
      }
    }
    return loss;
  }
};

}  // namespace
}  // namespace jyppx::optim

#define JYPPX_STATEFUL_OPTIMIZER_NADAM
#define JYPPX_STATEFUL_OPTIMIZER_ADAPTER NAdamOptimizerAdapter
#define JYPPX_STATEFUL_OPTIMIZER_OPTIONS NAdamAdapterOptions
#define JYPPX_STATEFUL_OPTIMIZER_GROUP NAdamParameterGroup
#define JYPPX_STATEFUL_OPTIMIZER_CHECKPOINT_OPTIONS \
  NAdamOptimizerCheckpointOptions
#define JYPPX_STATEFUL_OPTIMIZER_PARTICIPANT NAdamOptimizerCheckpointParticipant
#define JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIMIZER NativeNAdamOptimizer
#define JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIONS NativeNAdamOptions
#define JYPPX_STATEFUL_OPTIMIZER_NATIVE_STATE NativeNAdamParamState
#define JYPPX_STATEFUL_OPTIMIZER_NAME "NAdam"
#define JYPPX_STATEFUL_OPTIMIZER_COMPONENT "optim.nadam_adapter"
#define JYPPX_STATEFUL_OPTIMIZER_N1_MIGRATION
#define JYPPX_STATEFUL_OPTIMIZER_MAGIC_TAG 'N'
#define JYPPX_STATEFUL_OPTIMIZER_PRODUCER_MAGIC_TAG 'N'
#define JYPPX_STATEFUL_OPTIMIZER_PRODUCER_MAGIC "JYPPXANP"
#define JYPPX_STATEFUL_OPTIMIZER_TOPOLOGY_MAGIC "JYPPXANM"
#define JYPPX_STATEFUL_OPTIMIZER_CONTROL_MAGIC "JYPPXANC"
#define JYPPX_STATEFUL_OPTIMIZER_MEDIA_STEM "nadam"
#define JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIONS_NAME "NativeNAdamOptions"
#define JYPPX_STATEFUL_OPTIMIZER_NATIVE_STATE_NAME "NativeNAdamParamState"
#define JYPPX_STATEFUL_OPTIMIZER_ADAPTER_DESCRIPTION \
  "a non-null shared NAdamOptimizerAdapter"

#include "detail/stateful_optimizer_adapter.inc"

#undef JYPPX_STATEFUL_OPTIMIZER_ADAPTER_DESCRIPTION
#undef JYPPX_STATEFUL_OPTIMIZER_NATIVE_STATE_NAME
#undef JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIONS_NAME
#undef JYPPX_STATEFUL_OPTIMIZER_MEDIA_STEM
#undef JYPPX_STATEFUL_OPTIMIZER_CONTROL_MAGIC
#undef JYPPX_STATEFUL_OPTIMIZER_TOPOLOGY_MAGIC
#undef JYPPX_STATEFUL_OPTIMIZER_MAGIC_TAG
#undef JYPPX_STATEFUL_OPTIMIZER_COMPONENT
#undef JYPPX_STATEFUL_OPTIMIZER_PRODUCER_MAGIC
#undef JYPPX_STATEFUL_OPTIMIZER_PRODUCER_MAGIC_TAG
#undef JYPPX_STATEFUL_OPTIMIZER_N1_MIGRATION
#undef JYPPX_STATEFUL_OPTIMIZER_NAME
#undef JYPPX_STATEFUL_OPTIMIZER_NATIVE_STATE
#undef JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIONS
#undef JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIMIZER
#undef JYPPX_STATEFUL_OPTIMIZER_PARTICIPANT
#undef JYPPX_STATEFUL_OPTIMIZER_CHECKPOINT_OPTIONS
#undef JYPPX_STATEFUL_OPTIMIZER_GROUP
#undef JYPPX_STATEFUL_OPTIMIZER_OPTIONS
#undef JYPPX_STATEFUL_OPTIMIZER_ADAPTER
#undef JYPPX_STATEFUL_OPTIMIZER_NADAM
