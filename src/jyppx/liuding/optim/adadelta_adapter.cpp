// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/optim/adadelta_adapter.hpp>

#include <torch/optim/optimizer.h>

#include <ATen/core/grad_mode.h>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace jyppx::liuding::optim {
namespace {

/** @brief Private option object used by the common stateful adapter engine. */
class NativeAdadeltaOptions final : public torch::optim::OptimizerOptions {
 public:
  /** @brief Creates options with the supplied learning rate. */
  explicit NativeAdadeltaOptions(double learning_rate = 1.0)
      : learning_rate_(learning_rate) {}

  /** @brief Returns the mutable learning rate. */
  double lr() const noexcept { return learning_rate_; }

  /** @brief Replaces the mutable learning rate and supports fluent setup. */
  NativeAdadeltaOptions& lr(double value) noexcept {
    learning_rate_ = value;
    return *this;
  }

  /** @brief Returns the running-average decay factor. */
  double rho() const noexcept { return rho_; }

  /** @brief Replaces the decay factor during private construction. */
  NativeAdadeltaOptions& rho(double value) noexcept {
    rho_ = value;
    return *this;
  }

  /** @brief Returns the numerical stability constant. */
  double eps() const noexcept { return epsilon_; }

  /** @brief Replaces the numerical stability constant. */
  NativeAdadeltaOptions& eps(double value) noexcept {
    epsilon_ = value;
    return *this;
  }

  /** @brief Returns the coupled regularization coefficient. */
  double weight_decay() const noexcept { return weight_decay_; }

  /** @brief Replaces the regularization coefficient. */
  NativeAdadeltaOptions& weight_decay(double value) noexcept {
    weight_decay_ = value;
    return *this;
  }

  /** @brief Reports whether gradients are negated to maximize the objective. */
  bool maximize() const noexcept { return maximize_; }

  /** @brief Selects objective minimization or maximization. */
  NativeAdadeltaOptions& maximize(bool value) noexcept {
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
    return std::make_unique<NativeAdadeltaOptions>(*this);
  }

  double learning_rate_ = 1.0;  ///< Current group learning rate.
  double rho_ = 0.9;            ///< Immutable running-average decay factor.
  double epsilon_ = 1e-6;       ///< Immutable stability constant.
  double weight_decay_ = 0.0;   ///< Immutable coupled L2 coefficient.
  bool maximize_ = false;       ///< Whether the update maximizes the objective.
};

/** @brief Private per-parameter Adadelta state owned by the native optimizer.
 */
class NativeAdadeltaParamState final
    : public torch::optim::OptimizerCloneableParamState<
          NativeAdadeltaParamState> {
 public:
  /** @brief Returns the number of updates applied to this parameter. */
  std::int64_t step() const noexcept { return step_; }

  /** @brief Replaces the per-parameter update count. */
  NativeAdadeltaParamState& step(std::int64_t value) noexcept {
    step_ = value;
    return *this;
  }

  /** @brief Returns the running average of squared gradients. */
  const Tensor& square_avg() const noexcept { return square_avg_; }

  /** @brief Returns mutable squared-gradient average storage. */
  Tensor& square_avg() noexcept { return square_avg_; }

  /** @brief Replaces squared-gradient storage during restore. */
  NativeAdadeltaParamState& square_avg(Tensor value) {
    square_avg_ = std::move(value);
    return *this;
  }

  /** @brief Returns the running average of squared parameter deltas. */
  const Tensor& acc_delta() const noexcept { return acc_delta_; }

  /** @brief Returns mutable accumulated-delta storage. */
  Tensor& acc_delta() noexcept { return acc_delta_; }

  /** @brief Replaces accumulated-delta storage during restore. */
  NativeAdadeltaParamState& acc_delta(Tensor value) {
    acc_delta_ = std::move(value);
    return *this;
  }

 private:
  std::int64_t step_ = 0;  ///< Successful updates for this parameter.
  Tensor square_avg_;      ///< Exponential average of squared gradients.
  Tensor acc_delta_;       ///< Exponential average of squared deltas.
};

/** @brief Private tensor kernel aligned with torch.optim.Adadelta. */
class NativeAdadeltaOptimizer final : public torch::optim::Optimizer {
 public:
  /** @brief Creates an optimizer over explicit LibTorch parameter groups. */
  NativeAdadeltaOptimizer(
      const std::vector<torch::optim::OptimizerParamGroup>& groups,
      NativeAdadeltaOptions defaults)
      : Optimizer(groups, std::make_unique<NativeAdadeltaOptions>(
                              std::move(defaults))) {}

  /** @brief Executes one Adadelta update over parameters with gradients. */
  Tensor step(LossClosure closure = nullptr) override {
    Tensor loss;
    if (closure) {
      at::AutoGradMode enable_grad(true);
      loss = closure();
    }

    at::NoGradGuard no_grad;
    for (auto& group : param_groups()) {
      auto* options =
          dynamic_cast<NativeAdadeltaOptions*>(std::addressof(group.options()));
      TORCH_CHECK(options != nullptr, "Invalid native Adadelta option type");

      for (auto& parameter : group.params()) {
        auto gradient = parameter.grad();
        if (!gradient.defined()) {
          continue;
        }

        const auto key = parameter.unsafeGetTensorImpl();
        auto iterator = state().find(key);
        if (iterator == state().end()) {
          auto created = std::make_unique<NativeAdadeltaParamState>();
          created->square_avg(
              torch::zeros_like(parameter, at::MemoryFormat::Preserve));
          created->acc_delta(
              torch::zeros_like(parameter, at::MemoryFormat::Preserve));
          iterator = state().emplace(key, std::move(created)).first;
        }
        auto* parameter_state =
            dynamic_cast<NativeAdadeltaParamState*>(iterator->second.get());
        TORCH_CHECK(parameter_state != nullptr,
                    "Invalid native Adadelta parameter state type");
        TORCH_CHECK(
            parameter_state->step() < std::numeric_limits<std::int64_t>::max(),
            "Native Adadelta parameter step counter exhausted");
        parameter_state->step(parameter_state->step() + 1);

        if (options->maximize()) {
          gradient = gradient.neg();
        }
        if (options->weight_decay() != 0.0) {
          gradient = gradient.add(parameter, options->weight_decay());
        }

        auto& square_average = parameter_state->square_avg();
        auto& accumulated_delta = parameter_state->acc_delta();
        square_average.mul_(options->rho())
            .addcmul_(gradient, gradient, 1.0 - options->rho());
        auto standard_deviation = square_average.add(options->eps()).sqrt();
        auto delta = accumulated_delta.add(options->eps())
                         .sqrt()
                         .div_(standard_deviation)
                         .mul_(gradient);
        accumulated_delta.mul_(options->rho())
            .addcmul_(delta, delta, 1.0 - options->rho());
        parameter.add_(delta, -options->lr());
      }
    }
    return loss;
  }
};

}  // namespace
}  // namespace jyppx::liuding::optim

#define JYPPX_STATEFUL_OPTIMIZER_ADADELTA
#define JYPPX_STATEFUL_OPTIMIZER_ADAPTER AdadeltaOptimizerAdapter
#define JYPPX_STATEFUL_OPTIMIZER_OPTIONS AdadeltaAdapterOptions
#define JYPPX_STATEFUL_OPTIMIZER_GROUP AdadeltaParameterGroup
#define JYPPX_STATEFUL_OPTIMIZER_CHECKPOINT_OPTIONS \
  AdadeltaOptimizerCheckpointOptions
#define JYPPX_STATEFUL_OPTIMIZER_PARTICIPANT \
  AdadeltaOptimizerCheckpointParticipant
#define JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIMIZER NativeAdadeltaOptimizer
#define JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIONS NativeAdadeltaOptions
#define JYPPX_STATEFUL_OPTIMIZER_NATIVE_STATE NativeAdadeltaParamState
#define JYPPX_STATEFUL_OPTIMIZER_NAME "Adadelta"
#define JYPPX_STATEFUL_OPTIMIZER_COMPONENT "optim.adadelta_adapter"
#define JYPPX_STATEFUL_OPTIMIZER_N1_MIGRATION
#define JYPPX_STATEFUL_OPTIMIZER_MAGIC_TAG 'D'
#define JYPPX_STATEFUL_OPTIMIZER_PRODUCER_MAGIC_TAG 'D'
#define JYPPX_STATEFUL_OPTIMIZER_PRODUCER_MAGIC "JYPPXDEP"
#define JYPPX_STATEFUL_OPTIMIZER_TOPOLOGY_MAGIC "JYPPXADM"
#define JYPPX_STATEFUL_OPTIMIZER_CONTROL_MAGIC "JYPPXADC"
#define JYPPX_STATEFUL_OPTIMIZER_MEDIA_STEM "adadelta"
#define JYPPX_STATEFUL_OPTIMIZER_NATIVE_OPTIONS_NAME "NativeAdadeltaOptions"
#define JYPPX_STATEFUL_OPTIMIZER_NATIVE_STATE_NAME "NativeAdadeltaParamState"
#define JYPPX_STATEFUL_OPTIMIZER_ADAPTER_DESCRIPTION \
  "a non-null shared AdadeltaOptimizerAdapter"

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
#undef JYPPX_STATEFUL_OPTIMIZER_ADADELTA
