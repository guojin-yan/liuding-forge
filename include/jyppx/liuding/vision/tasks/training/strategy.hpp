// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_VISION_TASKS_TRAINING_STRATEGY_HPP
#define JYPPX_LIUDING_VISION_TASKS_TRAINING_STRATEGY_HPP
/** @file strategy.hpp @brief Vision parameter grouping and update-based LR
 * recipes. */

#include <jyppx/liuding/optim/lr_scheduler.hpp>
#include <jyppx/liuding/optim/sgd_adapter.hpp>

#include <torch/nn/module.h>

namespace jyppx::liuding::vision::tasks {

/** @brief Explicit SGD recipe, resolved before constructing an optimizer. */
struct DetectionSgdOptions {
  double learning_rate =
      0.01;               ///< Finite positive base rate, shared by all groups.
  double momentum = 0.9;  ///< Finite value in [0,1).
  double weight_decay = 0.0005;  ///< Finite nonnegative nominal decay.
  bool nesterov = false;         ///< Requires nonzero momentum.
  bool scale_weight_decay =
      true;  ///< Multiply by effective_batch/nominal_batch.
  std::uint64_t effective_batch =
      64;  ///< Configured batch*accumulation, positive.
  std::uint64_t nominal_batch =
      64;  ///< Positive reference batch for decay scaling.
};

/** @brief Resolved disjoint groups; tensor handles remain owned by the model.
 */
struct DetectionSgdGroups {
  std::vector<optim::SgdParameterGroup>
      groups;  ///< Nonempty groups in decay/no_decay/bias order.
  double effective_weight_decay = 0;    ///< Applied only to the decay group.
  std::int64_t parameter_elements = 0;  ///< Trainable parameter element count.
};

/** @brief Builds stable SGD groups without mutating model state or rates.
 * @param model Already placed model; frozen parameters are excluded.
 * @param options Explicit algorithm and effective-batch convention.
 * @return Biases, normalization parameters and <=1D parameters have zero decay;
 * all other parameters use the resolved decay. Known LibTorch BatchNorm,
 * InstanceNorm, GroupNorm and LayerNorm modules are recognized by type.
 * @throws InvalidArgument Invalid options or no trainable parameters.
 * @throws ResourceExhausted Trainable element count overflows int64.
 * @note Keep the model topology/placement fixed after binding. Custom
 * normalization modules with multidimensional parameters need a separate
 * recipe.
 */
DetectionSgdGroups make_detection_sgd_groups(
    torch::nn::Module& model, const DetectionSgdOptions& options = {});

/** @brief Successful-update warmup followed by a clamped half-cosine decay. */
struct WarmupCosineOptions {
  std::string scheduler_id =
      "scheduler.vision.warmup-cosine";  ///< Stable participant ID.
  std::uint64_t total_updates =
      1;  ///< Positive planned update count, <= INT64_MAX.
  std::uint64_t warmup_updates =
      0;  ///< Must be strictly less than total_updates.
  double warmup_start_factor = 0.1;  ///< Construction-time factor in (0,1].
  double final_factor = 0.01;        ///< Final/base LR in [0,1].
};

/** @brief Evaluates the frozen rate factor without reading or changing state.
 * @param options Validated warmup/cosine recipe.
 * @param completed_updates Updates already committed, zero before training.
 * @return Initial warmup factor (or 1 with no warmup), 1 at the warmup
 * boundary, and final_factor at/after total_updates. Rate changes apply to the
 * next update.
 * @throws InvalidArgument Invalid recipe.
 */
double warmup_cosine_factor(const WarmupCosineOptions& options,
                            std::uint64_t completed_updates);

/** @brief Descriptor-first scheduler using the existing transactional owner.
 *
 * Bind through training::SchedulerAdapter. Trainer advances it once per
 * successful logical optimizer update, including an accumulation-tail flush;
 * AMP skips do not advance it. Momentum is fixed by the optimizer recipe.
 * Exact descriptor/state checkpointing uses the existing generic participant.
 */
class WarmupCosineLrScheduler final : public optim::LrScheduler {
 public:
  /** @brief Captures optimizer bases, freezes the recipe and applies event
   * zero.
   * @param optimizer Nonnull stable optimizer rate-control owner.
   * @param options Frozen recipe and checkpoint identity.
   * @throws InvalidArgument Invalid recipe, optimizer or topology.
   */
  WarmupCosineLrScheduler(
      std::shared_ptr<optim::OptimizerRateControl> optimizer,
      WarmupCosineOptions options = {});
  /** @brief Releases immutable recipe and scheduler state. */
  ~WarmupCosineLrScheduler() override;

 private:
  /** @brief Previews the next counter and absolute base-scaled group rates. */
  optim::detail::SchedulerTransition preview_step_impl(
      std::span<const double> learning_rates) const override;
  /** @brief Captures the successful event counter. */
  optim::detail::SchedulerState capture_state_impl() const override;
  /** @brief Returns the committed successful event count. */
  std::uint64_t completed_steps_impl() const noexcept override;
  /** @brief Requires a counter-only leaf with no child/control state. */
  void validate_state_impl(
      const optim::detail::SchedulerState& state) const override;
  /** @brief Applies a prevalidated counter without throwing. */
  void apply_state_impl(
      const optim::detail::SchedulerState& state) noexcept override;
  WarmupCosineOptions options_;  ///< Frozen checkpoint-compatible recipe.
  std::uint64_t completed_ = 0;  ///< Successfully committed update events.
};

}  // namespace jyppx::liuding::vision::tasks
#endif  // JYPPX_LIUDING_VISION_TASKS_TRAINING_STRATEGY_HPP
