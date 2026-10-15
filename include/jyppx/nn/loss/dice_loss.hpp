// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_NN_LOSS_DICE_LOSS_HPP
#define JYPPX_NN_LOSS_DICE_LOSS_HPP

/**
 * @file dice_loss.hpp
 * @brief Defines configurable binary, multiclass, and multilabel Dice loss.
 */

#include <jyppx/core/types.hpp>

#include <torch/nn/module.h>

#include <cstdint>
#include <optional>
#include <string>

namespace jyppx::nn {

/**
 * @brief Selects how Dice loss interprets logits and targets.
 */
enum class DiceMode {
  binary,  ///< One sigmoid channel represented by tensors shaped `(N, ...)`.
  multiclass,  ///< Mutually exclusive softmax classes shaped `(N, C, ...)`.
  multilabel,  ///< Independent sigmoid classes shaped `(N, C, ...)`.
};

/**
 * @brief Selects how selected class scores are combined within each sample.
 */
enum class DiceAverage {
  micro,  ///< Pool intersections and cardinalities before computing one score.
  macro,  ///< Compute one score per selected class and average the scores.
};

/**
 * @brief Configures Dice probability conversion, aggregation, and reduction.
 *
 * Dice statistics are always computed independently for each batch item over
 * its non-batch spatial dimensions. `average` combines class channels inside
 * each item, and `reduction` then combines the resulting `(N)` loss vector.
 */
struct DiceLossOptions {
  DiceMode mode = DiceMode::multiclass;  ///< Logit and target interpretation.
  DiceAverage average = DiceAverage::macro;  ///< Per-sample class aggregation.
  double smooth =
      1.0;  ///< Finite non-negative numerator/denominator smoothing.
  double epsilon = 1e-7;  ///< Positive finite denominator clamp threshold.
  std::optional<std::int64_t> ignore_index =
      std::nullopt;  ///< Multiclass target value excluded from all statistics.
  bool include_background =
      true;  ///< Whether explicit channel zero participates.
  bool include_empty_classes =
      true;  ///< Whether target-empty classes participate in class aggregation.
  std::string reduction =
      "mean";  ///< Batch reduction: `"none"`, `"mean"`, or `"sum"`.
};

namespace functional {

/**
 * @brief Computes soft Dice loss from logits using an explicit task contract.
 *
 * Binary mode accepts floating logits and binary targets with identical shape
 * `(N, ...)`. Multiclass mode accepts logits `(N, C, ...)` and integral class
 * indices `(N, ...)`. Multilabel mode accepts logits and binary targets with
 * identical shape `(N, C, ...)`. Input and target must share a device, and no
 * tensor dimension may be empty.
 *
 * For each selected class the score is
 * `(2 * intersection + smooth) /
 *  clamp_min(prediction_sum + target_sum + smooth, epsilon)`.
 * Multiclass ignored locations are removed from prediction and target sums.
 * Excluding empty classes removes channels whose target sum is zero; a sample
 * with no selected class or only ignored locations contributes exactly zero.
 * Background exclusion removes explicit channel zero and is unsupported in
 * binary mode, which has no explicit class axis.
 *
 * @param input Floating-point logits following `options.mode`.
 * @param target Binary values or integral multiclass indices on the same
 *     device as `input`.
 * @param options Validated probability, class-selection, and reduction policy.
 * @return Per-sample losses with shape `(N)` for `"none"`, otherwise a scalar.
 * @throws InvalidArgument If configuration, shape, dtype, device, target value,
 *     or empty-tensor contracts are violated.
 * @throws BackendError If a validated LibTorch operation fails; the original
 *     backend exception is retained as a nested cause.
 */
Tensor dice_loss(const Tensor& input, const Tensor& target,
                 const DiceLossOptions& options = {});

}  // namespace functional

/**
 * @brief Stateless criterion module for configurable soft Dice loss.
 *
 * The module owns an immutable-by-interface copy of the validated options and
 * delegates every forward call to the functional implementation.
 */
class DiceLoss : public torch::nn::Module {
 public:
  /**
   * @brief Creates a Dice-loss criterion with the supplied semantic contract.
   * @param options Probability, aggregation, masking, and reduction policy.
   * @throws InvalidArgument If any option is unsupported or non-finite.
   */
  explicit DiceLoss(DiceLossOptions options = {});

  /**
   * @brief Computes Dice loss using the stored options.
   * @param input Floating-point logits following the configured mode.
   * @param target Binary values or multiclass indices matching the mode.
   * @return Per-sample losses or the configured scalar batch reduction.
   * @throws InvalidArgument If the tensor contract is violated.
   * @throws BackendError If a validated tensor operation fails in LibTorch.
   */
  Tensor forward(const Tensor& input, const Tensor& target) const;

  /**
   * @brief Returns the complete validated Dice semantic contract.
   * @return A reference valid for the lifetime of this module.
   */
  const DiceLossOptions& options() const noexcept { return options_; }

 private:
  DiceLossOptions options_;  ///< Validated Dice computation policy.
};

}  // namespace jyppx::nn

#endif  // JYPPX_NN_LOSS_DICE_LOSS_HPP
