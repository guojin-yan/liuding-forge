// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_NN_LOSS_FOCAL_LOSS_HPP
#define JYPPX_LIUDING_NN_LOSS_FOCAL_LOSS_HPP
/**
 * @file focal_loss.hpp
 * @brief Defines the binary and multiclass focal loss and criterion module.
 */

#include <jyppx/liuding/core/types.hpp>

#include <torch/nn/module.h>

#include <string>

namespace jyppx::liuding::nn {
namespace functional {

/**
 * @brief Computes focal loss for binary or class-index multiclass targets.
 *
 * Binary inputs have identical shape and use sigmoid focal loss. Multiclass
 * inputs have shape `(N, C, ...)`, class-index targets have shape `(N, ...)`,
 * and the per-class focal term is derived from cross entropy. The `alpha`
 * value weights the positive binary class and the whole multiclass objective;
 * `gamma = 0` therefore recovers the corresponding weighted cross entropy.
 *
 * @param input Floating-point logits. Binary input and target shapes match;
 *     multiclass input is `(N, C, ...)`.
 * @param target Binary values in `{0, 1}` with the input shape, or integral
 *     class indices shaped `(N, ...)`.
 * @param alpha Finite class-balancing weight in the closed interval `[0, 1]`.
 * @param gamma Finite non-negative focusing exponent.
 * @param reduction Output reduction: `"none"`, `"mean"`, or `"sum"`.
 * @return Per-element/per-sample loss for `"none"`, otherwise one scalar.
 * @throws InvalidArgument If configuration, shape, device, dtype, target
 *     values, or class indices violate the documented contract.
 * @throws BackendError If LibTorch rejects an operation after validation; the
 *     original backend exception is retained as a nested cause.
 */
Tensor focal_loss(const Tensor& input, const Tensor& target, double alpha = 1.0,
                  double gamma = 2.0, const std::string& reduction = "mean");

}  // namespace functional

/**
 * @brief Stateless criterion module for binary and multiclass focal loss.
 */
class FocalLoss : public torch::nn::Module {
 public:
  /**
   * @brief Creates a focal-loss criterion.
   * @param alpha Finite class-balancing weight in `[0, 1]`.
   * @param gamma Finite non-negative focusing exponent.
   * @param reduction Output reduction: `"none"`, `"mean"`, or `"sum"`.
   * @throws InvalidArgument If configuration is invalid.
   */
  explicit FocalLoss(double alpha = 1.0, double gamma = 2.0,
                     std::string reduction = "mean");

  /**
   * @brief Computes focal loss using the stored configuration.
   * @param input Floating-point binary or multiclass logits.
   * @param target Binary or class-index target matching the input contract.
   * @return Per-element/per-sample loss or the configured scalar reduction.
   * @throws InvalidArgument If the tensor contract is violated.
   * @throws BackendError If a validated tensor operation fails in LibTorch.
   */
  Tensor forward(const Tensor& input, const Tensor& target) const;

  /**
   * @brief Returns the validated class-balancing weight.
   * @return The finite class-balancing weight in `[0, 1]`.
   */
  double alpha() const noexcept { return alpha_; }

  /**
   * @brief Returns the validated focusing exponent.
   * @return The finite non-negative focusing exponent.
   */
  double gamma() const noexcept { return gamma_; }

  /**
   * @brief Returns the configured reduction name.
   * @return The reduction name, one of `"none"`, `"mean"`, or `"sum"`.
   */
  const std::string& reduction() const noexcept { return reduction_; }

 private:
  double alpha_;           ///< Finite class-balancing weight in `[0, 1]`.
  double gamma_;           ///< Finite non-negative focusing exponent.
  std::string reduction_;  ///< Reduction applied to per-example losses.
};

}  // namespace jyppx::liuding::nn

#endif  // JYPPX_LIUDING_NN_LOSS_FOCAL_LOSS_HPP