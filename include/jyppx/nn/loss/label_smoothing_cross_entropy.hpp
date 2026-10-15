// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_NN_LOSS_LABEL_SMOOTHING_CROSS_ENTROPY_HPP
#define JYPPX_NN_LOSS_LABEL_SMOOTHING_CROSS_ENTROPY_HPP

/**
 * @file label_smoothing_cross_entropy.hpp
 * @brief Defines functional and module forms of label-smoothed cross entropy.
 */

#include <jyppx/core/types.hpp>

#include <torch/nn/module.h>

#include <string>

namespace jyppx::nn {
namespace functional {

/**
 * @brief Computes cross entropy after mixing the target with a uniform label
 * distribution.
 *
 * The function delegates the numerical operation to LibTorch and therefore
 * follows `torch::nn::functional::cross_entropy` for supported shapes, scalar
 * types, devices, class-index validation, empty tensors, and gradient
 * behavior. `smoothing` is the probability mass assigned to the uniform class
 * distribution. Class-index targets use LibTorch's default `ignore_index` of
 * `-100`.
 *
 * @param input Unnormalized logits with shape `(C)`, `(N, C)`, or
 * `(N, C, d1, ..., dK)` and a floating-point scalar type.
 * @param target Class indices with shape `()`, `(N)`, or `(N, d1, ..., dK)`
 * corresponding to `input`.
 * @param smoothing Label-smoothing probability in the closed interval
 * `[0, 1]`; zero is ordinary cross entropy and one uses only the uniform
 * target distribution.
 * @param reduction Output reduction: `"none"`, `"mean"`, or `"sum"`.
 * @return The unreduced loss tensor or the requested scalar reduction.
 * @throws InvalidArgument If an input is undefined, `smoothing` is not finite
 * or lies outside `[0, 1]`, or `reduction` is unsupported.
 * @throws BackendError If LibTorch rejects an otherwise forwarded tensor
 * configuration or its backend operation fails.
 */
Tensor label_smoothing_cross_entropy(const Tensor& input, const Tensor& target,
                                     double smoothing = 0.1,
                                     const std::string& reduction = "mean");

}  // namespace functional

/**
 * @brief Stateless criterion module for label-smoothed cross entropy.
 *
 * The module stores only immutable-by-convention configuration values; it has
 * no parameters or buffers. Concurrent `forward()` calls are safe provided
 * callers do not mutate the module or its input tensors concurrently.
 */
class LabelSmoothingCrossEntropy : public torch::nn::Module {
 public:
  /**
   * @brief Creates a label-smoothed cross-entropy criterion.
   * @param smoothing Label-smoothing probability in `[0, 1]`.
   * @param reduction Output reduction: `"none"`, `"mean"`, or `"sum"`.
   * @throws InvalidArgument If either configuration value is invalid.
   */
  explicit LabelSmoothingCrossEntropy(double smoothing = 0.1,
                                      std::string reduction = "mean");

  /**
   * @brief Computes the configured loss for logits and class-index targets.
   * @param input Unnormalized logits accepted by the functional form.
   * @param target Class indices corresponding to `input`.
   * @return The unreduced loss tensor or configured scalar reduction.
   * @throws InvalidArgument If an input is undefined.
   * @throws BackendError If the LibTorch operation fails.
   */
  Tensor forward(const Tensor& input, const Tensor& target) const;

  /**
   * @brief Returns the configured label-smoothing probability.
   * @return A value in the closed interval `[0, 1]`.
   */
  double smoothing() const noexcept { return smoothing_; }

  /**
   * @brief Returns the configured reduction name.
   * @return One of `"none"`, `"mean"`, or `"sum"`.
   */
  const std::string& reduction() const noexcept { return reduction_; }

 private:
  double smoothing_;       ///< Probability mass assigned uniformly to classes.
  std::string reduction_;  ///< Reduction applied to the per-target loss.
};

}  // namespace jyppx::nn

#endif  // JYPPX_NN_LOSS_LABEL_SMOOTHING_CROSS_ENTROPY_HPP
