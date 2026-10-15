// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_NN_LOSS_CONTRASTIVE_LOSS_HPP
#define JYPPX_NN_LOSS_CONTRASTIVE_LOSS_HPP

/**
 * @file contrastive_loss.hpp
 * @brief Defines the pairwise contrastive loss and criterion module.
 */

#include <jyppx/core/types.hpp>

#include <torch/nn/module.h>

#include <string>

namespace jyppx::nn {
namespace functional {

/**
 * @brief Computes the pairwise contrastive loss for feature pairs.
 *
 * For a matching pair (`target = 1`) the loss is the squared Euclidean
 * distance. For a non-matching pair (`target = 0`) the loss is the squared
 * hinge distance `max(0, margin - distance)^2`. This is a JYPPX-native loss:
 * LibTorch exposes the primitive tensor operations but no equivalent
 * contrastive-loss functional with this contract.
 *
 * The API intentionally accepts exactly two-dimensional feature tensors and a
 * one-dimensional target. All tensors must use the same device; feature
 * tensors must have the same floating-point dtype. Targets may use an
 * integral, boolean, or floating-point dtype, but every value must be exactly
 * zero or one.
 *
 * @param x1 First feature tensor with shape `(N, D)`.
 * @param x2 Second feature tensor with shape `(N, D)`.
 * @param target Binary pair labels with shape `(N)`; one marks a matching
 *     pair and zero marks a non-matching pair.
 * @param margin Positive distance margin for non-matching pairs.
 * @param reduction Output reduction: `"none"`, `"mean"`, or `"sum"`.
 * @return Per-pair losses for `"none"`, otherwise one scalar loss.
 * @throws InvalidArgument If tensors are undefined, shapes/devices/dtypes are
 *     incompatible, labels are not binary, `margin` is not finite and
 *     positive, or `reduction` is unsupported.
 * @throws BackendError If LibTorch rejects a tensor operation after JYPPX
 *     validation; the original backend exception is retained as a nested
 *     cause.
 */
Tensor contrastive_loss(const Tensor& x1, const Tensor& x2,
                        const Tensor& target, double margin = 1.0,
                        const std::string& reduction = "mean");

}  // namespace functional

/**
 * @brief Stateless criterion module for pairwise contrastive loss.
 *
 * The module has no parameters or buffers. It stores only the validated
 * margin and reduction configuration and delegates computation to the
 * functional API.
 */
class ContrastiveLoss : public torch::nn::Module {
 public:
  /**
   * @brief Creates a contrastive-loss criterion.
   * @param margin Positive finite distance margin.
   * @param reduction Output reduction: `"none"`, `"mean"`, or `"sum"`.
   * @throws InvalidArgument If either configuration value is invalid.
   */
  explicit ContrastiveLoss(double margin = 1.0, std::string reduction = "mean");

  /**
   * @brief Computes the configured contrastive loss.
   * @param x1 First feature tensor with shape `(N, D)`.
   * @param x2 Second feature tensor with shape `(N, D)`.
   * @param target Binary pair labels with shape `(N)`.
   * @return Per-pair losses or the configured scalar reduction.
   * @throws InvalidArgument If the tensor contract is violated.
   * @throws BackendError If a validated tensor operation fails in LibTorch.
   */
  Tensor forward(const Tensor& x1, const Tensor& x2,
                 const Tensor& target) const;

  /**
   * @brief Returns the configured distance margin.
   * @return A finite positive margin.
   */
  double margin() const noexcept { return margin_; }

  /**
   * @brief Returns the configured reduction name.
   * @return One of `"none"`, `"mean"`, or `"sum"`.
   */
  const std::string& reduction() const noexcept { return reduction_; }

 private:
  double margin_;          ///< Positive distance margin for negative pairs.
  std::string reduction_;  ///< Reduction applied to per-pair losses.
};

}  // namespace jyppx::nn

#endif  // JYPPX_NN_LOSS_CONTRASTIVE_LOSS_HPP
