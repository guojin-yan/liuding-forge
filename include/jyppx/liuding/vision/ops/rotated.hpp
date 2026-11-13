// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
#ifndef JYPPX_LIUDING_VISION_OPS_ROTATED_HPP
#define JYPPX_LIUDING_VISION_OPS_ROTATED_HPP
/** @file rotated.hpp @brief Rotated rectangle training and exact CPU geometry.
 */
#include <jyppx/liuding/vision/ops/boxes.hpp>
namespace jyppx::liuding::vision_ops {
/** @brief Gaussian/Hellinger overlap for broadcast [...,5] cx/cy/w/h/radians.
 * Keeps autograd. Positive finite width/height are caller preconditions.
 * This differentiable training similarity is distinct from exact rotated IoU.
 */
Tensor probabilistic_box_iou(const Tensor& left, const Tensor& right,
                             double epsilon = 1e-7);
/** @brief Exact convex clipping IoU of CPU [N,5] and [M,5] rectangles.
 * Coordinates and positive dimensions are validated; output is CPU float64.
 * No gradients. The explicit N*M bound applies before allocation.
 */
Tensor pairwise_rotated_iou(const Tensor& left, const Tensor& right,
                            const PairwiseBoxOptions& options = {});
/** @brief Exact CPU convex-quadrilateral IoU for [N,4,2] corner tensors.
 * Handles anisotropically restored rectangles without fitting new rectangles.
 */
Tensor pairwise_quadrilateral_iou(const Tensor& left, const Tensor& right,
                                  const PairwiseBoxOptions& options = {});
/** @brief Stable class-agnostic exact rotated IoU NMS, CPU only. */
Tensor rotated_nms(const Tensor& boxes, const Tensor& scores,
                   const NmsOptions& options = {});
}  // namespace jyppx::liuding::vision_ops
#endif  // JYPPX_LIUDING_VISION_OPS_ROTATED_HPP
