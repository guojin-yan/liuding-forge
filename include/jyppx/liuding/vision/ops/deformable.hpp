// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_OPS_DEFORMABLE_HPP
#define JYPPX_LIUDING_VISION_OPS_DEFORMABLE_HPP
/** @file deformable.hpp @brief Tensor-native differentiable multiscale sampling. */
#include <jyppx/liuding/core/types.hpp>
#include <array>
#include <span>
namespace jyppx::liuding::vision_ops {
/** @brief GridSample attention: value [B,S,heads,D], locations
 * [B,Q,heads,levels,points,2] in [0,1] coordinates, weights with no final axis.
 * Shapes sum to S; zeros padding/bilinear/align_corners=false. Returns [B,Q,H*D].
 * All floating inputs share device/dtype; no detach, custom kernel or host
 * index inference. Limits bound the sampled temporary before allocation.
 */
Tensor multiscale_deformable_attention(
    const Tensor& value, std::span<const std::array<std::int64_t, 2>> shapes,
    const Tensor& locations, const Tensor& weights,
    std::int64_t max_sample_elements = 67108864);
/** @brief Classical ROI max-pooling, detached [K,5] batch/xyxy coordinates.
 * Uses rounded endpoints, inclusive ROI width/height, floor/ceil bins and zero
 * empty bins. Gradients reach the selected feature maxima. */
Tensor roi_pool(const Tensor& features, const Tensor& rois,
                std::array<std::int64_t, 2> output_size = {7, 7},
                double spatial_scale = 1.,
                std::int64_t max_output_elements = 67108864);
/** @brief Aligned generalized IoU for corresponding continuous xyxy rows. */
Tensor aligned_generalized_iou(const Tensor& left, const Tensor& right,
                               double epsilon = 1e-7);
}  // namespace jyppx::liuding::vision_ops
#endif
