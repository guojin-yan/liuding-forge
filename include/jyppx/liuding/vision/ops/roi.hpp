// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
#ifndef JYPPX_LIUDING_VISION_OPS_ROI_HPP
#define JYPPX_LIUDING_VISION_OPS_ROI_HPP
/** @file roi.hpp @brief Native differentiable ROIAlign and box coding. */
#include <jyppx/liuding/core/types.hpp>

#include <array>
#include <vector>
namespace jyppx::liuding::vision_ops {
/** @brief Bounded bilinear ROIAlign settings, with continuous xyxy coordinates.
 */
struct RoiAlignOptions {
  std::array<std::int64_t, 2> output_size{7, 7};
  double spatial_scale = 1.;
  std::int64_t sampling_ratio = 2;  ///< [1,8], fixed samples per bin axis.
  bool aligned = false;             ///< Half-pixel offset when true.
  std::int64_t max_rois = 4096;
  std::int64_t max_output_elements = 67108864;
};
/** @brief ROIAlign via native ATen bilinear sampling; gradients reach features.
 * @param features Floating NCHW CPU/CUDA features.
 * @param rois Floating [K,5] batch/x1/y1/x2/y2 on feature device, finite with
 * valid batch indices. ROI coordinates are detached, as in the reference op.
 * @return [K,C,pooled_h,pooled_w]; an empty ROI set preserves autograd.
 * @throws InvalidArgument Invalid metadata/coordinates/options.
 * @throws ResourceExhausted Explicit ROI/output/sampling bound exceeded.
 * @note Host ROI checks and per-ROI batch selection synchronize CUDA.
 */
Tensor roi_align(const Tensor& features, const Tensor& rois,
                 const RoiAlignOptions& options = {});
/** @brief FPN level selection (canonical scale 224, level 4), P2 upwards.
 * @param features Ordered equal-channel NCHW pyramid levels.
 * @param rois [K,5] canvas coordinates.
 * @param strides Matching positive power-of-two reductions.
 * @return ROIAlign outputs in original ROI order; no feature state copies.
 */
Tensor multiscale_roi_align(const std::vector<Tensor>& features,
                            const Tensor& rois,
                            const std::vector<std::int64_t>& strides,
                            RoiAlignOptions options = {});
/** @brief Encodes corresponding positive-area xyxy rows as dx/dy/dw/dh. */
Tensor encode_box_deltas(const Tensor& anchors, const Tensor& targets,
                         std::array<double, 4> weights = {1, 1, 1, 1});
/** @brief Decodes [N,4*K] deltas against [N,4] anchors, with bounded log-scale.
 */
Tensor decode_box_deltas(const Tensor& anchors, const Tensor& deltas,
                         std::array<double, 4> weights = {1, 1, 1, 1},
                         double max_log_scale = 4.135166556742356);
}  // namespace jyppx::liuding::vision_ops
#endif  // JYPPX_LIUDING_VISION_OPS_ROI_HPP
