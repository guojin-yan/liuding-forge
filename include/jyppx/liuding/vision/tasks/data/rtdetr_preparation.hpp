// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_DATA_RTDETR_PREPARATION_HPP
#define JYPPX_LIUDING_VISION_TASKS_DATA_RTDETR_PREPARATION_HPP
/** @file rtdetr_preparation.hpp @brief Explicit RT-DETR v1 box-data profile. */
#include <jyppx/liuding/data_image/joint_transform.hpp>

#include <string>

namespace jyppx::liuding::vision::tasks {
/** @brief PIL-style uint8 photometry, zoom-out, IoU crop, flip and stretch.
 * Per-operation photo probabilities follow torchvision
 * RandomPhotometricDistort. The profile accepts axis-aligned box targets; it
 * does not reinterpret rich masks, keypoints, paired targets or task payloads.
 */
struct RtDetrDataOptions {
  bool enabled = false;
  bool training = true;  ///< False selects stretch without training sanitation.
  double photometric_probability = .5;
  double zoom_out_probability = .5;
  double iou_crop_probability = .8;
  double horizontal_flip_probability = .5;
  std::uint32_t crop_trials = 40;       ///< Trials per sampled IoU threshold.
  std::uint32_t max_crop_rounds = 128;  ///< Explicit rejection-work bound.
};
/** @brief Validates and fingerprints the sampling and uint8 arithmetic policy.
 * @throws InvalidArgument Invalid probabilities or rejection-work bounds. */
std::string rtdetr_data_identity(const RtDetrDataOptions& options);
/** @brief Applies the detection profile without changing source storage.
 * @param sample CPU uint8 RGB with optional canonical box annotations.
 * @param canvas Exact stretched output extent.
 * @param options Explicit training/evaluation policy.
 * @param random Immutable logical sample stream, independent of worker order.
 * @param geometry Existing resource limits and optional aggregate budget.
 * @param stop Cooperative cancellation checked in sampling and pixel loops.
 * @return uint8 sample with filtered boxes and composed source geometry.
 * @throws InvalidArgument Invalid pixels, unsupported targets or options.
 * @throws ResourceExhausted Canvas, payload or sampling bound exceeded.
 * @throws Cancelled Cancellation requested before publication. */
data::image::VisionSample prepare_rtdetr_sample(
    const data::image::VisionSample& sample, data::image::ImageExtent canvas,
    const RtDetrDataOptions& options, const data::image::SampleRandom& random,
    const data::image::JointTransformOptions& geometry = {},
    std::stop_token stop = {});
}  // namespace jyppx::liuding::vision::tasks
#endif  // JYPPX_LIUDING_VISION_TASKS_DATA_RTDETR_PREPARATION_HPP
