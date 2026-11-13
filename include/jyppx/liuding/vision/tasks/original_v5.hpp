// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
#ifndef JYPPX_LIUDING_VISION_TASKS_ORIGINAL_V5_HPP
#define JYPPX_LIUDING_VISION_TASKS_ORIGINAL_V5_HPP
/** @file original_v5.hpp @brief Original anchor matching, objectness and CIoU.
 */
#include <jyppx/liuding/vision/tasks/task.hpp>
namespace jyppx::liuding::vision::tasks {
/** @brief Explicit original YOLOv5 loss recipe; anchors are canvas pixel sizes.
 */
struct OriginalV5TaskOptions {
  std::int64_t num_classes = 80;
  std::vector<double> class_weights;  ///< Training-only class-column weights.
  std::vector<std::int64_t> strides{8, 16, 32};
  std::vector<std::array<double, 6>> anchors{{10, 13, 16, 30, 33, 23},
                                             {30, 61, 62, 45, 59, 119},
                                             {116, 90, 156, 198, 373, 326}};
  std::vector<double> objectness_balance{4., 1., .4};
  double anchor_ratio_threshold = 4.;
  double box_gain = .05;
  double objectness_gain = 1.;
  double class_gain = .5;
  double label_smoothing = 0;
  double focal_gamma = 0;
  double focal_alpha = .25;
  std::int64_t max_targets = 4096;
  std::int64_t max_prediction_elements = 67108864;
};
/** @brief Packed raw [B,A,C+5] to boxes and objectness*class probabilities. */
YoloDecodedPrediction decode_original_v5(
    const Tensor& raw, std::array<std::int64_t, 2> image_size,
    const OriginalV5TaskOptions& options = {});
/** @brief Ratio matching plus neighboring grid cells, CIoU/BCE/focal terms.
 * Targets use the common [T,6] absolute xyxy representation. Matching is
 * detached and deterministic. Duplicate objectness cells use maximum IoU.
 */
TaskLoss original_v5_loss(const Tensor& raw, const Tensor& targets,
                          std::array<std::int64_t, 2> image_size,
                          const OriginalV5TaskOptions& options = {});
/** @brief Existing Trainer step; owns no optimizer or mutable autobalance
 * state. */
training::StepFunction make_original_v5_step(
    const OriginalV5TaskOptions& options = {});
}  // namespace jyppx::liuding::vision::tasks
#endif  // JYPPX_LIUDING_VISION_TASKS_ORIGINAL_V5_HPP
