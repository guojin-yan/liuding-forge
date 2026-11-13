// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
#ifndef JYPPX_LIUDING_VISION_TASKS_FASTER_RCNN_HPP
#define JYPPX_LIUDING_VISION_TASKS_FASTER_RCNN_HPP
/** @file faster_rcnn.hpp @brief RPN/ROI matching, sampling and task loss. */
#include <jyppx/liuding/vision/tasks/task.hpp>
namespace jyppx::liuding::vision::tasks {
/** @brief Fixed Faster R-CNN loss and sampling configuration. */
struct FasterRcnnTaskOptions {
  std::int64_t num_classes = 80;
  std::vector<double> class_weights;  ///< Training-only class-column weights.
  std::int64_t rpn_batch_size = 256;
  double rpn_positive_fraction = .5;
  double rpn_foreground_iou = .7;
  double rpn_background_iou = .3;
  double smooth_l1_beta = 1. / 9;
  std::int64_t max_assignment_pairs = 16777216;
  std::int64_t max_targets = 4096;
};
/** @brief Four-term RPN/ROI loss for the fixed nine-tensor detector output.
 * @param training Random balanced RPN sampling when true; deterministic prefix
 * selection in validation. Existing saved Torch RNG owns sampling state.
 * @note Evaluation ROI labels are matched from proposals without modifying
 * model mode or BN state; validation loss is a distinct proposal-based profile.
 */
TaskLoss faster_rcnn_loss(std::span<const Tensor> outputs,
                          const Tensor& targets,
                          const FasterRcnnTaskOptions& options = {},
                          bool training = true);
/** @brief Existing Trainer step; invocation carries images/targets/image_sizes.
 */
training::StepFunction make_faster_rcnn_step(
    const FasterRcnnTaskOptions& options = {});
}  // namespace jyppx::liuding::vision::tasks
#endif  // JYPPX_LIUDING_VISION_TASKS_FASTER_RCNN_HPP
