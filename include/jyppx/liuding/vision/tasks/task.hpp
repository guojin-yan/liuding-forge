// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
#ifndef JYPPX_LIUDING_VISION_TASKS_TASK_HPP
#define JYPPX_LIUDING_VISION_TASKS_TASK_HPP
/** @file task.hpp @brief Fixed task targets, native criteria and Trainer
 * bindings. */
#include <jyppx/liuding/vision/tasks/yolo_detection.hpp>

#include <functional>
#include <map>
#include <span>
#include <string>
#include <vector>
namespace jyppx::liuding::vision::tasks {
/** @brief Task identity is independent of model packages. */
enum class TaskKind {
  detect = 0, classify = 1, instance_segment = 2, pose = 3, obb = 4,
  classify_multilabel = 5, semantic_segment = 6, depth = 7,
  pretrain_simmim = 8, reid = 9
};
/** @brief Stable serialized task names, independent of enum ordinals. */
std::string_view task_name(TaskKind task);
/** @brief Parses an exact stable name and rejects unknown values. */
TaskKind parse_task(std::string_view name);
/** @brief Tasks whose output is a dense field, rather than object rows. */
bool is_dense_task(TaskKind task) noexcept;
/** @brief Rich targets retain one common box-row/object order. */
struct TaskTargets {
  Tensor boxes;  ///< Floating [T,6] batch/class/xyxy canvas pixels.
  Tensor masks;  ///< Optional [T,H,W] bool or floating binary instance masks.
  Tensor keypoints;  ///< Optional [T,K,3] xy/COCO visibility, canvas pixels.
  Tensor oriented_boxes;  ///< Optional [T,5] cx/cy/w/h/angle in radians.
};
/** @brief Frozen raw-layout, criterion and dual-branch schedule. */
struct YoloTaskOptions {
  TaskKind task = TaskKind::detect;
  YoloDetectionOptions detection;
  bool end_to_end = false;
  bool generation26 = false;
  std::int64_t mask_channels = 32;
  std::int64_t keypoints = 17;
  std::int64_t keypoint_dimensions = 3;
  double mask_gain = 7.5;
  double pose_gain = 12.;
  double visibility_gain = 1.;
  double semantic_gain = 1.;
  double flow_gain = 1.;
  double one_to_many_initial = 1.;
  double one_to_many_final = .1;
  std::uint64_t schedule_epochs = 100;
  std::vector<double> keypoint_sigmas;  ///< Empty selects uniform 1/K.
  std::int64_t max_mask_elements = 67108864;
};
/** @brief Scalar differentiable loss and unscaled named observations. */
struct TaskLoss {
  Tensor total;
  std::map<std::string, Tensor> components;
};
/** @brief Optional differentiable YOLO26 pose flow owned by the same model
 * state. */
using PoseLogProbability = std::function<Tensor(const Tensor&)>;
/** @brief Applies exact raw-layout checks and task-specific native criterion.
 * No optimizer, backward, random draws or mode mutations. Epoch selects the
 * declared dual-branch loss schedule. All sensitive arithmetic is float32.
 */
TaskLoss yolo_task_loss(std::span<const Tensor> outputs,
                        const TaskTargets& targets,
                        std::array<std::int64_t, 2> image_size,
                        const YoloTaskOptions& options = {},
                        std::uint64_t epoch = 0,
                        PoseLogProbability pose_flow = {});
/** @brief Hard [B] int64 or soft [B,C] probability cross entropy with
 * smoothing. Class weights, when supplied, are fixed train-only nonnegative [C]
 * weights.
 */
Tensor classification_loss(const Tensor& logits, const Tensor& targets,
                           double label_smoothing = 0,
                           const Tensor& class_weights = {});
/** @brief Binds fixed tuple outputs and a tensor/TaskTargets tuple to Trainer.
 * Target tuple order is boxes,masks,keypoints,oriented_boxes; absent auxiliary
 * targets are empty tensors. Trainer retains exclusive backward/update
 * ownership.
 */
training::StepFunction make_yolo_task_step(const YoloTaskOptions& options = {},
                                           PoseLogProbability pose_flow = {});
/** @brief Existing classification adapters can use this same native task
 * binding. */
training::StepFunction make_classification_step(
    double label_smoothing = 0, const Tensor& class_weights = {});
}  // namespace jyppx::liuding::vision::tasks
#endif  // JYPPX_LIUDING_VISION_TASKS_TASK_HPP
