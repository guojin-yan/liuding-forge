// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_VISION_TASKS_YOLO_DETECTION_HPP
#define JYPPX_LIUDING_VISION_TASKS_YOLO_DETECTION_HPP
/** @file yolo_detection.hpp
 * @brief Native P5 distribution decoding, assignment, detection loss and steps.
 */

#include <jyppx/liuding/core/types.hpp>
#include <jyppx/liuding/nn/model_adapter.hpp>
#include <jyppx/liuding/training/trainer.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace jyppx::liuding::vision::tasks {

/** @brief Explicit P5 detection recipe and logical allocation/work bounds.
 *
 * Plain value options own no training state. Bounds exclude allocator/backend
 * workspace and do not claim a process RSS/VRAM ceiling. This development
 * criterion has not been qualified against an independent upstream reference.
 */
struct YoloDetectionOptions {
  std::int64_t num_classes = 80;
  std::vector<double> class_weights;  ///< Training-only class-column weights.
                                      ///< ///< Positive count, at most 10000.
  std::int64_t reg_max = 16;  ///< [1,32]; 1 is direct distance regression.
  std::vector<std::int64_t> strides{8, 16,
                                    32};  ///< Ordered P2/P3/P4/P5/P6 levels.
  std::int64_t assignment_topk =
      10;                         ///< Positive anchors considered per target.
  double assignment_alpha = 0.5;  ///< Nonnegative class-probability exponent.
  double assignment_beta = 6.0;   ///< Positive overlap exponent.
  double box_gain = 7.5;          ///< Nonnegative CIoU term weight.
  double class_gain = 0.5;        ///< Nonnegative soft-target BCE term weight.
  double dfl_gain = 1.5;  ///< Nonnegative distribution focal term weight.
  std::int64_t max_targets = 4096;  ///< Maximum target rows per batch.
  std::int64_t max_assignment_pairs =
      1048576;  ///< Maximum GT*anchors per image.
  std::int64_t max_prediction_elements = 67108864;  ///< Raw element bound.
};

/** @brief Owning tensor handles for decoded dense predictions; no NMS. */
struct YoloDecodedPrediction {
  Tensor boxes;  ///< [B,A,4] absolute xyxy boxes in input-canvas coordinates.
  Tensor class_scores;   ///< [B,A,C] independent sigmoid scores, no objectness.
  Tensor anchor_points;  ///< [A,2] feature-grid centers, offset by 0.5.
  Tensor strides;        ///< [A,1] pixel stride for each packed anchor.
};

/** @brief Differentiable losses plus detached assignment diagnostics. */
struct YoloDetectionLoss {
  Tensor total;  ///< Scalar weighted loss multiplied by batch size.
  Tensor box;    ///< Scalar normalized CIoU loss before gain/batch scaling.
  Tensor classification;  ///< Scalar normalized soft-target BCE before gain.
  Tensor distribution;    ///< Scalar normalized edge-bin loss before gain.
  std::int64_t foreground_anchors = 0;  ///< Positive anchor count in this call.
};

/** @brief Detached task-aligned targets shared by detection and richer heads.
 */
struct YoloAssignment {
  Tensor boxes;   ///< [B,A,4] canvas xyxy; background entries are unspecified.
  Tensor scores;  ///< [B,A,C] normalized soft class labels.
  Tensor foreground;      ///< [B,A] bool.
  Tensor target_indices;  ///< [B,A] int64 global target row; -1 for background.
};

/** @brief Deterministic assignment without backward, model mutation or RNG.
 * @param prediction Decoded predictions from the same options and input canvas.
 * @param targets Validated [T,6] batch/class/xyxy, on prediction device.
 * @param options Matching policy and pair bounds.
 * @return Detached labels and original global target-row identities.
 */
YoloAssignment assign_yolo_targets(const YoloDecodedPrediction& prediction,
                                   const Tensor& targets,
                                   const YoloDetectionOptions& options = {});

/** @brief Decodes packed raw P3/P4/P5 predictions in native tensor operations.
 * @param raw Floating [B,4*reg_max+C,A], with positive B and matching options.
 * @param image_size Input canvas {height,width}, positive multiples of 32.
 * @param options Explicit head/stride definition and allocation bounds.
 * @return FP32 boxes/scores/grid on raw.device(), retaining the gradient graph.
 * @throws InvalidArgument Shape, image dimensions or options are inconsistent.
 * @throws ResourceExhausted Raw predictions exceed the configured bound.
 * @note Does not alter train/eval mode, filter predictions or read model state.
 */
YoloDecodedPrediction decode_yolo_detection(
    const Tensor& raw, std::array<std::int64_t, 2> image_size,
    const YoloDetectionOptions& options = {});

/** @brief Computes task-aligned soft targets, CIoU, BCE and distribution loss.
 * @param raw Packed raw predictions as in decode_yolo_detection().
 * @param targets Floating [T,6] on raw.device(): batch index, contiguous class
 * ID, x1,y1,x2,y2 in canvas pixels. Indices must be exact integers; boxes must
 * have positive area and lie within the canvas. [0,6] is a valid negative
 * batch.
 * @param image_size Input canvas {height,width}.
 * @param options Explicit recipe; matching occurs on detached predictions.
 * @return FP32 scalar losses with model gradients; diagnostics are not state.
 * @throws InvalidArgument Target values/metadata or recipe fields are invalid.
 * @throws ResourceExhausted Prediction/target/assignment bounds are exceeded.
 * @note Does no backward/update and consumes no RNG. Value validation and
 * assignment diagnostics can synchronize CUDA. No AP/quality is inferred.
 */
YoloDetectionLoss yolo_detection_loss(const Tensor& raw, const Tensor& targets,
                                      std::array<std::int64_t, 2> image_size,
                                      const YoloDetectionOptions& options = {});

/** @brief Builds a task step for the existing Trainer; never owns an optimizer.
 * @param options Immutable loss recipe copied into the callable.
 * @return Step requiring one image tensor in Invocation and one [T,6] tensor
 * target; produces loss and per-batch diagnostics in training/validation.
 * @throws InvalidArgument Invocation/target contracts are invalid when called.
 * @note The Trainer exclusively owns AMP, backward, updates and checkpoints.
 */
training::StepFunction make_yolo_detection_step(
    const YoloDetectionOptions& options = {});

}  // namespace jyppx::liuding::vision::tasks

#endif  // JYPPX_LIUDING_VISION_TASKS_YOLO_DETECTION_HPP
