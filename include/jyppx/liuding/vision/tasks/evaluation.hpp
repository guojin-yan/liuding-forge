// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_VISION_TASKS_EVALUATION_HPP
#define JYPPX_LIUDING_VISION_TASKS_EVALUATION_HPP
/** @file evaluation.hpp @brief Raw YOLO models to existing dataset box AP.
 */

#include <jyppx/liuding/training/detection_evaluation.hpp>
#include <jyppx/liuding/vision/tasks/data/preparation.hpp>
#include <jyppx/liuding/vision/tasks/prediction.hpp>

namespace jyppx::liuding::vision::tasks {

/** @brief Frozen preprocessing, suppression, scoring and provenance policies.
 */
struct YoloEvaluationOptions {
  YoloDataOptions data;  ///< Random augmentation is always disabled for AP.
  YoloPredictionOptions prediction;  ///< Explicit AP score/NMS policy.
  training::DetectionEvaluationOptions
      evaluation;  ///< Required dataset/class IDs, class count and metric
                   ///< limits.
  Device device{
      torch::kCPU};  ///< Must match the raw model's existing placement.
  std::size_t batch_size = 4;    ///< Sequential batch size, in [1,256].
  bool ignore_difficult = true;  ///< VOC difficult becomes ignored GT.
};

/** @brief Runs one complete canonical split through existing box AP evaluation.
 * @param model Raw P5 YOLO adapter; original module modes are restored.
 * @param dataset Complete image/annotation selection in stable manifest order.
 * @param options Explicit preprocessing, predictions and metric provenance.
 * @param stop Cooperative cancellation during decode/preparation/evaluation.
 * @param deadline Optional monotonic evaluation deadline.
 * @return Complete dataset AP/recall in original image coordinates.
 * @throws Error For mismatched classes, annotations, resources or backend work.
 *
 * Uses the existing evaluate_detection() and DetectionAveragePrecisionMetric;
 * never averages batch AP. Images use stable one-based manifest ordinals.
 * Preserves original GT area, crowd and ignore flags, including empty images.
 * Predictive boxes are mapped back through each image's GeometryTrace.
 * Evaluation confidence is explicit and independent of display confidence.
 * This entry accepts one complete split without source slices; an explicit
 * expected_images value must match the manifest. Access must be serialized.
 * Identity strings describe caller-frozen inputs, not image-content hashes.
 */
training::DetectionEvaluationResult evaluate_yolo_detection(
    nn::ModelAdapter& model, const data::image::ImageDataset& dataset,
    const YoloEvaluationOptions& options, std::stop_token stop = {},
    TelemetryDeadline deadline = std::nullopt);

}  // namespace jyppx::liuding::vision::tasks
#endif  // JYPPX_LIUDING_VISION_TASKS_EVALUATION_HPP
