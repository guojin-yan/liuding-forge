// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_VISION_TASKS_PREDICTION_HPP
#define JYPPX_LIUDING_VISION_TASKS_PREDICTION_HPP
/** @file prediction.hpp @brief Detection postprocessing using existing metric
 * values. */

#include <jyppx/liuding/data_image/geometry_trace.hpp>
#include <jyppx/liuding/training/metrics/detection.hpp>
#include <jyppx/liuding/vision/tasks/yolo_detection.hpp>

#include <span>
#include <vector>

namespace jyppx::liuding::vision::tasks {

/** @brief Explicit score selection, CPU suppression and output bounds. */
struct YoloPredictionOptions {
  YoloDetectionOptions detection;  ///< Raw head definition and decode bounds.
  double confidence_threshold =
      0.25;                     ///< Inclusive sigmoid score threshold [0,1].
  double iou_threshold = 0.45;  ///< NMS threshold [0,1].
  std::int64_t max_candidates =
      4096;  ///< Highest-scoring anchors retained per image.
  std::int64_t max_detections = 300;  ///< Final detection count per image.
  bool class_agnostic = false;  ///< Otherwise suppress independently per class.
};

/** @brief Decodes raw P5 output, chooses one class per anchor and applies NMS.
 * @param raw Same schema as decode_yolo_detection(); CPU/CUDA is accepted.
 * @param image_size Model input {height,width} in canvas pixels.
 * @param options Explicit score, suppression and finite allocation policies.
 * @return One existing metric prediction vector per image, in canvas xyxy
 * coordinates, clipped to the canvas and ordered by descending score.
 * @note Runs under NoGrad. Only bounded selected candidates move to CPU for
 * suppression; a CUDA call synchronizes. Score ties retain packed anchor order.
 * @throws InvalidArgument Invalid options, metadata or nonfinite candidates.
 * @throws ResourceExhausted Decoder bounds are exceeded.
 */
std::vector<std::vector<training::metrics::DetectionPrediction>>
postprocess_yolo_detection(const Tensor& raw,
                           std::array<std::int64_t, 2> image_size,
                           const YoloPredictionOptions& options = {});

/** @brief Clips and maps canvas detections back through the existing trace.
 * @param predictions Canvas-space detections in caller-selected order.
 * @param geometry Invertible single-source image resize/crop/pad/flip trace.
 * @return Source-space detections; padding-only boxes are omitted.
 * @throws InvalidArgument Invalid/nonfinite box coordinates.
 */
std::vector<training::metrics::DetectionPrediction>
restore_detection_coordinates(
    std::span<const training::metrics::DetectionPrediction> predictions,
    const data::image::GeometryTrace& geometry);

}  // namespace jyppx::liuding::vision::tasks
#endif  // JYPPX_LIUDING_VISION_TASKS_PREDICTION_HPP
