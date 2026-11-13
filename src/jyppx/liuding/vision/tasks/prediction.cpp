// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/vision/ops/boxes.hpp>
#include <jyppx/liuding/vision/tasks/prediction.hpp>

#include <algorithm>
#include <cmath>
#include <map>

namespace jyppx::liuding::vision::tasks {

std::vector<std::vector<training::metrics::DetectionPrediction>>
postprocess_yolo_detection(const Tensor& raw,
                           std::array<std::int64_t, 2> image_size,
                           const YoloPredictionOptions& options) {
  if (!std::isfinite(options.confidence_threshold) ||
      options.confidence_threshold < 0 || options.confidence_threshold > 1 ||
      !std::isfinite(options.iou_threshold) || options.iou_threshold < 0 ||
      options.iou_threshold > 1 || options.max_candidates <= 0 ||
      options.max_detections <= 0 ||
      options.max_detections > options.max_candidates)
    throw InvalidArgument(
        "Invalid YOLO prediction thresholds or candidate/output bounds");
  const torch::NoGradGuard no_grad;
  auto decoded = decode_yolo_detection(raw, image_size, options.detection);
  std::vector<std::vector<training::metrics::DetectionPrediction>> output;
  output.reserve(static_cast<std::size_t>(raw.size(0)));
  for (std::int64_t image = 0; image < raw.size(0); ++image) {
    auto [scores, labels] = decoded.class_scores[image].max(-1);
    auto indices = (scores >= options.confidence_threshold).nonzero().flatten();
    auto order =
        torch::argsort(scores.index_select(0, indices), true, -1, true);
    order =
        order.slice(0, 0, (std::min)(order.size(0), options.max_candidates));
    indices = indices.index_select(0, order);
    auto boxes = decoded.boxes[image]
                     .index_select(0, indices)
                     .to(torch::kCPU)
                     .contiguous();
    scores = scores.index_select(0, indices).to(torch::kCPU).contiguous();
    labels = labels.index_select(0, indices).to(torch::kCPU).contiguous();
    if (!torch::isfinite(boxes).all().item<bool>() ||
        !torch::isfinite(scores).all().item<bool>())
      throw InvalidArgument("YOLO prediction candidates must be finite");
    boxes.select(1, 0).clamp_(0, image_size[1]);
    boxes.select(1, 2).clamp_(0, image_size[1]);
    boxes.select(1, 1).clamp_(0, image_size[0]);
    boxes.select(1, 3).clamp_(0, image_size[0]);
    auto coordinates = boxes.accessor<float, 2>();
    auto classes = labels.accessor<std::int64_t, 1>();
    auto confidence = scores.accessor<float, 1>();
    std::map<std::int64_t, std::vector<std::int64_t>> groups;
    for (std::int64_t row = 0; row < boxes.size(0); ++row)
      if (coordinates[row][2] > coordinates[row][0] &&
          coordinates[row][3] > coordinates[row][1])
        groups[options.class_agnostic ? 0 : classes[row]].push_back(row);
    std::vector<bool> selected(static_cast<std::size_t>(boxes.size(0)), false);
    for (const auto& [category, rows] : groups) {
      static_cast<void>(category);
      auto group_indices =
          torch::tensor(rows, torch::TensorOptions().dtype(torch::kInt64));
      const auto keep =
          vision_ops::nms(boxes.index_select(0, group_indices),
                          scores.index_select(0, group_indices),
                          {.iou_threshold = options.iou_threshold,
                           .max_candidates = options.max_candidates,
                           .max_output = options.max_detections});
      const auto* kept = keep.data_ptr<std::int64_t>();
      for (std::int64_t i = 0; i < keep.numel(); ++i)
        selected[static_cast<std::size_t>(
            rows[static_cast<std::size_t>(kept[i])])] = true;
    }
    std::vector<training::metrics::DetectionPrediction> predictions;
    for (std::int64_t row = 0;
         row < boxes.size(0) &&
         predictions.size() < static_cast<std::size_t>(options.max_detections);
         ++row)
      if (selected[static_cast<std::size_t>(row)])
        predictions.push_back(
            {.box = {coordinates[row][0], coordinates[row][1],
                     coordinates[row][2], coordinates[row][3]},
             .score = confidence[row],
             .class_index = static_cast<std::size_t>(classes[row])});
    output.push_back(std::move(predictions));
  }
  return output;
}

std::vector<training::metrics::DetectionPrediction>
restore_detection_coordinates(
    std::span<const training::metrics::DetectionPrediction> predictions,
    const data::image::GeometryTrace& geometry) {
  std::vector<training::metrics::DetectionPrediction> result;
  result.reserve(predictions.size());
  for (const auto& prediction : predictions) {
    const auto& box = prediction.box;
    const auto restored =
        geometry.inverse_box({box[0], box[1], box[2], box[3]});
    if (!restored) continue;
    auto output = prediction;
    output.box = {restored->x_min, restored->y_min, restored->x_max,
                  restored->y_max};
    result.push_back(std::move(output));
  }
  return result;
}

}  // namespace jyppx::liuding::vision::tasks
