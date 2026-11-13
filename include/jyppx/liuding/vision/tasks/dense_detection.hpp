// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_DENSE_DETECTION_HPP
#define JYPPX_LIUDING_VISION_TASKS_DENSE_DETECTION_HPP
/** @file dense_detection.hpp @brief FCOS, RetinaNet and SSD native criteria. */
#include <jyppx/liuding/vision/tasks/results.hpp>
namespace jyppx::liuding::vision::tasks {
enum class DenseDetectionCriterion { fcos, retinanet_v1, ssd };
struct DenseDetectionLossOptions {
  DenseDetectionCriterion kind=DenseDetectionCriterion::fcos;
  std::int64_t num_classes=80;
  std::int64_t max_assignment_pairs=16777216;
  double center_sampling_radius=1.5;
  double negative_positive_ratio=3.;
};
/** @brief Targets [T,6] batch/class/xyxy; no hidden sampling engine. */
TaskLoss dense_detection_loss(std::span<const Tensor> raw,const Tensor& targets,
                               const DenseDetectionLossOptions& options={});
training::StepFunction make_dense_detection_step(DenseDetectionLossOptions options={});
std::vector<ImageResult> decode_dense_detection(std::span<const Tensor> raw,
    std::array<std::int64_t,2> size,DenseDetectionCriterion kind,
    const ResultOptions& options={});
}
#endif
