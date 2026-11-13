// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
#ifndef JYPPX_LIUDING_VISION_TASKS_METRICS_HPP
#define JYPPX_LIUDING_VISION_TASKS_METRICS_HPP
/** @file metrics.hpp @brief Dataset AP for masks, OKS and exact rotated
 * corners. */
#include <jyppx/liuding/training/metrics/detection.hpp>
#include <jyppx/liuding/vision/tasks/results.hpp>

#include <optional>
namespace jyppx::liuding::vision::tasks {
/** @brief Complete metric identity and finite aggregation limits. */
struct TaskMetricOptions {
  TaskKind task = TaskKind::instance_segment;
  std::string dataset_id;
  std::string classes_id;
  std::size_t num_classes = 80;
  std::vector<double> thresholds{.5, .55, .6, .65, .7, .75, .8, .85, .9, .95};
  std::vector<double> keypoint_sigmas;
  std::size_t max_detections_per_image_class = 100;
  std::size_t max_images = 1000000;
  std::size_t max_retained_predictions = 10000000;
  std::size_t max_similarity_pairs = 1048576;
  std::int64_t max_mask_elements = 67108864;
  training::metrics::DetectionNoGroundTruth no_ground_truth =
      training::metrics::DetectionNoGroundTruth::exclude;
};
/** @brief One class's exact dataset aggregate, with precision/recall curves. */
struct TaskClassMetric {
  std::uint64_t targets = 0;
  std::uint64_t predictions = 0;
  std::optional<double> average_precision;
  std::vector<double> ap_by_threshold;
  std::vector<double>
      precision;  ///< First threshold, descending global scores.
  std::vector<double> recall;
};
struct TaskMetricReport {
  std::optional<double> mean_average_precision;
  std::optional<double> ap50;
  std::size_t images = 0;
  std::vector<TaskClassMetric> classes;
  std::string
      protocol;  ///< Explicit native protocol; not pycocotools qualification.
};
/** @brief Dataset aggregator; never averages AP computed on separate batches.
 * Masks use full-resolution binary intersection/union, pose uses target-area
 * normalized OKS, OBB uses exact convex corner intersection. Crowd mask regions
 * allow repeated intersection/prediction-area matching. Pose zero-visible
 * targets are ignored. Callers serialize updates; inputs are read-only.
 */
class TaskMetric final {
 public:
  explicit TaskMetric(TaskMetricOptions options);
  ~TaskMetric();
  /** @brief Atomically adds one whole image and rejects duplicate image IDs.
   * Targets reuse ImageResult geometry; scores are ignored. Optional target
   * areas=[T], ignore/crowd=[T] CPU flags preserve source evaluation semantics.
   */
  void update(std::uint64_t image_id, const ImageResult& prediction,
              const ImageResult& target, const Tensor& areas = {},
              const Tensor& ignore = {}, const Tensor& crowd = {});
  TaskMetricReport compute() const;
  /** @brief Versioned bounded state for an existing checkpoint/storage owner.
   */
  std::string save_state(std::size_t max_bytes = 256U << 20U) const;
  void load_state(std::string_view bytes);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace jyppx::liuding::vision::tasks
#endif  // JYPPX_LIUDING_VISION_TASKS_METRICS_HPP
