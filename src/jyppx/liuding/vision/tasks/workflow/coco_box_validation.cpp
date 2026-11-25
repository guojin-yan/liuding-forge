// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "coco_box_validation.hpp"

#include <nlohmann/json.hpp>
#include <set>
namespace jyppx::liuding::vision::tasks::detail {
namespace im = data::image;
namespace tm = training::metrics;
CocoBoxValidation::CocoBoxValidation(
    std::shared_ptr<const im::AnnotationIndex> targets,
    std::string dataset_identity, const RecipeSpec& recipe,
    std::string execution)
    : targets_(std::move(targets)),
      results_(recipe.evaluation),
      execution_(std::move(execution)) {
  if (!targets_ || targets_->images.empty())
    throw InvalidArgument("COCO box validation requires source annotations");
  std::set<std::int64_t> ids;
  for (std::size_t i = 0; i < targets_->images.size(); ++i) {
    if (targets_->images[i].source_image_id <= 0 ||
        !ids.insert(targets_->images[i].source_image_id).second)
      throw InvalidArgument(
          "COCO box validation requires unique positive source IDs");
    if (!by_path_.emplace(targets_->images[i].image_path, i).second)
      throw InvalidArgument("Duplicate COCO box validation source path");
  }
  options_.dataset_id = std::move(dataset_identity);
  options_.class_vocabulary_id = options_.dataset_id + ":source-category-order";
  options_.num_classes = static_cast<std::size_t>(recipe.model.num_classes);
  options_.max_detections_per_image_class = 100;
  reset();
}
void CocoBoxValidation::reset() {
  metrics_.clear();
  // COCO summarizes all-area recall at maxDet1/10/100 and AP/recall for
  // small/medium/large at maxDet100. Boundaries are inclusive, as in COCOeval.
  for (std::size_t i = 0; i < 6; ++i) {
    auto options = options_;
    options.max_detections_per_image_class = i == 0 ? 1 : i == 1 ? 10 : 100;
    if (i == 3) options.max_area = 32 * 32;
    if (i == 4) {
      options.min_area = 32 * 32;
      options.max_area = 96 * 96;
    }
    if (i == 5) options.min_area = 96 * 96;
    metrics_.push_back(
        std::make_unique<tm::DetectionAveragePrecisionMetric>(options));
  }
  images_ = 0;
}
void CocoBoxValidation::update(const im::ImageBatch& batch,
                               std::span<const ImageResult> results,
                               std::uint64_t epoch) {
  if (!epoch_ || *epoch_ != epoch) {
    reset();
    epoch_ = epoch;
  }
  if (results.size() != batch.items.size())
    throw InvalidArgument("COCO box validation output cardinality mismatch");
  std::vector<tm::DetectionImage> updates;
  updates.reserve(results.size());
  for (std::size_t i = 0; i < results.size(); ++i) {
    const auto& metadata = batch.items[i];
    const auto found = by_path_.find(metadata.image_path);
    if (found == by_path_.end() || !metadata.geometry)
      throw InvalidArgument("COCO box validation source/geometry missing");
    const auto& source = targets_->images[found->second];
    auto result =
        restore_result(results[i], *metadata.geometry,
                       results_.max_mask_elements, results_.clip_boxes);
    if (result.extent.width != source.width ||
        result.extent.height != source.height)
      throw InvalidArgument("COCO box validation source extent mismatch");
    tm::DetectionImage image;
    image.image_id = static_cast<std::uint64_t>(source.source_image_id);
    for (const auto& target : source.objects) {
      if (!target.box) throw InvalidArgument("COCO source target has no box");
      const auto& b = *target.box;
      image.targets.push_back({{b.x_min, b.y_min, b.x_max, b.y_max},
                               target.category,
                               target.crowd,
                               target.ignore || target.difficult,
                               target.area});
    }
    const auto boxes = result.boxes.accessor<float, 2>();
    const auto scores = result.scores.accessor<float, 1>();
    const auto classes = result.classes.accessor<std::int64_t, 1>();
    for (std::int64_t row = 0; row < result.boxes.size(0); ++row)
      image.predictions.push_back(
          {{boxes[row][0], boxes[row][1], boxes[row][2], boxes[row][3]},
           scores[row],
           static_cast<std::size_t>(classes[row])});
    updates.push_back(std::move(image));
  }
  for (auto& metric : metrics_) metric->update(updates);
  images_ += updates.size();
}
std::string CocoBoxValidation::finish(std::stop_token stop) {
  if (stop.stop_requested()) throw Cancelled("COCO box validation cancelled");
  if (images_ != targets_->images.size())
    throw InvalidArgument("COCO box validation did not visit the entire split");
  std::vector<tm::DetectionSummary> summaries;
  for (const auto& metric : metrics_) summaries.push_back(metric->summarize());
  const auto& summary = summaries[2];
  nlohmann::json report{{"schema", "vision.evaluation/v1"},
                        {"protocol", "axis_aligned_box_ap.r101/v1"},
                        {"execution", execution_},
                        {"dataset_identity", options_.dataset_id},
                        {"images", images_},
                        {"metric_max_detections_per_image_class", 100},
                        {"prediction_max_detections", results_.max_detections},
                        {"map", nullptr},
                        {"ap50", nullptr}};
  if (summary.mean_average_precision)
    report["map"] = *summary.mean_average_precision;
  double total = 0;
  std::size_t classes = 0;
  for (const auto& c : summary.per_class)
    if (!c.ap_by_iou.empty() && c.ap_by_iou.front()) {
      total += *c.ap_by_iou.front();
      ++classes;
    }
  if (classes) report["ap50"] = total / classes;
  const auto recall = [](const tm::DetectionSummary& s) {
    double sum = 0;
    std::size_t count = 0;
    for (const auto& c : s.per_class)
      for (const auto value : c.recall_by_iou)
        if (value) {
          sum += *value;
          ++count;
        }
    return count ? sum / count : -1.;
  };
  double ap75 = 0;
  std::size_t ap75_classes = 0;
  for (const auto& c : summary.per_class)
    if (c.ap_by_iou[5]) {
      ap75 += *c.ap_by_iou[5];
      ++ap75_classes;
    }
  report["coco_metrics"] = {
      {"AP", summary.mean_average_precision.value_or(-1.)},
      {"AP50", classes ? total / classes : -1.},
      {"AP75", ap75_classes ? ap75 / ap75_classes : -1.},
      {"AP_small", summaries[3].mean_average_precision.value_or(-1.)},
      {"AP_medium", summaries[4].mean_average_precision.value_or(-1.)},
      {"AP_large", summaries[5].mean_average_precision.value_or(-1.)},
      {"AR1", recall(summaries[0])},
      {"AR10", recall(summaries[1])},
      {"AR100", recall(summary)},
      {"AR_small", recall(summaries[3])},
      {"AR_medium", recall(summaries[4])},
      {"AR_large", recall(summaries[5])}};
  epoch_.reset();
  return report.dump(2);
}
}  // namespace jyppx::liuding::vision::tasks::detail
