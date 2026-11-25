// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "rtdetr_validation.hpp"

#include <nlohmann/json.hpp>
#include <set>
namespace jyppx::liuding::vision::tasks::detail {
namespace im = data::image;
namespace tm = training::metrics;
RtDetrValidation::RtDetrValidation(
    std::shared_ptr<const im::AnnotationIndex> targets,
    std::string dataset_identity, const RecipeSpec& recipe)
    : targets_(std::move(targets)), results_(recipe.evaluation) {
  if (!targets_ || targets_->images.empty())
    throw InvalidArgument("RT-DETR validation requires source annotations");
  std::set<std::int64_t> ids;
  for (std::size_t i = 0; i < targets_->images.size(); ++i) {
    if (targets_->images[i].source_image_id <= 0 ||
        !ids.insert(targets_->images[i].source_image_id).second)
      throw InvalidArgument(
          "RT-DETR validation requires unique positive source IDs");
    if (!by_path_.emplace(targets_->images[i].image_path, i).second)
      throw InvalidArgument("Duplicate RT-DETR validation source path");
  }
  options_.dataset_id = std::move(dataset_identity);
  options_.class_vocabulary_id = options_.dataset_id + ":source-category-order";
  options_.num_classes = static_cast<std::size_t>(recipe.model.num_classes);
  options_.max_detections_per_image_class = 100;
  reset();
}
void RtDetrValidation::reset() {
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
training::StepOutput RtDetrValidation::step(TaskBinding& binding,
                                            const training::StepInput& input) {
  if (input.context.phase != training::StepPhase::validation)
    throw InvalidArgument("RT-DETR inference validation is validation-only");
  if (!epoch_ || *epoch_ != input.context.epoch) {
    reset();
    epoch_ = input.context.epoch;
  }
  const auto batch = im::image_batch_from_value(input.batch.value);
  if (!batch.images.defined() || batch.images.dim() != 4 || batch.items.empty())
    throw InvalidArgument("RT-DETR validation requires a packed image batch");
  auto invocation = input.invocation;
  // Evaluation does not need supervised targets or denoising queries.
  invocation.positional.at(1) = torch::empty({0, 6}, batch.images.options());
  auto prediction = binding.model->invoke(invocation);
  auto results = binding.decode(
      prediction.value, {batch.images.size(2), batch.images.size(3)}, results_);
  if (results.size() != batch.items.size())
    throw InvalidArgument("RT-DETR validation output cardinality mismatch");
  std::vector<tm::DetectionImage> updates;
  updates.reserve(results.size());
  for (std::size_t i = 0; i < results.size(); ++i) {
    const auto& metadata = batch.items[i];
    const auto found = by_path_.find(metadata.image_path);
    if (found == by_path_.end() || !metadata.geometry)
      throw InvalidArgument("RT-DETR validation source/geometry missing");
    const auto& source = targets_->images[found->second];
    auto result = restore_result(results[i], *metadata.geometry,
                                 results_.max_mask_elements, false);
    if (result.extent.width != source.width ||
        result.extent.height != source.height)
      throw InvalidArgument("RT-DETR validation source extent mismatch");
    tm::DetectionImage image;
    image.image_id = static_cast<std::uint64_t>(source.source_image_id);
    for (const auto& target : source.objects) {
      if (!target.box)
        throw InvalidArgument("RT-DETR source target has no box");
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
  training::StepOutput output;
  output.sample_count = batch.items.size();
  return output;
}
std::string RtDetrValidation::finish(std::stop_token stop) {
  if (stop.stop_requested()) throw Cancelled("RT-DETR validation cancelled");
  if (images_ != targets_->images.size())
    throw InvalidArgument("RT-DETR validation did not visit the entire split");
  std::vector<tm::DetectionSummary> summaries;
  for (const auto& metric : metrics_) summaries.push_back(metric->summarize());
  const auto& summary = summaries[2];
  nlohmann::json report{{"schema", "vision.evaluation/v1"},
                        {"protocol", "axis_aligned_box_ap.r101/v1"},
                        {"execution", "rtdetr.batched_inference/v1"},
                        {"dataset_identity", options_.dataset_id},
                        {"images", images_},
                        {"metric_max_detections_per_image_class", 100},
                        {"prediction_max_detections", 300},
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
