// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/training/detection_evaluation.hpp>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <limits>
#include <memory>
#include <utility>

namespace jyppx::liuding::training {
namespace {
void check(std::stop_token stop, TelemetryDeadline deadline) {
  if (stop.stop_requested()) throw Cancelled("Detection evaluation cancelled");
  if (deadline && std::chrono::steady_clock::now() >= *deadline)
    throw DeadlineExceeded("Detection evaluation deadline elapsed");
}
class Modes final {
 public:
  explicit Modes(nn::ModelAdapter& model) {
    for (const auto& module : model.state().module().modules())
      values_.emplace_back(module, module->is_training());
  }
  void restore() {
    for (const auto& [m, training] : values_) m->train(training);
    values_.clear();
  }
  ~Modes() {
    try {
      restore();
    } catch (...) {
    }
  }

 private:
  std::vector<std::pair<std::shared_ptr<torch::nn::Module>, bool>> values_;
};
bool integral(torch::ScalarType t) {
  return t == torch::kUInt8 || t == torch::kInt8 || t == torch::kInt16 ||
         t == torch::kInt32 || t == torch::kInt64;
}
std::string number(double v) {
  std::array<char, 64> b{};
  const auto [p, ec] = std::to_chars(b.data(), b.data() + b.size(), v,
                                     std::chars_format::general,
                                     std::numeric_limits<double>::max_digits10);
  if (ec != std::errc{})
    throw InvalidArgument("Cannot encode detection parameter");
  return {b.data(), p};
}
void observe(DetectionEvaluationResult& r,
             const metrics::DetectionSummary& summary,
             const metrics::DetectionMetricOptions& o, const std::string& slice,
             const std::string& source) {
  std::string thresholds, sources;
  for (double t : o.iou_thresholds) {
    if (!thresholds.empty()) thresholds += ',';
    thresholds += number(t);
  }
  for (const auto& s : o.sources) {
    if (!sources.empty()) sources += ',';
    sources += s;
  }
  std::uint64_t total_gt = 0;
  for (const auto& c : summary.per_class) total_gt += c.ground_truth;
  for (std::size_t i = 0; i <= summary.per_class.size(); ++i) {
    const auto value = i ? summary.per_class[i - 1].average_precision
                         : summary.mean_average_precision;
    const auto samples = i ? summary.per_class[i - 1].ground_truth : total_gt;
    TelemetryMetricPoint p;
    p.definition = {
        .metric_id = "detection.box_ap",
        .version = 1,
        .unit = TelemetryMetricUnit::ratio,
        .aggregation = TelemetryAggregation::mean,
        .direction = TelemetryMetricDirection::maximize,
        .parameters = {{"task", "axis_aligned_box_detection"},
                       {"dataset_id", o.dataset_id},
                       {"class_vocabulary_id", o.class_vocabulary_id},
                       {"classes", std::to_string(o.num_classes)},
                       {"iou_thresholds", thresholds},
                       {"min_area", number(o.min_area)},
                       {"max_area", number(o.max_area)},
                       {"max_detections_per_image_class",
                        std::to_string(o.max_detections_per_image_class)},
                       {"no_ground_truth", std::to_string(static_cast<unsigned>(
                                               o.no_ground_truth))},
                       {"sources", sources},
                       {"interpolation", "101_point"},
                       {"coordinates", "continuous_xyxy"},
                       {"matching", "greedy_eligible_first_last_gt_iou_tie"},
                       {"crowd",
                        "ignored_repeat_intersection_over_detection_area"},
                       {"score_ties", "image_id_then_prediction_row"}},
        .context_keys = {"slice", "source", "class"}};
    p.step = {.kind = TelemetryStepKind::sample, .value = samples};
    p.observation = {
        .status = value ? TelemetryValueStatus::finite
                        : TelemetryValueStatus::missing,
        .value = value,
        .reason = value ? "" : "No eligible detection ground truth"};
    p.context = {{"slice", slice},
                 {"source", source},
                 {"class", i ? std::to_string(i - 1) : "all"}};
    p.sample_count = samples;
    r.observations.push_back(std::move(p));
  }
}
}  // namespace
DetectionEvaluationResult evaluate_detection(
    nn::ModelAdapter& model, const DetectionEvaluationSource& source,
    const DetectionEvaluationOptions& o, std::stop_token stop,
    TelemetryDeadline deadline) {
  if (!source || !o.max_batches || !o.max_conversion_bytes ||
      !o.max_summary_values ||
      (o.expected_images && *o.expected_images > o.limits.max_images))
    throw InvalidArgument("Invalid detection evaluation limits/source");
  metrics::DetectionAveragePrecisionMetric metric(o.metric, o.limits);
  const auto copies = o.metric.sources.size() + 1;
  if (o.metric.num_classes >
      o.max_summary_values / copies / (1 + 2 * o.metric.iou_thresholds.size()))
    throw ResourceExhausted("Detection result scalar count exceeds limit");
  check(stop, deadline);
  Modes modes(model);
  torch::NoGradGuard no_grad;
  model.state().eval();
  DetectionEvaluationResult result;
  for (;;) {
    check(stop, deadline);
    auto batch = source(stop);
    check(stop, deadline);
    if (!batch) break;
    if (batch->images.empty())
      throw InvalidArgument(
          "Detection evaluation batches must contain complete images");
    if (result.batches >= o.max_batches ||
        batch->images.size() > o.limits.max_images - metric.image_count() ||
        (o.expected_images &&
         batch->images.size() > *o.expected_images - metric.image_count()))
      throw ResourceExhausted("Detection evaluation input limit exceeded");
    for (const auto& image : batch->images) {
      if (!image.predictions.empty())
        throw InvalidArgument(
            "Detection evaluator source must not supply predictions");
      if (image.targets.size() > o.limits.max_targets_per_image)
        throw ResourceExhausted("Detection target input limit exceeded");
      if ((o.metric.sources.empty() && image.source_index) ||
          (!o.metric.sources.empty() &&
           (!image.source_index ||
            *image.source_index >= o.metric.sources.size())))
        throw InvalidArgument("Detection evaluation source ID differs");
    }
    const auto output = model.invoke(batch->invocation).value;
    check(stop, deadline);
    if (!output.isList() || output.toList().size() != batch->images.size())
      throw InvalidArgument(
          "Detection model must return one box/score/label tuple per image");
    const auto list = output.toList();
    // Preflight ALL shapes and logical staging before any output host copies.
    std::uint64_t payload = 0;
    for (const auto& entry : list) {
      const c10::IValue value = entry;
      if (!value.isTuple() || value.toTupleRef().elements().size() != 3)
        throw InvalidArgument(
            "Detection image output must be a tuple of three tensors");
      const auto& tuple = value.toTupleRef().elements();
      if (!tuple[0].isTensor() || !tuple[1].isTensor() || !tuple[2].isTensor())
        throw InvalidArgument("Detection outputs must be tensors");
      const auto boxes = tuple[0].toTensor(), scores = tuple[1].toTensor(),
                 labels = tuple[2].toTensor();
      if (!boxes.defined() || !scores.defined() || !labels.defined() ||
          boxes.layout() != torch::kStrided ||
          scores.layout() != torch::kStrided ||
          labels.layout() != torch::kStrided ||
          !(boxes.device().is_cpu() || boxes.device().is_cuda()) ||
          scores.device() != boxes.device() ||
          labels.device() != boxes.device() || !boxes.is_floating_point() ||
          !scores.is_floating_point() || !integral(labels.scalar_type()) ||
          boxes.dim() != 2 || boxes.size(1) != 4 || scores.dim() != 1 ||
          labels.dim() != 1 || scores.size(0) != boxes.size(0) ||
          labels.size(0) != boxes.size(0))
        throw InvalidArgument("Invalid detection output tensor contract");
      const auto count = static_cast<std::uint64_t>(boxes.size(0));
      constexpr auto per_prediction =
          96ULL + sizeof(metrics::DetectionPrediction);
      if (count > o.limits.max_predictions_per_image ||
          count > (o.max_conversion_bytes - payload) / per_prediction)
        throw ResourceExhausted(
            "Detection prediction conversion payload exceeds limit");
      payload += count * per_prediction;
    }
    for (std::size_t i = 0; i < batch->images.size(); ++i) {
      check(stop, deadline);
      const c10::IValue value = list.get(i);
      const auto& tuple = value.toTupleRef().elements();
      const auto boxes =
          tuple[0].toTensor().to(torch::kCPU, torch::kFloat64).contiguous();
      const auto scores =
          tuple[1].toTensor().to(torch::kCPU, torch::kFloat64).contiguous();
      const auto labels =
          tuple[2].toTensor().to(torch::kCPU, torch::kInt64).contiguous();
      const auto* b = boxes.const_data_ptr<double>();
      const auto* v = scores.const_data_ptr<double>();
      const auto* c = labels.const_data_ptr<std::int64_t>();
      auto& predictions = batch->images[i].predictions;
      predictions.reserve(static_cast<std::size_t>(boxes.size(0)));
      for (std::int64_t j = 0; j < boxes.size(0); ++j) {
        if (c[j] < 0 ||
            static_cast<std::uint64_t>(c[j]) >= o.metric.num_classes)
          throw InvalidArgument("Detection output class is outside vocabulary");
        predictions.push_back(
            {.box = {b[j * 4], b[j * 4 + 1], b[j * 4 + 2], b[j * 4 + 3]},
             .score = v[j],
             .class_index = static_cast<std::size_t>(c[j])});
      }
    }
    check(stop, deadline);
    metric.update(batch->images);
    check(stop, deadline);
    ++result.batches;
  }
  if (o.expected_images && metric.image_count() != *o.expected_images)
    throw InvalidArgument(
        "Detection EOF does not match expected dataset image count");
  result.overall = metric.summarize();
  observe(result, result.overall, o.metric, "overall", "all");
  for (std::size_t i = 0; i < o.metric.sources.size(); ++i) {
    check(stop, deadline);
    result.by_source.push_back(metric.summarize(i));
    observe(result, result.by_source.back(), o.metric, "source",
            o.metric.sources[i]);
  }
  check(stop, deadline);
  modes.restore();
  return result;
}
}  // namespace jyppx::liuding::training
