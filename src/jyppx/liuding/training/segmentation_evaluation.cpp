// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/training/segmentation_evaluation.hpp>

#include <algorithm>
#include <chrono>
#include <memory>
#include <set>
#include <utility>

namespace jyppx::liuding::training {
namespace {
using SegmentationCounts = metrics::SemanticSegmentationMetric;
void check(std::stop_token stop, TelemetryDeadline deadline) {
  if (stop.stop_requested())
    throw Cancelled("Segmentation evaluation cancelled");
  if (deadline && std::chrono::steady_clock::now() >= *deadline)
    throw DeadlineExceeded("Segmentation evaluation deadline elapsed");
}
class Modes final {
 public:
  explicit Modes(nn::ModelAdapter& model) {
    for (const auto& module : model.state().module().modules())
      values_.emplace_back(module, module->is_training());
  }
  void restore() {
    for (const auto& [module, training] : values_) module->train(training);
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
SegmentationEvaluationSummary summary(SegmentationCounts& metric) {
  SegmentationEvaluationSummary result;
  result.pixels = metric.total_pixels();
  result.counts = metric.class_counts();
  if (!result.pixels) {
    result.per_class.resize(result.counts.size());
    return result;
  }
  result.per_class = metric.per_class_scores();
  try {
    result.score = metric.compute();
  } catch (
      const NotReady&) { /* No included scoring denominator remains missing. */
  }
  return result;
}
void observe(SegmentationEvaluationResult& result,
             const SegmentationEvaluationSummary& value,
             const metrics::SegmentationMetricOptions& o,
             const std::string& slice, const std::string& source) {
  for (std::size_t i = 0; i <= value.per_class.size(); ++i) {
    const auto score = i ? value.per_class[i - 1] : value.score;
    const auto samples = static_cast<std::uint64_t>(
        i ? value.counts[i - 1].target : value.pixels);
    TelemetryMetricPoint point;
    point.definition =
        {.metric_id = "semantic." +
                      std::string(metrics::segmentation_score_name(o.score)),
         .version = 1,
         .unit = TelemetryMetricUnit::ratio,
         .aggregation = TelemetryAggregation::mean,
         .direction = TelemetryMetricDirection::maximize,
         .parameters =
             {{"task", "semantic_segmentation"},
              {"classes", std::to_string(o.num_classes)},
              {"input", std::string(metrics::segmentation_input_name(o.input))},
              {"average",
               std::string(metrics::classification_average_name(o.average))},
              {"absent",
               std::string(metrics::absent_class_policy_name(o.absent))},
              {"ignore_index",
               o.ignore_index ? std::to_string(*o.ignore_index) : "none"},
              {"background_index", std::to_string(o.background_index)},
              {"include_background", o.include_background ? "true" : "false"},
              {"ties", "lowest_class_index"}},
         .context_keys = {"slice", "source", "class"}};
    point.step = {.kind = TelemetryStepKind::sample, .value = samples};
    point.observation = {
        .status = score ? TelemetryValueStatus::finite
                        : TelemetryValueStatus::missing,
        .value = score,
        .reason = score ? "" : "No included scoring denominator"};
    point.context = {{"slice", slice},
                     {"source", source},
                     {"class", i ? std::to_string(i - 1) : "all"}};
    point.sample_count = samples;
    result.observations.push_back(std::move(point));
  }
}
}  // namespace

SegmentationEvaluationResult evaluate_semantic_segmentation(
    nn::ModelAdapter& model, const SegmentationEvaluationSource& source,
    const SegmentationEvaluationOptions& o, std::stop_token stop,
    TelemetryDeadline deadline) {
  if (!source || !o.max_batches || !o.max_images || !o.max_pixels ||
      !o.max_counts_bytes || o.sources.size() > o.max_sources ||
      o.metric.average == metrics::ClassificationAverage::none)
    throw InvalidArgument("Invalid segmentation evaluation options");
  std::set<std::string> names;
  for (const auto& name : o.sources)
    if (name.empty() || name.size() > 256 || !names.insert(name).second ||
        !std::all_of(name.begin(), name.end(), [](unsigned char c) {
          return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                 (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
        }))
      throw InvalidArgument("Invalid or duplicate segmentation source name");
  // Live class counts, returned count copies, and one atomic-update delta.
  const auto copies =
      2ULL * (static_cast<std::uint64_t>(o.sources.size()) + 1) + 1;
  if (o.metric.num_classes > o.max_counts_bytes / 24 / copies)
    throw ResourceExhausted(
        "Segmentation evaluation count payload exceeds limit");
  check(stop, deadline);
  SegmentationCounts overall(o.metric, o.limits);
  std::vector<std::unique_ptr<SegmentationCounts>> slices;
  for (std::size_t i = 0; i < o.sources.size(); ++i)
    slices.push_back(std::make_unique<SegmentationCounts>(o.metric, o.limits));
  SegmentationEvaluationResult result;
  Modes modes(model);
  torch::NoGradGuard no_grad;
  model.state().eval();
  for (;;) {
    check(stop, deadline);
    auto batch = source(stop);
    check(stop, deadline);
    if (!batch) break;
    const auto& targets = batch->targets;
    if (!targets.defined() || targets.dim() != 3 || targets.size(0) <= 0 ||
        targets.size(1) <= 0 || targets.size(2) <= 0)
      throw InvalidArgument(
          "Segmentation evaluation targets must be nonempty N,H,W");
    const auto images = static_cast<std::uint64_t>(targets.size(0));
    const auto pixels = static_cast<std::uint64_t>(targets.numel());
    if (result.batches >= o.max_batches ||
        images > o.max_images - result.images ||
        pixels > o.max_pixels - result.pixels_seen ||
        pixels > o.limits.max_batch_pixels)
      throw ResourceExhausted("Segmentation evaluation input limit exceeded");
    if ((o.sources.empty() && !batch->source_indices.empty()) ||
        (!o.sources.empty() && batch->source_indices.size() != images) ||
        std::any_of(batch->source_indices.begin(), batch->source_indices.end(),
                    [&](auto index) { return index >= o.sources.size(); }))
      throw InvalidArgument("Segmentation source IDs must cover every image");
    const auto output = model.invoke(batch->invocation).value;
    check(stop, deadline);
    if (!output.isTensor())
      throw InvalidArgument("Segmentation model must return a tensor");
    const auto predictions = output.toTensor();
    if (slices.empty())
      overall.update(predictions, targets);
    else {
      // View each image once: no repeated argmax or source index_select copies.
      // Preflight full output shape before taking an image view.
      const bool logits = o.metric.input == metrics::SegmentationInput::logits;
      if (!predictions.defined() || predictions.dim() != (logits ? 4 : 3) ||
          predictions.size(0) != targets.size(0))
        throw InvalidArgument("Segmentation model output batch shape differs");
      for (std::size_t i = 0; i < batch->source_indices.size(); ++i) {
        check(stop, deadline);
        slices[batch->source_indices[i]]->update(
            predictions.narrow(0, static_cast<std::int64_t>(i), 1),
            targets.narrow(0, static_cast<std::int64_t>(i), 1));
      }
    }
    check(stop, deadline);
    ++result.batches;
    result.images += images;
    result.pixels_seen += pixels;
  }
  for (const auto& slice : slices) overall.merge(*slice);
  result.overall = summary(overall);
  observe(result, result.overall, o.metric, "overall", "all");
  for (std::size_t i = 0; i < slices.size(); ++i) {
    result.by_source.push_back(summary(*slices[i]));
    observe(result, result.by_source.back(), o.metric, "source", o.sources[i]);
  }
  check(stop, deadline);
  modes.restore();
  return result;
}
}  // namespace jyppx::liuding::training
