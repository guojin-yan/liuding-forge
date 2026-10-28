// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "evaluation_session_state.hpp"

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
namespace {
namespace state = detail::evaluation_session;
using DetectionCounts = metrics::DetectionAveragePrecisionMetric;
std::unique_ptr<DetectionCounts> make_metric(
    const DetectionEvaluationOptions& o) {
  if (!o.max_batches || !o.max_conversion_bytes || !o.max_summary_values ||
      (o.expected_images && *o.expected_images > o.limits.max_images))
    throw InvalidArgument("Invalid detection evaluation limits/source");
  auto metric = std::make_unique<metrics::DetectionAveragePrecisionMetric>(
      o.metric, o.limits);
  const auto copies = o.metric.sources.size() + 1;
  if (o.metric.num_classes >
      o.max_summary_values / copies / (1 + 2 * o.metric.iou_thresholds.size()))
    throw ResourceExhausted("Detection result scalar count exceeds limit");
  return metric;
}
void infer_batch(nn::ModelAdapter& model, DetectionEvaluationBatch& batch,
                 DetectionCounts& metric, const DetectionEvaluationOptions& o,
                 std::uint64_t batches, std::stop_token stop,
                 TelemetryDeadline deadline) {
  if (batch.images.empty())
    throw InvalidArgument(
        "Detection evaluation batches must contain complete images");
  if (batches >= o.max_batches ||
      batch.images.size() > o.limits.max_images - metric.image_count() ||
      (o.expected_images &&
       batch.images.size() > *o.expected_images - metric.image_count()))
    throw ResourceExhausted("Detection evaluation input limit exceeded");
  for (const auto& image : batch.images) {
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
  const auto output = model.invoke(batch.invocation).value;
  check(stop, deadline);
  if (!output.isList() || output.toList().size() != batch.images.size())
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
  for (std::size_t i = 0; i < batch.images.size(); ++i) {
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
    auto& predictions = batch.images[i].predictions;
    predictions.reserve(static_cast<std::size_t>(boxes.size(0)));
    for (std::int64_t j = 0; j < boxes.size(0); ++j) {
      if (c[j] < 0 || static_cast<std::uint64_t>(c[j]) >= o.metric.num_classes)
        throw InvalidArgument("Detection output class is outside vocabulary");
      predictions.push_back(
          {.box = {b[j * 4], b[j * 4 + 1], b[j * 4 + 2], b[j * 4 + 3]},
           .score = v[j],
           .class_index = static_cast<std::size_t>(c[j])});
    }
  }
  check(stop, deadline);
  metric.update(batch.images);
}
DetectionEvaluationResult make_result(
    const DetectionCounts& metric, const DetectionEvaluationOptions& o,
    std::uint64_t batches, std::stop_token stop = {},
    TelemetryDeadline deadline = std::nullopt) {
  DetectionEvaluationResult result;
  result.batches = batches;
  result.overall = metric.summarize();
  observe(result, result.overall, o.metric, "overall", "all");
  for (std::size_t i = 0; i < o.metric.sources.size(); ++i) {
    check(stop, deadline);
    result.by_source.push_back(metric.summarize(i));
    observe(result, result.by_source.back(), o.metric, "source",
            o.metric.sources[i]);
  }
  check(stop, deadline);
  return result;
}
constexpr std::uint64_t kProgressBytes = 32;
constexpr std::uint64_t kProgressMagic = 0x31544544454AULL;
}  // namespace
DetectionEvaluationResult evaluate_detection(
    nn::ModelAdapter& model, const DetectionEvaluationSource& source,
    const DetectionEvaluationOptions& o, std::stop_token stop,
    TelemetryDeadline deadline) {
  if (!source) throw InvalidArgument("Detection source is empty");
  auto metric = make_metric(o);
  check(stop, deadline);
  Modes modes(model);
  torch::NoGradGuard no_grad;
  model.state().eval();
  std::uint64_t batches = 0;
  for (;;) {
    check(stop, deadline);
    auto batch = source(stop);
    check(stop, deadline);
    if (!batch) break;
    infer_batch(model, *batch, *metric, o, batches, stop, deadline);
    ++batches;
  }
  if (o.expected_images && metric->image_count() != *o.expected_images)
    throw InvalidArgument(
        "Detection EOF does not match expected dataset image count");
  auto result = make_result(*metric, o, batches, stop, deadline);
  modes.restore();
  return result;
}

class DetectionEvaluationSession::Impl {
 public:
  Impl(DetectionEvaluationIdentity identity, DetectionEvaluationOptions options)
      : identity_(std::move(identity)), options_(std::move(options)) {
    state::append_text(topology_, "jyppx.detection-evaluation/v1");
    state::append_identity(topology_, identity_);
    for (auto value :
         {options_.max_batches, options_.max_conversion_bytes,
          options_.max_summary_values,
          static_cast<std::uint64_t>(options_.expected_images.has_value()),
          options_.expected_images.value_or(0),
          static_cast<std::uint64_t>(options_.limits.max_classes),
          static_cast<std::uint64_t>(options_.limits.max_sources),
          options_.limits.max_images, options_.limits.max_predictions_per_image,
          options_.limits.max_targets_per_image,
          options_.limits.max_retained_predictions,
          options_.limits.max_state_bytes, options_.limits.max_workspace_bytes,
          options_.limits.max_match_operations_per_update})
      state::append_integer(topology_, value);
    metric_ = make_metric(options_);
  }
  DetectionEvaluationIdentity identity_;
  DetectionEvaluationOptions options_;
  state::Bytes topology_;
  std::unique_ptr<DetectionCounts> metric_;
  DetectionEvaluationProgress progress_;
  bool running_ = false, boundary_ = false;
};
DetectionEvaluationSession::DetectionEvaluationSession(
    DetectionEvaluationIdentity identity, DetectionEvaluationOptions options)
    : impl_(std::make_unique<Impl>(std::move(identity), std::move(options))) {}
DetectionEvaluationSession::~DetectionEvaluationSession() = default;
DetectionEvaluationProgress DetectionEvaluationSession::progress()
    const noexcept {
  return impl_->progress_;
}
DetectionEvaluationResult DetectionEvaluationSession::result() const {
  if (!impl_->progress_.complete)
    throw NotReady("Detection evaluation requires valid EOF");
  return make_result(*impl_->metric_, impl_->options_,
                     impl_->progress_.batches);
}
std::string DetectionEvaluationSession::participant_id() const {
  return impl_->identity_.participant_id;
}
StateManifest DetectionEvaluationSession::describe_state() const {
  StateManifest manifest{
      .participant_id = participant_id(),
      .fields = {{.name = "topology", .max_bytes = impl_->topology_.size()},
                 {.name = "progress", .max_bytes = kProgressBytes}},
      .degradation_reasons = {}};
  for (auto field : impl_->metric_->describe_state().fields) {
    field.name = state::prefix(0) + field.name;
    manifest.fields.push_back(std::move(field));
  }
  return manifest;
}
void DetectionEvaluationSession::save_state(StateWriter& writer) const {
  if (impl_->running_ && !impl_->boundary_)
    throw NotReady("Evaluation save requires a committed boundary");
  state::Bytes progress;
  for (auto value :
       {kProgressMagic, impl_->progress_.batches, impl_->progress_.images,
        std::uint64_t{impl_->progress_.complete ? 1U : 0U}})
    state::append_integer(progress, value);
  writer.write_blob("topology", impl_->topology_);
  writer.write_blob("progress", progress);
  state::SliceWriter slice(writer, 0);
  impl_->metric_->save_state(slice);
}
void DetectionEvaluationSession::load_state(const StateReader& reader,
                                            RestorePolicy policy) {
  if (impl_->running_) throw NotReady("Cannot restore a running evaluation");
  state::validate_reader(reader, describe_state(), impl_->topology_, policy);
  const auto progress = reader.read_blob("progress", kProgressBytes);
  if (progress.size() != kProgressBytes ||
      state::read_integer(progress, 0) != kProgressMagic ||
      state::read_integer(progress, 24) > 1)
    throw SerializationError("Invalid detection evaluation progress");
  DetectionEvaluationProgress next{
      .batches = state::read_integer(progress, 8),
      .images = state::read_integer(progress, 16),
      .complete = state::read_integer(progress, 24) != 0};
  const auto& o = impl_->options_;
  if (next.batches > o.max_batches || next.images > o.limits.max_images ||
      (next.batches == 0) != (next.images == 0) || next.batches > next.images ||
      (o.expected_images &&
       (next.images > *o.expected_images ||
        (next.complete && next.images != *o.expected_images))))
    throw SerializationError("Inconsistent detection evaluation counters");
  auto candidate = make_metric(o);
  state::SliceReader slice(reader, 0, candidate->describe_state());
  candidate->load_state(slice, policy);
  if (candidate->image_count() != next.images)
    throw SerializationError(
        "Detection image identities do not cover committed cursor");
  if (next.complete)
    static_cast<void>(make_result(*candidate, o, next.batches));
  impl_->metric_.swap(candidate);
  impl_->progress_ = next;
}
DetectionEvaluationResult DetectionEvaluationSession::run(
    nn::ModelAdapter& model, const IndexedDetectionEvaluationSource& source,
    const std::function<void(const DetectionEvaluationSession&)>& committed,
    std::stop_token stop, TelemetryDeadline deadline) {
  if (!source) throw InvalidArgument("Detection indexed source is empty");
  check(stop, deadline);
  state::RunningGuard running(impl_->running_, impl_->boundary_);
  const auto notify = [&] {
    state::notify(*this, committed, impl_->boundary_);
    check(stop, deadline);
  };
  if (impl_->progress_.complete) {
    notify();
    return result();
  }
  const auto& o = impl_->options_;
  Modes modes(model);
  torch::NoGradGuard no_grad;
  model.state().eval();
  for (;;) {
    check(stop, deadline);
    auto batch = source(impl_->progress_.batches, stop);
    check(stop, deadline);
    if (!batch) {
      if (o.expected_images && impl_->progress_.images != *o.expected_images)
        throw InvalidArgument(
            "Detection EOF does not match expected input total");
      auto final = make_result(*impl_->metric_, o, impl_->progress_.batches,
                               stop, deadline);
      modes.restore();
      impl_->progress_.complete = true;
      notify();
      return final;
    }
    // Matching is internally atomic; no throwing check separates it from cursor
    // commit, and prior dataset records are never copied for a batch retry.
    infer_batch(model, *batch, *impl_->metric_, o, impl_->progress_.batches,
                stop, deadline);
    ++impl_->progress_.batches;
    impl_->progress_.images = impl_->metric_->image_count();
    notify();
  }
}
}  // namespace jyppx::liuding::training
