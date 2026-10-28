// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "evaluation_session_state.hpp"

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

namespace {
namespace state = detail::evaluation_session;
using EvaluationMetrics = std::vector<std::unique_ptr<SegmentationCounts>>;
void validate_options(const SegmentationEvaluationOptions& o) {
  if (!o.max_batches || !o.max_images || !o.max_pixels || !o.max_counts_bytes ||
      o.sources.size() > o.max_sources ||
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
}
EvaluationMetrics make_metrics(const SegmentationEvaluationOptions& o) {
  EvaluationMetrics values;
  for (std::size_t i = 0; i <= o.sources.size(); ++i)
    values.push_back(std::make_unique<SegmentationCounts>(o.metric, o.limits));
  return values;
}
SegmentationEvaluationResult make_result(
    const EvaluationMetrics& values, const SegmentationEvaluationOptions& o,
    SegmentationEvaluationProgress progress) {
  SegmentationEvaluationResult result;
  result.batches = progress.batches;
  result.images = progress.images;
  result.pixels_seen = progress.pixels_seen;
  result.overall = summary(*values[0]);
  observe(result, result.overall, o.metric, "overall", "all");
  for (std::size_t i = 0; i < o.sources.size(); ++i) {
    result.by_source.push_back(summary(*values[i + 1]));
    observe(result, result.by_source.back(), o.metric, "source", o.sources[i]);
  }
  return result;
}
constexpr std::uint64_t kProgressBytes = 40;
constexpr std::uint64_t kProgressMagic = 0x31474553454AULL;
}  // namespace

SegmentationEvaluationResult evaluate_semantic_segmentation(
    nn::ModelAdapter& model, const SegmentationEvaluationSource& source,
    const SegmentationEvaluationOptions& o, std::stop_token stop,
    TelemetryDeadline deadline) {
  validate_options(o);
  if (!source) throw InvalidArgument("Segmentation batch source is empty");
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

class SegmentationEvaluationSession::Impl {
 public:
  Impl(SegmentationEvaluationIdentity identity,
       SegmentationEvaluationOptions options,
       std::optional<std::uint64_t> images, std::optional<std::uint64_t> pixels)
      : identity_(std::move(identity)),
        options_(std::move(options)),
        expected_images_(images),
        expected_pixels_(pixels) {
    validate_options(options_);
    if ((images && *images > options_.max_images) ||
        (pixels && *pixels > options_.max_pixels) ||
        (images && pixels &&
         (*images > *pixels || ((*images == 0) != (*pixels == 0)))))
      throw InvalidArgument(
          "Expected segmentation totals exceed input contract");
    state::append_text(topology_, "jyppx.segmentation-evaluation/v1");
    state::append_identity(topology_, identity_);
    for (auto value :
         {options_.max_batches, options_.max_images, options_.max_pixels,
          options_.max_counts_bytes,
          static_cast<std::uint64_t>(options_.max_sources),
          static_cast<std::uint64_t>(options_.limits.max_classes),
          options_.limits.max_state_bytes, options_.limits.max_batch_pixels,
          static_cast<std::uint64_t>(options_.limits.chunk_pixels),
          static_cast<std::uint64_t>(images.has_value()), images.value_or(0),
          static_cast<std::uint64_t>(pixels.has_value()), pixels.value_or(0),
          static_cast<std::uint64_t>(options_.sources.size())})
      state::append_integer(topology_, value);
    for (const auto& source : options_.sources)
      state::append_text(topology_, source);
    values_ = make_metrics(options_);
  }
  SegmentationEvaluationIdentity identity_;
  SegmentationEvaluationOptions options_;
  std::optional<std::uint64_t> expected_images_, expected_pixels_;
  state::Bytes topology_;
  EvaluationMetrics values_;
  SegmentationEvaluationProgress progress_;
  bool running_ = false, boundary_ = false;
};
SegmentationEvaluationSession::SegmentationEvaluationSession(
    SegmentationEvaluationIdentity identity,
    SegmentationEvaluationOptions options, std::optional<std::uint64_t> images,
    std::optional<std::uint64_t> pixels)
    : impl_(std::make_unique<Impl>(std::move(identity), std::move(options),
                                   images, pixels)) {}
SegmentationEvaluationSession::~SegmentationEvaluationSession() = default;
SegmentationEvaluationProgress SegmentationEvaluationSession::progress()
    const noexcept {
  return impl_->progress_;
}
SegmentationEvaluationResult SegmentationEvaluationSession::result() const {
  if (!impl_->progress_.complete)
    throw NotReady("Segmentation evaluation requires valid EOF");
  return make_result(impl_->values_, impl_->options_, impl_->progress_);
}
std::string SegmentationEvaluationSession::participant_id() const {
  return impl_->identity_.participant_id;
}
StateManifest SegmentationEvaluationSession::describe_state() const {
  StateManifest manifest{
      .participant_id = participant_id(),
      .fields = {{.name = "topology", .max_bytes = impl_->topology_.size()},
                 {.name = "progress", .max_bytes = kProgressBytes}},
      .degradation_reasons = {}};
  for (std::size_t i = 0; i < impl_->values_.size(); ++i)
    for (auto field : impl_->values_[i]->describe_state().fields) {
      field.name = state::prefix(i) + field.name;
      manifest.fields.push_back(std::move(field));
    }
  return manifest;
}
void SegmentationEvaluationSession::save_state(StateWriter& writer) const {
  if (impl_->running_ && !impl_->boundary_)
    throw NotReady("Evaluation save requires a committed boundary");
  state::Bytes progress;
  for (auto value : {kProgressMagic, impl_->progress_.batches,
                     impl_->progress_.images, impl_->progress_.pixels_seen,
                     std::uint64_t{impl_->progress_.complete ? 1U : 0U}})
    state::append_integer(progress, value);
  writer.write_blob("topology", impl_->topology_);
  writer.write_blob("progress", progress);
  for (std::size_t i = 0; i < impl_->values_.size(); ++i) {
    state::SliceWriter slice(writer, i);
    impl_->values_[i]->save_state(slice);
  }
}
void SegmentationEvaluationSession::load_state(const StateReader& reader,
                                               RestorePolicy policy) {
  if (impl_->running_) throw NotReady("Cannot restore a running evaluation");
  state::validate_reader(reader, describe_state(), impl_->topology_, policy);
  const auto progress = reader.read_blob("progress", kProgressBytes);
  if (progress.size() != kProgressBytes ||
      state::read_integer(progress, 0) != kProgressMagic ||
      state::read_integer(progress, 32) > 1)
    throw SerializationError("Invalid segmentation evaluation progress");
  SegmentationEvaluationProgress next{
      .batches = state::read_integer(progress, 8),
      .images = state::read_integer(progress, 16),
      .pixels_seen = state::read_integer(progress, 24),
      .complete = state::read_integer(progress, 32) != 0};
  const auto& o = impl_->options_;
  if (next.batches > o.max_batches || next.images > o.max_images ||
      next.pixels_seen > o.max_pixels ||
      (next.batches == 0) != (next.images == 0) ||
      (next.images == 0) != (next.pixels_seen == 0) ||
      next.batches > next.images || next.images > next.pixels_seen ||
      (next.pixels_seen &&
       (next.pixels_seen - 1) / o.limits.max_batch_pixels >= next.batches) ||
      (impl_->expected_images_ &&
       (next.images > *impl_->expected_images_ ||
        (next.complete && next.images != *impl_->expected_images_))) ||
      (impl_->expected_pixels_ &&
       (next.pixels_seen > *impl_->expected_pixels_ ||
        (next.complete && next.pixels_seen != *impl_->expected_pixels_))))
    throw SerializationError("Inconsistent segmentation evaluation counters");
  auto candidate = make_metrics(o);
  for (std::size_t i = 0; i < candidate.size(); ++i) {
    state::SliceReader slice(reader, i, candidate[i]->describe_state());
    candidate[i]->load_state(slice, policy);
    if (static_cast<std::uint64_t>(candidate[i]->total_pixels()) >
        next.pixels_seen)
      throw SerializationError("Segmentation counts exceed committed pixels");
  }
  if (!o.metric.ignore_index &&
      static_cast<std::uint64_t>(candidate[0]->total_pixels()) !=
          next.pixels_seen)
    throw SerializationError(
        "Segmentation counts do not cover committed pixels");
  if (!o.sources.empty())
    for (std::size_t c = 0; c < o.metric.num_classes; ++c) {
      const auto& total = candidate[0]->class_counts()[c];
      auto remaining = total;
      for (std::size_t i = 1; i < candidate.size(); ++i) {
        const auto& part = candidate[i]->class_counts()[c];
        if (part.intersection > remaining.intersection ||
            part.target > remaining.target ||
            part.predicted > remaining.predicted)
          throw SerializationError("Segmentation source counts exceed overall");
        remaining.intersection -= part.intersection;
        remaining.target -= part.target;
        remaining.predicted -= part.predicted;
      }
      if (remaining.intersection || remaining.target || remaining.predicted)
        throw SerializationError(
            "Segmentation source counts do not cover overall");
    }
  if (next.complete)
    for (const auto& value : candidate) static_cast<void>(summary(*value));
  impl_->values_.swap(candidate);
  impl_->progress_ = next;
}
SegmentationEvaluationResult SegmentationEvaluationSession::run(
    nn::ModelAdapter& model, const IndexedSegmentationEvaluationSource& source,
    const std::function<void(const SegmentationEvaluationSession&)>& committed,
    std::stop_token stop, TelemetryDeadline deadline) {
  if (!source) throw InvalidArgument("Segmentation indexed source is empty");
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
      if ((impl_->expected_images_ &&
           impl_->progress_.images != *impl_->expected_images_) ||
          (impl_->expected_pixels_ &&
           impl_->progress_.pixels_seen != *impl_->expected_pixels_))
        throw InvalidArgument(
            "Segmentation EOF does not match expected input totals");
      // Validate one summary at a time so the final hook can build its own
      // result without retaining a second complete set of count copies.
      for (const auto& value : impl_->values_)
        static_cast<void>(summary(*value));
      check(stop, deadline);
      modes.restore();
      impl_->progress_.complete = true;
      notify();
      return result();
    }
    const auto& targets = batch->targets;
    if (!targets.defined() || targets.dim() != 3 || targets.size(0) <= 0 ||
        targets.size(1) <= 0 || targets.size(2) <= 0)
      throw InvalidArgument(
          "Segmentation evaluation targets must be nonempty N,H,W");
    const auto images = static_cast<std::uint64_t>(targets.size(0));
    const auto pixels = static_cast<std::uint64_t>(targets.numel());
    if (impl_->progress_.batches >= o.max_batches ||
        images > o.max_images - impl_->progress_.images ||
        pixels > o.max_pixels - impl_->progress_.pixels_seen ||
        pixels > o.limits.max_batch_pixels)
      throw ResourceExhausted("Segmentation evaluation input limit exceeded");
    if ((impl_->expected_images_ &&
         images > *impl_->expected_images_ - impl_->progress_.images) ||
        (impl_->expected_pixels_ &&
         pixels > *impl_->expected_pixels_ - impl_->progress_.pixels_seen))
      throw InvalidArgument("Segmentation batch exceeds expected input totals");
    if ((o.sources.empty() && !batch->source_indices.empty()) ||
        (!o.sources.empty() && batch->source_indices.size() != images) ||
        std::any_of(batch->source_indices.begin(), batch->source_indices.end(),
                    [&](auto i) { return i >= o.sources.size(); }))
      throw InvalidArgument("Segmentation source IDs must cover every image");
    const auto output = model.invoke(batch->invocation).value;
    check(stop, deadline);
    if (!output.isTensor())
      throw InvalidArgument("Segmentation model must return a tensor");
    const auto predictions = output.toTensor();
    auto candidate = make_metrics(o);
    for (std::size_t i = o.sources.empty() ? 0 : 1; i < candidate.size(); ++i)
      candidate[i]->merge(*impl_->values_[i]);
    if (o.sources.empty())
      candidate[0]->update(predictions, targets);
    else {
      const bool logits = o.metric.input == metrics::SegmentationInput::logits;
      if (!predictions.defined() || predictions.dim() != (logits ? 4 : 3) ||
          predictions.size(0) != targets.size(0))
        throw InvalidArgument("Segmentation model output batch shape differs");
      for (std::size_t i = 0; i < batch->source_indices.size(); ++i) {
        check(stop, deadline);
        candidate[batch->source_indices[i] + 1]->update(
            predictions.narrow(0, static_cast<std::int64_t>(i), 1),
            targets.narrow(0, static_cast<std::int64_t>(i), 1));
      }
      for (std::size_t i = 1; i < candidate.size(); ++i)
        candidate[0]->merge(*candidate[i]);
    }
    check(stop, deadline);
    impl_->values_.swap(candidate);
    ++impl_->progress_.batches;
    impl_->progress_.images += images;
    impl_->progress_.pixels_seen += pixels;
    candidate.clear();
    notify();
  }
}
}  // namespace jyppx::liuding::training
