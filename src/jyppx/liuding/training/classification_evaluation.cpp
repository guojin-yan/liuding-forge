// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/training/classification_evaluation.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <memory>
#include <set>
#include <utility>

namespace jyppx::liuding::training {
namespace {

void check_stop(std::stop_token stop, TelemetryDeadline deadline) {
  if (stop.stop_requested())
    throw Cancelled("Classification evaluation cancelled");
  if (deadline && std::chrono::steady_clock::now() >= *deadline)
    throw DeadlineExceeded("Classification evaluation deadline elapsed");
}

class ModeGuard final {
 public:
  explicit ModeGuard(nn::ModelAdapter& model) {
    for (const auto& module : model.state().module().modules())
      modes_.emplace_back(module, module->is_training());
  }
  void restore() {
    for (const auto& [module, training] : modes_) module->train(training);
    modes_.clear();
  }
  ~ModeGuard() {
    try {
      restore();
    } catch (...) { /* Preserve the active execution error. */
    }
  }

 private:
  std::vector<std::pair<std::shared_ptr<torch::nn::Module>, bool>> modes_;
};

void validate(const ClassificationEvaluationOptions& o) {
  if (!o.num_classes || o.num_classes > o.max_classes ||
      o.num_classes >
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
      !o.max_batches || !o.max_samples || !o.max_batch_samples ||
      o.sources.size() > o.max_sources || !o.max_confusion_bytes)
    throw InvalidArgument(
        "Invalid classification evaluation topology or limits");
  std::set<std::string> names;
  for (const auto& name : o.sources) {
    if (name.empty() || name.size() > o.max_source_bytes ||
        !names.insert(name).second ||
        !std::all_of(name.begin(), name.end(), [](unsigned char c) {
          return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                 (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
        }))
      throw InvalidArgument("Invalid or duplicate classification source name");
  }
  if (o.zero_division != metrics::ZeroDivisionPolicy::zero &&
      o.zero_division != metrics::ZeroDivisionPolicy::one &&
      o.zero_division != metrics::ZeroDivisionPolicy::error)
    throw InvalidArgument("Invalid classification zero-division policy");
  // Metrics + returned matrices + one update's candidate/delta matrices.
  // Division avoids overflow before any fixed-topology matrix is allocated.
  const auto copies =
      2ULL * (static_cast<std::uint64_t>(o.sources.size()) + 1) + 2;
  const auto cells = o.max_confusion_bytes / sizeof(std::int64_t) / copies;
  if (o.num_classes > cells / o.num_classes)
    throw ResourceExhausted("Classification confusion payload exceeds budget");
}

double divide(long double n, long double d,
              metrics::ZeroDivisionPolicy policy) {
  if (d != 0) return static_cast<double>(n / d);
  if (policy == metrics::ZeroDivisionPolicy::error)
    throw NumericalError("Classification score has an empty denominator");
  return policy == metrics::ZeroDivisionPolicy::one ? 1.0 : 0.0;
}

ClassificationEvaluationSummary summarize(const metrics::F1Score& metric,
                                          metrics::ZeroDivisionPolicy policy) {
  ClassificationEvaluationSummary result;
  result.samples = static_cast<std::uint64_t>(metric.total_observations());
  result.confusion = metric.confusion_matrix();
  if (!result.samples) return result;
  const auto count = metric.classes();
  result.classes.resize(count);
  long double correct = 0, p = 0, r = 0, f = 0;
  for (std::size_t i = 0; i < count; ++i) {
    auto& cls = result.classes[i];
    for (std::size_t j = 0; j < count; ++j) {
      cls.support += result.confusion[i][j];
      cls.predicted += result.confusion[j][i];
    }
    const auto tp = static_cast<long double>(result.confusion[i][i]);
    cls.precision = divide(tp, static_cast<long double>(cls.predicted), policy);
    cls.recall = divide(tp, static_cast<long double>(cls.support), policy);
    cls.f1 = divide(
        2 * tp, static_cast<long double>(cls.predicted) + cls.support, policy);
    correct += tp;
    p += cls.precision;
    r += cls.recall;
    f += cls.f1;
  }
  result.accuracy = static_cast<double>(correct / result.samples);
  result.macro_precision = static_cast<double>(p / count);
  result.macro_recall = static_cast<double>(r / count);
  result.macro_f1 = static_cast<double>(f / count);
  return result;
}

void observations(ClassificationEvaluationResult& result,
                  const ClassificationEvaluationSummary& summary,
                  const ClassificationEvaluationOptions& o,
                  const std::string& slice, const std::string& source) {
  const std::array<std::pair<const char*, std::optional<double>>, 4> values{
      {{"accuracy", summary.accuracy},
       {"macro_precision", summary.macro_precision},
       {"macro_recall", summary.macro_recall},
       {"macro_f1", summary.macro_f1}}};
  for (const auto& [name, value] : values) {
    TelemetryMetricPoint point;
    point.definition = {
        .metric_id = std::string("classification.") + name,
        .version = 1,
        .unit = TelemetryMetricUnit::ratio,
        .aggregation = TelemetryAggregation::mean,
        .direction = TelemetryMetricDirection::maximize,
        .parameters =
            {{"classes", std::to_string(o.num_classes)},
             {"ignore_index",
              o.ignore_index ? std::to_string(*o.ignore_index) : "none"},
             {"zero_division",
              std::string(metrics::zero_division_policy_name(o.zero_division))},
             {"ties", "lowest_class_index"},
             {"macro_classes", "all"}},
        .context_keys = {"slice", "source"}};
    point.step = {.kind = TelemetryStepKind::sample, .value = summary.samples};
    point.observation = {.status = value ? TelemetryValueStatus::finite
                                         : TelemetryValueStatus::missing,
                         .value = value,
                         .reason = value ? "" : "No non-ignored samples"};
    point.context = {{"slice", slice}, {"source", source}};
    point.sample_count = summary.samples;
    result.observations.push_back(std::move(point));
  }
}

}  // namespace

ClassificationEvaluationResult evaluate_classification(
    nn::ModelAdapter& model, const ClassificationEvaluationSource& source,
    const ClassificationEvaluationOptions& o, std::stop_token stop,
    TelemetryDeadline deadline) {
  validate(o);
  if (!source) throw InvalidArgument("Classification batch source is empty");
  check_stop(stop, deadline);
  std::vector<std::unique_ptr<metrics::F1Score>> metrics;
  for (std::size_t i = 0; i <= o.sources.size(); ++i)
    metrics.push_back(std::make_unique<metrics::F1Score>(
        metrics::ClassificationMetricOptions{.metric_id = "evaluation.f1",
                                             .num_classes = o.num_classes,
                                             .zero_division = o.zero_division,
                                             .ignore_index = o.ignore_index},
        metrics::ClassificationMetricCheckpointOptions{
            .max_classes = o.max_classes,
            .max_counts_bytes = o.max_confusion_bytes}));
  ClassificationEvaluationResult result;
  ModeGuard modes(model);
  torch::NoGradGuard no_grad;
  model.state().eval();
  for (;;) {
    check_stop(stop, deadline);
    auto batch = source(stop);
    check_stop(stop, deadline);
    if (!batch) break;
    const auto& targets = batch->targets;
    if (!targets.defined() || targets.dim() != 1 || targets.numel() <= 0 ||
        !c10::isIntegralType(targets.scalar_type(), false))
      throw InvalidArgument(
          "Classification targets must be a nonempty integral vector");
    const auto n = static_cast<std::uint64_t>(targets.numel());
    if (n > o.max_batch_samples || result.batches >= o.max_batches ||
        n > o.max_samples - result.rows_seen)
      throw ResourceExhausted(
          "Classification input exceeds batch/sample limits");
    if ((o.sources.empty() && !batch->source_indices.empty()) ||
        (!o.sources.empty() && batch->source_indices.size() != n) ||
        std::any_of(batch->source_indices.begin(), batch->source_indices.end(),
                    [&](auto index) { return index >= o.sources.size(); }))
      throw InvalidArgument("Classification source IDs must cover every row");
    const auto output = model.invoke(batch->invocation).value;
    check_stop(stop, deadline);
    if (!output.isTensor())
      throw InvalidArgument("Classification model must return logits");
    const auto logits = output.toTensor();
    metrics[0]->update(logits, targets);
    for (std::size_t i = 0; i < o.sources.size(); ++i) {
      std::vector<std::int64_t> rows;
      for (std::size_t j = 0; j < batch->source_indices.size(); ++j)
        if (batch->source_indices[j] == i)
          rows.push_back(static_cast<std::int64_t>(j));
      if (rows.empty()) continue;
      const auto indices =
          torch::tensor(rows, torch::kInt64).to(logits.device());
      metrics[i + 1]->update(logits.index_select(0, indices),
                             targets.index_select(0, indices));
    }
    ++result.batches;
    result.rows_seen += n;
  }
  result.overall = summarize(*metrics[0], o.zero_division);
  observations(result, result.overall, o, "overall", "all");
  for (std::size_t i = 0; i < o.sources.size(); ++i) {
    result.by_source.push_back(summarize(*metrics[i + 1], o.zero_division));
    observations(result, result.by_source.back(), o, "source", o.sources[i]);
  }
  check_stop(stop, deadline);
  modes.restore();
  return result;
}

}  // namespace jyppx::liuding::training
