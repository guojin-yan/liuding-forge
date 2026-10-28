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

using EvaluationMetrics = std::vector<std::unique_ptr<metrics::F1Score>>;

EvaluationMetrics make_metrics(const ClassificationEvaluationOptions& o) {
  EvaluationMetrics result;
  for (std::size_t i = 0; i <= o.sources.size(); ++i)
    result.push_back(std::make_unique<metrics::F1Score>(
        metrics::ClassificationMetricOptions{.metric_id = "evaluation.f1",
                                             .num_classes = o.num_classes,
                                             .zero_division = o.zero_division,
                                             .ignore_index = o.ignore_index},
        metrics::ClassificationMetricCheckpointOptions{
            .max_classes = o.max_classes,
            .max_counts_bytes = static_cast<std::uint64_t>(o.num_classes) *
                                o.num_classes * sizeof(std::int64_t)}));
  return result;
}

std::uint64_t validate_batch(const ClassificationEvaluationBatch& batch,
                             const ClassificationEvaluationOptions& o,
                             std::uint64_t batches, std::uint64_t rows) {
  const auto& targets = batch.targets;
  if (!targets.defined() || targets.dim() != 1 || targets.numel() <= 0 ||
      !c10::isIntegralType(targets.scalar_type(), false))
    throw InvalidArgument(
        "Classification targets must be a nonempty integral vector");
  const auto n = static_cast<std::uint64_t>(targets.numel());
  if (n > o.max_batch_samples || batches >= o.max_batches ||
      n > o.max_samples - rows)
    throw ResourceExhausted("Classification input exceeds batch/sample limits");
  if ((o.sources.empty() && !batch.source_indices.empty()) ||
      (!o.sources.empty() && batch.source_indices.size() != n) ||
      std::any_of(batch.source_indices.begin(), batch.source_indices.end(),
                  [&](auto index) { return index >= o.sources.size(); }))
    throw InvalidArgument("Classification source IDs must cover every row");
  return n;
}

void update_metrics(EvaluationMetrics& values, const c10::IValue& output,
                    const ClassificationEvaluationBatch& batch,
                    const ClassificationEvaluationOptions& o) {
  if (!output.isTensor())
    throw InvalidArgument("Classification model must return logits");
  const auto logits = output.toTensor();
  values[0]->update(logits, batch.targets);
  for (std::size_t i = 0; i < o.sources.size(); ++i) {
    std::vector<std::int64_t> rows;
    for (std::size_t j = 0; j < batch.source_indices.size(); ++j)
      if (batch.source_indices[j] == i)
        rows.push_back(static_cast<std::int64_t>(j));
    if (rows.empty()) continue;
    const auto indices = torch::tensor(rows, torch::kInt64).to(logits.device());
    values[i + 1]->update(logits.index_select(0, indices),
                          batch.targets.index_select(0, indices));
  }
}

ClassificationEvaluationResult make_result(
    const EvaluationMetrics& values, const ClassificationEvaluationOptions& o,
    std::uint64_t batches, std::uint64_t rows) {
  ClassificationEvaluationResult result;
  result.batches = batches;
  result.rows_seen = rows;
  result.overall = summarize(*values[0], o.zero_division);
  observations(result, result.overall, o, "overall", "all");
  for (std::size_t i = 0; i < o.sources.size(); ++i) {
    result.by_source.push_back(summarize(*values[i + 1], o.zero_division));
    observations(result, result.by_source.back(), o, "source", o.sources[i]);
  }
  return result;
}

}  // namespace

ClassificationEvaluationResult evaluate_classification(
    nn::ModelAdapter& model, const ClassificationEvaluationSource& source,
    const ClassificationEvaluationOptions& o, std::stop_token stop,
    TelemetryDeadline deadline) {
  validate(o);
  if (!source) throw InvalidArgument("Classification batch source is empty");
  check_stop(stop, deadline);
  auto values = make_metrics(o);
  ClassificationEvaluationResult result;
  ModeGuard modes(model);
  torch::NoGradGuard no_grad;
  model.state().eval();
  for (;;) {
    check_stop(stop, deadline);
    auto batch = source(stop);
    check_stop(stop, deadline);
    if (!batch) break;
    const auto n = validate_batch(*batch, o, result.batches, result.rows_seen);
    const auto output = model.invoke(batch->invocation).value;
    check_stop(stop, deadline);
    update_metrics(values, output, *batch, o);
    ++result.batches;
    result.rows_seen += n;
  }
  result = make_result(values, o, result.batches, result.rows_seen);
  check_stop(stop, deadline);
  modes.restore();
  return result;
}

namespace {
using EvaluationBytes = std::vector<std::byte>;
constexpr std::uint64_t kSessionTopologyBytes = 64ULL << 10U;
constexpr std::uint64_t kProgressBytes = 32;
constexpr std::uint64_t kProgressMagic = 0x315353414C43454AULL;

void append_integer(EvaluationBytes& bytes, std::uint64_t value) {
  for (unsigned i = 0; i < 8; ++i)
    bytes.push_back(static_cast<std::byte>((value >> (8U * i)) & 0xffU));
}
std::uint64_t read_integer(std::span<const std::byte> bytes,
                           std::size_t offset) {
  if (offset > bytes.size() || bytes.size() - offset < 8)
    throw SerializationError("Truncated classification evaluation progress");
  std::uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i)
    value |= static_cast<std::uint64_t>(bytes[offset + i]) << (8U * i);
  return value;
}
void append_text(EvaluationBytes& bytes, std::string_view value) {
  if (bytes.size() > kSessionTopologyBytes - 8 ||
      value.size() > kSessionTopologyBytes - 8 - bytes.size())
    throw ResourceExhausted(
        "Classification evaluation topology exceeds 64 KiB");
  append_integer(bytes, value.size());
  for (const auto c : value) bytes.push_back(static_cast<std::byte>(c));
}
void validate_text(std::string_view value) {
  if (value.empty() || value.size() > 1024 ||
      !std::all_of(value.begin(), value.end(),
                   [](unsigned char c) { return c >= 32 && c <= 126; }))
    throw InvalidArgument(
        "Evaluation identity requires bounded printable ASCII");
}
std::string prefix(std::size_t index) {
  return "slice." + std::to_string(index) + ".";
}
class SliceWriter final : public StateWriter {
 public:
  SliceWriter(StateWriter& writer, std::size_t index)
      : writer_(writer), prefix_(prefix(index)) {}
  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    writer_.write_blob(prefix_ + std::string(name), data);
  }

 private:
  StateWriter& writer_;
  std::string prefix_;
};
class SliceReader final : public StateReader {
 public:
  SliceReader(const StateReader& reader, std::size_t index,
              StateManifest manifest)
      : reader_(reader),
        prefix_(prefix(index)),
        manifest_(std::move(manifest)) {}
  const StateManifest& manifest() const noexcept override { return manifest_; }
  bool contains(std::string_view name) const override {
    return reader_.contains(prefix_ + std::string(name));
  }
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t maximum) const override {
    return reader_.read_blob(prefix_ + std::string(name), maximum);
  }

 private:
  const StateReader& reader_;
  std::string prefix_;
  StateManifest manifest_;
};
}  // namespace

class ClassificationEvaluationSession::Impl {
 public:
  Impl(ClassificationEvaluationIdentity identity,
       ClassificationEvaluationOptions options,
       std::optional<std::uint64_t> expected_rows)
      : identity_(std::move(identity)),
        options_(std::move(options)),
        expected_rows_(expected_rows) {
    validate(options_);
    validate_text(identity_.evaluation_id);
    validate_text(identity_.build_fingerprint);
    validate_artifact_digest(identity_.dataset_digest);
    validate_artifact_digest(identity_.model_digest);
    validate_artifact_digest(identity_.config_digest);
    if (expected_rows_ && *expected_rows_ > options_.max_samples)
      throw InvalidArgument("Expected rows exceed evaluation sample bound");
    // Validate the registry ID before allocating a confusion matrix.
    validate_state_manifest({.participant_id = identity_.participant_id,
                             .fields = {{.name = "topology", .max_bytes = 1}},
                             .degradation_reasons = {}});
    append_text(topology_, "jyppx.classification-evaluation/v1");
    append_text(topology_, identity_.participant_id);
    append_text(topology_, identity_.evaluation_id);
    append_text(topology_, identity_.dataset_digest.hex);
    append_text(topology_, identity_.model_digest.hex);
    append_text(topology_, identity_.config_digest.hex);
    append_text(topology_, identity_.build_fingerprint);
    for (auto value :
         {static_cast<std::uint64_t>(options_.num_classes),
          static_cast<std::uint64_t>(options_.zero_division),
          static_cast<std::uint64_t>(options_.ignore_index.has_value()),
          static_cast<std::uint64_t>(options_.ignore_index.value_or(0)),
          options_.max_batches, options_.max_samples,
          static_cast<std::uint64_t>(options_.max_batch_samples),
          static_cast<std::uint64_t>(options_.max_classes),
          static_cast<std::uint64_t>(options_.max_sources),
          static_cast<std::uint64_t>(options_.max_source_bytes),
          options_.max_confusion_bytes,
          static_cast<std::uint64_t>(expected_rows_.has_value()),
          expected_rows_.value_or(0),
          static_cast<std::uint64_t>(options_.sources.size())})
      append_integer(topology_, value);
    if (topology_.size() > kSessionTopologyBytes)
      throw ResourceExhausted(
          "Classification evaluation topology exceeds 64 KiB");
    for (const auto& name : options_.sources) append_text(topology_, name);
    values_ = make_metrics(options_);
  }
  ClassificationEvaluationIdentity identity_;
  ClassificationEvaluationOptions options_;
  std::optional<std::uint64_t> expected_rows_;
  EvaluationBytes topology_;
  EvaluationMetrics values_;
  ClassificationEvaluationProgress progress_;
  bool running_ = false;
  bool checkpoint_boundary_ = false;
};

ClassificationEvaluationSession::ClassificationEvaluationSession(
    ClassificationEvaluationIdentity identity,
    ClassificationEvaluationOptions options,
    std::optional<std::uint64_t> expected_rows)
    : impl_(std::make_unique<Impl>(std::move(identity), std::move(options),
                                   expected_rows)) {}
ClassificationEvaluationSession::~ClassificationEvaluationSession() = default;

ClassificationEvaluationProgress ClassificationEvaluationSession::progress()
    const noexcept {
  return impl_->progress_;
}
ClassificationEvaluationResult ClassificationEvaluationSession::result() const {
  if (!impl_->progress_.complete)
    throw NotReady("Classification evaluation has not reached valid EOF");
  return make_result(impl_->values_, impl_->options_, impl_->progress_.batches,
                     impl_->progress_.rows_seen);
}
std::string ClassificationEvaluationSession::participant_id() const {
  return impl_->identity_.participant_id;
}
StateManifest ClassificationEvaluationSession::describe_state() const {
  StateManifest manifest{
      .participant_id = participant_id(),
      .fields = {{.name = "topology", .max_bytes = impl_->topology_.size()},
                 {.name = "progress", .max_bytes = kProgressBytes}},
      .degradation_reasons = {}};
  for (std::size_t i = 0; i < impl_->values_.size(); ++i)
    for (auto field : impl_->values_[i]->describe_state().fields) {
      field.name = prefix(i) + field.name;
      manifest.fields.push_back(std::move(field));
    }
  return manifest;
}
void ClassificationEvaluationSession::save_state(StateWriter& writer) const {
  if (impl_->running_ && !impl_->checkpoint_boundary_)
    throw NotReady("Evaluation checkpoint requires a committed batch boundary");
  EvaluationBytes control;
  append_integer(control, kProgressMagic);
  append_integer(control, impl_->progress_.batches);
  append_integer(control, impl_->progress_.rows_seen);
  append_integer(control, impl_->progress_.complete ? 1 : 0);
  writer.write_blob("topology", impl_->topology_);
  writer.write_blob("progress", control);
  for (std::size_t i = 0; i < impl_->values_.size(); ++i) {
    SliceWriter slice(writer, i);
    impl_->values_[i]->save_state(slice);
  }
}
void ClassificationEvaluationSession::load_state(const StateReader& reader,
                                                 RestorePolicy policy) {
  if (impl_->running_) throw NotReady("Cannot restore a running evaluation");
  const auto expected = describe_state();
  const auto& stored = reader.manifest();
  validate_state_manifest(stored);
  if (stored.participant_id != expected.participant_id ||
      stored.schema_version != 1 ||
      stored.restore_level != RestoreLevel::exact ||
      !restore_policy_accepts(policy, stored.restore_level) ||
      !state_fields_equal(stored.fields, expected.fields))
    throw CompatibilityError(
        "Classification evaluation state contract differs");
  const auto topology = reader.read_blob("topology", impl_->topology_.size());
  if (topology.size() != impl_->topology_.size() ||
      !std::equal(topology.begin(), topology.end(), impl_->topology_.begin()))
    throw CompatibilityError(
        "Classification evaluation input/config identity differs");
  const auto control = reader.read_blob("progress", kProgressBytes);
  if (control.size() != kProgressBytes ||
      read_integer(control, 0) != kProgressMagic ||
      read_integer(control, 24) > 1)
    throw SerializationError(
        "Invalid classification evaluation progress header");
  ClassificationEvaluationProgress next{
      .batches = read_integer(control, 8),
      .rows_seen = read_integer(control, 16),
      .complete = read_integer(control, 24) != 0};
  const auto& o = impl_->options_;
  if (next.batches > o.max_batches || next.rows_seen > o.max_samples ||
      (next.batches == 0) != (next.rows_seen == 0) ||
      next.batches > next.rows_seen ||
      (next.rows_seen &&
       (next.rows_seen - 1) / o.max_batch_samples >= next.batches) ||
      (impl_->expected_rows_ &&
       (next.rows_seen > *impl_->expected_rows_ ||
        (next.complete && next.rows_seen != *impl_->expected_rows_))))
    throw SerializationError("Inconsistent classification evaluation counters");
  auto candidate = make_metrics(o);
  for (std::size_t i = 0; i < candidate.size(); ++i) {
    SliceReader slice(reader, i, candidate[i]->describe_state());
    candidate[i]->load_state(slice, policy);
    if (static_cast<std::uint64_t>(candidate[i]->total_observations()) >
        next.rows_seen)
      throw SerializationError("Classification counts exceed committed rows");
  }
  if (!o.ignore_index &&
      static_cast<std::uint64_t>(candidate[0]->total_observations()) !=
          next.rows_seen)
    throw SerializationError(
        "Classification counts do not cover committed rows");
  if (!o.sources.empty()) {
    const auto& overall = candidate[0]->confusion_matrix();
    for (std::size_t row = 0; row < o.num_classes; ++row)
      for (std::size_t col = 0; col < o.num_classes; ++col) {
        auto remaining = overall[row][col];
        for (std::size_t i = 1; i < candidate.size(); ++i) {
          const auto value = candidate[i]->confusion_matrix()[row][col];
          if (value > remaining)
            throw SerializationError(
                "Classification source counts exceed overall");
          remaining -= value;
        }
        if (remaining != 0)
          throw SerializationError(
              "Classification source counts do not cover overall");
      }
  }
  if (next.complete)
    for (const auto& value : candidate) {
      // One temporary summary at a time keeps the declared matrix budget.
      static_cast<void>(summarize(*value, o.zero_division));
    }
  impl_->values_.swap(candidate);
  impl_->progress_ = next;
}

ClassificationEvaluationResult ClassificationEvaluationSession::run(
    nn::ModelAdapter& model,
    const IndexedClassificationEvaluationSource& source,
    const std::function<void(const ClassificationEvaluationSession&)>&
        committed,
    std::stop_token stop, TelemetryDeadline deadline) {
  if (impl_->running_) throw NotReady("Evaluation is already running");
  if (!source) throw InvalidArgument("Classification indexed source is empty");
  check_stop(stop, deadline);
  struct RunningGuard {
    Impl& impl;
    explicit RunningGuard(Impl& value) : impl(value) { impl.running_ = true; }
    ~RunningGuard() {
      impl.running_ = false;
      impl.checkpoint_boundary_ = false;
    }
  } running(*impl_);
  const auto notify = [&] {
    if (!committed) return;
    impl_->checkpoint_boundary_ = true;
    try {
      committed(*this);
    } catch (...) {
      impl_->checkpoint_boundary_ = false;
      throw;
    }
    impl_->checkpoint_boundary_ = false;
    check_stop(stop, deadline);
  };
  if (impl_->progress_.complete) {
    notify();
    return result();
  }
  ModeGuard modes(model);
  torch::NoGradGuard no_grad;
  model.state().eval();
  for (;;) {
    check_stop(stop, deadline);
    auto batch = source(impl_->progress_.batches, stop);
    check_stop(stop, deadline);
    if (!batch) {
      if (impl_->expected_rows_ &&
          impl_->progress_.rows_seen != *impl_->expected_rows_)
        throw InvalidArgument(
            "Classification EOF does not match expected rows");
      // Score policies must succeed before EOF becomes a completed result.
      auto final =
          make_result(impl_->values_, impl_->options_, impl_->progress_.batches,
                      impl_->progress_.rows_seen);
      check_stop(stop, deadline);
      modes.restore();
      impl_->progress_.complete = true;
      notify();
      return final;
    }
    const auto n =
        validate_batch(*batch, impl_->options_, impl_->progress_.batches,
                       impl_->progress_.rows_seen);
    if (impl_->expected_rows_ &&
        n > *impl_->expected_rows_ - impl_->progress_.rows_seen)
      throw InvalidArgument("Classification batch exceeds expected rows");
    const auto output = model.invoke(batch->invocation).value;
    check_stop(stop, deadline);
    auto candidate = make_metrics(impl_->options_);
    for (std::size_t i = 0; i < candidate.size(); ++i)
      candidate[i]->merge(*impl_->values_[i]);
    update_metrics(candidate, output, *batch, impl_->options_);
    check_stop(stop, deadline);
    impl_->values_.swap(candidate);
    ++impl_->progress_.batches;
    impl_->progress_.rows_seen += n;
    // Release old matrices before hooks can allocate a checkpoint/result.
    candidate.clear();
    notify();
  }
}

}  // namespace jyppx::liuding::training
