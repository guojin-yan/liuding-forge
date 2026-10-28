// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/training/metrics/segmentation.hpp>

#include <algorithm>
#include <limits>
#include <span>
#include <utility>

namespace jyppx::liuding::training::metrics {
namespace {
constexpr auto kMax = std::numeric_limits<std::int64_t>::max();
constexpr char kTopologyMedia[] =
    "application/vnd.jyppx.liuding.metric.semantic-topology-v1";
constexpr char kCountsMedia[] =
    "application/vnd.jyppx.liuding.metric.semantic-int64-le-v1";

bool integral(torch::ScalarType type) {
  return type == torch::kUInt8 || type == torch::kInt8 ||
         type == torch::kInt16 || type == torch::kInt32 ||
         type == torch::kInt64;
}

std::string topology(const SegmentationMetricDescriptor& d) {
  const auto& o = d.options;
  return d.type_id + "\nid=" + o.metric_id +
         "\nclasses=" + std::to_string(o.num_classes) +
         "\ninput=" + std::string(segmentation_input_name(o.input)) +
         "\naverage=" + std::string(classification_average_name(o.average)) +
         "\nabsent=" + std::string(absent_class_policy_name(o.absent)) +
         "\nignore=" +
         (o.ignore_index ? std::to_string(*o.ignore_index) : "none") +
         "\nbackground=" + std::to_string(o.background_index) +
         "\ninclude_background=" + (o.include_background ? "true" : "false") +
         "\nties=lowest_class_index\n";
}

void append(std::vector<std::byte>& bytes, std::int64_t value) {
  const auto encoded = static_cast<std::uint64_t>(value);
  for (unsigned i = 0; i < 8; ++i)
    bytes.push_back(static_cast<std::byte>((encoded >> (8U * i)) & 255U));
}

std::int64_t read(std::span<const std::byte> bytes, std::size_t& offset) {
  std::uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i)
    value |= static_cast<std::uint64_t>(
                 std::to_integer<unsigned char>(bytes[offset++]))
             << (8U * i);
  if (value > static_cast<std::uint64_t>(kMax))
    throw SerializationError("Segmentation counts must be nonnegative int64");
  return static_cast<std::int64_t>(value);
}

long double denominator(const SegmentationClassCounts& c,
                        SegmentationScore score) {
  const auto supports = static_cast<long double>(c.target) + c.predicted;
  return score == SegmentationScore::iou ? supports - c.intersection : supports;
}

double ratio(long double intersection, long double den,
             SegmentationScore score) {
  return static_cast<double>(
      (score == SegmentationScore::iou ? intersection : 2 * intersection) /
      den);
}
}  // namespace

std::string_view segmentation_score_name(SegmentationScore score) noexcept {
  switch (score) {
    case SegmentationScore::iou:
      return "iou";
    case SegmentationScore::dice:
      return "dice";
  }
  return "unknown";
}
std::string_view segmentation_input_name(SegmentationInput input) noexcept {
  switch (input) {
    case SegmentationInput::logits:
      return "logits";
    case SegmentationInput::class_indices:
      return "class_indices";
  }
  return "unknown";
}
std::string_view absent_class_policy_name(AbsentClassPolicy policy) noexcept {
  switch (policy) {
    case AbsentClassPolicy::exclude:
      return "exclude";
    case AbsentClassPolicy::zero:
      return "zero";
    case AbsentClassPolicy::one:
      return "one";
    case AbsentClassPolicy::error:
      return "error";
  }
  return "unknown";
}

SemanticSegmentationMetric::SemanticSegmentationMetric(
    SegmentationMetricOptions options, SegmentationMetricLimits limits)
    : Metric("semantic_" + std::string(segmentation_score_name(options.score))),
      limits_(limits) {
  if (options.metric_id.empty()) options.metric_id = "metric." + name();
  if (options.metric_id.size() > 256 ||
      !std::all_of(options.metric_id.begin(), options.metric_id.end(),
                   [](unsigned char c) {
                     return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                            (c >= '0' && c <= '9') || c == '.' || c == '_' ||
                            c == '-';
                   }) ||
      segmentation_score_name(options.score) == "unknown" ||
      segmentation_input_name(options.input) == "unknown" ||
      classification_average_name(options.average) == "unknown" ||
      absent_class_policy_name(options.absent) == "unknown" ||
      !options.num_classes || options.background_index >= options.num_classes ||
      !limits.max_classes || !limits.max_state_bytes ||
      !limits.max_batch_pixels || !limits.chunk_pixels ||
      limits.chunk_pixels > static_cast<std::uint64_t>(kMax))
    throw InvalidArgument("Invalid semantic segmentation options or limits");
  if (options.num_classes > limits.max_classes || limits.max_state_bytes < 8 ||
      options.num_classes > (limits.max_state_bytes - 8) / 24 ||
      options.num_classes > static_cast<std::uint64_t>(kMax))
    throw ResourceExhausted("Semantic segmentation class/state limit exceeded");
  auto type_id = "jyppx.liuding.metric.semantic_" +
                 std::string(segmentation_score_name(options.score)) + "/v1";
  descriptor_ = {.schema_version = 1,
                 .type_id = std::move(type_id),
                 .options = std::move(options)};
  counts_.resize(descriptor_.options.num_classes);
}

void SemanticSegmentationMetric::update(const Tensor& predictions,
                                        const Tensor& targets) {
  const auto& o = descriptor_.options;
  const bool logits = o.input == SegmentationInput::logits;
  if (!predictions.defined() || !targets.defined() || targets.dim() != 3 ||
      predictions.dim() != (logits ? 4 : 3) ||
      predictions.layout() != torch::kStrided ||
      targets.layout() != torch::kStrided || !integral(targets.scalar_type()) ||
      (logits ? !predictions.is_floating_point()
              : !integral(predictions.scalar_type())) ||
      predictions.device() != targets.device() ||
      (!targets.device().is_cpu() && !targets.device().is_cuda()))
    throw InvalidArgument(
        "Semantic segmentation requires strided same-device image batches");
  if (predictions.size(0) != targets.size(0) ||
      (logits &&
       predictions.size(1) != static_cast<std::int64_t>(o.num_classes)) ||
      predictions.size(logits ? 2 : 1) != targets.size(1) ||
      predictions.size(logits ? 3 : 2) != targets.size(2))
    throw InvalidArgument(
        "Semantic segmentation N,C,H,W / N,H,W shapes differ");
  if (static_cast<std::uint64_t>(targets.numel()) > limits_.max_batch_pixels)
    throw ResourceExhausted("Semantic segmentation batch pixel limit exceeded");
  if (!targets.numel()) return;
  torch::NoGradGuard no_grad;
  std::vector<SegmentationClassCounts> delta(counts_.size());
  std::int64_t accepted = 0;
  const auto width = targets.size(2), height = targets.size(1);
  const auto chunk = static_cast<std::int64_t>(limits_.chunk_pixels);
  const auto row_step = std::max(std::int64_t{1}, chunk / width);
  for (std::int64_t n = 0; n < targets.size(0); ++n) {
    for (std::int64_t y = 0; y < height;) {
      const auto rows = std::min(row_step, height - y);
      for (std::int64_t x = 0; x < width;) {
        const auto columns = std::min(chunk, width - x);
        auto predicted = predictions.narrow(0, n, 1)
                             .narrow(logits ? 2 : 1, y, rows)
                             .narrow(logits ? 3 : 2, x, columns);
        if (logits) {
          if (!torch::isfinite(predicted).all().item<bool>())
            throw NumericalError("Semantic segmentation logits must be finite");
          predicted = predicted.argmax(1);
        }
        predicted =
            predicted.to(torch::kCPU, torch::kInt64).contiguous().view({-1});
        const auto expected = targets.narrow(0, n, 1)
                                  .narrow(1, y, rows)
                                  .narrow(2, x, columns)
                                  .to(torch::kCPU, torch::kInt64)
                                  .contiguous()
                                  .view({-1});
        const auto* p = predicted.const_data_ptr<std::int64_t>();
        const auto* t = expected.const_data_ptr<std::int64_t>();
        for (std::int64_t i = 0; i < expected.numel(); ++i) {
          if (o.ignore_index && t[i] == *o.ignore_index) continue;
          if (t[i] < 0 || p[i] < 0 ||
              static_cast<std::uint64_t>(t[i]) >= o.num_classes ||
              static_cast<std::uint64_t>(p[i]) >= o.num_classes)
            throw InvalidArgument(
                "Semantic segmentation class ID out of range");
          ++delta[static_cast<std::size_t>(t[i])].target;
          ++delta[static_cast<std::size_t>(p[i])].predicted;
          if (t[i] == p[i])
            ++delta[static_cast<std::size_t>(t[i])].intersection;
          ++accepted;
        }
        x += columns;
      }
      y += rows;
    }
  }
  if (accepted > kMax - total_)
    throw ResourceExhausted("Semantic segmentation int64 pixel count overflow");
  for (std::size_t i = 0; i < counts_.size(); ++i) {
    counts_[i].intersection += delta[i].intersection;
    counts_[i].target += delta[i].target;
    counts_[i].predicted += delta[i].predicted;
  }
  total_ += accepted;
}

std::vector<std::optional<double>>
SemanticSegmentationMetric::per_class_scores() const {
  if (!total_)
    throw NotReady("Semantic segmentation has no non-ignored pixels");
  const auto& o = descriptor_.options;
  std::vector<std::optional<double>> result(counts_.size());
  for (std::size_t i = 0; i < counts_.size(); ++i) {
    if (!o.include_background && i == o.background_index) continue;
    const auto den = denominator(counts_[i], o.score);
    if (den)
      result[i] = ratio(counts_[i].intersection, den, o.score);
    else if (o.absent == AbsentClassPolicy::zero)
      result[i] = 0;
    else if (o.absent == AbsentClassPolicy::one)
      result[i] = 1;
    else if (o.absent == AbsentClassPolicy::error)
      throw NumericalError("Semantic segmentation includes an absent class");
  }
  return result;
}

double SemanticSegmentationMetric::compute() {
  const auto& o = descriptor_.options;
  if (o.average == ClassificationAverage::none)
    throw InvalidArgument("Use per_class_scores for segmentation average=none");
  const auto scores = per_class_scores();
  long double numerator = 0, den = 0;
  for (std::size_t i = 0; i < counts_.size(); ++i) {
    if (!scores[i]) continue;
    if (o.average == ClassificationAverage::micro) {
      numerator += counts_[i].intersection;
      den += denominator(counts_[i], o.score);
    } else {
      const auto weight = o.average == ClassificationAverage::weighted
                              ? static_cast<long double>(counts_[i].target)
                              : 1.L;
      numerator += *scores[i] * weight;
      den += weight;
    }
  }
  if (!den)
    throw NotReady("Semantic segmentation has no included scoring denominator");
  return o.average == ClassificationAverage::micro
             ? ratio(numerator, den, o.score)
             : static_cast<double>(numerator / den);
}

void SemanticSegmentationMetric::reset() noexcept {
  std::fill(counts_.begin(), counts_.end(), SegmentationClassCounts{});
  total_ = 0;
}
void SemanticSegmentationMetric::merge(
    const SemanticSegmentationMetric& other) {
  if (topology(descriptor_) != topology(other.descriptor_))
    throw CompatibilityError("Semantic segmentation descriptors differ");
  if (other.total_ > kMax - total_)
    throw ResourceExhausted("Semantic segmentation merge int64 count overflow");
  for (std::size_t i = 0; i < counts_.size(); ++i) {
    counts_[i].intersection += other.counts_[i].intersection;
    counts_[i].target += other.counts_[i].target;
    counts_[i].predicted += other.counts_[i].predicted;
  }
  total_ += other.total_;
}
const std::vector<SegmentationClassCounts>&
SemanticSegmentationMetric::class_counts() const noexcept {
  return counts_;
}
std::int64_t SemanticSegmentationMetric::total_pixels() const noexcept {
  return total_;
}
const SegmentationMetricDescriptor& SemanticSegmentationMetric::descriptor()
    const noexcept {
  return descriptor_;
}
std::string SemanticSegmentationMetric::participant_id() const {
  return descriptor_.options.metric_id;
}

StateManifest SemanticSegmentationMetric::describe_state() const {
  StateManifest manifest{.participant_id = participant_id(),
                         .schema_version = 1,
                         .restore_level = RestoreLevel::exact,
                         .fields = {{.name = "topology",
                                     .media_type = kTopologyMedia,
                                     .max_bytes = topology(descriptor_).size(),
                                     .required = true},
                                    {.name = "counts",
                                     .media_type = kCountsMedia,
                                     .max_bytes = 8 + 24 * counts_.size(),
                                     .required = true}},
                         .degradation_reasons = {}};
  validate_state_manifest(manifest);
  return manifest;
}
void SemanticSegmentationMetric::save_state(StateWriter& writer) const {
  const auto text = topology(descriptor_);
  writer.write_blob("topology", std::as_bytes(std::span(text)));
  std::vector<std::byte> bytes;
  bytes.reserve(8 + 24 * counts_.size());
  append(bytes, total_);
  for (const auto& c : counts_) {
    append(bytes, c.intersection);
    append(bytes, c.target);
    append(bytes, c.predicted);
  }
  writer.write_blob("counts", bytes);
}
void SemanticSegmentationMetric::load_state(const StateReader& reader,
                                            RestorePolicy policy) {
  const auto expected = describe_state();
  const auto& stored = reader.manifest();
  validate_state_manifest(stored);
  if (!restore_policy_accepts(policy, stored.restore_level) ||
      stored.restore_level != RestoreLevel::exact ||
      stored.schema_version != 1 || stored.participant_id != participant_id() ||
      !stored.degradation_reasons.empty() ||
      !state_fields_equal(stored.fields, expected.fields))
    throw CompatibilityError(
        "Semantic segmentation checkpoint manifest differs");
  const auto text = topology(descriptor_);
  const auto encoded_topology = reader.read_blob("topology", text.size());
  const auto topology_bytes = std::as_bytes(std::span(text));
  if (encoded_topology.size() != topology_bytes.size() ||
      !std::equal(encoded_topology.begin(), encoded_topology.end(),
                  topology_bytes.begin()))
    throw CompatibilityError(
        "Semantic segmentation checkpoint topology differs");
  const auto bytes = reader.read_blob("counts", limits_.max_state_bytes);
  if (bytes.size() != 8 + 24 * counts_.size())
    throw SerializationError(
        "Semantic segmentation checkpoint count size differs");
  std::size_t offset = 0;
  const auto total = read(bytes, offset);
  std::vector<SegmentationClassCounts> candidate(counts_.size());
  std::int64_t targets = 0, predicted = 0, intersection = 0;
  for (auto& c : candidate) {
    c = {.intersection = read(bytes, offset),
         .target = read(bytes, offset),
         .predicted = read(bytes, offset)};
    if (c.intersection > c.target || c.intersection > c.predicted ||
        c.target > total - targets || c.predicted > total - predicted ||
        c.intersection > total - intersection ||
        c.predicted - c.intersection > total - c.target)
      throw SerializationError(
          "Semantic segmentation counts violate pixel conservation");
    targets += c.target;
    predicted += c.predicted;
    intersection += c.intersection;
  }
  if (targets != total || predicted != total)
    throw SerializationError(
        "Semantic segmentation supports do not equal total pixels");
  const auto mistakes = total - intersection;
  for (const auto& c : candidate)
    if (c.target - c.intersection > mistakes - (c.predicted - c.intersection))
      throw SerializationError(
          "Semantic segmentation residual counts cannot match");
  counts_.swap(candidate);
  total_ = total;
}
}  // namespace jyppx::liuding::training::metrics
