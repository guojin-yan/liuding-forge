// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/training/metrics/ranking.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <span>
#include <string>
#include <type_traits>
#include <utility>

namespace jyppx::training::metrics {
namespace {

constexpr char kComponent[] = "training.metrics.ranking";
constexpr char kAuRocTypeId[] = "jyppx.metric.binary-auroc/v1";
constexpr char kAveragePrecisionTypeId[] =
    "jyppx.metric.binary-average-precision/v1";
constexpr char kMulticlassAuRocTypeId[] = "jyppx.metric.multiclass-auroc/v1";
constexpr char kMulticlassAveragePrecisionTypeId[] =
    "jyppx.metric.multiclass-average-precision/v1";
constexpr char kMultilabelAuRocTypeId[] = "jyppx.metric.multilabel-auroc/v1";
constexpr char kMultilabelAveragePrecisionTypeId[] =
    "jyppx.metric.multilabel-average-precision/v1";
constexpr char kTopologyField[] = "topology";
constexpr char kObservationsField[] = "observations";
constexpr char kTopologyMediaType[] =
    "application/vnd.jyppx.metric.binary-ranking-topology-v1";
constexpr char kObservationsMediaType[] =
    "application/vnd.jyppx.metric.binary-ranking-observations-v1";
constexpr char kClasswiseTopologyMediaType[] =
    "application/vnd.jyppx.metric.classwise-ranking-topology-v1";
constexpr char kClasswiseObservationsMediaType[] =
    "application/vnd.jyppx.metric.classwise-ranking-observations-v1";
constexpr std::uint64_t kObservationHeaderBytes = 16;
constexpr std::uint64_t kObservationBytes = 9;
constexpr std::array<std::byte, 8> kTopologyMagic{
    std::byte{'J'}, std::byte{'Y'}, std::byte{'P'}, std::byte{'R'},
    std::byte{'A'}, std::byte{'N'}, std::byte{'K'}, std::byte{'1'}};
constexpr std::array<std::byte, 8> kObservationsMagic{
    std::byte{'J'}, std::byte{'Y'}, std::byte{'P'}, std::byte{'O'},
    std::byte{'B'}, std::byte{'S'}, std::byte{'0'}, std::byte{'1'}};
constexpr std::array<std::byte, 8> kClasswiseTopologyMagic{
    std::byte{'J'}, std::byte{'Y'}, std::byte{'P'}, std::byte{'C'},
    std::byte{'R'}, std::byte{'A'}, std::byte{'N'}, std::byte{'1'}};
constexpr std::array<std::byte, 8> kClasswiseObservationsMagic{
    std::byte{'J'}, std::byte{'Y'}, std::byte{'P'}, std::byte{'C'},
    std::byte{'O'}, std::byte{'B'}, std::byte{'S'}, std::byte{'1'}};

static_assert(sizeof(double) == sizeof(std::uint64_t));
static_assert(std::numeric_limits<double>::is_iec559);

[[noreturn]] void throw_invalid(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
  throw InvalidArgument("Invalid ranking metric contract",
                        ErrorDetails{.component = kComponent,
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

[[noreturn]] void throw_incompatible(std::string operation,
                                     std::string expected, std::string actual,
                                     std::string hint) {
  throw CompatibilityError("Ranking metric state is incompatible",
                           ErrorDetails{.component = kComponent,
                                        .operation = std::move(operation),
                                        .expected = std::move(expected),
                                        .actual = std::move(actual),
                                        .hint = std::move(hint)});
}

[[noreturn]] void throw_serialization(std::string operation,
                                      std::string expected, std::string actual,
                                      std::string hint) {
  throw SerializationError("Ranking metric state is malformed",
                           ErrorDetails{.component = kComponent,
                                        .operation = std::move(operation),
                                        .expected = std::move(expected),
                                        .actual = std::move(actual),
                                        .hint = std::move(hint)});
}

[[noreturn]] void throw_exhausted(std::string operation, std::string expected,
                                  std::string actual, std::string hint) {
  throw ResourceExhausted("Ranking metric resource limit exceeded",
                          ErrorDetails{.component = kComponent,
                                       .operation = std::move(operation),
                                       .expected = std::move(expected),
                                       .actual = std::move(actual),
                                       .hint = std::move(hint)});
}

bool valid_identifier(std::string_view value) {
  return !value.empty() &&
         std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return (character >= 'a' && character <= 'z') ||
                  (character >= 'A' && character <= 'Z') ||
                  (character >= '0' && character <= '9') || character == '.' ||
                  character == '_' || character == '-';
         });
}

bool valid_type_identifier(std::string_view value) {
  return !value.empty() && value.front() != '/' && value.back() != '/' &&
         std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return (character >= 'a' && character <= 'z') ||
                  (character >= 'A' && character <= 'Z') ||
                  (character >= '0' && character <= '9') || character == '.' ||
                  character == '_' || character == '-' || character == '/';
         });
}

bool integral_labels(torch::ScalarType type) noexcept {
  switch (type) {
    case torch::kUInt8:
    case torch::kInt8:
    case torch::kInt16:
    case torch::kInt32:
    case torch::kInt64:
      return true;
    default:
      return false;
  }
}

std::uint64_t checked_observation_bytes(std::size_t observations) {
  if (observations >
      (std::numeric_limits<std::uint64_t>::max() - kObservationHeaderBytes) /
          kObservationBytes) {
    throw_exhausted(
        "options.max_observations", "a checkpoint size fitting uint64",
        std::to_string(observations), "Reduce the exact observation limit.");
  }
  return kObservationHeaderBytes +
         static_cast<std::uint64_t>(observations) * kObservationBytes;
}

std::uint64_t checked_classwise_observation_bytes(std::size_t classes,
                                                  std::size_t observations) {
  constexpr auto kHeaderBytes =
      static_cast<std::uint64_t>(kClasswiseObservationsMagic.size()) +
      sizeof(std::uint64_t);
  if (classes > (std::numeric_limits<std::uint64_t>::max() - kHeaderBytes) /
                    sizeof(std::uint64_t)) {
    throw_exhausted("options.max_classes", "a checkpoint size fitting uint64",
                    std::to_string(classes), "Reduce the fixed class limit.");
  }
  const auto class_bytes =
      static_cast<std::uint64_t>(classes) * sizeof(std::uint64_t);
  if (observations >
      (std::numeric_limits<std::uint64_t>::max() - kHeaderBytes - class_bytes) /
          kObservationBytes) {
    throw_exhausted(
        "options.max_observations", "a checkpoint size fitting uint64",
        std::to_string(observations), "Reduce the exact observation limit.");
  }
  return kHeaderBytes + class_bytes +
         static_cast<std::uint64_t>(observations) * kObservationBytes;
}

template <typename Unsigned>
void append_unsigned(std::vector<std::byte>& output, Unsigned value) {
  static_assert(std::is_unsigned_v<Unsigned>);
  for (std::size_t index = 0; index < sizeof(Unsigned); ++index) {
    output.push_back(
        static_cast<std::byte>((value >> (index * 8U)) & Unsigned{0xff}));
  }
}

void append_string(std::vector<std::byte>& output, std::string_view value) {
  append_unsigned(output, static_cast<std::uint64_t>(value.size()));
  for (const auto character : value) {
    output.push_back(static_cast<std::byte>(character));
  }
}

std::uint64_t read_u64(std::span<const std::byte> input, std::size_t& offset,
                       std::string_view operation) {
  if (offset > input.size() || input.size() - offset < sizeof(std::uint64_t)) {
    throw_serialization(std::string(operation),
                        "a complete little-endian uint64 value", "truncated",
                        "Use an intact current-schema checkpoint.");
  }
  std::uint64_t value = 0;
  for (std::size_t index = 0; index < sizeof(std::uint64_t); ++index) {
    value |= static_cast<std::uint64_t>(
                 std::to_integer<unsigned char>(input[offset + index]))
             << (index * 8U);
  }
  offset += sizeof(std::uint64_t);
  return value;
}

std::vector<std::byte> encode_topology(
    const BinaryRankingMetricDescriptor& descriptor) {
  std::vector<std::byte> output(kTopologyMagic.begin(), kTopologyMagic.end());
  append_unsigned(output, descriptor.schema_version);
  append_unsigned(output, static_cast<std::uint8_t>(descriptor.score));
  append_unsigned(output,
                  static_cast<std::uint8_t>(descriptor.undefined_policy));
  append_unsigned(
      output, static_cast<std::uint8_t>(descriptor.ignore_index.has_value()));
  append_unsigned(output, std::bit_cast<std::uint64_t>(
                              descriptor.ignore_index.value_or(0)));
  append_string(output, descriptor.type_id);
  append_string(output, descriptor.metric_id);
  return output;
}

std::vector<std::byte> encode_topology(
    const ClasswiseRankingMetricDescriptor& descriptor) {
  std::vector<std::byte> output(kClasswiseTopologyMagic.begin(),
                                kClasswiseTopologyMagic.end());
  append_unsigned(output, descriptor.schema_version);
  append_unsigned(output, static_cast<std::uint8_t>(descriptor.score));
  append_unsigned(output, static_cast<std::uint8_t>(descriptor.task));
  append_unsigned(output, static_cast<std::uint64_t>(descriptor.num_classes));
  append_unsigned(output, static_cast<std::uint8_t>(descriptor.average));
  append_unsigned(output,
                  static_cast<std::uint8_t>(descriptor.undefined_policy));
  append_unsigned(
      output, static_cast<std::uint8_t>(descriptor.ignore_index.has_value()));
  append_unsigned(output, std::bit_cast<std::uint64_t>(
                              descriptor.ignore_index.value_or(0)));
  append_string(output, descriptor.type_id);
  append_string(output, descriptor.metric_id);
  return output;
}

template <typename Observations>
std::vector<std::byte> encode_observations(const Observations& observations) {
  const auto encoded_bytes = checked_observation_bytes(observations.size());
  if (encoded_bytes > std::numeric_limits<std::size_t>::max()) {
    throw_exhausted("save_state.observations", "an allocation fitting size_t",
                    std::to_string(encoded_bytes),
                    "Reduce the exact observation limit.");
  }
  std::vector<std::byte> output;
  output.reserve(static_cast<std::size_t>(encoded_bytes));
  output.insert(output.end(), kObservationsMagic.begin(),
                kObservationsMagic.end());
  append_unsigned(output, static_cast<std::uint64_t>(observations.size()));
  for (const auto& observation : observations) {
    append_unsigned(output, std::bit_cast<std::uint64_t>(observation.score));
    output.push_back(observation.positive ? std::byte{1} : std::byte{0});
  }
  return output;
}

template <typename ClassObservations>
std::vector<std::byte> encode_classwise_observations(
    const ClassObservations& class_observations, std::size_t observations) {
  const auto encoded_bytes = checked_classwise_observation_bytes(
      class_observations.size(), observations);
  if (encoded_bytes > std::numeric_limits<std::size_t>::max()) {
    throw_exhausted("save_state.observations", "an allocation fitting size_t",
                    std::to_string(encoded_bytes),
                    "Reduce the exact observation limit.");
  }
  std::vector<std::byte> output;
  output.reserve(static_cast<std::size_t>(encoded_bytes));
  output.insert(output.end(), kClasswiseObservationsMagic.begin(),
                kClasswiseObservationsMagic.end());
  append_unsigned(output,
                  static_cast<std::uint64_t>(class_observations.size()));
  for (const auto& items : class_observations) {
    append_unsigned(output, static_cast<std::uint64_t>(items.size()));
    for (const auto& observation : items) {
      append_unsigned(output, std::bit_cast<std::uint64_t>(observation.score));
      output.push_back(observation.positive ? std::byte{1} : std::byte{0});
    }
  }
  return output;
}

void validate_manifest_shape(const StateManifest& stored,
                             const StateManifest& expected,
                             RestorePolicy policy) {
  validate_state_manifest(stored);
  if (!restore_policy_accepts(policy, stored.restore_level)) {
    throw_incompatible("restore.policy", "a policy accepting exact state",
                       std::string(restore_policy_name(policy)),
                       "Use exact_only or another policy accepting exact.");
  }
  if (stored.participant_id != expected.participant_id ||
      stored.schema_version != expected.schema_version ||
      stored.restore_level != RestoreLevel::exact ||
      !stored.degradation_reasons.empty() ||
      stored.fields.size() != expected.fields.size()) {
    throw_incompatible("restore.manifest", expected.participant_id,
                       stored.participant_id,
                       "Restore current exact state for the same metric ID.");
  }
  for (const auto& expected_field : expected.fields) {
    const auto actual_field =
        std::find_if(stored.fields.begin(), stored.fields.end(),
                     [&](const StateFieldDescriptor& candidate) {
                       return candidate.name == expected_field.name;
                     });
    if (actual_field == stored.fields.end() ||
        actual_field->media_type != expected_field.media_type ||
        actual_field->required != expected_field.required) {
      throw_incompatible(
          "restore.fields", expected_field.name,
          actual_field == stored.fields.end() ? "missing" : actual_field->name,
          "Restore the current ranking field schema.");
    }
  }
}

std::string default_metric_id(BinaryRankingScore score) {
  return "metric.binary_" + std::string(binary_ranking_score_name(score));
}

double undefined_result(UndefinedRankingPolicy policy, BinaryRankingScore score,
                        std::string actual) {
  switch (policy) {
    case UndefinedRankingPolicy::zero:
      return 0.0;
    case UndefinedRankingPolicy::one:
      return 1.0;
    case UndefinedRankingPolicy::error:
      throw NumericalError(
          "Ranking metric is undefined for the observed classes",
          ErrorDetails{
              .component = kComponent,
              .operation = std::string(binary_ranking_score_name(score)),
              .expected = score == BinaryRankingScore::auroc
                              ? "at least one positive and one negative"
                              : "at least one positive",
              .actual = std::move(actual),
              .hint = "Provide both required target classes or choose "
                      "an explicit zero/one undefined policy."});
  }
  throw InternalError("Unknown ranking undefined-result policy");
}

template <typename Observations>
std::vector<BinaryRankingCurvePoint> build_threshold_curve(
    const Observations& observations, std::string_view operation) {
  if (observations.empty()) {
    throw NotReady(
        "Ranking metric has no observations",
        ErrorDetails{.component = kComponent,
                     .operation = std::string(operation),
                     .expected = "at least one non-ignored observation",
                     .actual = "0",
                     .hint = "Call update() before requesting the curve."});
  }
  auto sorted = observations;
  std::sort(sorted.begin(), sorted.end(),
            [](const auto& left, const auto& right) {
              return left.score > right.score;
            });
  const auto total_positive = static_cast<std::uint64_t>(
      std::count_if(sorted.begin(), sorted.end(),
                    [](const auto& item) { return item.positive; }));
  const auto total_negative =
      static_cast<std::uint64_t>(sorted.size()) - total_positive;
  std::uint64_t true_positive = 0;
  std::uint64_t false_positive = 0;
  std::vector<BinaryRankingCurvePoint> result;
  result.reserve(sorted.size());
  std::size_t begin = 0;
  while (begin < sorted.size()) {
    auto end = begin;
    while (end < sorted.size() && sorted[end].score == sorted[begin].score) {
      if (sorted[end].positive) {
        ++true_positive;
      } else {
        ++false_positive;
      }
      ++end;
    }
    result.push_back({.threshold = sorted[begin].score,
                      .true_positives = true_positive,
                      .false_positives = false_positive,
                      .true_negatives = total_negative - false_positive,
                      .false_negatives = total_positive - true_positive});
    begin = end;
  }
  return result;
}

template <typename Observations>
double compute_score(const Observations& observations, BinaryRankingScore score,
                     UndefinedRankingPolicy policy) {
  const auto positive = static_cast<std::size_t>(
      std::count_if(observations.begin(), observations.end(),
                    [](const auto& item) { return item.positive; }));
  const auto negative = observations.size() - positive;
  if (score == BinaryRankingScore::auroc && (positive == 0 || negative == 0)) {
    return undefined_result(policy, score,
                            "positive=" + std::to_string(positive) +
                                ", negative=" + std::to_string(negative));
  }
  if (score == BinaryRankingScore::average_precision && positive == 0) {
    return undefined_result(policy, score, "positive=0");
  }

  const auto curve = build_threshold_curve(observations, "compute");
  if (score == BinaryRankingScore::average_precision) {
    long double result = 0.0L;
    std::uint64_t previous_true_positive = 0;
    for (const auto& point : curve) {
      const auto group_positive = point.true_positives - previous_true_positive;
      if (group_positive != 0) {
        const auto selected = point.true_positives + point.false_positives;
        const auto precision = static_cast<long double>(point.true_positives) /
                               static_cast<long double>(selected);
        const auto recall_increment = static_cast<long double>(group_positive) /
                                      static_cast<long double>(positive);
        result += recall_increment * precision;
      }
      previous_true_positive = point.true_positives;
    }
    return static_cast<double>(result);
  }

  long double wins = 0.0L;
  std::uint64_t previous_true_positive = 0;
  std::uint64_t previous_false_positive = 0;
  for (const auto& point : curve) {
    const auto group_positive = point.true_positives - previous_true_positive;
    const auto group_negative = point.false_positives - previous_false_positive;
    wins += static_cast<long double>(group_positive) *
            static_cast<long double>(point.true_negatives);
    wins += static_cast<long double>(group_positive) *
            static_cast<long double>(group_negative) * 0.5L;
    previous_true_positive = point.true_positives;
    previous_false_positive = point.false_positives;
  }
  return static_cast<double>(wins / (static_cast<long double>(positive) *
                                     static_cast<long double>(negative)));
}

}  // namespace

std::string_view binary_ranking_score_name(BinaryRankingScore score) noexcept {
  switch (score) {
    case BinaryRankingScore::auroc:
      return "auroc";
    case BinaryRankingScore::average_precision:
      return "average_precision";
  }
  return "unknown";
}

std::string_view undefined_ranking_policy_name(
    UndefinedRankingPolicy policy) noexcept {
  switch (policy) {
    case UndefinedRankingPolicy::error:
      return "error";
    case UndefinedRankingPolicy::zero:
      return "zero";
    case UndefinedRankingPolicy::one:
      return "one";
  }
  return "unknown";
}

BinaryRankingMetric::BinaryRankingMetric(
    std::string name, std::string type_id, BinaryRankingScore score,
    BinaryRankingMetricOptions options,
    ExactRankingResourceOptions resource_options)
    : Metric(name),
      descriptor_{.schema_version = 1,
                  .type_id = std::move(type_id),
                  .metric_id = options.metric_id.empty()
                                   ? default_metric_id(score)
                                   : std::move(options.metric_id),
                  .score = score,
                  .undefined_policy = options.undefined_policy,
                  .ignore_index = options.ignore_index},
      resource_options_(resource_options) {
  if (!valid_type_identifier(descriptor_.type_id)) {
    throw_invalid("descriptor.type_id", "a versioned ASCII type identifier",
                  descriptor_.type_id,
                  "Use stable segments such as jyppx.metric.name/v1.");
  }
  if (!valid_identifier(descriptor_.metric_id)) {
    throw_invalid("descriptor.metric_id", "a stable ASCII identifier",
                  descriptor_.metric_id,
                  "Use letters, digits, dot, underscore, or hyphen.");
  }
  if (binary_ranking_score_name(score) == "unknown") {
    throw_invalid("options.score", "auroc or average_precision", "unknown",
                  "Use a declared BinaryRankingScore value.");
  }
  if (undefined_ranking_policy_name(options.undefined_policy) == "unknown") {
    throw_invalid("options.undefined_policy", "error, zero, or one", "unknown",
                  "Use a declared UndefinedRankingPolicy value.");
  }
  if (resource_options.max_observations == 0 ||
      resource_options.max_topology_bytes == 0 ||
      resource_options.max_observation_bytes == 0) {
    throw_invalid("options.limits", "positive exact-state resource limits",
                  "zero", "Choose positive metric limits.");
  }
  const auto required_observation_bytes =
      checked_observation_bytes(resource_options.max_observations);
  if (required_observation_bytes > resource_options.max_observation_bytes) {
    throw_invalid("options.max_observation_bytes",
                  "at least " + std::to_string(required_observation_bytes),
                  std::to_string(resource_options.max_observation_bytes),
                  "Size the byte limit for every permitted observation.");
  }
  const auto topology = encode_topology(descriptor_);
  if (topology.size() > resource_options.max_topology_bytes) {
    throw_exhausted("options.max_topology_bytes",
                    "at least " + std::to_string(topology.size()),
                    std::to_string(resource_options.max_topology_bytes),
                    "Raise the explicit topology limit.");
  }
}

BinaryRankingMetric::~BinaryRankingMetric() = default;

void BinaryRankingMetric::update(const Tensor& predictions,
                                 const Tensor& targets) {
  if (!predictions.defined() || !targets.defined()) {
    throw_invalid("update.inputs", "defined prediction and target tensors",
                  "undefined", "Provide both tensors.");
  }
  if (!predictions.is_floating_point()) {
    throw_invalid("update.predictions.dtype", "a floating score tensor",
                  c10::toString(predictions.scalar_type()),
                  "Provide finite scalar scores before thresholding.");
  }
  if (!integral_labels(targets.scalar_type())) {
    throw_invalid("update.targets.dtype", "an integral binary label tensor",
                  c10::toString(targets.scalar_type()),
                  "Convert labels explicitly to an integer dtype.");
  }
  if (predictions.dim() != 1 || targets.dim() != 1 ||
      predictions.numel() != targets.numel()) {
    throw_invalid("update.shape", "equal one-dimensional tensors shaped {N}",
                  "predictions.dim=" + std::to_string(predictions.dim()) +
                      ", targets.dim=" + std::to_string(targets.dim()),
                  "Flatten scores and labels explicitly before update.");
  }
  if (predictions.device() != targets.device()) {
    throw_invalid("update.device", "scores and labels on the same device",
                  predictions.device().str() + " vs " + targets.device().str(),
                  "Move both inputs to one device explicitly.");
  }
  if (predictions.numel() == 0) {
    return;
  }
  if (!torch::isfinite(predictions).all().item<bool>()) {
    throw NumericalError(
        "Binary ranking scores must be finite",
        ErrorDetails{.component = kComponent,
                     .operation = "update.predictions",
                     .expected = "only finite scores",
                     .actual = "NaN or infinity",
                     .hint = "Validate model output before metric update."});
  }

  const auto scores = predictions.to(torch::kCPU, torch::kFloat64).contiguous();
  const auto labels = targets.to(torch::kCPU, torch::kInt64).contiguous();
  const auto score_values = scores.accessor<double, 1>();
  const auto label_values = labels.accessor<std::int64_t, 1>();
  std::vector<Observation> additions;
  additions.reserve(static_cast<std::size_t>(scores.numel()));
  for (std::int64_t index = 0; index < scores.numel(); ++index) {
    const auto label = label_values[index];
    if (descriptor_.ignore_index.has_value() &&
        label == *descriptor_.ignore_index) {
      continue;
    }
    if (label != 0 && label != 1) {
      throw_invalid("update.targets.range", "binary labels 0 or 1",
                    std::to_string(label),
                    "Fix labels or configure an explicit ignore_index.");
    }
    additions.push_back({.score = score_values[index], .positive = label == 1});
  }

  if (additions.size() >
      resource_options_.max_observations -
          std::min(observations_.size(), resource_options_.max_observations)) {
    throw_exhausted(
        "update.observations",
        "at most " + std::to_string(resource_options_.max_observations),
        std::to_string(observations_.size() + additions.size()),
        "Reset the metric, raise the explicit limit, or use a bounded mode.");
  }
  if (additions.empty()) {
    return;
  }
  auto candidate = observations_;
  candidate.reserve(observations_.size() + additions.size());
  candidate.insert(candidate.end(), additions.begin(), additions.end());
  observations_.swap(candidate);
}

void BinaryRankingMetric::reset() noexcept { observations_.clear(); }

void BinaryRankingMetric::merge(const BinaryRankingMetric& other) {
  if (encode_topology(descriptor_) != encode_topology(other.descriptor_)) {
    throw_incompatible("merge.descriptor", descriptor_.metric_id,
                       other.descriptor_.metric_id,
                       "Merge only metrics with exactly equal semantics.");
  }
  if (other.observations_.size() >
      resource_options_.max_observations -
          std::min(observations_.size(), resource_options_.max_observations)) {
    throw_exhausted(
        "merge.observations",
        "at most " + std::to_string(resource_options_.max_observations),
        std::to_string(observations_.size() + other.observations_.size()),
        "Raise the destination limit or reduce before merging.");
  }
  auto candidate = observations_;
  candidate.reserve(observations_.size() + other.observations_.size());
  candidate.insert(candidate.end(), other.observations_.begin(),
                   other.observations_.end());
  observations_.swap(candidate);
}

std::vector<BinaryRankingCurvePoint> BinaryRankingMetric::threshold_curve()
    const {
  return build_threshold_curve(observations_, "threshold_curve");
}

std::size_t BinaryRankingMetric::observations() const noexcept {
  return observations_.size();
}

std::size_t BinaryRankingMetric::positive_observations() const noexcept {
  return static_cast<std::size_t>(
      std::count_if(observations_.begin(), observations_.end(),
                    [](const Observation& item) { return item.positive; }));
}

std::size_t BinaryRankingMetric::negative_observations() const noexcept {
  return observations_.size() - positive_observations();
}

const BinaryRankingMetricDescriptor& BinaryRankingMetric::descriptor()
    const noexcept {
  return descriptor_;
}

std::string BinaryRankingMetric::participant_id() const {
  return descriptor_.metric_id;
}

StateManifest BinaryRankingMetric::describe_state() const {
  return {.participant_id = descriptor_.metric_id,
          .schema_version = 1,
          .restore_level = RestoreLevel::exact,
          .fields = {{.name = kTopologyField,
                      .media_type = kTopologyMediaType,
                      .max_bytes = resource_options_.max_topology_bytes},
                     {.name = kObservationsField,
                      .media_type = kObservationsMediaType,
                      .max_bytes = resource_options_.max_observation_bytes}},
          .degradation_reasons = {}};
}

void BinaryRankingMetric::save_state(StateWriter& writer) const {
  const auto topology = encode_topology(descriptor_);
  const auto observations = encode_observations(observations_);
  if (topology.size() > resource_options_.max_topology_bytes ||
      observations.size() > resource_options_.max_observation_bytes) {
    throw_exhausted("save_state", "state within configured limits", "oversized",
                    "Raise explicit limits before constructing the metric.");
  }
  writer.write_blob(kTopologyField, std::span<const std::byte>(
                                        topology.data(), topology.size()));
  writer.write_blob(
      kObservationsField,
      std::span<const std::byte>(observations.data(), observations.size()));
}

void BinaryRankingMetric::load_state(const StateReader& reader,
                                     RestorePolicy policy) {
  const auto expected_manifest = describe_state();
  validate_manifest_shape(reader.manifest(), expected_manifest, policy);
  const auto stored_topology =
      reader.read_blob(kTopologyField, resource_options_.max_topology_bytes);
  const auto expected_topology = encode_topology(descriptor_);
  if (stored_topology.size() != expected_topology.size() ||
      !std::equal(stored_topology.begin(), stored_topology.end(),
                  expected_topology.begin())) {
    throw_incompatible("restore.topology", descriptor_.metric_id,
                       "different canonical descriptor",
                       "Restore state from exactly equal metric semantics.");
  }

  const auto encoded = reader.read_blob(
      kObservationsField, resource_options_.max_observation_bytes);
  if (encoded.size() < kObservationHeaderBytes ||
      !std::equal(kObservationsMagic.begin(), kObservationsMagic.end(),
                  encoded.begin())) {
    throw_serialization("observations.header", "current observation magic",
                        "missing or truncated",
                        "Use an intact current-schema checkpoint.");
  }
  std::size_t offset = kObservationsMagic.size();
  const auto encoded_count = read_u64(encoded, offset, "observations.count");
  if (encoded_count > resource_options_.max_observations) {
    throw_exhausted(
        "restore.observations",
        "at most " + std::to_string(resource_options_.max_observations),
        std::to_string(encoded_count),
        "Raise the explicit limit before restoring this metric.");
  }
  const auto expected_bytes =
      checked_observation_bytes(static_cast<std::size_t>(encoded_count));
  if (encoded.size() != expected_bytes) {
    throw_serialization("observations.size", std::to_string(expected_bytes),
                        std::to_string(encoded.size()),
                        "Reject truncated or trailing observation bytes.");
  }

  std::vector<Observation> candidate;
  candidate.reserve(static_cast<std::size_t>(encoded_count));
  for (std::uint64_t index = 0; index < encoded_count; ++index) {
    const auto score =
        std::bit_cast<double>(read_u64(encoded, offset, "observations.score"));
    const auto label = std::to_integer<unsigned char>(encoded[offset]);
    ++offset;
    if (!std::isfinite(score)) {
      throw_serialization("observations.score", "a finite IEEE-754 double",
                          "NaN or infinity",
                          "Reject corrupt exact ranking state.");
    }
    if (label > 1) {
      throw_serialization("observations.label", "zero or one",
                          std::to_string(label),
                          "Reject corrupt binary labels.");
    }
    candidate.push_back({.score = score, .positive = label == 1});
  }
  observations_.swap(candidate);
}

double BinaryRankingMetric::compute_ranking_score() const {
  if (observations_.empty()) {
    throw NotReady(
        "Binary ranking metric has no observations",
        ErrorDetails{.component = kComponent,
                     .operation = "compute",
                     .expected = "at least one non-ignored observation",
                     .actual = "0",
                     .hint = "Call update() before computing the metric."});
  }
  return compute_score(observations_, descriptor_.score,
                       descriptor_.undefined_policy);
}

BinaryAUROC::BinaryAUROC(BinaryRankingMetricOptions options,
                         ExactRankingResourceOptions resource_options)
    : BinaryRankingMetric("auroc", kAuRocTypeId, BinaryRankingScore::auroc,
                          std::move(options), resource_options) {}

double BinaryAUROC::compute() { return compute_ranking_score(); }

BinaryAveragePrecision::BinaryAveragePrecision(
    BinaryRankingMetricOptions options,
    ExactRankingResourceOptions resource_options)
    : BinaryRankingMetric("average_precision", kAveragePrecisionTypeId,
                          BinaryRankingScore::average_precision,
                          std::move(options), resource_options) {}

double BinaryAveragePrecision::compute() { return compute_ranking_score(); }

std::string_view ranking_average_name(RankingAverage average) noexcept {
  switch (average) {
    case RankingAverage::micro:
      return "micro";
    case RankingAverage::macro:
      return "macro";
    case RankingAverage::weighted:
      return "weighted";
    case RankingAverage::none:
      return "none";
  }
  return "unknown";
}

std::string_view classwise_ranking_task_name(
    ClasswiseRankingTask task) noexcept {
  switch (task) {
    case ClasswiseRankingTask::multiclass:
      return "multiclass";
    case ClasswiseRankingTask::multilabel:
      return "multilabel";
  }
  return "unknown";
}

namespace {

std::string default_classwise_metric_id(BinaryRankingScore score,
                                        ClasswiseRankingTask task) {
  return "metric." + std::string(classwise_ranking_task_name(task)) + "_" +
         std::string(binary_ranking_score_name(score));
}

}  // namespace

ClasswiseRankingMetric::ClasswiseRankingMetric(
    std::string name, std::string type_id, BinaryRankingScore score,
    ClasswiseRankingTask task, ClasswiseRankingMetricOptions options,
    ExactRankingResourceOptions resource_options)
    : Metric(name),
      descriptor_{.schema_version = 1,
                  .type_id = std::move(type_id),
                  .metric_id = options.metric_id.empty()
                                   ? default_classwise_metric_id(score, task)
                                   : std::move(options.metric_id),
                  .score = score,
                  .task = task,
                  .num_classes = options.num_classes,
                  .average = options.average,
                  .undefined_policy = options.undefined_policy,
                  .ignore_index = options.ignore_index},
      resource_options_(resource_options) {
  if (!valid_type_identifier(descriptor_.type_id)) {
    throw_invalid("descriptor.type_id", "a versioned ASCII type identifier",
                  descriptor_.type_id,
                  "Use stable segments such as jyppx.metric.name/v1.");
  }
  if (!valid_identifier(descriptor_.metric_id)) {
    throw_invalid("descriptor.metric_id", "a stable ASCII identifier",
                  descriptor_.metric_id,
                  "Use letters, digits, dot, underscore, or hyphen.");
  }
  if (binary_ranking_score_name(score) == "unknown") {
    throw_invalid("options.score", "auroc or average_precision", "unknown",
                  "Use a declared BinaryRankingScore value.");
  }
  if (classwise_ranking_task_name(task) == "unknown") {
    throw_invalid("options.task", "multiclass or multilabel", "unknown",
                  "Use a declared ClasswiseRankingTask value.");
  }
  if (ranking_average_name(options.average) == "unknown") {
    throw_invalid("options.average", "micro, macro, weighted, or none",
                  "unknown", "Use a declared RankingAverage value.");
  }
  if (undefined_ranking_policy_name(options.undefined_policy) == "unknown") {
    throw_invalid("options.undefined_policy", "error, zero, or one", "unknown",
                  "Use a declared UndefinedRankingPolicy value.");
  }
  if (options.num_classes < 2 || resource_options.max_classes == 0 ||
      resource_options.max_observations == 0 ||
      resource_options.max_topology_bytes == 0 ||
      resource_options.max_observation_bytes == 0) {
    throw_invalid("options.limits",
                  "at least two classes and positive resource limits", "zero",
                  "Choose a fixed class topology and positive limits.");
  }
  if (options.num_classes > resource_options.max_classes ||
      options.num_classes >
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw_exhausted("options.num_classes",
                    "at most " + std::to_string(resource_options.max_classes),
                    std::to_string(options.num_classes),
                    "Raise the explicit class limit or reduce class count.");
  }
  const auto required_observation_bytes = checked_classwise_observation_bytes(
      options.num_classes, resource_options.max_observations);
  if (required_observation_bytes > resource_options.max_observation_bytes) {
    throw_invalid("options.max_observation_bytes",
                  "at least " + std::to_string(required_observation_bytes),
                  std::to_string(resource_options.max_observation_bytes),
                  "Size the byte limit for all permitted classes and pairs.");
  }
  const auto topology = encode_topology(descriptor_);
  if (topology.size() > resource_options.max_topology_bytes) {
    throw_exhausted("options.max_topology_bytes",
                    "at least " + std::to_string(topology.size()),
                    std::to_string(resource_options.max_topology_bytes),
                    "Raise the explicit topology limit.");
  }
  class_observations_.resize(descriptor_.num_classes);
}

ClasswiseRankingMetric::~ClasswiseRankingMetric() = default;

void ClasswiseRankingMetric::update(const Tensor& predictions,
                                    const Tensor& targets) {
  if (!predictions.defined() || !targets.defined()) {
    throw_invalid("update.inputs", "defined prediction and target tensors",
                  "undefined", "Provide both tensors.");
  }
  if (!predictions.is_floating_point()) {
    throw_invalid("update.predictions.dtype", "a floating score matrix",
                  c10::toString(predictions.scalar_type()),
                  "Provide finite scores before thresholding.");
  }
  if (!integral_labels(targets.scalar_type())) {
    throw_invalid("update.targets.dtype", "an integral target tensor",
                  c10::toString(targets.scalar_type()),
                  "Convert labels explicitly to an integer dtype.");
  }
  if (predictions.dim() != 2 ||
      predictions.size(1) !=
          static_cast<std::int64_t>(descriptor_.num_classes)) {
    throw_invalid("update.predictions.shape",
                  "{N," + std::to_string(descriptor_.num_classes) + "}",
                  std::to_string(predictions.dim()) + " dimensions",
                  "Use the fixed descriptor class topology.");
  }
  const auto multiclass = descriptor_.task == ClasswiseRankingTask::multiclass;
  const auto valid_target_shape =
      multiclass ? targets.dim() == 1 && targets.size(0) == predictions.size(0)
                 : targets.dim() == 2 && targets.sizes() == predictions.sizes();
  if (!valid_target_shape) {
    throw_invalid("update.targets.shape",
                  multiclass ? "{N}" : "the same {N,C} shape as scores",
                  std::to_string(targets.dim()) + " dimensions",
                  "Provide targets matching the configured task.");
  }
  if (predictions.device() != targets.device()) {
    throw_invalid("update.device", "scores and labels on the same device",
                  predictions.device().str() + " vs " + targets.device().str(),
                  "Move both inputs to one device explicitly.");
  }
  if (predictions.numel() == 0) {
    return;
  }
  if (!torch::isfinite(predictions).all().item<bool>()) {
    throw NumericalError(
        "Classwise ranking scores must be finite",
        ErrorDetails{.component = kComponent,
                     .operation = "update.predictions",
                     .expected = "only finite scores",
                     .actual = "NaN or infinity",
                     .hint = "Validate model output before metric update."});
  }

  const auto scores = predictions.to(torch::kCPU, torch::kFloat64).contiguous();
  const auto labels = targets.to(torch::kCPU, torch::kInt64).contiguous();
  const auto score_values = scores.accessor<double, 2>();
  std::size_t additions_count = 0;
  if (multiclass) {
    const auto label_values = labels.accessor<std::int64_t, 1>();
    for (std::int64_t row = 0; row < labels.size(0); ++row) {
      const auto label = label_values[row];
      if (descriptor_.ignore_index.has_value() &&
          label == *descriptor_.ignore_index) {
        continue;
      }
      if (label < 0 ||
          label >= static_cast<std::int64_t>(descriptor_.num_classes)) {
        throw_invalid("update.targets.range",
                      "class indices in [0," +
                          std::to_string(descriptor_.num_classes) + ")",
                      std::to_string(label),
                      "Fix labels or configure an explicit ignore_index.");
      }
      if (descriptor_.num_classes >
          resource_options_.max_observations -
              std::min(additions_count, resource_options_.max_observations)) {
        throw_exhausted("update.observations", "the configured global limit",
                        "batch expansion exceeds the limit",
                        "Raise the exact-pair limit or reduce the batch.");
      }
      additions_count += descriptor_.num_classes;
    }
  } else {
    const auto label_values = labels.accessor<std::int64_t, 2>();
    for (std::int64_t row = 0; row < labels.size(0); ++row) {
      for (std::size_t class_index = 0; class_index < descriptor_.num_classes;
           ++class_index) {
        const auto label =
            label_values[row][static_cast<std::int64_t>(class_index)];
        if (descriptor_.ignore_index.has_value() &&
            label == *descriptor_.ignore_index) {
          continue;
        }
        if (label != 0 && label != 1) {
          throw_invalid("update.targets.range", "binary labels 0 or 1",
                        std::to_string(label),
                        "Fix labels or configure an explicit ignore_index.");
        }
        ++additions_count;
      }
    }
  }
  if (additions_count >
      resource_options_.max_observations -
          std::min(observations_, resource_options_.max_observations)) {
    throw_exhausted(
        "update.observations",
        "at most " + std::to_string(resource_options_.max_observations),
        std::to_string(observations_) + "+" + std::to_string(additions_count),
        "Reset the metric or raise the global exact-pair limit.");
  }
  if (additions_count == 0) {
    return;
  }

  std::vector<std::vector<Observation>> additions(descriptor_.num_classes);
  if (multiclass) {
    const auto label_values = labels.accessor<std::int64_t, 1>();
    for (std::int64_t row = 0; row < labels.size(0); ++row) {
      const auto label = label_values[row];
      if (descriptor_.ignore_index.has_value() &&
          label == *descriptor_.ignore_index) {
        continue;
      }
      for (std::size_t class_index = 0; class_index < descriptor_.num_classes;
           ++class_index) {
        additions[class_index].push_back(
            {.score = score_values[row][static_cast<std::int64_t>(class_index)],
             .positive = label == static_cast<std::int64_t>(class_index)});
      }
    }
  } else {
    const auto label_values = labels.accessor<std::int64_t, 2>();
    for (std::int64_t row = 0; row < labels.size(0); ++row) {
      for (std::size_t class_index = 0; class_index < descriptor_.num_classes;
           ++class_index) {
        const auto label =
            label_values[row][static_cast<std::int64_t>(class_index)];
        if (descriptor_.ignore_index.has_value() &&
            label == *descriptor_.ignore_index) {
          continue;
        }
        additions[class_index].push_back(
            {.score = score_values[row][static_cast<std::int64_t>(class_index)],
             .positive = label == 1});
      }
    }
  }

  auto candidate = class_observations_;
  for (std::size_t class_index = 0; class_index < descriptor_.num_classes;
       ++class_index) {
    auto& destination = candidate[class_index];
    const auto& source = additions[class_index];
    destination.reserve(destination.size() + source.size());
    destination.insert(destination.end(), source.begin(), source.end());
  }
  class_observations_.swap(candidate);
  observations_ += additions_count;
}

double ClasswiseRankingMetric::compute() {
  if (descriptor_.average == RankingAverage::none) {
    throw InvalidArgument(
        "Scalar compute is unavailable for average=none",
        ErrorDetails{.component = kComponent,
                     .operation = "compute.average",
                     .expected = "micro, macro, or weighted",
                     .actual = "none",
                     .hint = "Call per_class_scores() for class-wise values."});
  }
  if (observations_ == 0) {
    throw NotReady(
        "Classwise ranking metric has no observations",
        ErrorDetails{.component = kComponent,
                     .operation = "compute",
                     .expected = "at least one non-ignored score/label pair",
                     .actual = "0",
                     .hint = "Call update() before computing the metric."});
  }
  if (descriptor_.average == RankingAverage::micro) {
    std::vector<Observation> pooled;
    pooled.reserve(observations_);
    for (const auto& items : class_observations_) {
      pooled.insert(pooled.end(), items.begin(), items.end());
    }
    return compute_score(pooled, descriptor_.score,
                         descriptor_.undefined_policy);
  }

  const auto scores = per_class_scores();
  if (descriptor_.average == RankingAverage::macro) {
    long double sum = 0.0L;
    for (const auto score : scores) {
      sum += score;
    }
    return static_cast<double>(sum / static_cast<long double>(scores.size()));
  }

  long double weighted_sum = 0.0L;
  std::size_t total_support = 0;
  for (std::size_t class_index = 0; class_index < scores.size();
       ++class_index) {
    const auto support = positive_observations(class_index);
    weighted_sum += static_cast<long double>(scores[class_index]) *
                    static_cast<long double>(support);
    total_support += support;
  }
  if (total_support == 0) {
    return undefined_result(descriptor_.undefined_policy, descriptor_.score,
                            "positive=0 across all classes");
  }
  return static_cast<double>(weighted_sum /
                             static_cast<long double>(total_support));
}

void ClasswiseRankingMetric::reset() noexcept {
  for (auto& items : class_observations_) {
    items.clear();
  }
  observations_ = 0;
}

void ClasswiseRankingMetric::merge(const ClasswiseRankingMetric& other) {
  if (encode_topology(descriptor_) != encode_topology(other.descriptor_)) {
    throw_incompatible("merge.descriptor", descriptor_.metric_id,
                       other.descriptor_.metric_id,
                       "Merge only metrics with exactly equal semantics.");
  }
  if (other.observations_ >
      resource_options_.max_observations -
          std::min(observations_, resource_options_.max_observations)) {
    throw_exhausted(
        "merge.observations",
        "at most " + std::to_string(resource_options_.max_observations),
        std::to_string(observations_) + "+" +
            std::to_string(other.observations_),
        "Raise the destination limit or reduce before merging.");
  }
  auto candidate = class_observations_;
  for (std::size_t class_index = 0; class_index < descriptor_.num_classes;
       ++class_index) {
    auto& destination = candidate[class_index];
    const auto& source = other.class_observations_[class_index];
    destination.reserve(destination.size() + source.size());
    destination.insert(destination.end(), source.begin(), source.end());
  }
  class_observations_.swap(candidate);
  observations_ += other.observations_;
}

std::vector<double> ClasswiseRankingMetric::per_class_scores() const {
  if (observations_ == 0) {
    throw NotReady(
        "Classwise ranking metric has no observations",
        ErrorDetails{.component = kComponent,
                     .operation = "per_class_scores",
                     .expected = "at least one non-ignored score/label pair",
                     .actual = "0",
                     .hint = "Call update() before computing the metric."});
  }
  std::vector<double> result;
  result.reserve(descriptor_.num_classes);
  for (const auto& items : class_observations_) {
    result.push_back(
        compute_score(items, descriptor_.score, descriptor_.undefined_policy));
  }
  return result;
}

std::vector<BinaryRankingCurvePoint> ClasswiseRankingMetric::threshold_curve(
    std::size_t class_index) const {
  validate_class_index(class_index);
  return build_threshold_curve(class_observations_[class_index],
                               "threshold_curve");
}

std::vector<BinaryRankingCurvePoint>
ClasswiseRankingMetric::micro_threshold_curve() const {
  std::vector<Observation> pooled;
  pooled.reserve(observations_);
  for (const auto& items : class_observations_) {
    pooled.insert(pooled.end(), items.begin(), items.end());
  }
  return build_threshold_curve(pooled, "micro_threshold_curve");
}

std::size_t ClasswiseRankingMetric::observations() const noexcept {
  return observations_;
}

std::size_t ClasswiseRankingMetric::observations(
    std::size_t class_index) const {
  validate_class_index(class_index);
  return class_observations_[class_index].size();
}

std::size_t ClasswiseRankingMetric::positive_observations(
    std::size_t class_index) const {
  validate_class_index(class_index);
  return static_cast<std::size_t>(
      std::count_if(class_observations_[class_index].begin(),
                    class_observations_[class_index].end(),
                    [](const Observation& item) { return item.positive; }));
}

const ClasswiseRankingMetricDescriptor& ClasswiseRankingMetric::descriptor()
    const noexcept {
  return descriptor_;
}

std::string ClasswiseRankingMetric::participant_id() const {
  return descriptor_.metric_id;
}

StateManifest ClasswiseRankingMetric::describe_state() const {
  return {.participant_id = descriptor_.metric_id,
          .schema_version = 1,
          .restore_level = RestoreLevel::exact,
          .fields = {{.name = kTopologyField,
                      .media_type = kClasswiseTopologyMediaType,
                      .max_bytes = resource_options_.max_topology_bytes},
                     {.name = kObservationsField,
                      .media_type = kClasswiseObservationsMediaType,
                      .max_bytes = resource_options_.max_observation_bytes}},
          .degradation_reasons = {}};
}

void ClasswiseRankingMetric::save_state(StateWriter& writer) const {
  const auto topology = encode_topology(descriptor_);
  const auto observations =
      encode_classwise_observations(class_observations_, observations_);
  if (topology.size() > resource_options_.max_topology_bytes ||
      observations.size() > resource_options_.max_observation_bytes) {
    throw_exhausted("save_state", "state within configured limits", "oversized",
                    "Raise explicit limits before constructing the metric.");
  }
  writer.write_blob(kTopologyField, std::span<const std::byte>(
                                        topology.data(), topology.size()));
  writer.write_blob(
      kObservationsField,
      std::span<const std::byte>(observations.data(), observations.size()));
}

void ClasswiseRankingMetric::load_state(const StateReader& reader,
                                        RestorePolicy policy) {
  const auto expected_manifest = describe_state();
  validate_manifest_shape(reader.manifest(), expected_manifest, policy);
  const auto stored_topology =
      reader.read_blob(kTopologyField, resource_options_.max_topology_bytes);
  const auto expected_topology = encode_topology(descriptor_);
  if (stored_topology.size() != expected_topology.size() ||
      !std::equal(stored_topology.begin(), stored_topology.end(),
                  expected_topology.begin())) {
    throw_incompatible("restore.topology", descriptor_.metric_id,
                       "different canonical descriptor",
                       "Restore state from exactly equal metric semantics.");
  }

  const auto encoded = reader.read_blob(
      kObservationsField, resource_options_.max_observation_bytes);
  constexpr auto kHeaderBytes =
      kClasswiseObservationsMagic.size() + sizeof(std::uint64_t);
  if (encoded.size() < kHeaderBytes ||
      !std::equal(kClasswiseObservationsMagic.begin(),
                  kClasswiseObservationsMagic.end(), encoded.begin())) {
    throw_serialization("observations.header", "current classwise state magic",
                        "missing or truncated",
                        "Use an intact current-schema checkpoint.");
  }
  std::size_t offset = kClasswiseObservationsMagic.size();
  const auto encoded_classes =
      read_u64(encoded, offset, "observations.classes");
  if (encoded_classes != descriptor_.num_classes) {
    throw_serialization("observations.classes",
                        std::to_string(descriptor_.num_classes),
                        std::to_string(encoded_classes),
                        "Reject state whose class table is malformed.");
  }

  std::vector<std::vector<Observation>> candidate(descriptor_.num_classes);
  std::size_t candidate_count = 0;
  for (std::size_t class_index = 0; class_index < descriptor_.num_classes;
       ++class_index) {
    const auto encoded_count =
        read_u64(encoded, offset, "observations.class_count");
    if (encoded_count > resource_options_.max_observations - candidate_count) {
      throw_exhausted(
          "restore.observations",
          "at most " + std::to_string(resource_options_.max_observations),
          std::to_string(candidate_count) + "+" + std::to_string(encoded_count),
          "Raise the global exact-pair limit before restoring.");
    }
    auto& items = candidate[class_index];
    items.reserve(static_cast<std::size_t>(encoded_count));
    for (std::uint64_t index = 0; index < encoded_count; ++index) {
      const auto score = std::bit_cast<double>(
          read_u64(encoded, offset, "observations.score"));
      if (offset >= encoded.size()) {
        throw_serialization("observations.label", "one binary label byte",
                            "truncated",
                            "Use an intact current-schema checkpoint.");
      }
      const auto label = std::to_integer<unsigned char>(encoded[offset]);
      ++offset;
      if (!std::isfinite(score)) {
        throw_serialization("observations.score", "a finite IEEE-754 double",
                            "NaN or infinity",
                            "Reject corrupt exact ranking state.");
      }
      if (label > 1) {
        throw_serialization("observations.label", "zero or one",
                            std::to_string(label),
                            "Reject corrupt binary labels.");
      }
      items.push_back({.score = score, .positive = label == 1});
    }
    candidate_count += static_cast<std::size_t>(encoded_count);
  }
  if (offset != encoded.size() ||
      encoded.size() != checked_classwise_observation_bytes(
                            descriptor_.num_classes, candidate_count)) {
    throw_serialization("observations.size", "exact current-schema bytes",
                        std::to_string(encoded.size()),
                        "Reject truncated or trailing observation bytes.");
  }
  class_observations_.swap(candidate);
  observations_ = candidate_count;
}

void ClasswiseRankingMetric::validate_class_index(
    std::size_t class_index) const {
  if (class_index >= descriptor_.num_classes) {
    throw_invalid("class_index",
                  "an index below " + std::to_string(descriptor_.num_classes),
                  std::to_string(class_index),
                  "Use the fixed descriptor class topology.");
  }
}

MulticlassAUROC::MulticlassAUROC(ClasswiseRankingMetricOptions options,
                                 ExactRankingResourceOptions resource_options)
    : ClasswiseRankingMetric("auroc", kMulticlassAuRocTypeId,
                             BinaryRankingScore::auroc,
                             ClasswiseRankingTask::multiclass,
                             std::move(options), resource_options) {}

MulticlassAveragePrecision::MulticlassAveragePrecision(
    ClasswiseRankingMetricOptions options,
    ExactRankingResourceOptions resource_options)
    : ClasswiseRankingMetric("average_precision",
                             kMulticlassAveragePrecisionTypeId,
                             BinaryRankingScore::average_precision,
                             ClasswiseRankingTask::multiclass,
                             std::move(options), resource_options) {}

MultilabelAUROC::MultilabelAUROC(ClasswiseRankingMetricOptions options,
                                 ExactRankingResourceOptions resource_options)
    : ClasswiseRankingMetric("auroc", kMultilabelAuRocTypeId,
                             BinaryRankingScore::auroc,
                             ClasswiseRankingTask::multilabel,
                             std::move(options), resource_options) {}

MultilabelAveragePrecision::MultilabelAveragePrecision(
    ClasswiseRankingMetricOptions options,
    ExactRankingResourceOptions resource_options)
    : ClasswiseRankingMetric("average_precision",
                             kMultilabelAveragePrecisionTypeId,
                             BinaryRankingScore::average_precision,
                             ClasswiseRankingTask::multilabel,
                             std::move(options), resource_options) {}

}  // namespace jyppx::training::metrics
