// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/training/metrics/classification.hpp>

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

constexpr char kComponent[] = "training.metrics.classification";
constexpr char kPrecisionTypeId[] = "jyppx.metric.precision/v1";
constexpr char kRecallTypeId[] = "jyppx.metric.recall/v1";
constexpr char kF1TypeId[] = "jyppx.metric.f1/v1";
constexpr char kTopologyField[] = "topology";
constexpr char kCountsField[] = "counts";
constexpr char kTopologyMediaType[] =
    "application/vnd.jyppx.metric.classification-topology-v1";
constexpr char kCountsMediaType[] =
    "application/vnd.jyppx.metric.confusion-int64-le-v1";
constexpr std::array<std::byte, 8> kTopologyMagic{
    std::byte{'J'}, std::byte{'Y'}, std::byte{'P'}, std::byte{'M'},
    std::byte{'E'}, std::byte{'T'}, std::byte{'0'}, std::byte{'1'}};

[[noreturn]] void throw_invalid(std::string operation, std::string expected,
                                std::string actual, std::string hint) {
  throw InvalidArgument("Invalid classification metric contract",
                        ErrorDetails{.component = kComponent,
                                     .operation = std::move(operation),
                                     .expected = std::move(expected),
                                     .actual = std::move(actual),
                                     .hint = std::move(hint)});
}

[[noreturn]] void throw_incompatible(std::string operation,
                                     std::string expected, std::string actual,
                                     std::string hint) {
  throw CompatibilityError("Classification metric checkpoint is incompatible",
                           ErrorDetails{.component = kComponent,
                                        .operation = std::move(operation),
                                        .expected = std::move(expected),
                                        .actual = std::move(actual),
                                        .hint = std::move(hint)});
}

[[noreturn]] void throw_serialization(std::string operation,
                                      std::string expected, std::string actual,
                                      std::string hint) {
  throw SerializationError("Classification metric state is malformed",
                           ErrorDetails{.component = kComponent,
                                        .operation = std::move(operation),
                                        .expected = std::move(expected),
                                        .actual = std::move(actual),
                                        .hint = std::move(hint)});
}

[[noreturn]] void throw_exhausted(std::string operation, std::string expected,
                                  std::string actual, std::string hint) {
  throw ResourceExhausted("Classification metric resource limit exceeded",
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

std::size_t checked_cells(std::size_t classes) {
  if (classes > std::numeric_limits<std::size_t>::max() / classes) {
    throw_exhausted("options.num_classes", "a square matrix fitting size_t",
                    std::to_string(classes), "Reduce the fixed class count.");
  }
  return classes * classes;
}

std::uint64_t checked_counts_bytes(std::size_t classes) {
  const auto cells = checked_cells(classes);
  if (cells >
      std::numeric_limits<std::uint64_t>::max() / sizeof(std::int64_t)) {
    throw_exhausted("options.num_classes", "an int64 matrix fitting uint64",
                    std::to_string(classes), "Reduce the fixed class count.");
  }
  return static_cast<std::uint64_t>(cells) * sizeof(std::int64_t);
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

std::uint64_t read_u64(std::span<const std::byte> input, std::size_t& offset) {
  if (offset > input.size() || input.size() - offset < sizeof(std::uint64_t)) {
    throw_serialization("counts", "a complete little-endian int64 value",
                        "truncated", "Use an intact current-schema state.");
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
    const ClassificationMetricDescriptor& descriptor) {
  std::vector<std::byte> output(kTopologyMagic.begin(), kTopologyMagic.end());
  append_unsigned(output, descriptor.schema_version);
  append_unsigned(output, static_cast<std::uint8_t>(descriptor.score));
  append_unsigned(output, static_cast<std::uint8_t>(descriptor.average));
  append_unsigned(output, static_cast<std::uint8_t>(descriptor.zero_division));
  append_unsigned(output, static_cast<std::uint64_t>(descriptor.num_classes));
  append_unsigned(
      output, static_cast<std::uint8_t>(descriptor.ignore_index.has_value()));
  append_unsigned(output, std::bit_cast<std::uint64_t>(
                              descriptor.ignore_index.value_or(0)));
  append_string(output, descriptor.type_id);
  append_string(output, descriptor.metric_id);
  return output;
}

std::vector<std::byte> encode_counts(
    const std::vector<std::vector<std::int64_t>>& matrix) {
  const auto classes = matrix.size();
  std::vector<std::byte> output;
  output.reserve(checked_cells(classes) * sizeof(std::int64_t));
  for (const auto& row : matrix) {
    for (const auto count : row) {
      append_unsigned(output, static_cast<std::uint64_t>(count));
    }
  }
  return output;
}

std::vector<std::vector<std::int64_t>> decode_counts(
    std::span<const std::byte> encoded, std::size_t classes) {
  const auto expected_bytes = checked_counts_bytes(classes);
  if (encoded.size() != expected_bytes) {
    throw_serialization("counts.size", std::to_string(expected_bytes),
                        std::to_string(encoded.size()),
                        "Restore counts from the same fixed class topology.");
  }
  std::vector<std::vector<std::int64_t>> candidate(
      classes, std::vector<std::int64_t>(classes, 0));
  std::size_t offset = 0;
  std::int64_t total = 0;
  for (auto& row : candidate) {
    for (auto& count : row) {
      const auto encoded_count = read_u64(encoded, offset);
      if (encoded_count > static_cast<std::uint64_t>(
                              std::numeric_limits<std::int64_t>::max())) {
        throw_serialization("counts.value", "a non-negative int64 count",
                            std::to_string(encoded_count),
                            "Reject corrupt or overflowed counts.");
      }
      count = static_cast<std::int64_t>(encoded_count);
      if (count > std::numeric_limits<std::int64_t>::max() - total) {
        throw_serialization("counts.total", "an int64 observation count",
                            "overflow",
                            "Reject state with an unrepresentable total.");
      }
      total += count;
    }
  }
  return candidate;
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
  for (const auto& right : expected.fields) {
    const auto left = std::find_if(stored.fields.begin(), stored.fields.end(),
                                   [&](const StateFieldDescriptor& candidate) {
                                     return candidate.name == right.name;
                                   });
    if (left == stored.fields.end() || left->media_type != right.media_type ||
        left->required != right.required) {
      throw_incompatible("restore.fields", right.name,
                         left == stored.fields.end() ? "missing" : left->name,
                         "Restore the current metric field schema.");
    }
  }
}

std::string default_metric_id(ClassificationScore score) {
  return "metric." + std::string(classification_score_name(score));
}

double zero_division_value(ZeroDivisionPolicy policy, std::size_t class_index,
                           ClassificationScore score) {
  switch (policy) {
    case ZeroDivisionPolicy::zero:
      return 0.0;
    case ZeroDivisionPolicy::one:
      return 1.0;
    case ZeroDivisionPolicy::error:
      throw NumericalError(
          "Classification metric denominator is zero",
          ErrorDetails{
              .component = kComponent,
              .operation = std::string(classification_score_name(score)) +
                           ".class[" + std::to_string(class_index) + "]",
              .expected = "a positive denominator",
              .actual = "0",
              .hint = "Choose zero/one policy or provide observations."});
  }
  throw InternalError("Unknown zero-division policy");
}

double score_from_counts(ClassificationScore score, long double true_positive,
                         long double false_positive, long double false_negative,
                         ZeroDivisionPolicy policy, std::size_t class_index) {
  long double numerator = 0.0L;
  long double denominator = 0.0L;
  switch (score) {
    case ClassificationScore::precision:
      numerator = true_positive;
      denominator = true_positive + false_positive;
      break;
    case ClassificationScore::recall:
      numerator = true_positive;
      denominator = true_positive + false_negative;
      break;
    case ClassificationScore::f1:
      numerator = 2.0L * true_positive;
      denominator = numerator + false_positive + false_negative;
      break;
  }
  if (denominator == 0.0L) {
    return zero_division_value(policy, class_index, score);
  }
  return static_cast<double>(numerator / denominator);
}

}  // namespace

std::string_view classification_average_name(
    ClassificationAverage average) noexcept {
  switch (average) {
    case ClassificationAverage::micro:
      return "micro";
    case ClassificationAverage::macro:
      return "macro";
    case ClassificationAverage::weighted:
      return "weighted";
    case ClassificationAverage::none:
      return "none";
  }
  return "unknown";
}

std::string_view zero_division_policy_name(ZeroDivisionPolicy policy) noexcept {
  switch (policy) {
    case ZeroDivisionPolicy::zero:
      return "zero";
    case ZeroDivisionPolicy::one:
      return "one";
    case ZeroDivisionPolicy::error:
      return "error";
  }
  return "unknown";
}

std::string_view classification_score_name(ClassificationScore score) noexcept {
  switch (score) {
    case ClassificationScore::precision:
      return "precision";
    case ClassificationScore::recall:
      return "recall";
    case ClassificationScore::f1:
      return "f1";
  }
  return "unknown";
}

ConfusionMatrixMetric::ConfusionMatrixMetric(
    std::string name, std::string type_id, ClassificationScore score,
    ClassificationMetricOptions options,
    ClassificationMetricCheckpointOptions checkpoint_options)
    : Metric(name),
      descriptor_{.schema_version = 1,
                  .type_id = std::move(type_id),
                  .metric_id = options.metric_id.empty()
                                   ? default_metric_id(score)
                                   : std::move(options.metric_id),
                  .score = score,
                  .num_classes = options.num_classes,
                  .average = options.average,
                  .zero_division = options.zero_division,
                  .ignore_index = options.ignore_index},
      checkpoint_options_(checkpoint_options) {
  if (!valid_type_identifier(descriptor_.type_id)) {
    throw_invalid("descriptor.type_id", "a nonempty versioned type identifier",
                  descriptor_.type_id,
                  "Use stable ASCII segments such as jyppx.metric.name/v1.");
  }
  if (!valid_identifier(descriptor_.metric_id)) {
    throw_invalid("descriptor.metric_id", "a nonempty stable ASCII identifier",
                  descriptor_.metric_id,
                  "Use letters, digits, dot, underscore, or hyphen.");
  }
  if (classification_score_name(score) == "unknown") {
    throw_invalid("options.score", "precision, recall, or f1", "unknown",
                  "Use a declared ClassificationScore value.");
  }
  if (classification_average_name(options.average) == "unknown") {
    throw_invalid("options.average", "micro, macro, weighted, or none",
                  "unknown", "Use a declared ClassificationAverage value.");
  }
  if (zero_division_policy_name(options.zero_division) == "unknown") {
    throw_invalid("options.zero_division", "zero, one, or error", "unknown",
                  "Use a declared ZeroDivisionPolicy value.");
  }
  if (options.num_classes == 0 || checkpoint_options.max_classes == 0 ||
      checkpoint_options.max_topology_bytes == 0 ||
      checkpoint_options.max_counts_bytes == 0) {
    throw_invalid("options.limits", "positive topology and resource limits",
                  "zero", "Choose positive metric limits.");
  }
  if (options.num_classes > checkpoint_options.max_classes) {
    throw_exhausted("options.num_classes",
                    "at most " + std::to_string(checkpoint_options.max_classes),
                    std::to_string(options.num_classes),
                    "Raise the explicit limit or reduce class count.");
  }
  const auto count_bytes = checked_counts_bytes(options.num_classes);
  if (count_bytes > checkpoint_options.max_counts_bytes) {
    throw_exhausted("options.max_counts_bytes",
                    "at least " + std::to_string(count_bytes),
                    std::to_string(checkpoint_options.max_counts_bytes),
                    "Raise the explicit state limit or reduce class count.");
  }
  const auto topology = encode_topology(descriptor_);
  if (topology.size() > checkpoint_options.max_topology_bytes) {
    throw_exhausted("options.max_topology_bytes",
                    "at least " + std::to_string(topology.size()),
                    std::to_string(checkpoint_options.max_topology_bytes),
                    "Raise the explicit topology limit.");
  }
  matrix_.assign(options.num_classes,
                 std::vector<std::int64_t>(options.num_classes, 0));
}

ConfusionMatrixMetric::~ConfusionMatrixMetric() = default;

void ConfusionMatrixMetric::update(const Tensor& predictions,
                                   const Tensor& targets) {
  if (!predictions.defined() || !targets.defined()) {
    throw_invalid("update.inputs", "defined prediction and target tensors",
                  "undefined", "Provide both tensors.");
  }
  if (!predictions.is_floating_point()) {
    throw_invalid("update.predictions.dtype", "a floating logits tensor",
                  c10::toString(predictions.scalar_type()),
                  "Provide finite class scores before argmax.");
  }
  if (!integral_labels(targets.scalar_type())) {
    throw_invalid("update.targets.dtype", "an integral label tensor",
                  c10::toString(targets.scalar_type()),
                  "Convert labels explicitly to an integer dtype.");
  }
  if (predictions.dim() != 2 ||
      predictions.size(1) != static_cast<std::int64_t>(classes())) {
    throw_invalid("update.predictions.shape",
                  "{batch," + std::to_string(classes()) + "}",
                  std::to_string(predictions.dim()) + " dimensions",
                  "Use the fixed descriptor class topology.");
  }
  if (targets.numel() != predictions.size(0)) {
    throw_invalid("update.targets.shape",
                  std::to_string(predictions.size(0)) + " labels",
                  std::to_string(targets.numel()),
                  "Provide one target for every prediction row.");
  }
  if (predictions.numel() == 0) {
    return;
  }
  if (!torch::isfinite(predictions).all().item<bool>()) {
    throw NumericalError(
        "Classification logits must be finite",
        ErrorDetails{.component = kComponent,
                     .operation = "update.predictions",
                     .expected = "only finite logits",
                     .actual = "NaN or infinity",
                     .hint = "Validate model output before metric update."});
  }

  const auto predicted =
      predictions.argmax(1).to(torch::kCPU, torch::kInt64).contiguous();
  const auto expected =
      targets.reshape({-1}).to(torch::kCPU, torch::kInt64).contiguous();
  const auto predicted_values = predicted.accessor<std::int64_t, 1>();
  const auto expected_values = expected.accessor<std::int64_t, 1>();
  std::vector<std::vector<std::int64_t>> delta(
      classes(), std::vector<std::int64_t>(classes(), 0));
  std::int64_t accepted = 0;
  for (std::int64_t index = 0; index < expected.numel(); ++index) {
    const auto target = expected_values[index];
    if (descriptor_.ignore_index.has_value() &&
        target == *descriptor_.ignore_index) {
      continue;
    }
    if (target < 0 ||
        target >= static_cast<std::int64_t>(descriptor_.num_classes)) {
      throw_invalid(
          "update.targets.range",
          "labels in [0," + std::to_string(descriptor_.num_classes) + ")",
          std::to_string(target),
          "Fix labels or configure an explicit ignore_index.");
    }
    const auto prediction = predicted_values[index];
    auto& cell = delta[static_cast<std::size_t>(target)]
                      [static_cast<std::size_t>(prediction)];
    ++cell;
    ++accepted;
  }
  if (accepted == 0) {
    return;
  }
  const auto current_total = total_observations();
  if (accepted > std::numeric_limits<std::int64_t>::max() - current_total) {
    throw_exhausted("update.total", "an int64 observation count", "overflow",
                    "Reset or checkpoint and begin a new metric identity.");
  }

  auto candidate = matrix_;
  for (std::size_t row = 0; row < classes(); ++row) {
    for (std::size_t column = 0; column < classes(); ++column) {
      if (delta[row][column] >
          std::numeric_limits<std::int64_t>::max() - candidate[row][column]) {
        throw_exhausted("update.counts", "an int64 confusion count", "overflow",
                        "Start a new metric identity.");
      }
      candidate[row][column] += delta[row][column];
    }
  }
  matrix_.swap(candidate);
}

void ConfusionMatrixMetric::reset() noexcept {
  for (auto& row : matrix_) {
    std::fill(row.begin(), row.end(), 0);
  }
}

void ConfusionMatrixMetric::merge(const ConfusionMatrixMetric& other) {
  if (encode_topology(descriptor_) != encode_topology(other.descriptor_)) {
    throw_incompatible("merge.descriptor", descriptor_.metric_id,
                       other.descriptor_.metric_id,
                       "Merge only metrics with exactly equal semantics.");
  }
  const auto other_total = other.total_observations();
  const auto current_total = total_observations();
  if (other_total > std::numeric_limits<std::int64_t>::max() - current_total) {
    throw_exhausted("merge.total", "an int64 observation count", "overflow",
                    "Start a new reduced metric identity.");
  }
  auto candidate = matrix_;
  for (std::size_t row = 0; row < classes(); ++row) {
    for (std::size_t column = 0; column < classes(); ++column) {
      const auto addition = other.matrix_[row][column];
      if (addition >
          std::numeric_limits<std::int64_t>::max() - candidate[row][column]) {
        throw_exhausted("merge.counts", "an int64 confusion count", "overflow",
                        "Start a new reduced metric identity.");
      }
      candidate[row][column] += addition;
    }
  }
  matrix_.swap(candidate);
}

std::vector<double> ConfusionMatrixMetric::per_class_scores() const {
  if (total_observations() == 0) {
    throw NotReady(
        "Classification metric has no observations",
        ErrorDetails{.component = kComponent,
                     .operation = "compute",
                     .expected = "at least one non-ignored observation",
                     .actual = "0",
                     .hint = "Call update() before computing the metric."});
  }
  std::vector<double> result;
  result.reserve(classes());
  for (std::size_t class_index = 0; class_index < classes(); ++class_index) {
    long double predicted_positive = 0.0L;
    long double actual_positive = 0.0L;
    for (std::size_t other = 0; other < classes(); ++other) {
      predicted_positive +=
          static_cast<long double>(matrix_[other][class_index]);
      actual_positive += static_cast<long double>(matrix_[class_index][other]);
    }
    const auto true_positive =
        static_cast<long double>(matrix_[class_index][class_index]);
    result.push_back(score_from_counts(descriptor_.score, true_positive,
                                       predicted_positive - true_positive,
                                       actual_positive - true_positive,
                                       descriptor_.zero_division, class_index));
  }
  return result;
}

double ConfusionMatrixMetric::compute_aggregate() const {
  if (descriptor_.average == ClassificationAverage::none) {
    throw InvalidArgument(
        "Scalar compute is unavailable for average=none",
        ErrorDetails{.component = kComponent,
                     .operation = "compute.average",
                     .expected = "micro, macro, or weighted",
                     .actual = "none",
                     .hint = "Call per_class_scores() for class-wise values."});
  }
  if (total_observations() == 0) {
    throw NotReady(
        "Classification metric has no observations",
        ErrorDetails{.component = kComponent,
                     .operation = "compute",
                     .expected = "at least one non-ignored observation",
                     .actual = "0",
                     .hint = "Call update() before computing the metric."});
  }
  if (descriptor_.average == ClassificationAverage::micro) {
    long double true_positive = 0.0L;
    long double false_positive = 0.0L;
    long double false_negative = 0.0L;
    for (std::size_t class_index = 0; class_index < classes(); ++class_index) {
      true_positive +=
          static_cast<long double>(matrix_[class_index][class_index]);
      for (std::size_t other = 0; other < classes(); ++other) {
        if (other != class_index) {
          false_positive +=
              static_cast<long double>(matrix_[other][class_index]);
          false_negative +=
              static_cast<long double>(matrix_[class_index][other]);
        }
      }
    }
    return score_from_counts(descriptor_.score, true_positive, false_positive,
                             false_negative, descriptor_.zero_division, 0);
  }

  const auto class_scores = per_class_scores();
  if (descriptor_.average == ClassificationAverage::macro) {
    long double sum = 0.0L;
    for (const auto score : class_scores) {
      sum += score;
    }
    return static_cast<double>(sum /
                               static_cast<long double>(class_scores.size()));
  }

  long double weighted_sum = 0.0L;
  long double weight_total = 0.0L;
  for (std::size_t class_index = 0; class_index < classes(); ++class_index) {
    std::int64_t support = 0;
    for (const auto count : matrix_[class_index]) {
      support += count;
    }
    if (support > 0) {
      weighted_sum += static_cast<long double>(class_scores[class_index]) *
                      static_cast<long double>(support);
      weight_total += static_cast<long double>(support);
    }
  }
  return static_cast<double>(weighted_sum / weight_total);
}

const std::vector<std::vector<std::int64_t>>&
ConfusionMatrixMetric::confusion_matrix() const noexcept {
  return matrix_;
}

std::size_t ConfusionMatrixMetric::classes() const noexcept {
  return descriptor_.num_classes;
}

std::int64_t ConfusionMatrixMetric::total_observations() const noexcept {
  std::int64_t total = 0;
  for (const auto& row : matrix_) {
    for (const auto count : row) {
      total += count;
    }
  }
  return total;
}

const ClassificationMetricDescriptor& ConfusionMatrixMetric::descriptor()
    const noexcept {
  return descriptor_;
}

std::string ConfusionMatrixMetric::participant_id() const {
  return descriptor_.metric_id;
}

StateManifest ConfusionMatrixMetric::describe_state() const {
  return {.participant_id = descriptor_.metric_id,
          .schema_version = 1,
          .restore_level = RestoreLevel::exact,
          .fields = {{.name = kTopologyField,
                      .media_type = kTopologyMediaType,
                      .max_bytes = checkpoint_options_.max_topology_bytes},
                     {.name = kCountsField,
                      .media_type = kCountsMediaType,
                      .max_bytes = checkpoint_options_.max_counts_bytes}},
          .degradation_reasons = {}};
}

void ConfusionMatrixMetric::save_state(StateWriter& writer) const {
  const auto topology = encode_topology(descriptor_);
  const auto counts = encode_counts(matrix_);
  if (topology.size() > checkpoint_options_.max_topology_bytes ||
      counts.size() > checkpoint_options_.max_counts_bytes) {
    throw_exhausted("save_state", "state within configured limits", "oversized",
                    "Raise explicit limits before constructing the metric.");
  }
  writer.write_blob(kTopologyField, std::span<const std::byte>(
                                        topology.data(), topology.size()));
  writer.write_blob(kCountsField,
                    std::span<const std::byte>(counts.data(), counts.size()));
}

void ConfusionMatrixMetric::load_state(const StateReader& reader,
                                       RestorePolicy policy) {
  const auto expected_manifest = describe_state();
  validate_manifest_shape(reader.manifest(), expected_manifest, policy);
  const auto stored_topology =
      reader.read_blob(kTopologyField, checkpoint_options_.max_topology_bytes);
  const auto expected_topology = encode_topology(descriptor_);
  if (stored_topology.size() != expected_topology.size() ||
      !std::equal(stored_topology.begin(), stored_topology.end(),
                  expected_topology.begin())) {
    throw_incompatible("restore.topology", descriptor_.metric_id,
                       "different canonical descriptor",
                       "Restore state from exactly equal metric semantics.");
  }
  auto candidate = decode_counts(
      reader.read_blob(kCountsField, checkpoint_options_.max_counts_bytes),
      classes());
  matrix_.swap(candidate);
}

Precision::Precision(ClassificationMetricOptions options,
                     ClassificationMetricCheckpointOptions checkpoint_options)
    : ConfusionMatrixMetric("precision", kPrecisionTypeId,
                            ClassificationScore::precision, std::move(options),
                            checkpoint_options) {}

double Precision::compute() { return compute_aggregate(); }

Recall::Recall(ClassificationMetricOptions options,
               ClassificationMetricCheckpointOptions checkpoint_options)
    : ConfusionMatrixMetric("recall", kRecallTypeId,
                            ClassificationScore::recall, std::move(options),
                            checkpoint_options) {}

double Recall::compute() { return compute_aggregate(); }

F1Score::F1Score(ClassificationMetricOptions options,
                 ClassificationMetricCheckpointOptions checkpoint_options)
    : ConfusionMatrixMetric("f1", kF1TypeId, ClassificationScore::f1,
                            std::move(options), checkpoint_options) {}

double F1Score::compute() { return compute_aggregate(); }

}  // namespace jyppx::training::metrics
