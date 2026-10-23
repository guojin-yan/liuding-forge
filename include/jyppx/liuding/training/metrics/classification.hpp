// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_TRAINING_METRICS_CLASSIFICATION_HPP
#define JYPPX_LIUDING_TRAINING_METRICS_CLASSIFICATION_HPP
/**
 * @file classification.hpp
 * @brief Defines mergeable confusion-count classification metrics.
 */

#include <jyppx/liuding/foundation/checkpoint.hpp>
#include <jyppx/liuding/training/metric.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::liuding::training::metrics {

/**
 * @brief Selects how per-class scores become the published metric value.
 *
 * Numeric values are persisted in metric topology and must remain stable.
 */
enum class ClassificationAverage : std::uint8_t {
  micro = 1,     ///< Aggregates all class counts before division.
  macro = 2,     ///< Averages equally weighted per-class scores.
  weighted = 3,  ///< Weights each per-class score by true-class support.
  none = 4,      ///< Publishes only `per_class_scores()`; scalar compute fails.
};

/**
 * @brief Returns the stable lowercase name of a classification average.
 * @param average Average policy to render.
 * @return A process-lifetime name, or `"unknown"` for an invalid value.
 */
std::string_view classification_average_name(
    ClassificationAverage average) noexcept;

/**
 * @brief Selects a deterministic result for a zero score denominator.
 *
 * Numeric values are persisted in metric topology and must remain stable.
 */
enum class ZeroDivisionPolicy : std::uint8_t {
  zero = 1,   ///< Produces zero for an included zero-denominator class.
  one = 2,    ///< Produces one for an included zero-denominator class.
  error = 3,  ///< Throws `NumericalError` instead of inventing a score.
};

/**
 * @brief Returns the stable lowercase name of a zero-division policy.
 * @param policy Policy to render.
 * @return A process-lifetime name, or `"unknown"` for an invalid value.
 */
std::string_view zero_division_policy_name(ZeroDivisionPolicy policy) noexcept;

/**
 * @brief Identifies the scalar derived from a confusion matrix.
 *
 * Numeric values are persisted in metric topology and must remain stable.
 */
enum class ClassificationScore : std::uint8_t {
  precision = 1,  ///< True positives divided by predicted positives.
  recall = 2,     ///< True positives divided by actual positives.
  f1 = 3,         ///< Harmonic mean derived from TP, FP, and FN counts.
};

/**
 * @brief Returns the stable lowercase name of a classification score.
 * @param score Score kind to render.
 * @return A process-lifetime name, or `"unknown"` for an invalid value.
 */
std::string_view classification_score_name(ClassificationScore score) noexcept;

/** @brief Configures immutable classification metric semantics. */
struct ClassificationMetricOptions {
  /** Stable checkpoint identity; empty selects a score-specific default. */
  std::string metric_id;
  std::size_t num_classes = 2;  ///< Fixed positive class topology.
  ClassificationAverage average =
      ClassificationAverage::macro;  ///< Published aggregation policy.
  ZeroDivisionPolicy zero_division =
      ZeroDivisionPolicy::zero;  ///< Explicit zero-denominator behavior.
  std::optional<std::int64_t>
      ignore_index;  ///< Target label excluded before range validation.
};

/** @brief Configures checkpoint and allocation resource limits. */
struct ClassificationMetricCheckpointOptions {
  std::size_t max_classes = 4096;  ///< Positive maximum fixed class count.
  std::uint64_t max_topology_bytes =
      64ULL * 1024ULL;  ///< Maximum canonical topology blob size.
  std::uint64_t max_counts_bytes =
      128ULL * 1024ULL *
      1024ULL;  ///< Maximum flattened int64 confusion-state size.
};

/**
 * @brief Describes exact metric identity, topology, and aggregation semantics.
 */
struct ClassificationMetricDescriptor {
  std::uint32_t schema_version = 1;  ///< Positive descriptor schema version.
  std::string type_id;    ///< Stable algorithm and semantic schema identity.
  std::string metric_id;  ///< Stable checkpoint participant identity.
  ClassificationScore score =
      ClassificationScore::precision;  ///< Derived score definition.
  std::size_t num_classes = 0;         ///< Fixed rows and columns in state.
  ClassificationAverage average =
      ClassificationAverage::macro;  ///< Scalar aggregation policy.
  ZeroDivisionPolicy zero_division =
      ZeroDivisionPolicy::zero;  ///< Zero-denominator policy.
  std::optional<std::int64_t>
      ignore_index;  ///< Target label excluded during updates.
};

/**
 * @brief Base for fixed-topology Precision, Recall, and F1 metrics.
 *
 * Updates accept finite floating logits of shape `{batch, num_classes}` and
 * integral targets containing one label per row. Counts use signed int64
 * storage with checked addition. `merge()` combines sufficient statistics,
 * so arbitrary batch partitions and distributed rank partitions produce the
 * same result as one process. The complete topology and count matrix form a
 * current-only exact checkpoint participant.
 *
 * Instances are not internally synchronized. Callers must serialize update,
 * merge, reset, compute, and checkpoint operations.
 */
class ConfusionMatrixMetric : public Metric, public CheckpointParticipant {
 public:
  /** @brief Enables destruction through either public interface. */
  ~ConfusionMatrixMetric() override;

  /** @brief Prevents copying mutable accumulated counts. */
  ConfusionMatrixMetric(const ConfusionMatrixMetric&) = delete;

  /** @brief Prevents assigning mutable accumulated counts. */
  ConfusionMatrixMetric& operator=(const ConfusionMatrixMetric&) = delete;

  /** @brief Prevents moving an object retained by a participant registry. */
  ConfusionMatrixMetric(ConfusionMatrixMetric&&) = delete;

  /** @brief Prevents move assignment of registered metric state. */
  ConfusionMatrixMetric& operator=(ConfusionMatrixMetric&&) = delete;

  /**
   * @brief Accumulates one validated batch of confusion counts atomically.
   * @param predictions Finite floating logits shaped `{N, num_classes}`.
   * @param targets Integral labels containing exactly `N` elements.
   * @throws InvalidArgument If dtype, shape, label, or topology is invalid.
   * @throws NumericalError If logits contain NaN or infinity.
   * @throws ResourceExhausted If an int64 count would overflow.
   */
  void update(const Tensor& predictions, const Tensor& targets) override;

  /** @brief Clears every confusion count without changing topology. */
  void reset() noexcept override;

  /**
   * @brief Merges compatible sufficient statistics atomically.
   * @param other Metric with exactly equal immutable descriptor semantics.
   * @throws CompatibilityError If descriptors differ.
   * @throws ResourceExhausted If a merged int64 count would overflow.
   */
  void merge(const ConfusionMatrixMetric& other);

  /**
   * @brief Returns one score for every fixed class.
   * @return `num_classes` values in stable class-index order.
   * @throws NotReady If no non-ignored observations were accumulated.
   * @throws NumericalError If an included zero denominator uses `error`.
   */
  std::vector<double> per_class_scores() const;

  /**
   * @brief Returns the fixed int64 confusion matrix.
   * @return Rows are true labels and columns are predicted labels.
   */
  const std::vector<std::vector<std::int64_t>>& confusion_matrix()
      const noexcept;

  /**
   * @brief Returns the fixed class topology.
   * @return Positive row and column count.
   */
  std::size_t classes() const noexcept;

  /**
   * @brief Returns the number of accumulated non-ignored observations.
   * @return Checked sum of every confusion count.
   */
  std::int64_t total_observations() const noexcept;

  /**
   * @brief Returns immutable metric topology and policy metadata.
   * @return A reference valid for the metric lifetime.
   */
  const ClassificationMetricDescriptor& descriptor() const noexcept;

  /**
   * @brief Returns the stable checkpoint participant identity.
   * @return An owned identifier matching `describe_state()`.
   */
  std::string participant_id() const override;

  /**
   * @brief Describes canonical topology and flattened int64 counts.
   * @return A validated current-only exact participant manifest.
   */
  StateManifest describe_state() const override;

  /**
   * @brief Writes canonical topology and complete confusion counts.
   * @param writer Storage-neutral checkpoint destination.
   */
  void save_state(StateWriter& writer) const override;

  /**
   * @brief Restores compatible counts after complete preflight validation.
   * @param reader Storage-neutral checkpoint source.
   * @param policy Restore policy that must accept exact state.
   * @throws CompatibilityError If manifest or topology differs.
   * @throws SerializationError If count encoding is malformed.
   * @throws ResourceExhausted If configured limits are exceeded.
   */
  void load_state(const StateReader& reader, RestorePolicy policy) override;

 protected:
  /**
   * @brief Constructs one score-specific confusion metric.
   * @param name Stable scalar value name.
   * @param type_id Stable score algorithm and schema identity.
   * @param score Score derived from accumulated counts.
   * @param options Fixed topology and semantic policies.
   * @param checkpoint_options Allocation and checkpoint limits.
   * @throws InvalidArgument If any identity, enum, topology, or limit is
   * invalid.
   * @throws ResourceExhausted If the requested matrix exceeds its limit.
   */
  ConfusionMatrixMetric(
      std::string name, std::string type_id, ClassificationScore score,
      ClassificationMetricOptions options,
      ClassificationMetricCheckpointOptions checkpoint_options);

  /**
   * @brief Computes the configured scalar aggregation.
   * @return Finite micro, macro, or support-weighted score.
   * @throws InvalidArgument If aggregation is `none`.
   * @throws NotReady If no non-ignored observations were accumulated.
   * @throws NumericalError If zero-division policy is `error`.
   */
  double compute_aggregate() const;

 private:
  ClassificationMetricDescriptor
      descriptor_;  ///< Immutable identity and semantic topology.
  ClassificationMetricCheckpointOptions
      checkpoint_options_;  ///< Immutable state resource limits.
  std::vector<std::vector<std::int64_t>>
      matrix_;  ///< Fixed rows=true, columns=predicted sufficient statistics.
};

/** @brief Computes configurable precision from mergeable confusion counts. */
class Precision final : public ConfusionMatrixMetric {
 public:
  /**
   * @brief Creates a fixed-topology precision metric.
   * @param options Identity, class count, averaging, zero, and ignore policy.
   * @param checkpoint_options State resource limits.
   */
  explicit Precision(
      ClassificationMetricOptions options = {},
      ClassificationMetricCheckpointOptions checkpoint_options = {});

  /**
   * @brief Computes configured precision aggregation.
   * @return Finite scalar precision.
   */
  double compute() override;
};

/** @brief Computes configurable recall from mergeable confusion counts. */
class Recall final : public ConfusionMatrixMetric {
 public:
  /**
   * @brief Creates a fixed-topology recall metric.
   * @param options Identity, class count, averaging, zero, and ignore policy.
   * @param checkpoint_options State resource limits.
   */
  explicit Recall(
      ClassificationMetricOptions options = {},
      ClassificationMetricCheckpointOptions checkpoint_options = {});

  /**
   * @brief Computes configured recall aggregation.
   * @return Finite scalar recall.
   */
  double compute() override;
};

/** @brief Computes configurable F1 from mergeable confusion counts. */
class F1Score final : public ConfusionMatrixMetric {
 public:
  /**
   * @brief Creates a fixed-topology F1 metric.
   * @param options Identity, class count, averaging, zero, and ignore policy.
   * @param checkpoint_options State resource limits.
   */
  explicit F1Score(
      ClassificationMetricOptions options = {},
      ClassificationMetricCheckpointOptions checkpoint_options = {});

  /**
   * @brief Computes configured F1 aggregation.
   * @return Finite scalar F1.
   */
  double compute() override;
};

}  // namespace jyppx::liuding::training::metrics

#endif  // JYPPX_LIUDING_TRAINING_METRICS_CLASSIFICATION_HPP