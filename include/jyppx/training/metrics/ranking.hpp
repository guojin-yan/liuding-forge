// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_TRAINING_METRICS_RANKING_HPP
#define JYPPX_TRAINING_METRICS_RANKING_HPP

/**
 * @file ranking.hpp
 * @brief Defines exact binary, multiclass, and multilabel ranking metrics.
 */

#include <jyppx/foundation/checkpoint.hpp>
#include <jyppx/training/metric.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::training::metrics {

/**
 * @brief Identifies the scalar derived from a binary score ordering.
 *
 * Numeric values are persisted in metric topology and must remain stable.
 */
enum class BinaryRankingScore : std::uint8_t {
  auroc = 1,              ///< Area under the receiver-operating curve.
  average_precision = 2,  ///< Non-interpolated area under the PR step curve.
};

/**
 * @brief Returns the stable lowercase name of a binary ranking score.
 * @param score Score kind to render.
 * @return A process-lifetime name, or `"unknown"` for an invalid value.
 */
std::string_view binary_ranking_score_name(BinaryRankingScore score) noexcept;

/**
 * @brief Selects the result when a ranking metric lacks a required class.
 *
 * Empty metric state always raises `NotReady`; this policy applies only after
 * at least one non-ignored observation has been accumulated. Numeric values
 * are persisted in metric topology and must remain stable.
 */
enum class UndefinedRankingPolicy : std::uint8_t {
  error = 1,  ///< Throws `NumericalError` for a missing required class.
  zero = 2,   ///< Returns zero for an otherwise undefined scalar metric.
  one = 3,    ///< Returns one for an otherwise undefined scalar metric.
};

/**
 * @brief Returns the stable lowercase name of an undefined-result policy.
 * @param policy Policy to render.
 * @return A process-lifetime name, or `"unknown"` for an invalid value.
 */
std::string_view undefined_ranking_policy_name(
    UndefinedRankingPolicy policy) noexcept;

/** @brief Configures immutable binary ranking semantics. */
struct BinaryRankingMetricOptions {
  /** Stable checkpoint identity; empty selects an algorithm-specific default.
   */
  std::string metric_id;

  /** Result used when the selected score lacks a required target class. */
  UndefinedRankingPolicy undefined_policy = UndefinedRankingPolicy::error;

  /** Target value excluded before binary label range validation. */
  std::optional<std::int64_t> ignore_index;
};

/** @brief Configures exact-observation memory and checkpoint resource limits.
 */
struct ExactRankingResourceOptions {
  /** Maximum fixed class count accepted by a classwise descriptor. */
  std::size_t max_classes = 100'000;

  /** Maximum non-ignored score/label pairs retained by update, merge, or
   * restore. */
  std::size_t max_observations = 1'000'000;

  /** Maximum canonical topology blob size accepted during checkpoint I/O. */
  std::uint64_t max_topology_bytes = 64ULL * 1024ULL;

  /** Maximum encoded observation blob size accepted during checkpoint I/O. */
  std::uint64_t max_observation_bytes = 13'000'016ULL;
};

/**
 * @brief Describes exact binary ranking identity and undefined-result policy.
 */
struct BinaryRankingMetricDescriptor {
  /** Current-only descriptor schema version. */
  std::uint32_t schema_version = 1;

  /** Stable algorithm and semantic schema identity. */
  std::string type_id;

  /** Stable checkpoint participant identity. */
  std::string metric_id;

  /** Scalar ranking score derived from the stored observations. */
  BinaryRankingScore score = BinaryRankingScore::auroc;

  /** Result policy used when required target classes are absent. */
  UndefinedRankingPolicy undefined_policy = UndefinedRankingPolicy::error;

  /** Target value excluded before binary label range validation. */
  std::optional<std::int64_t> ignore_index;
};

/**
 * @brief Contains cumulative binary counts at one distinct score threshold.
 *
 * Points are returned in descending threshold order. Each point includes every
 * observation whose score is greater than or equal to `threshold`; tied scores
 * are processed as one group, so results never depend on input order.
 */
struct BinaryRankingCurvePoint {
  double threshold = 0.0;            ///< Finite threshold represented by point.
  std::uint64_t true_positives = 0;  ///< Positive labels accepted so far.
  std::uint64_t false_positives = 0;  ///< Negative labels accepted so far.
  std::uint64_t true_negatives = 0;   ///< Negative labels below the threshold.
  std::uint64_t false_negatives = 0;  ///< Positive labels below the threshold.

  /**
   * @brief Compares threshold and all cumulative counts exactly.
   * @param other Curve point to compare.
   * @return True when every field has the same value.
   */
  bool operator==(const BinaryRankingCurvePoint& other) const = default;
};

/**
 * @brief Base for exact binary AUROC and Average Precision metrics.
 *
 * Updates accept finite floating scores and integral binary labels, both shaped
 * `{N}` on the same device. Exact observations are retained under explicit
 * limits because ordering and tied-threshold semantics cannot be recovered
 * from a confusion matrix. `merge()` concatenates compatible sufficient state;
 * computation sorts a copy and therefore does not mutate checkpoint state.
 * The canonical topology and every score/label pair form a current-only exact
 * checkpoint participant.
 *
 * Instances are not internally synchronized. Callers must serialize update,
 * merge, reset, compute, curve, and checkpoint operations.
 */
class BinaryRankingMetric : public Metric, public CheckpointParticipant {
 public:
  /** @brief Enables destruction through either public interface. */
  ~BinaryRankingMetric() override;

  /** @brief Prevents copying potentially large accumulated observations. */
  BinaryRankingMetric(const BinaryRankingMetric&) = delete;

  /** @brief Prevents assigning potentially large accumulated observations. */
  BinaryRankingMetric& operator=(const BinaryRankingMetric&) = delete;

  /** @brief Prevents moving an object retained by a participant registry. */
  BinaryRankingMetric(BinaryRankingMetric&&) = delete;

  /** @brief Prevents move assignment of registered metric state. */
  BinaryRankingMetric& operator=(BinaryRankingMetric&&) = delete;

  /**
   * @brief Appends one validated score/label batch atomically.
   * @param predictions Finite floating scores shaped `{N}`.
   * @param targets Integral labels shaped `{N}` with values zero or one.
   * @throws InvalidArgument If dtype, shape, device, or labels are invalid.
   * @throws NumericalError If any score is NaN or infinity.
   * @throws ResourceExhausted If the configured exact-state limit is exceeded.
   */
  void update(const Tensor& predictions, const Tensor& targets) override;

  /** @brief Clears all observations without changing descriptor or limits. */
  void reset() noexcept override;

  /**
   * @brief Merges exactly compatible observations atomically.
   * @param other Metric with equal immutable descriptor semantics.
   * @throws CompatibilityError If descriptors differ.
   * @throws ResourceExhausted If the combined state exceeds explicit limits.
   */
  void merge(const BinaryRankingMetric& other);

  /**
   * @brief Builds cumulative counts for every distinct score threshold.
   * @return Points in descending finite-threshold order with ties grouped.
   * @throws NotReady If no non-ignored observations were accumulated.
   */
  std::vector<BinaryRankingCurvePoint> threshold_curve() const;

  /**
   * @brief Returns the number of retained non-ignored observations.
   * @return Exact score/label pair count.
   */
  std::size_t observations() const noexcept;

  /**
   * @brief Returns the number of retained positive labels.
   * @return Count of target value one.
   */
  std::size_t positive_observations() const noexcept;

  /**
   * @brief Returns the number of retained negative labels.
   * @return Count of target value zero.
   */
  std::size_t negative_observations() const noexcept;

  /**
   * @brief Returns immutable ranking topology and policy metadata.
   * @return A reference valid for the metric lifetime.
   */
  const BinaryRankingMetricDescriptor& descriptor() const noexcept;

  /**
   * @brief Returns the stable checkpoint participant identity.
   * @return An owned identifier matching `describe_state()`.
   */
  std::string participant_id() const override;

  /**
   * @brief Describes canonical topology and bounded exact observations.
   * @return A validated current-only exact participant manifest.
   */
  StateManifest describe_state() const override;

  /**
   * @brief Writes canonical topology and every exact score/label pair.
   * @param writer Storage-neutral checkpoint destination.
   */
  void save_state(StateWriter& writer) const override;

  /**
   * @brief Restores compatible observations after complete preflight checks.
   * @param reader Storage-neutral checkpoint source.
   * @param policy Restore policy that must accept exact state.
   * @throws CompatibilityError If manifest or topology differs.
   * @throws SerializationError If observation encoding is malformed.
   * @throws ResourceExhausted If configured limits are exceeded.
   */
  void load_state(const StateReader& reader, RestorePolicy policy) override;

 protected:
  /**
   * @brief Constructs one algorithm-specific exact ranking metric.
   * @param name Stable scalar value name.
   * @param type_id Stable algorithm and schema identity.
   * @param score Score derived from retained observations.
   * @param options Metric identity, undefined, and ignore policies.
   * @param resource_options Exact-state allocation and checkpoint limits.
   * @throws InvalidArgument If identity, enums, or limits are invalid.
   * @throws ResourceExhausted If topology exceeds its configured limit.
   */
  BinaryRankingMetric(std::string name, std::string type_id,
                      BinaryRankingScore score,
                      BinaryRankingMetricOptions options,
                      ExactRankingResourceOptions resource_options);

  /**
   * @brief Computes the selected scalar over a sorted snapshot.
   * @return Finite AUROC or Average Precision under frozen tie semantics.
   * @throws NotReady If no observations were accumulated.
   * @throws NumericalError If required classes are absent under error policy.
   */
  double compute_ranking_score() const;

 private:
  /** @brief Stores one canonical finite double score and binary target. */
  struct Observation {
    double score = 0.0;  ///< Finite ordering value converted from input dtype.
    bool positive = false;  ///< True when the original target value was one.
  };

  BinaryRankingMetricDescriptor
      descriptor_;  ///< Immutable identity and ranking semantics.
  ExactRankingResourceOptions
      resource_options_;  ///< Immutable exact-state resource limits.
  std::vector<Observation>
      observations_;  ///< Bounded exact score/label state in update order.
};

/** @brief Computes exact binary area under the ROC curve. */
class BinaryAUROC final : public BinaryRankingMetric {
 public:
  /**
   * @brief Creates an exact binary AUROC metric.
   * @param options Identity, undefined-result, and ignore policies.
   * @param resource_options Exact-state allocation and checkpoint limits.
   */
  explicit BinaryAUROC(BinaryRankingMetricOptions options = {},
                       ExactRankingResourceOptions resource_options = {});

  /**
   * @brief Computes tie-aware exact AUROC.
   * @return Probability that a positive outranks a negative, with half credit
   * for ties.
   */
  double compute() override;
};

/** @brief Computes exact binary non-interpolated Average Precision. */
class BinaryAveragePrecision final : public BinaryRankingMetric {
 public:
  /**
   * @brief Creates an exact binary Average Precision metric.
   * @param options Identity, undefined-result, and ignore policies.
   * @param resource_options Exact-state allocation and checkpoint limits.
   */
  explicit BinaryAveragePrecision(
      BinaryRankingMetricOptions options = {},
      ExactRankingResourceOptions resource_options = {});

  /**
   * @brief Computes non-interpolated AP using distinct-score step points.
   * @return Sum of recall increments multiplied by threshold precision.
   */
  double compute() override;
};

/**
 * @brief Selects how independent one-vs-rest class scores are aggregated.
 *
 * Numeric values are persisted in checkpoint topology and must remain stable.
 */
enum class RankingAverage : std::uint8_t {
  micro = 1,     ///< Pools every retained class decision before scoring.
  macro = 2,     ///< Averages class scores with equal weight.
  weighted = 3,  ///< Weights class scores by positive target support.
  none = 4,      ///< Exposes class scores only; scalar compute is unavailable.
};

/**
 * @brief Returns the stable lowercase name of a classwise aggregation.
 * @param average Aggregation to render.
 * @return A process-lifetime name, or `"unknown"` for an invalid value.
 */
std::string_view ranking_average_name(RankingAverage average) noexcept;

/**
 * @brief Identifies the tensor and ignore semantics of a classwise metric.
 *
 * Numeric values are persisted in checkpoint topology and must remain stable.
 */
enum class ClasswiseRankingTask : std::uint8_t {
  multiclass = 1,  ///< One class index per row, expanded one-vs-rest.
  multilabel = 2,  ///< Independent binary target at every row/class position.
};

/**
 * @brief Returns the stable lowercase name of a classwise ranking task.
 * @param task Task to render.
 * @return A process-lifetime name, or `"unknown"` for an invalid value.
 */
std::string_view classwise_ranking_task_name(
    ClasswiseRankingTask task) noexcept;

/** @brief Configures immutable multiclass or multilabel ranking semantics. */
struct ClasswiseRankingMetricOptions {
  /** Stable checkpoint identity; empty selects an algorithm-specific default.
   */
  std::string metric_id;

  /** Fixed number of score columns and one-vs-rest class states. */
  std::size_t num_classes = 2;

  /** Scalar aggregation; `none` requires `per_class_scores()`. */
  RankingAverage average = RankingAverage::macro;

  /** Per-class result used when required target classes are absent. */
  UndefinedRankingPolicy undefined_policy = UndefinedRankingPolicy::error;

  /** Multiclass row or multilabel element excluded before label validation. */
  std::optional<std::int64_t> ignore_index;
};

/** @brief Describes exact classwise ranking identity and aggregation semantics.
 */
struct ClasswiseRankingMetricDescriptor {
  /** Current-only descriptor schema version. */
  std::uint32_t schema_version = 1;

  /** Stable algorithm and semantic schema identity. */
  std::string type_id;

  /** Stable checkpoint participant identity. */
  std::string metric_id;

  /** Scalar ranking score computed for every one-vs-rest class. */
  BinaryRankingScore score = BinaryRankingScore::auroc;

  /** Input tensor and ignore interpretation. */
  ClasswiseRankingTask task = ClasswiseRankingTask::multiclass;

  /** Fixed score-column and one-vs-rest class count. */
  std::size_t num_classes = 2;

  /** Scalar aggregation over classwise scores. */
  RankingAverage average = RankingAverage::macro;

  /** Result policy for a class missing data required by the score. */
  UndefinedRankingPolicy undefined_policy = UndefinedRankingPolicy::error;

  /** Multiclass row or multilabel element excluded before label validation. */
  std::optional<std::int64_t> ignore_index;
};

/**
 * @brief Base for exact multiclass and multilabel AUROC/AP metrics.
 *
 * Multiclass updates accept finite scores shaped `{N,C}` and integral class
 * indices shaped `{N}`. Each accepted row contributes `C` one-vs-rest pairs.
 * Multilabel updates accept finite scores and integral binary targets both
 * shaped `{N,C}`; ignore is applied independently to each target element.
 * Finiteness is validated before ignore filtering so model-output corruption
 * cannot be hidden by target values. Exact score/label pairs are retained under
 * a global resource limit and form a current-only exact participant.
 *
 * Instances are not internally synchronized. Callers must serialize every
 * update, merge, reset, compute, curve, and checkpoint operation.
 */
class ClasswiseRankingMetric : public Metric, public CheckpointParticipant {
 public:
  /** @brief Enables destruction through either public interface. */
  ~ClasswiseRankingMetric() override;

  /** @brief Prevents copying potentially large accumulated observations. */
  ClasswiseRankingMetric(const ClasswiseRankingMetric&) = delete;

  /** @brief Prevents assigning potentially large accumulated observations. */
  ClasswiseRankingMetric& operator=(const ClasswiseRankingMetric&) = delete;

  /** @brief Prevents moving an object retained by a participant registry. */
  ClasswiseRankingMetric(ClasswiseRankingMetric&&) = delete;

  /** @brief Prevents move assignment of registered metric state. */
  ClasswiseRankingMetric& operator=(ClasswiseRankingMetric&&) = delete;

  /**
   * @brief Appends one validated multiclass or multilabel batch atomically.
   * @param predictions Finite floating score matrix shaped `{N,C}`.
   * @param targets Integral class indices `{N}` or binary matrix `{N,C}`.
   * @throws InvalidArgument If dtype, shape, device, or labels are invalid.
   * @throws NumericalError If any score is NaN or infinity.
   * @throws ResourceExhausted If the global exact-pair limit is exceeded.
   */
  void update(const Tensor& predictions, const Tensor& targets) override;

  /**
   * @brief Computes the configured micro, macro, or weighted scalar.
   * @return Exact aggregate AUROC or Average Precision.
   * @throws InvalidArgument If descriptor aggregation is `none`.
   * @throws NotReady If no non-ignored pairs were accumulated.
   * @throws NumericalError Under `error` policy when a class is undefined.
   */
  double compute() override;

  /** @brief Clears every class state without changing descriptor or limits. */
  void reset() noexcept override;

  /**
   * @brief Merges exactly compatible class states atomically.
   * @param other Metric with equal immutable descriptor semantics.
   * @throws CompatibilityError If descriptors differ.
   * @throws ResourceExhausted If combined pairs exceed destination limits.
   */
  void merge(const ClasswiseRankingMetric& other);

  /**
   * @brief Computes every one-vs-rest class score in class-index order.
   * @return Vector containing exactly `num_classes` exact scores.
   * @throws NotReady If no non-ignored pairs were accumulated.
   * @throws NumericalError Under `error` policy when a class is undefined.
   */
  std::vector<double> per_class_scores() const;

  /**
   * @brief Builds one class's cumulative counts at distinct thresholds.
   * @param class_index Zero-based one-vs-rest class index.
   * @return Points in descending finite-threshold order with ties grouped.
   * @throws InvalidArgument If `class_index` is out of range.
   * @throws NotReady If the selected class has no retained observations.
   */
  std::vector<BinaryRankingCurvePoint> threshold_curve(
      std::size_t class_index) const;

  /**
   * @brief Builds the curve over every class decision pooled together.
   * @return Micro curve in descending finite-threshold order.
   * @throws NotReady If no non-ignored pairs were accumulated.
   */
  std::vector<BinaryRankingCurvePoint> micro_threshold_curve() const;

  /**
   * @brief Returns the total retained score/label pair count.
   * @return Global exact-state pair count across all classes.
   */
  std::size_t observations() const noexcept;

  /**
   * @brief Returns retained pairs for one class.
   * @param class_index Zero-based class index.
   * @return Exact pair count for the selected class.
   * @throws InvalidArgument If `class_index` is out of range.
   */
  std::size_t observations(std::size_t class_index) const;

  /**
   * @brief Returns positive target support for one class.
   * @param class_index Zero-based class index.
   * @return Positive label count for the selected class.
   * @throws InvalidArgument If `class_index` is out of range.
   */
  std::size_t positive_observations(std::size_t class_index) const;

  /**
   * @brief Returns immutable task, topology, and aggregation metadata.
   * @return A reference valid for the metric lifetime.
   */
  const ClasswiseRankingMetricDescriptor& descriptor() const noexcept;

  /**
   * @brief Returns the stable checkpoint participant identity.
   * @return An owned identifier matching `describe_state()`.
   */
  std::string participant_id() const override;

  /**
   * @brief Describes canonical topology and bounded exact class state.
   * @return A validated current-only exact participant manifest.
   */
  StateManifest describe_state() const override;

  /**
   * @brief Writes canonical topology and every retained class pair.
   * @param writer Storage-neutral checkpoint destination.
   */
  void save_state(StateWriter& writer) const override;

  /**
   * @brief Restores compatible class state after complete preflight checks.
   * @param reader Storage-neutral checkpoint source.
   * @param policy Restore policy that must accept exact state.
   * @throws CompatibilityError If manifest or topology differs.
   * @throws SerializationError If exact state encoding is malformed.
   * @throws ResourceExhausted If configured limits are exceeded.
   */
  void load_state(const StateReader& reader, RestorePolicy policy) override;

 protected:
  /**
   * @brief Constructs one task- and algorithm-specific classwise metric.
   * @param name Stable scalar value name.
   * @param type_id Stable algorithm and schema identity.
   * @param score Ranking score computed from each class state.
   * @param task Multiclass or multilabel input interpretation.
   * @param options Fixed classes, aggregation, undefined, and ignore policies.
   * @param resource_options Exact-state allocation and checkpoint limits.
   */
  ClasswiseRankingMetric(std::string name, std::string type_id,
                         BinaryRankingScore score, ClasswiseRankingTask task,
                         ClasswiseRankingMetricOptions options,
                         ExactRankingResourceOptions resource_options);

 private:
  /** @brief Stores one canonical finite score and one-vs-rest binary target. */
  struct Observation {
    double score = 0.0;     ///< Finite score converted from input dtype.
    bool positive = false;  ///< True for this one-vs-rest positive class.
  };

  /**
   * @brief Throws when a caller selects an out-of-range class index.
   * @param class_index Candidate zero-based class index.
   */
  void validate_class_index(std::size_t class_index) const;

  ClasswiseRankingMetricDescriptor
      descriptor_;  ///< Immutable task and aggregation semantics.
  ExactRankingResourceOptions
      resource_options_;  ///< Immutable global exact-state limits.
  std::vector<std::vector<Observation>>
      class_observations_;  ///< Bounded exact pairs grouped by class index.
  std::size_t observations_ = 0;  ///< Cached total pair count across classes.
};

/** @brief Computes exact multiclass one-vs-rest AUROC. */
class MulticlassAUROC final : public ClasswiseRankingMetric {
 public:
  /**
   * @brief Creates an exact multiclass one-vs-rest AUROC metric.
   * @param options Fixed classes, aggregation, undefined, and ignore policies.
   * @param resource_options Global exact-pair and checkpoint resource limits.
   */
  explicit MulticlassAUROC(ClasswiseRankingMetricOptions options = {},
                           ExactRankingResourceOptions resource_options = {});
};

/** @brief Computes exact multiclass one-vs-rest Average Precision. */
class MulticlassAveragePrecision final : public ClasswiseRankingMetric {
 public:
  /**
   * @brief Creates an exact multiclass one-vs-rest AP metric.
   * @param options Fixed classes, aggregation, undefined, and ignore policies.
   * @param resource_options Global exact-pair and checkpoint resource limits.
   */
  explicit MulticlassAveragePrecision(
      ClasswiseRankingMetricOptions options = {},
      ExactRankingResourceOptions resource_options = {});
};

/** @brief Computes exact multilabel one-vs-rest AUROC. */
class MultilabelAUROC final : public ClasswiseRankingMetric {
 public:
  /**
   * @brief Creates an exact multilabel AUROC metric.
   * @param options Fixed classes, aggregation, undefined, and ignore policies.
   * @param resource_options Global exact-pair and checkpoint resource limits.
   */
  explicit MultilabelAUROC(ClasswiseRankingMetricOptions options = {},
                           ExactRankingResourceOptions resource_options = {});
};

/** @brief Computes exact multilabel one-vs-rest Average Precision. */
class MultilabelAveragePrecision final : public ClasswiseRankingMetric {
 public:
  /**
   * @brief Creates an exact multilabel AP metric.
   * @param options Fixed classes, aggregation, undefined, and ignore policies.
   * @param resource_options Global exact-pair and checkpoint resource limits.
   */
  explicit MultilabelAveragePrecision(
      ClasswiseRankingMetricOptions options = {},
      ExactRankingResourceOptions resource_options = {});
};

}  // namespace jyppx::training::metrics

#endif  // JYPPX_TRAINING_METRICS_RANKING_HPP
