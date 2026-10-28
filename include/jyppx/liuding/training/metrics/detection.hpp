// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_TRAINING_METRICS_DETECTION_HPP
#define JYPPX_LIUDING_TRAINING_METRICS_DETECTION_HPP
/** @file detection.hpp
 * @brief Offline dataset-level axis-aligned box AP with compact exact state.
 */
#include <jyppx/liuding/foundation/checkpoint.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace jyppx::liuding::training::metrics {
/** @brief Policy for a class with no eligible, non-crowd ground truth. */
enum class DetectionNoGroundTruth : std::uint8_t {
  exclude = 1,  ///< Missing class AP, excluded from macro AP.
  zero = 2,     ///< Include class AP zero, even without predictions.
  error = 3,    ///< Throw NumericalError for a nonempty dataset/slice.
};
/** @brief One positive-area continuous xyxy box; no clipping or +1 pixels. */
using DetectionBox = std::array<double, 4>;
/** @brief Explicit ground truth, in the same coordinate system as predictions.
 */
struct DetectionTarget {
  DetectionBox box{};           ///< Finite x1,y1,x2,y2 with x2>x1 and y2>y1.
  std::size_t class_index = 0;  ///< Fixed contiguous class ID [0,C).
  bool crowd =
      false;  ///< Ignored GT accepting repeated intersection/dt-area matches.
  bool ignore = false;  ///< Ignored non-crowd GT accepts at most one match.
  std::optional<double>
      area;  ///< Finite nonnegative annotation area; default box area.
};
/** @brief One already decoded prediction; row order is its stable tie identity.
 */
struct DetectionPrediction {
  DetectionBox box{};  ///< Finite positive-area continuous xyxy box.
  double score = 0;    ///< Finite ranking score, no implicit threshold or NMS.
  std::size_t class_index = 0;  ///< Fixed contiguous class ID [0,C).
};
/** @brief One complete image; updates must never split an image's detections.
 */
struct DetectionImage {
  std::uint64_t image_id =
      0;  ///< Unique stable ID, independent of batch/merge order.
  std::optional<std::size_t>
      source_index;  ///< Required exactly when sources are configured.
  std::vector<DetectionTarget>
      targets;  ///< Stable annotation order breaks GT ties.
  std::vector<DetectionPrediction>
      predictions;  ///< Stable row order breaks score ties.
};
/** @brief Complete immutable semantics, including caller-declared data
 * identity. */
struct DetectionMetricOptions {
  std::string metric_id = "metric.box_ap";  ///< Stable participant identifier.
  std::string
      dataset_id;  ///< Required immutable dataset/split/filter fingerprint.
  std::string
      class_vocabulary_id;  ///< Required ordered class-mapping identity.
  std::size_t num_classes =
      2;  ///< Fixed class topology, including absent classes.
  std::vector<double> iou_thresholds{
      .5,  .55, .6,  .65, .7,
      .75, .8,  .85, .9,  .95};  ///< Strictly increasing, (0,1], at most 64.
  double min_area =
      0;  ///< Inclusive eligible GT/unmatched-prediction area minimum.
  double max_area =
      1e10;  ///< Inclusive area maximum; GT may supply annotation area.
  std::size_t max_detections_per_image_class =
      100;  ///< Score truncation before matching, per image AND class.
  DetectionNoGroundTruth no_ground_truth =
      DetectionNoGroundTruth::exclude;  ///< Missing-class policy.
  std::vector<std::string>
      sources;  ///< Unique explicit source names; never inferred.
};
/** @brief Logical limits; caller inputs, allocator overhead and RSS are
 * separate. */
struct DetectionMetricLimits {
  std::size_t max_classes = 4096;  ///< Maximum fixed class count.
  std::size_t max_sources = 64;    ///< Maximum named source slices.
  std::uint64_t max_images =
      1'000'000;  ///< Unique images, including empty images.
  std::uint64_t max_predictions_per_image =
      100'000;  ///< Before maxDet truncation.
  std::uint64_t max_targets_per_image =
      100'000;  ///< Includes ignored/crowd GT.
  std::uint64_t max_retained_predictions =
      5'000'000;  ///< Dataset-wide matched records.
  std::uint64_t max_state_bytes =
      256ULL << 20U;  ///< Encoded match state, excluding topology.
  std::uint64_t max_workspace_bytes =
      128ULL << 20U;  ///< Known temporary/result payload per operation.
  std::uint64_t max_match_operations_per_update =
      100'000'000;  ///< Sum keptD*GT*IoU thresholds, before matching.
};
/** @brief Current-only versioned task identity; limits are not semantics. */
struct DetectionMetricDescriptor {
  std::uint32_t schema_version = 1;  ///< Current exact format.
  std::string type_id =
      "jyppx.liuding.metric.axis_aligned_box_ap/v1";  ///< Fixed
                                                      ///< task/matching/101-point
                                                      ///< version.
  DetectionMetricOptions options;  ///< Entire scoring and provenance contract.
};
/** @brief One class result, in configured threshold order. */
struct DetectionClassResult {
  std::uint64_t ground_truth =
      0;  ///< Eligible non-crowd, non-ignored GT count.
  std::uint64_t predictions =
      0;  ///< Retained predictions, including ignored ones.
  std::optional<double> average_precision;  ///< Mean of AP over IoU thresholds.
  std::vector<std::optional<double>>
      ap_by_iou;  ///< 101-point interpolated precision.
  std::vector<std::optional<double>>
      recall_by_iou;  ///< Maximum attained recall.
};
/** @brief Complete dataset or explicit-source aggregate. */
struct DetectionSummary {
  std::uint64_t images = 0;  ///< Selected unique images, including empty ones.
  std::optional<double>
      mean_average_precision;  ///< Macro mean of included classes/IoUs.
  std::vector<DetectionClassResult>
      per_class;  ///< Exactly C entries in class order.
};
/** @brief Exact offline bbox AP; callers serialize access to this participant.
 *
 * Greedy per-image/class matching prioritizes eligible GT, then ignored GT;
 * equal IoU chooses the last GT in annotation order. Crowd uses intersection
 * divided by detection area and accepts multiple matches. Unmatched detections
 * outside the inclusive area range are ignored. Score ties use image ID then
 * prediction row, so disjoint merge/batch order cannot change dataset AP.
 * The 101 recall points use the COCO FP64 grid k * 0.01 and FP64 recall
 * comparisons, including floating-point boundary effects. Retains threshold
 * bitmasks, not boxes or a D*G IoU matrix. Source summaries
 * share these records. Does not provide mask/rotated/pose AP, NMS, collective
 * transport, evaluator cursors, or process-crash recovery.
 */
class DetectionAveragePrecisionMetric final : public CheckpointParticipant {
 public:
  /** @brief Validates semantics/limits before allocating mutable state.
   * @param options Required dataset/class identities and explicit scoring
   * rules.
   * @param limits Local logical state/work/time bounds.
   * @throws InvalidArgument For invalid options/identities/limits.
   * @throws ResourceExhausted If topology exceeds local limits.
   */
  explicit DetectionAveragePrecisionMetric(DetectionMetricOptions options,
                                           DetectionMetricLimits limits = {});
  /** @brief Releases compact match records. */
  ~DetectionAveragePrecisionMetric() override;
  /** @brief Prevents copying registered mutable state. */
  DetectionAveragePrecisionMetric(const DetectionAveragePrecisionMetric&) =
      delete;
  /** @brief Prevents copying registered mutable state. */
  DetectionAveragePrecisionMetric& operator=(
      const DetectionAveragePrecisionMetric&) = delete;
  /** @brief Prevents relocating a registered participant. */
  DetectionAveragePrecisionMetric(DetectionAveragePrecisionMetric&&) = delete;
  /** @brief Prevents relocating a registered participant. */
  DetectionAveragePrecisionMetric& operator=(
      DetectionAveragePrecisionMetric&&) = delete;
  /** @brief Matches complete images atomically; empty spans are no-ops.
   * @param images Unique complete images; duplicates, even empty, are errors.
   * @throws InvalidArgument For IDs, boxes, classes, sources or nonfinite
   * scores.
   * @throws ResourceExhausted Before exceeding local logical limits.
   */
  void update(std::span<const DetectionImage> images);
  /** @brief Computes one dataset-level mean AP; never averages batch AP.
   * @return Macro AP over classes and configured IoUs.
   * @throws NotReady If no images or no included classes.
   * @throws NumericalError For no-GT error policy.
   * @throws ResourceExhausted If compute workspace exceeds its limit.
   */
  double compute() const;
  /** @brief Computes overall or one explicitly configured source, without state
   * copies.
   * @param source_index nullopt for overall; otherwise a configured source ID.
   * @return Per-class AP/recall and macro AP; empty slices remain missing.
   * @throws InvalidArgument For an invalid source ID.
   * @throws NumericalError For no-GT error policy on nonempty selections.
   * @throws ResourceExhausted For known result/sort workspace payload limits.
   */
  DetectionSummary summarize(
      std::optional<std::size_t> source_index = std::nullopt) const;
  /** @brief Atomically merges disjoint images with identical descriptors.
   * @param other Compatible participant; overlapping image IDs are rejected.
   * @throws CompatibilityError For different semantics or data identities.
   * @throws InvalidArgument For overlap, including nonempty self-merge.
   * @throws ResourceExhausted For destination resource limits.
   */
  void merge(const DetectionAveragePrecisionMetric& other);
  /** @brief Releases all accumulated image records, preserving identity. */
  void reset() noexcept;
  /** @brief Returns unique image count.
   * @return Number of complete images accepted.
   */
  std::uint64_t image_count() const noexcept;
  /** @brief Returns exact logical encoded state payload, including header.
   * @return State bytes; does not include topology or allocator overhead.
   */
  std::uint64_t state_bytes() const noexcept;
  /** @brief Returns immutable complete task/data identity.
   * @return Borrowed descriptor valid for this participant's lifetime.
   */
  const DetectionMetricDescriptor& descriptor() const noexcept;
  /** @brief Returns the stable checkpoint participant identity.
   * @return Configured metric_id.
   */
  std::string participant_id() const override;
  /** @brief Describes current-only exact compact matching state.
   * @return Validated manifest, with local storage bounds.
   */
  StateManifest describe_state() const override;
  /** @brief Saves canonical image/class/score order and all match bits.
   * @param writer Storage-neutral bounded destination.
   */
  void save_state(StateWriter& writer) const override;
  /** @brief Validates complete state before replacing current records.
   * @param reader Current-schema manifest and both required blobs.
   * @param policy Restore policy accepting exact state.
   * @throws CompatibilityError For identity/schema/policy mismatch.
   * @throws SerializationError For malformed records/counts/flags/order.
   * @throws ResourceExhausted For destination resource bounds.
   */
  void load_state(const StateReader& reader, RestorePolicy policy) override;

 private:
  /** @brief Private compact per-image matched state and local resource limits.
   */
  struct Impl;
  std::unique_ptr<Impl>
      impl_;  ///< O(images + occupied image/classes + kept predictions).
};
}  // namespace jyppx::liuding::training::metrics
#endif  // JYPPX_LIUDING_TRAINING_METRICS_DETECTION_HPP
