// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_TRAINING_METRICS_SEGMENTATION_HPP
#define JYPPX_LIUDING_TRAINING_METRICS_SEGMENTATION_HPP
/** @file segmentation.hpp
 * @brief Dataset-level semantic IoU/Dice with linear-size exact state.
 */

#include <jyppx/liuding/foundation/checkpoint.hpp>
#include <jyppx/liuding/training/metrics/classification.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::liuding::training::metrics {

/** @brief Stable persisted single-label semantic segmentation score. */
enum class SegmentationScore : std::uint8_t {
  iou = 1,   ///< Intersection divided by union.
  dice = 2,  ///< Twice intersection divided by true plus predicted support.
};

/** @brief Stable persisted prediction representation. */
enum class SegmentationInput : std::uint8_t {
  logits = 1,  ///< Finite floating N,C,H,W; argmax ties select lowest ID.
  class_indices =
      2,  ///< Integral N,H,W class IDs without implicit thresholding.
};

/** @brief Behavior when a class is absent in both truth and predictions. */
enum class AbsentClassPolicy : std::uint8_t {
  exclude = 1,  ///< Missing per-class value, omitted from macro averaging.
  zero = 2,     ///< Included score zero.
  one = 3,      ///< Included score one.
  error = 4,    ///< NumericalError for an included absent class.
};

/** @brief Returns a stable score name, or unknown for an invalid enum.
 * @param score Persisted score kind.
 * @return Process-lifetime name.
 */
std::string_view segmentation_score_name(SegmentationScore score) noexcept;
/** @brief Returns a stable input name, or unknown for an invalid enum.
 * @param input Persisted representation.
 * @return Process-lifetime name.
 */
std::string_view segmentation_input_name(SegmentationInput input) noexcept;
/** @brief Returns a stable absent-class name, or unknown for an invalid enum.
 * @param policy Persisted absent-class policy.
 * @return Process-lifetime name.
 */
std::string_view absent_class_policy_name(AbsentClassPolicy policy) noexcept;

/** @brief Immutable semantic contract; background exclusion affects scores
 * only. */
struct SegmentationMetricOptions {
  std::string metric_id;  ///< Empty selects metric.semantic_iou/semantic_dice.
  std::size_t num_classes = 2;  ///< Fixed contiguous label IDs [0,C).
  SegmentationScore score = SegmentationScore::iou;  ///< Published score kind.
  SegmentationInput input =
      SegmentationInput::logits;  ///< Required input form.
  ClassificationAverage average =
      ClassificationAverage::macro;  ///< Aggregation.
  AbsentClassPolicy absent =
      AbsentClassPolicy::exclude;  ///< Both supports zero.
  std::optional<std::int64_t>
      ignore_index;                  ///< Target ID removed before counts.
  bool include_background = true;    ///< Include background in scalar averages.
  std::size_t background_index = 0;  ///< Background ID, always inside [0,C).
};

/** @brief Logical resource limits, independent of metric identity. */
struct SegmentationMetricLimits {
  std::size_t max_classes = 4096;  ///< Positive fixed-topology limit.
  std::uint64_t max_state_bytes = 1ULL << 20U;    ///< 8 + 24*C encoded counts.
  std::uint64_t max_batch_pixels = 16ULL << 20U;  ///< Includes ignored pixels.
  std::size_t chunk_pixels =
      65536;  ///< Positive maximum pixels per update tile.
};

/** @brief Exact sufficient statistics for one class, including background. */
struct SegmentationClassCounts {
  std::int64_t intersection = 0;  ///< Pixels with prediction == target == ID.
  std::int64_t target = 0;        ///< Non-ignored true pixels of this class.
  std::int64_t predicted = 0;  ///< Non-ignored predicted pixels of this class.
};

/** @brief Versioned immutable metric descriptor for comparison and reporting.
 */
struct SegmentationMetricDescriptor {
  std::uint32_t schema_version = 1;   ///< Current-only semantic schema.
  std::string type_id;                ///< Versioned task/score identity.
  SegmentationMetricOptions options;  ///< Complete persisted scoring contract.
};

/** @brief Dataset-level single-label semantic IoU/Dice, not instance/mask AP.
 *
 * Counts are int64 with checked totals. Updates use bounded spatial tiles on
 * CPU/CUDA, retain O(C) state and never create pixel-by-class one-hot tensors.
 * Background pixels remain in counts so foreground mistakes remain visible.
 * No autograd graph, per-image scores or full prediction history is retained.
 * Limits bound logical counts and tile size, not allocator/RSS or runtime
 * workspace. Input tensors are borrowed for update only; serialize every
 * operation on the instance. Merge transports no collective communications.
 */
class SemanticSegmentationMetric final : public Metric,
                                         public CheckpointParticipant {
 public:
  /** @brief Validates topology and limits before allocating class counts.
   * @param options Complete scoring semantics.
   * @param limits Positive allocation and per-update limits.
   * @throws InvalidArgument For invalid IDs, enums or zero limits.
   * @throws ResourceExhausted If class/count limits are exceeded.
   */
  explicit SemanticSegmentationMetric(SegmentationMetricOptions options = {},
                                      SegmentationMetricLimits limits = {});
  /** @brief Destroys owned class counts. */
  ~SemanticSegmentationMetric() override = default;
  /** @brief Prevents copying registered mutable state. */
  SemanticSegmentationMetric(const SemanticSegmentationMetric&) = delete;
  /** @brief Prevents assigning registered mutable state. */
  SemanticSegmentationMetric& operator=(const SemanticSegmentationMetric&) =
      delete;
  /** @brief Prevents relocating a registered participant. */
  SemanticSegmentationMetric(SemanticSegmentationMetric&&) = delete;
  /** @brief Prevents move assignment of a registered participant. */
  SemanticSegmentationMetric& operator=(SemanticSegmentationMetric&&) = delete;

  /** @brief Atomically accumulates one image batch; empty batches are no-ops.
   * @param predictions N,C,H,W finite logits or integral N,H,W class IDs.
   * @param targets Integral N,H,W, same CPU/CUDA device as predictions.
   * @throws InvalidArgument For shape, layout, dtype, device or label errors.
   * @throws NumericalError For any nonfinite logit, including ignored pixels.
   * @throws ResourceExhausted For batch pixel limit or int64 count overflow.
   * Predictions at ignored targets are not counted; class-index values at
   * those positions need not be valid. Noncontiguous strided inputs are valid.
   */
  void update(const Tensor& predictions, const Tensor& targets) override;
  /** @brief Computes one dataset score, never an average of per-batch scores.
   * @return Micro, macro or target-support-weighted score.
   * @throws NotReady For no valid pixels or no included scoring denominator.
   * @throws InvalidArgument If average is none.
   * @throws NumericalError If an included absent class uses error policy.
   */
  double compute() override;
  /** @brief Clears every count without changing semantics or allocation. */
  void reset() noexcept override;
  /** @brief Returns class-index scores; background/excluded absent are missing.
   * @return Exactly C values in class-ID order.
   * @throws NotReady If no valid pixels have been accumulated.
   * @throws NumericalError If an included absent class uses error policy.
   */
  std::vector<std::optional<double>> per_class_scores() const;
  /** @brief Adds compatible statistics atomically, including self-merge.
   * @param other Equal descriptor, including participant ID and all semantics.
   * @throws CompatibilityError If descriptors differ.
   * @throws ResourceExhausted If the total would exceed int64.
   */
  void merge(const SemanticSegmentationMetric& other);
  /** @brief Returns borrowed counts valid until update, reset or restore.
   * @return All C class counts, including an excluded background.
   */
  const std::vector<SegmentationClassCounts>& class_counts() const noexcept;
  /** @brief Returns the number of non-ignored pixels accumulated.
   * @return Exact non-negative int64 total, including background.
   */
  std::int64_t total_pixels() const noexcept;
  /** @brief Returns immutable identity for the instance lifetime.
   * @return Versioned complete scoring descriptor.
   */
  const SegmentationMetricDescriptor& descriptor() const noexcept;
  /** @brief Returns the stable participant ID.
   * @return Owned ID matching the descriptor and state manifest.
   */
  std::string participant_id() const override;
  /** @brief Describes topology and little-endian counts, current-only exact.
   * @return Validated exact state manifest.
   */
  StateManifest describe_state() const override;
  /** @brief Writes complete topology and sufficient statistics.
   * @param writer Bounded storage-neutral writer consuming each blob.
   */
  void save_state(StateWriter& writer) const override;
  /** @brief Validates every field and invariant before replacing counts.
   * @param reader Complete current-schema state.
   * @param policy Policy accepting exact state.
   * @throws CompatibilityError For schema, topology, policy or field mismatch.
   * @throws SerializationError For invalid size/counts or missing fields.
   * @throws ResourceExhausted If local read limits are exceeded.
   */
  void load_state(const StateReader& reader, RestorePolicy policy) override;

 private:
  SegmentationMetricDescriptor descriptor_;  ///< Immutable scoring semantics.
  SegmentationMetricLimits limits_;  ///< Local limits, not checkpoint identity.
  std::vector<SegmentationClassCounts> counts_;  ///< O(C) complete state.
  std::int64_t total_ = 0;  ///< Exact sum of all true/predicted support.
};

}  // namespace jyppx::liuding::training::metrics
#endif  // JYPPX_LIUDING_TRAINING_METRICS_SEGMENTATION_HPP
