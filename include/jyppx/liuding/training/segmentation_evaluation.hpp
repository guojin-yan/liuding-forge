// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_TRAINING_SEGMENTATION_EVALUATION_HPP
#define JYPPX_LIUDING_TRAINING_SEGMENTATION_EVALUATION_HPP
/** @file segmentation_evaluation.hpp
 * @brief Actual semantic segmentation inference with dataset-level reports.
 */
#include <jyppx/liuding/foundation/telemetry.hpp>
#include <jyppx/liuding/nn/model_adapter.hpp>
#include <jyppx/liuding/training/metrics/segmentation.hpp>

#include <functional>
#include <stop_token>

namespace jyppx::liuding::training {
/** @brief Fixed task and dataset/source resource limits. */
struct SegmentationEvaluationOptions {
  metrics::SegmentationMetricOptions metric;  ///< IoU or Dice and all policies.
  metrics::SegmentationMetricLimits limits;   ///< Per-update metric limits.
  std::vector<std::string> sources;  ///< Explicit per-image source names.
  std::uint64_t max_batches =
      1'000'000;  ///< Maximum nonempty model invocations.
  std::uint64_t max_images = 1'000'000;  ///< Maximum images, including ignored.
  std::uint64_t max_pixels = 1ULL
                             << 40U;  ///< Total target pixels, ignored too.
  std::uint64_t max_counts_bytes = 128ULL
                                   << 20U;  ///< Known count/result payload.
  std::size_t max_sources = 64;  ///< Maximum configured metadata slices.
};
/** @brief Borrowed complete image batch for one model invocation. */
struct SegmentationEvaluationBatch {
  nn::Invocation
      invocation;  ///< Caller-prepared model inputs, no implicit copy.
  Tensor targets;  ///< Integral N,H,W on model output CPU/CUDA device.
  std::vector<std::size_t>
      source_indices;  ///< Per-image IDs, empty without slices.
};
/** @brief One complete dataset/source result; no valid pixels means missing. */
struct SegmentationEvaluationSummary {
  std::int64_t pixels = 0;      ///< Non-ignored pixels, including background.
  std::optional<double> score;  ///< Configured dataset-level aggregate.
  std::vector<metrics::SegmentationClassCounts>
      counts;  ///< All fixed class IDs.
  std::vector<std::optional<double>>
      per_class;  ///< Excluded/absent are missing.
};
/** @brief Complete inference result after EOF only, with no prediction history.
 */
struct SegmentationEvaluationResult {
  std::uint64_t batches = 0;  ///< Actual model invocation count.
  std::uint64_t images = 0;   ///< Input images, including all-ignored images.
  std::uint64_t pixels_seen =
      0;  ///< All input pixels, including ignored pixels.
  SegmentationEvaluationSummary overall;                 ///< Dataset aggregate.
  std::vector<SegmentationEvaluationSummary> by_source;  ///< Configured order.
  std::vector<TelemetryMetricPoint>
      observations;  ///< Scalar/class report inputs.
};
/** @brief Returns the next complete batch or nullopt for EOF.
 * Loading, preprocessing, provenance and device placement belong to the caller.
 */
using SegmentationEvaluationSource =
    std::function<std::optional<SegmentationEvaluationBatch>(std::stop_token)>;

/** @brief Performs real no-grad/eval inference, then dataset-level IoU or Dice.
 * @param model Borrowed adapter returning the configured tensor representation.
 * @param source Sequential complete image batches, honoring cancellation.
 * @param options Fixed scoring semantics, metadata slices and logical limits.
 * @param stop Cooperative cancellation checked around source/model/metric
 * calls.
 * @param deadline Optional monotonic deadline checked at those boundaries.
 * @return Complete dataset result and observations for run_evaluation.
 * @throws InvalidArgument For options, batches, outputs or invalid slice IDs.
 * @throws NumericalError For nonfinite logits or absent-class error policy.
 * @throws ResourceExhausted For resource limits or count overflow.
 * @throws Cancelled If cancellation was requested.
 * @throws DeadlineExceeded If the deadline elapsed.
 * Restores each original submodule mode on success/error; custom train()
 * overrides must support restoration. Callers serialize model access, keep
 * topology stable and bind report data/model/build identities. Does not reset
 * RNG, transfer devices, change autocast, roll back arbitrary model side
 * effects or promise allocator/RSS bounds. Errors never return a partial
 * complete report.
 */
SegmentationEvaluationResult evaluate_semantic_segmentation(
    nn::ModelAdapter& model, const SegmentationEvaluationSource& source,
    const SegmentationEvaluationOptions& options = {},
    std::stop_token stop = {}, TelemetryDeadline deadline = std::nullopt);
}  // namespace jyppx::liuding::training
#endif  // JYPPX_LIUDING_TRAINING_SEGMENTATION_EVALUATION_HPP
