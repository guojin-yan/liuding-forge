// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_TRAINING_DETECTION_EVALUATION_HPP
#define JYPPX_LIUDING_TRAINING_DETECTION_EVALUATION_HPP
/** @file detection_evaluation.hpp
 * @brief Actual model inference to dataset-level bbox AP and quality
 * observations.
 */
#include <jyppx/liuding/foundation/telemetry.hpp>
#include <jyppx/liuding/nn/model_adapter.hpp>
#include <jyppx/liuding/training/metrics/detection.hpp>

#include <functional>
#include <stop_token>

namespace jyppx::liuding::training {
/** @brief Fixed task/data identity and explicit resource limits. */
struct DetectionEvaluationOptions {
  metrics::DetectionMetricOptions
      metric;  ///< Required data/class identities and matching rules.
  metrics::DetectionMetricLimits
      limits;  ///< Compact state and matching/compute bounds.
  std::uint64_t max_batches =
      1'000'000;  ///< Maximum nonempty model invocations.
  std::optional<std::uint64_t>
      expected_images;  ///< If set, EOF must have exactly this many unique
                        ///< images.
  std::uint64_t max_conversion_bytes =
      128ULL << 20U;  ///< Known CPU tensor/prediction staging per batch.
  std::uint64_t max_summary_values =
      1'000'000;  ///< Class/IoU scalar slots across all returned summaries.
};
/** @brief Borrowed model invocation and owned complete ground-truth images. */
struct DetectionEvaluationBatch {
  nn::Invocation invocation;  ///< Caller-prepared input and device placement.
  std::vector<metrics::DetectionImage>
      images;  ///< Predictions must be empty; metadata/targets are explicit.
};
/** @brief Complete EOF-only result; no raw prediction/box history is returned.
 */
struct DetectionEvaluationResult {
  std::uint64_t batches = 0;          ///< Actual model invocation count.
  metrics::DetectionSummary overall;  ///< Entire declared dataset selection.
  std::vector<metrics::DetectionSummary>
      by_source;  ///< Configured source order.
  std::vector<TelemetryMetricPoint>
      observations;  ///< Macro/class AP for existing quality report APIs.
};
/** @brief Sequential complete batches; nullopt denotes dataset EOF. */
using DetectionEvaluationSource =
    std::function<std::optional<DetectionEvaluationBatch>(std::stop_token)>;
/** @brief Runs no-grad/eval inference and computes offline dataset box AP.
 * @param model Adapter returning a list of (boxes[D,4], scores[D], labels[D])
 * tuples, one per image.
 * @param source Complete ground truth with stable image/source IDs and explicit
 * EOF.
 * @param options Immutable scoring/provenance contract and local limits.
 * @param stop Cooperative cancellation checked around source, inference and
 * matching.
 * @param deadline Optional monotonic deadline checked at those boundaries.
 * @return Complete dataset/source/class AP and quality observations.
 * @throws InvalidArgument For options, duplicate IDs, targets, tensor output or
 * class/source errors.
 * @throws NumericalError For configured no-ground-truth error policy.
 * @throws ResourceExhausted For batch/state/workspace/result limits.
 * @throws Cancelled For cooperative cancellation.
 * @throws DeadlineExceeded For expired deadlines.
 * Floating boxes/scores and integer uint8/int8/int16/int32/int64 labels must
 * share a CPU/CUDA device per image. Noncontiguous strided outputs are valid;
 * conversion synchronizes to bounded CPU FP64/int64 staging before matching.
 * No threshold/NMS/clipping/coordinate transform is implicit. Callers bind
 * actual dataset/model/build identities, serialize model access and keep
 * topology stable. Each original submodule mode is restored on success/error;
 * custom train() must support restoration. Does not reset RNG, alter autocast,
 * undo model side effects, persist an evaluation cursor, or bound full RSS,
 * model/CUDA allocator memory or telemetry metadata. Partial reports are never
 * returned on errors, early cancellation, deadlines or expected-size mismatch.
 */
DetectionEvaluationResult evaluate_detection(
    nn::ModelAdapter& model, const DetectionEvaluationSource& source,
    const DetectionEvaluationOptions& options, std::stop_token stop = {},
    TelemetryDeadline deadline = std::nullopt);
}  // namespace jyppx::liuding::training
#endif  // JYPPX_LIUDING_TRAINING_DETECTION_EVALUATION_HPP
