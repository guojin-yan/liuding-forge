// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_TRAINING_SEGMENTATION_EVALUATION_HPP
#define JYPPX_LIUDING_TRAINING_SEGMENTATION_EVALUATION_HPP
/** @file segmentation_evaluation.hpp
 * @brief Actual semantic segmentation inference with dataset-level reports.
 */
#include <jyppx/liuding/foundation/artifact.hpp>
#include <jyppx/liuding/foundation/telemetry.hpp>
#include <jyppx/liuding/nn/model_adapter.hpp>
#include <jyppx/liuding/training/metrics/segmentation.hpp>

#include <functional>
#include <memory>
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

/** @brief Caller-asserted immutable content/build identity for resumed
 * inference. Config covers batching, preprocessing, seed and adapter semantics;
 * this declaration does not authenticate the actual model or dataset bytes.
 */
struct SegmentationEvaluationIdentity {
  std::string participant_id = "evaluation.segmentation";  ///< Registry ID.
  std::string
      evaluation_id;  ///< Stable execution ID, at most 1024 ASCII bytes.
  ArtifactDigest dataset_digest;  ///< Complete ordered dataset selection.
  ArtifactDigest model_digest;  ///< Fixed evaluated weights and model contract.
  ArtifactDigest config_digest;   ///< Frozen evaluator and batch configuration.
  std::string build_fingerprint;  ///< Exact supported runtime/build identity.
};
/** @brief Committed counters; ignored images/pixels remain in the input totals.
 */
struct SegmentationEvaluationProgress {
  std::uint64_t batches = 0;      ///< Next zero-based batch index to request.
  std::uint64_t images = 0;       ///< Images in successfully committed batches.
  std::uint64_t pixels_seen = 0;  ///< All target pixels, including ignored IDs.
  bool complete = false;  ///< Validated EOF and scoring policies committed.
};
/** @brief Seekable deterministic complete batches, with nullopt at EOF.
 * Repeated requests for one index must return identical inputs/labels/sources.
 * No source-owned mutable cursor is checkpointed by the session.
 */
using IndexedSegmentationEvaluationSource =
    std::function<std::optional<SegmentationEvaluationBatch>(std::uint64_t,
                                                             std::stop_token)>;
/** @brief Current-only exact semantic evaluation with atomic batch commits.
 * Retains O(classes * sources) counts, never pixels or predictions. Serialize
 * access and keep model/source deterministic, with fixed content and topology;
 * arbitrary mutable/stochastic producers require coordinated participants.
 * The count budget covers live/candidate/result counts and update delta, not
 * model/source/allocator/RSS/CUDA workspace or coordinator rollback buffers.
 */
class SegmentationEvaluationSession final : public CheckpointParticipant {
 public:
  /** @brief Validates immutable identities, scoring and positive local bounds.
   * @param identity Actual input/build identities; ID/build text is nonempty
   * printable ASCII, at most 1024 bytes; participant ID follows its registry.
   * @param options Fixed scoring, slices and resource contract.
   * @param expected_images Optional exact input total, including ignored
   * images.
   * @param expected_pixels Optional exact input total, including ignored
   * pixels.
   * @throws Error For invalid identities, policies, bounds or topology size.
   */
  explicit SegmentationEvaluationSession(
      SegmentationEvaluationIdentity identity,
      SegmentationEvaluationOptions options = {},
      std::optional<std::uint64_t> expected_images = std::nullopt,
      std::optional<std::uint64_t> expected_pixels = std::nullopt);
  /** @brief Releases owned sufficient statistics. */
  ~SegmentationEvaluationSession() override;
  /** @brief Registered participants have stable ownership and address. */
  SegmentationEvaluationSession(const SegmentationEvaluationSession&) = delete;
  /** @brief Registered participants cannot be assigned. */
  SegmentationEvaluationSession& operator=(
      const SegmentationEvaluationSession&) = delete;
  /** @brief Registered participants cannot be relocated. */
  SegmentationEvaluationSession(SegmentationEvaluationSession&&) = delete;
  /** @brief Registered participants cannot be move assigned. */
  SegmentationEvaluationSession& operator=(SegmentationEvaluationSession&&) =
      delete;
  /** @brief Continues from committed counts/cursor through validated EOF.
   * @param model Borrowed fixed model, using the free evaluator's mode
   * contract.
   * @param source Deterministic indexed source; retries repeat uncommitted
   * work.
   * @param committed Hook after each atomic batch and EOF; may save state and
   * publish results using stable idempotency keys, but must not run/load it.
   * @param stop Cooperative cancellation around loading, inference and metrics.
   * @param deadline Optional monotonic operation deadline.
   * @return EOF-only dataset/source/class observations.
   * @throws Error For input, scoring, resource, cancellation or state errors.
   * @throws std::exception Propagates source/model/hook exceptions.
   * Any batch failure leaves all statistics and counters unchanged. A hook
   * failure keeps the committed in-memory batch; another process restores the
   * last durable boundary. Complete state replays only the final hook, without
   * loading or inference. Save during run is allowed only inside the hook;
   * reentrant run/load is rejected. No arbitrary model/source effects are
   * undone.
   */
  SegmentationEvaluationResult run(
      nn::ModelAdapter& model,
      const IndexedSegmentationEvaluationSource& source,
      const std::function<void(const SegmentationEvaluationSession&)>&
          committed = {},
      std::stop_token stop = {}, TelemetryDeadline deadline = std::nullopt);
  /** @brief Returns a detached copy of the last committed progress.
   * @return Input counters and EOF state, without a partial batch.
   */
  SegmentationEvaluationProgress progress() const noexcept;
  /** @brief Rebuilds final observations from complete sufficient statistics.
   * @return The same result as uninterrupted inference under the bound
   * contract.
   * @throws NotReady Until validated EOF has been committed.
   */
  SegmentationEvaluationResult result() const;
  /** @copydoc CheckpointParticipant::participant_id */
  std::string participant_id() const override;
  /** @copydoc CheckpointParticipant::describe_state */
  StateManifest describe_state() const override;
  /** @copydoc CheckpointParticipant::save_state */
  void save_state(StateWriter& writer) const override;
  /** @copydoc CheckpointParticipant::load_state */
  void load_state(const StateReader& reader, RestorePolicy policy) override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;  ///< Immutable identity and committed counts.
};
}  // namespace jyppx::liuding::training
#endif  // JYPPX_LIUDING_TRAINING_SEGMENTATION_EVALUATION_HPP
