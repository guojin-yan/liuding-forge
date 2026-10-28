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
#include <jyppx/liuding/foundation/artifact.hpp>
#include <jyppx/liuding/foundation/telemetry.hpp>
#include <jyppx/liuding/nn/model_adapter.hpp>
#include <jyppx/liuding/training/metrics/detection.hpp>

#include <functional>
#include <memory>
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

/** @brief Caller-asserted immutable inputs for exact resumed detection.
 * Config covers batching, preprocessing, seed and adapter semantics; metric
 * dataset/class identities must also describe the actual selected data.
 * These declarations do not authenticate model or dataset bytes.
 */
struct DetectionEvaluationIdentity {
  std::string participant_id = "evaluation.detection";  ///< Registry ID.
  std::string
      evaluation_id;  ///< Stable execution ID, at most 1024 ASCII bytes.
  ArtifactDigest dataset_digest;  ///< Complete ordered selection of images.
  ArtifactDigest model_digest;    ///< Fixed weights and model contract.
  ArtifactDigest config_digest;   ///< Frozen evaluator/batch configuration.
  std::string build_fingerprint;  ///< Exact supported runtime/build identity.
};
/** @brief Counters at a fully committed matching boundary. */
struct DetectionEvaluationProgress {
  std::uint64_t batches = 0;  ///< Next zero-based indexed batch to request.
  std::uint64_t images = 0;   ///< Unique images, including empty/crowd images.
  bool complete = false;      ///< Valid EOF and all scoring policies committed.
};
/** @brief Deterministic seekable complete batches; nullopt denotes EOF.
 * Same-index retries must preserve input, image IDs, targets and source IDs.
 * The session does not checkpoint a mutable source-owned cursor.
 */
using IndexedDetectionEvaluationSource =
    std::function<std::optional<DetectionEvaluationBatch>(std::uint64_t,
                                                          std::stop_token)>;
/** @brief Exact resumed offline box AP with compact matching state.
 * Uses the metric's atomic matching update without cloning prior dataset state
 * each batch. Stores matched score/flag records and stable image IDs, never
 * input tensors or boxes. Serialize access and keep model/source deterministic;
 * mutable/stochastic state needs coordinated model/RNG/source participants.
 * Logical state/conversion/workspace bounds do not promise full allocator,
 * model/RSS/VRAM or coordinator rollback bounds. Restore temporarily holds old
 * and candidate match states; checkpoint encoding uses an additional blob.
 */
class DetectionEvaluationSession final : public CheckpointParticipant {
 public:
  /** @brief Validates bound input identity, matching semantics and local
   * limits.
   * @param identity Actual input/build identity; ID/build text is nonempty
   * printable ASCII of at most 1024 bytes; registry ID follows its own
   * contract.
   * @param options Fixed AP/conversion/result limits and optional expected
   * images.
   * @throws Error For invalid identities, policies, limits or topology size.
   */
  DetectionEvaluationSession(DetectionEvaluationIdentity identity,
                             DetectionEvaluationOptions options);
  /** @brief Releases compact sufficient statistics. */
  ~DetectionEvaluationSession() override;
  /** @brief Registered participants have stable ownership and address. */
  DetectionEvaluationSession(const DetectionEvaluationSession&) = delete;
  /** @brief Registered participants cannot be assigned. */
  DetectionEvaluationSession& operator=(const DetectionEvaluationSession&) =
      delete;
  /** @brief Registered participants cannot be relocated. */
  DetectionEvaluationSession(DetectionEvaluationSession&&) = delete;
  /** @brief Registered participants cannot be move assigned. */
  DetectionEvaluationSession& operator=(DetectionEvaluationSession&&) = delete;
  /** @brief Continues matching from committed progress through valid EOF.
   * @param model Borrowed fixed model, with the free evaluator's mode contract.
   * @param source Indexed deterministic batches of complete unique images.
   * @param committed Hook after each matching/cursor commit and EOF; may save
   * and publish with stable idempotency keys, but must not reenter run/load.
   * @param stop Cooperative cancellation at load/inference/matching boundaries.
   * @param deadline Optional monotonic operation deadline.
   * @return Complete dataset/source/class AP and quality observations.
   * @throws Error For input, scoring, state, resource or cancellation failures.
   * @throws std::exception Propagates caller source/model/hook errors.
   * Matching and cursor commit together after full validation; cancellation
   * arriving during atomic matching is observed after that commit. A hook
   * failure retains the committed batch; a new process restores the last
   * durable boundary. Complete state never reads/forwards, but retries the
   * final hook. Save during run is allowed only in that hook; run/load reentry
   * fails. Arbitrary model/source side effects and cross-store transactions are
   * not rolled back or made atomic by this participant.
   */
  DetectionEvaluationResult run(
      nn::ModelAdapter& model, const IndexedDetectionEvaluationSource& source,
      const std::function<void(const DetectionEvaluationSession&)>& committed =
          {},
      std::stop_token stop = {}, TelemetryDeadline deadline = std::nullopt);
  /** @brief Returns detached counters from the last atomic commit.
   * @return Batch/image cursor and validated EOF state.
   */
  DetectionEvaluationProgress progress() const noexcept;
  /** @brief Rebuilds final observations from saved compact matching state.
   * @return The complete result under the bound matching/input contract.
   * @throws NotReady Until validated EOF has been committed.
   */
  DetectionEvaluationResult result() const;
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
  std::unique_ptr<Impl> impl_;  ///< Immutable identity, counters and matches.
};
}  // namespace jyppx::liuding::training
#endif  // JYPPX_LIUDING_TRAINING_DETECTION_EVALUATION_HPP
