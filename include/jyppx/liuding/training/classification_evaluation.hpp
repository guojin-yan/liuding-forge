// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_TRAINING_CLASSIFICATION_EVALUATION_HPP
#define JYPPX_LIUDING_TRAINING_CLASSIFICATION_EVALUATION_HPP
/** @file classification_evaluation.hpp
 * @brief Streaming single-label inference and report-ready classification.
 */

#include <jyppx/liuding/foundation/artifact.hpp>
#include <jyppx/liuding/foundation/telemetry.hpp>
#include <jyppx/liuding/nn/model_adapter.hpp>
#include <jyppx/liuding/training/metrics/classification.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <vector>

namespace jyppx::liuding::training {

/** @brief Fixed task semantics and logical resource limits for one run. */
struct ClassificationEvaluationOptions {
  std::size_t num_classes = 2;  ///< Output columns, including absent classes.
  std::optional<std::int64_t> ignore_index;  ///< Labels excluded from counts.
  metrics::ZeroDivisionPolicy zero_division =
      metrics::ZeroDivisionPolicy::zero;  ///< Included empty denominators.
  std::vector<std::string> sources;  ///< Unique explicit source slice names.
  std::uint64_t max_batches = 1'000'000;  ///< Nonempty batches before EOF.
  std::uint64_t max_samples = 1'000'000;  ///< Rows, including ignored labels.
  std::size_t max_batch_samples = 65536;  ///< Maximum rows per forward call.
  std::size_t max_classes = 4096;         ///< Maximum fixed class topology.
  std::size_t max_sources = 64;           ///< Maximum configured source slices.
  std::size_t max_source_bytes = 256;  ///< Maximum bytes per ASCII slice name.
  std::uint64_t max_confusion_bytes =
      128ULL << 20U;  ///< Known matrix payload, including update/report copies.
};

/** @brief One borrowed inference batch; no prediction history is retained. */
struct ClassificationEvaluationBatch {
  nn::Invocation invocation;  ///< Prepared model arguments; no implicit copy.
  Tensor targets;  ///< Nonempty integral vector, same device as model logits.
  std::vector<std::size_t>
      source_indices;  ///< One configured source per row,
                       ///< or empty if no slices configured.
};

/** @brief Per-class sufficient statistics and explicit denominator policy. */
struct ClassificationClassResult {
  std::int64_t support = 0;    ///< Non-ignored true-label count.
  std::int64_t predicted = 0;  ///< Argmax prediction count.
  double precision = 0;        ///< TP / predicted, with configured zero policy.
  double recall = 0;           ///< TP / support, with configured zero policy.
  double f1 = 0;  ///< 2 TP / (predicted + support), with zero policy.
};

/** @brief One dataset/source aggregate; empty slices have missing scalars. */
struct ClassificationEvaluationSummary {
  std::uint64_t samples = 0;       ///< Accepted rows, excluding ignore_index.
  std::optional<double> accuracy;  ///< Top-1, ties choose lowest class index.
  std::optional<double> macro_precision;  ///< Includes every fixed class.
  std::optional<double> macro_recall;     ///< Includes every fixed class.
  std::optional<double> macro_f1;  ///< Mean class F1, not F1 of macro P/R.
  std::vector<ClassificationClassResult> classes;  ///< Class-index order;
                                                   ///< empty for empty slices.
  std::vector<std::vector<std::int64_t>>
      confusion;  ///< Rows=true, cols=argmax.
};

/** @brief Complete inference result, bounded independently of dataset size. */
struct ClassificationEvaluationResult {
  std::uint64_t batches = 0;    ///< Actual nonempty model invocations.
  std::uint64_t rows_seen = 0;  ///< Includes ignored target rows.
  ClassificationEvaluationSummary overall;  ///< All accepted observations.
  std::vector<ClassificationEvaluationSummary> by_source;  ///< Options order.
  std::vector<TelemetryMetricPoint>
      observations;  ///< Four scalars per slice,
                     ///< ready for run_evaluation.
};

/** @brief Supplies the next complete batch or EOF, honoring caller
 * cancellation.
 * @note The source owns loading, preprocessing, device placement and identity.
 */
using ClassificationEvaluationSource =
    std::function<std::optional<ClassificationEvaluationBatch>(
        std::stop_token)>;

/** @brief Runs real model inference and aggregates classification statistics.
 * @param model Borrowed adapter returning finite floating logits shaped N,C.
 * @param source Sequential batch producer; nullopt means complete dataset.
 * @param options Fixed labels, slices, ignore/zero policies and resource caps.
 * @param stop Cooperative cancellation, checked around source and model calls.
 * @param deadline Optional monotonic deadline, checked at the same boundaries.
 * @return Complete result after EOF; partial or cancelled runs never return.
 * @throws InvalidArgument For malformed options, labels, logits or slice IDs.
 * @throws NumericalError For nonfinite logits or an error-policy denominator.
 * @throws ResourceExhausted For a configured logical limit or count overflow.
 * @throws Cancelled If stop was requested; DeadlineExceeded on elapsed time.
 * @note Uses NoGradGuard and eval mode, restoring each original submodule mode
 * on success/failure. Custom train() overrides must support restoring modes.
 * No RNG reset, device transfer, autocast change, process isolation or model
 * mutation rollback is performed. Callers serialize access, keep the module
 * topology stable, and bind observations to actual dataset/model provenance.
 * Limits bound known matrix payload, not allocator/RSS or model workspace.
 */
ClassificationEvaluationResult evaluate_classification(
    nn::ModelAdapter& model, const ClassificationEvaluationSource& source,
    const ClassificationEvaluationOptions& options = {},
    std::stop_token stop = {}, TelemetryDeadline deadline = std::nullopt);

/** @brief Caller-asserted immutable inputs of a recoverable evaluation.
 * The config digest covers batching, preprocessing, seed and adapter semantics.
 * Digests identify actual content; this object does not verify model/data
 * bytes.
 */
struct ClassificationEvaluationIdentity {
  std::string participant_id = "evaluation.classification";  ///< Registry ID.
  std::string
      evaluation_id;  ///< Stable execution identity, at most 1024 bytes.
  ArtifactDigest dataset_digest;  ///< Complete ordered dataset selection.
  ArtifactDigest model_digest;    ///< Evaluated weights and model contract.
  ArtifactDigest config_digest;   ///< Frozen evaluator/batch configuration.
  std::string build_fingerprint;  ///< Exact supported runtime/build identity.
};

/** @brief Committed progress, including rows whose labels were ignored. */
struct ClassificationEvaluationProgress {
  std::uint64_t batches = 0;    ///< Next zero-based indexed batch to request.
  std::uint64_t rows_seen = 0;  ///< Rows in successfully committed batches.
  bool complete = false;  ///< True only after valid EOF and score validation.
};

/** @brief Seekable deterministic source of complete inference batches.
 * The supplied index is the next committed batch cursor. Repeated requests
 * for that index must return the same batch, including metadata and targets.
 * nullopt is EOF at that index. No source-owned mutable cursor is checkpointed.
 */
using IndexedClassificationEvaluationSource =
    std::function<std::optional<ClassificationEvaluationBatch>(
        std::uint64_t, std::stop_token)>;

/** @brief Atomic batch progress and current-only exact classification state.
 * Retains only confusion counts and counters, never predictions or inputs.
 * Save through CheckpointCoordinator/FileCheckpointStore at committed hooks.
 * Serialize access. Keep model content/topology fixed and inference/source
 * deterministic under the bound build/config; stochastic or mutable model
 * state requires coordinated model/RNG participants and caller quiescence.
 * Full model/RSS/CUDA workspace, source storage and coordinator rollback
 * buffers are outside the known confusion-payload budget.
 */
class ClassificationEvaluationSession final : public CheckpointParticipant {
 public:
  /** @brief Validates identity, fixed semantics and bounds before allocation.
   * @param identity Actual immutable input/build identities; text is printable
   * ASCII, nonempty and at most 1024 bytes (registry ID follows its contract).
   * @param options Scoring, slices and aggregate logical matrix budget.
   * @param expected_rows Optional exact total, including ignored target rows.
   * @throws Error For invalid identities, semantics, sizes or resource limits.
   */
  explicit ClassificationEvaluationSession(
      ClassificationEvaluationIdentity identity,
      ClassificationEvaluationOptions options = {},
      std::optional<std::uint64_t> expected_rows = std::nullopt);
  /** @brief Releases private sufficient statistics. */
  ~ClassificationEvaluationSession() override;
  /** @brief A registered participant has stable ownership/address. */
  ClassificationEvaluationSession(const ClassificationEvaluationSession&) =
      delete;
  /** @brief Mutable evaluation state cannot be assigned. */
  ClassificationEvaluationSession& operator=(
      const ClassificationEvaluationSession&) = delete;
  /** @brief Registry handles must not be invalidated by moving. */
  ClassificationEvaluationSession(ClassificationEvaluationSession&&) = delete;
  /** @brief Registry handles must not be invalidated by move assignment. */
  ClassificationEvaluationSession& operator=(
      ClassificationEvaluationSession&&) = delete;

  /** @brief Runs from committed progress through EOF, then returns final
   * scores.
   * @param model Borrowed fixed model returning finite N,C floating logits.
   * @param source Deterministic indexed batches; retries repeat uncommitted
   * work.
   * @param committed Optional hook after each atomic batch and valid EOF. May
   * save this participant and publish final evidence; must not run/load it.
   * @param stop Cooperative cancellation around source/model/metric work.
   * @param deadline Optional monotonic operation deadline.
   * @return EOF-only result equivalent to the uninterrupted evaluation.
   * @throws Error On invalid inputs, scoring, cancellation or resource limits.
   * @throws std::exception Propagates caller source/model/hook failures.
   * A batch failure leaves all counts/cursors unchanged. A hook failure keeps
   * that committed batch; a new process restores the last durable checkpoint.
   * Re-running complete state never invokes source/model and replays the final
   * hook for acknowledgement recovery. Hooks must use stable idempotency keys.
   * Uses the same no-grad/eval/mode-restoration contract as the free evaluator;
   * arbitrary model/source side effects are not rolled back. Checkpoint save
   * during execution is allowed only inside a committed hook; restore/run are
   * rejected until this call returns. Caller exceptions never return a report.
   */
  ClassificationEvaluationResult run(
      nn::ModelAdapter& model,
      const IndexedClassificationEvaluationSource& source,
      const std::function<void(const ClassificationEvaluationSession&)>&
          committed = {},
      std::stop_token stop = {}, TelemetryDeadline deadline = std::nullopt);
  /** @brief Returns counters at the last atomic commit.
   * @return A detached progress value, without partial-batch state.
   */
  ClassificationEvaluationProgress progress() const noexcept;
  /** @brief Builds final observations only after validated EOF.
   * @return Complete dataset/source scores, derived from saved counts.
   * @throws NotReady If EOF has not been committed.
   */
  ClassificationEvaluationResult result() const;
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
  std::unique_ptr<Impl> impl_;  ///< Immutable contract and committed counts.
};

}  // namespace jyppx::liuding::training
#endif  // JYPPX_LIUDING_TRAINING_CLASSIFICATION_EVALUATION_HPP
