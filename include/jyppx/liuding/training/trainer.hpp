// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_TRAINING_TRAINER_HPP
#define JYPPX_LIUDING_TRAINING_TRAINER_HPP
/**
 * @file trainer.hpp
 * @brief Defines the current-only transactional training-loop coordinator.
 */

#include <jyppx/liuding/data/data_pipeline.hpp>
#include <jyppx/liuding/foundation/checkpoint.hpp>
#include <jyppx/liuding/nn/invocation.hpp>
#include <jyppx/liuding/training/checkpoint_coordinator.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <vector>

namespace jyppx::liuding::core {
class RngCheckpointParticipant;
}  // namespace jyppx::liuding::core

namespace jyppx::liuding::nn {
class ModelAdapter;
class ModelCheckpointParticipant;
}  // namespace jyppx::liuding::nn

namespace jyppx::liuding::optim {
class OptimizerAdapter;
}  // namespace jyppx::liuding::optim

namespace jyppx::liuding::training {

namespace amp {
class GradScaler;
class GradScalerCheckpointParticipant;
}  // namespace amp

class CallbackDispatcher;
class CheckpointCoordinator;
struct CheckpointDescriptor;
class CheckpointStateSink;
class CheckpointStateSource;
class EventOutbox;
class SchedulerAdapter;
class TrainingStrategy;

/**
 * @brief Selects how an incomplete gradient-accumulation tail is handled.
 *
 * Numeric values are persisted by the Trainer participant and must remain
 * stable.
 */
enum class AccumulationTailPolicy : std::uint8_t {
  flush = 1,  ///< Rescales and commits the final partial accumulation window.
  drop = 2,   ///< Clears gradients without applying a partial update.
};

/**
 * @brief Selects the numeric execution policy owned by the Trainer.
 *
 * Numeric values are persisted in Trainer topology and must remain stable.
 * Float16 uses dynamic gradient scaling; bfloat16 modes deliberately do not.
 */
enum class PrecisionMode : std::uint8_t {
  full = 1,           ///< Disables autocast and gradient scaling.
  cpu_bfloat16 = 2,   ///< Enables CPU autocast with bfloat16 operators.
  cuda_float16 = 3,   ///< Enables CUDA float16 autocast and GradScaler.
  cuda_bfloat16 = 4,  ///< Enables CUDA bfloat16 autocast without scaling.
};

/**
 * @brief Identifies the immutable task phase supplied to a step function.
 *
 * Numeric values form part of the public step-function contract.
 */
enum class StepPhase : std::uint8_t {
  training = 1,    ///< The step may create a differentiable scalar loss.
  validation = 2,  ///< The step executes under a no-gradient evaluation guard.
};

/**
 * @brief Identifies the Trainer lifecycle captured by checkpoints and results.
 *
 * A restored `running` state resumes the exact committed data/optimizer
 * boundary. Terminal states cannot start a second run.
 */
enum class TrainerPhase : std::uint8_t {
  created = 1,    ///< No execution domain has been prepared.
  prepared = 2,   ///< Participants are registered and the strategy is ready.
  running = 3,    ///< A fresh or restored run owns the execution domain.
  completed = 4,  ///< The run reached an epoch, step, or callback stop limit.
  cancelled = 5,  ///< Cooperative cancellation ended the run.
  failed = 6,     ///< An exception ended the run after failure publication.
  closed = 7,     ///< Explicit resource finalization completed.
};

/** @brief Reports the non-failure terminal disposition of one fit call. */
enum class FitStatus : std::uint8_t {
  completed = 1,  ///< Configured epoch or optimizer-step limit was reached.
  stopped = 2,    ///< A callback requested a safe-boundary stop.
  cancelled = 3,  ///< The caller's stop token was honored.
};

/** @brief Configures one current-only Trainer instance and its resource limits.
 */
struct TrainerOptions {
  std::string trainer_id =
      "training.trainer";  ///< Stable checkpoint participant identity.
  std::string run_id = "run.default";  ///< Stable callback/event run identity.
  std::string trace_id =
      "trace.default";  ///< Nonempty trace identity propagated to events.
  std::uint64_t max_epochs = 1;  ///< Positive exclusive epoch upper bound.
  std::uint64_t max_steps =
      0;  ///< Positive optimizer-step limit; zero means epoch-limited.
  std::size_t accumulate_steps =
      1;  ///< Positive number of micro-batches per optimizer update.
  AccumulationTailPolicy accumulation_tail =
      AccumulationTailPolicy::flush;  ///< Partial epoch/stop window policy.
  PrecisionMode precision =
      PrecisionMode::full;  ///< Explicit autocast and scaling policy.
  std::optional<double>
      gradient_clip_norm;  ///< Positive finite global norm, or disabled.
  std::string scheduler_metric =
      "validation.loss";  ///< Metric supplied to metric-triggered schedulers.
  RestorePolicy restore_policy =
      RestorePolicy::exact_only;  ///< Minimum accepted participant guarantee.
  bool drain_events =
      true;  ///< Whether the loop synchronously drains its optional outbox.
  std::size_t max_metrics =
      4096;  ///< Maximum metric names retained in loop/checkpoint state.
  std::size_t max_optimizers =
      256;  ///< Maximum disjoint optimizers in one logical update cycle.
  std::size_t max_identifier_bytes =
      4096;  ///< Maximum bytes in one identity or metric name.
  std::uint64_t max_topology_bytes =
      1024ULL * 1024ULL;  ///< Maximum encoded Trainer topology bytes.
  std::uint64_t max_control_bytes =
      16ULL * 1024ULL * 1024ULL;  ///< Maximum mutable loop-state bytes.
  CheckpointCoordinatorOptions checkpoint;  ///< Checkpoint resource limits.
};

/** @brief Carries immutable cursor information to one task step. */
struct StepContext {
  StepPhase phase = StepPhase::training;  ///< Training or validation phase.
  std::uint64_t epoch = 0;                ///< Zero-based logical epoch.
  std::uint64_t global_step = 0;  ///< Optimizer updates committed before step.
  std::uint64_t batch_sequence = 0;  ///< Pipeline batch sequence in the epoch.
  std::size_t micro_step =
      0;  ///< One-based position in the active accumulation window.
};

/**
 * @brief Supplies mapped batch values and execution context to task code.
 *
 * The Trainer does not infer target, loss, or model-call semantics. A step may
 * call `model.invoke(invocation)` once or multiple times and return any
 * schema-validated prediction value.
 */
struct StepInput {
  nn::ModelAdapter& model;           ///< Prepared model invocation adapter.
  const data::BatchEnvelope& batch;  ///< Consumer-committed source batch.
  const nn::Invocation& invocation;  ///< Mapper-produced model arguments.
  const std::optional<c10::IValue>& target;  ///< Optional task target value.
  const StepContext& context;  ///< Immutable run and cursor snapshot.
};

/** @brief Returns task loss, prediction, metrics, and observation cardinality.
 */
struct StepOutput {
  Tensor loss;  ///< Scalar finite training loss; optional during validation.
  std::optional<nn::InvocationResult>
      prediction;  ///< Optional schema-checked model result for consumers.
  std::map<std::string, double>
      metrics;  ///< Finite per-batch observations averaged by sample count.
  std::uint64_t sample_count =
      0;  ///< Positive number of observations represented by this output.
};

/** @brief Converts an opaque data envelope into model invocation arguments. */
using BatchMapper =
    std::function<nn::Invocation(const data::BatchEnvelope& batch)>;

/** @brief Extracts an optional task target from an opaque data envelope. */
using TargetMapper =
    std::function<std::optional<c10::IValue>(const data::BatchEnvelope& batch)>;

/** @brief Implements task-specific forward, loss, and batch metric semantics.
 */
using StepFunction = std::function<StepOutput(const StepInput& input)>;

/**
 * @brief Publishes one participant-consistent checkpoint through an external
 * artifact boundary.
 *
 * The callback owns the storage/container format and must call
 * `CheckpointCoordinator::save()` or an equivalent interop adapter. It runs
 * only when no micro-step or validation aggregate is uncommitted.
 */
using CheckpointAction =
    std::function<void(const CheckpointCoordinator& coordinator)>;

/**
 * @brief Binds one optimizer to its exact state and optional scheduler.
 *
 * Trainer sorts bindings by `OptimizerAdapter::optimizer_id()` before
 * preparation and update. Optimizer parameter sets must be disjoint and their
 * union must cover every trainable model parameter. The checkpoint participant
 * must own the same optimizer; a scheduler, when present, must identify it in
 * its immutable descriptor.
 */
struct OptimizerExecutionBinding {
  std::shared_ptr<optim::OptimizerAdapter>
      optimizer;  ///< Required current optimizer adapter.
  std::shared_ptr<CheckpointParticipant>
      checkpoint;  ///< Required exact algorithm-state participant.
  std::shared_ptr<SchedulerAdapter>
      scheduler;  ///< Optional event-ordered scheduler for this optimizer.
};

/**
 * @brief Binds all shared objects required for one prepared training run.
 *
 * Shared ownership keeps every checkpoint participant alive for the prepared
 * Trainer lifetime. The model and optimizer participants are required so a
 * run cannot advertise exact recovery while omitting primary trainable state.
 */
struct FitRequest {
  std::shared_ptr<nn::ModelAdapter> model;  ///< Schema-checked model boundary.
  std::vector<OptimizerExecutionBinding>
      optimizers;  ///< Nonempty disjoint optimizer execution domain.
  std::shared_ptr<data::BatchPipeline>
      training_data;  ///< Required map- or iterable-style training stream.
  BatchMapper training_batch_mapper;  ///< Required training invocation mapper.
  TargetMapper training_target_mapper;  ///< Optional training target mapper.
  StepFunction training_step;  ///< Required task-specific training operation.
  std::shared_ptr<data::BatchPipeline>
      validation_data;                  ///< Optional validation stream.
  BatchMapper validation_batch_mapper;  ///< Required when validation is set.
  TargetMapper
      validation_target_mapper;  ///< Optional validation target mapper.
  StepFunction validation_step;  ///< Required when validation is set.
  std::shared_ptr<TrainingStrategy>
      strategy;  ///< Required device/backward/update execution owner.
  std::shared_ptr<amp::GradScaler>
      gradient_scaler;  ///< Required only for CUDA float16 training.
  std::shared_ptr<amp::GradScalerCheckpointParticipant>
      gradient_scaler_checkpoint;  ///< Exact state for the configured scaler.
  std::shared_ptr<CallbackDispatcher>
      callbacks;  ///< Optional immutable-event callback transaction owner.
  std::shared_ptr<EventOutbox>
      event_outbox;  ///< Optional bounded post-callback delivery queue.
  std::shared_ptr<nn::ModelCheckpointParticipant>
      model_checkpoint;  ///< Required participant for exact model state.
  std::shared_ptr<core::RngCheckpointParticipant>
      rng_checkpoint;  ///< Required process RNG participant.
  std::vector<std::shared_ptr<CheckpointParticipant>>
      additional_participants;  ///< Metric/scaler or task state participants.
  CheckpointAction
      checkpoint_action;  ///< Optional callback-command publication action.
};

/** @brief Summarizes one successful, stopped, or cancelled fit execution. */
struct FitResult {
  FitStatus status = FitStatus::completed;  ///< Terminal non-failure outcome.
  std::uint64_t completed_epochs = 0;       ///< Fully committed epoch count.
  std::uint64_t global_step = 0;  ///< Committed optimizer update count.
  std::map<std::string, double>
      training_metrics;  ///< Last fully completed training epoch metrics.
  std::map<std::string, double>
      validation_metrics;  ///< Last completed validation metrics, if any.
  std::string reason;  ///< Callback or cancellation rationale, if applicable.
};

/**
 * @brief Coordinates one transactional single-owner training run.
 *
 * The Trainer owns only loop/control state. Model, optimizer, data, strategy,
 * scheduler, callbacks, outbox, RNG, and task state remain independent
 * checkpoint participants registered with one `CheckpointCoordinator` during
 * `prepare()`. Checkpoints are allowed only when the accumulation window is
 * empty and validation is not active. Schema version 3 reads only the current
 * format; no legacy Trainer archive or callback API is accepted.
 */
class Trainer final : public CheckpointParticipant {
 public:
  /**
   * @brief Creates an unprepared training-loop owner.
   * @param options Stable identities, limits, and loop policies.
   * @throws InvalidArgument If an option is malformed or contradictory.
   */
  explicit Trainer(TrainerOptions options = {});

  /** @brief Leaves resource finalization to explicit `close()`. */
  ~Trainer() override;

  Trainer(const Trainer&) = delete;  ///< Loop state has unique ownership.
  Trainer& operator=(const Trainer&) = delete;  ///< Trainer is non-copyable.
  Trainer(Trainer&&) = delete;  ///< Registered self state cannot move safely.
  Trainer& operator=(Trainer&&) = delete;  ///< Trainer is non-movable.

  /**
   * @brief Validates and binds one complete execution/checkpoint domain.
   * @param request Shared model, task, pipeline, and participant objects.
   * @throws Error If topology, ownership, capability, or registration fails.
   *
   * Trainer-owned state remains created on failure. External participants
   * retain the failure guarantees documented by their own `prepare()` calls.
   */
  void prepare(FitRequest request);

  /**
   * @brief Executes or resumes the prepared run until a terminal boundary.
   * @param stop Cooperative cancellation checked across data, events, and work.
   * @return Completed, callback-stopped, or cancelled run summary.
   * @throws Error If task, strategy, callback, sink, or checkpoint work fails.
   */
  FitResult fit(std::stop_token stop = {});

  /**
   * @brief Saves every registered participant at a current safe boundary.
   * @param sink Transactional storage destination owned by interop code.
   * @return Aggregate descriptor committed by the coordinator.
   * @throws NotReady If preparation or consistency requirements are unmet.
   */
  CheckpointDescriptor save_checkpoint(CheckpointStateSink& sink);

  /**
   * @brief Restores the prepared execution domain before fit or resume.
   * @param source Complete participant source with stored descriptors.
   * @param policy Weakest accepted aggregate restore guarantee. `nullopt`
   * selects `TrainerOptions::restore_policy`.
   * @return Aggregate descriptor restored by the coordinator.
   * @throws NotReady If training already advanced after preparation.
   */
  CheckpointDescriptor restore_checkpoint(
      const CheckpointStateSource& source,
      std::optional<RestorePolicy> policy = std::nullopt);

  /**
   * @brief Drains events and closes data and strategy resources explicitly.
   * @param stop Cooperative shutdown cancellation.
   * @throws Error If event delivery, barrier, or strategy finalization fails.
   */
  void close(std::stop_token stop = {});

  /**
   * @brief Returns immutable validated construction options.
   * @return Options retained by this Trainer instance.
   */
  const TrainerOptions& options() const noexcept;

  /**
   * @brief Returns the current loop lifecycle.
   * @return Current prepared, running, terminal, or closed phase.
   */
  TrainerPhase phase() const noexcept;

  /**
   * @brief Returns fully committed optimizer updates.
   * @return Number of optimizer updates committed at safe boundaries.
   */
  std::uint64_t global_step() const noexcept;

  /**
   * @brief Returns fully completed training epochs.
   * @return Number of epochs whose final event and state were committed.
   */
  std::uint64_t completed_epochs() const noexcept;

  /**
   * @brief Returns last fully completed training epoch observations.
   * @return Immutable name-to-value map for the last completed epoch.
   */
  const std::map<std::string, double>& training_metrics() const noexcept;

  /**
   * @brief Returns last completed validation observations.
   * @return Immutable name-to-value map for the last validation pass.
   */
  const std::map<std::string, double>& validation_metrics() const noexcept;

  /**
   * @brief Returns the prepared coordinator for interop checkpoint actions.
   * @return Coordinator reference valid until Trainer destruction.
   * @throws NotReady If `prepare()` has not completed.
   */
  const CheckpointCoordinator& checkpoint_coordinator() const;

  /** @copydoc CheckpointParticipant::participant_id */
  std::string participant_id() const override;

  /** @copydoc CheckpointParticipant::describe_state */
  StateManifest describe_state() const override;

  /** @copydoc CheckpointParticipant::save_state */
  void save_state(StateWriter& writer) const override;

  /** @copydoc CheckpointParticipant::load_state */
  void load_state(const StateReader& reader, RestorePolicy policy) override;

 private:
  /** @brief Hides loop execution, aggregate, registry, and codec state. */
  struct Impl;

  std::unique_ptr<Impl> impl_;  ///< Owned current-only Trainer implementation.
};

}  // namespace jyppx::liuding::training

#endif  // JYPPX_LIUDING_TRAINING_TRAINER_HPP
