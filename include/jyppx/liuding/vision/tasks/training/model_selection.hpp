// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_VISION_TASKS_TRAINING_MODEL_SELECTION_HPP
#define JYPPX_LIUDING_VISION_TASKS_TRAINING_MODEL_SELECTION_HPP
/** @file model_selection.hpp @brief Checkpointed epoch metric selection.
 */

#include <jyppx/liuding/training/callback_dispatcher.hpp>

#include <cstdint>
#include <string>

namespace jyppx::liuding::vision::tasks {

/** @brief Fixed selection metric, comparison and publication frequency. */
struct ModelSelectionOptions {
  std::string callback_id = "callback.model_selection";  ///< Stable state ID.
  std::string monitor = "validation.loss";  ///< Required epoch metric key.
  bool maximize = false;  ///< False selects lower monitored values.
  double min_delta = 0;   ///< Finite nonnegative strict improvement threshold.
  std::uint64_t save_interval_epochs =
      1;  ///< Request periodic saves; zero requests saves only on improvement.
};

/** @brief Detached selection state; epoch numbers are zero based. */
struct ModelSelectionState {
  std::uint64_t observations = 0;  ///< Completed epoch observations.
  double best_value = 0;           ///< Meaningful only when observations>0.
  std::uint64_t best_epoch = 0;    ///< Epoch of the strict best observation.
  std::uint64_t best_step = 0;     ///< Successful updates at that observation.
  std::uint64_t last_step = 0;     ///< Successful updates at the latest epoch.
  bool last_improved = false;      ///< Latest observation replaced the best.
};

/** @brief Selects the best epoch and requests safe-boundary checkpoints.
 *
 * This callback performs no filesystem I/O. A Trainer checkpoint_action reads
 * state() to publish last/best artifacts through the existing coordinator.
 * State is small enough for dispatcher rollback and resumes with the run.
 * Restoring selection history does not recreate older best-weight artifacts;
 * retain those files separately when resuming into a new output directory.
 */
class ModelSelectionCallback final : public training::CallbackHandler,
                                     public CheckpointParticipant {
 public:
  /** @brief Validates a fixed, bounded selection recipe.
   * @param options Metric, direction, strict threshold and save interval.
   * @throws InvalidArgument For invalid identifiers or comparison options.
   */
  explicit ModelSelectionCallback(ModelSelectionOptions options = {});
  ModelSelectionCallback(const ModelSelectionCallback&) =
      delete;  ///< Unique state.
  ModelSelectionCallback& operator=(const ModelSelectionCallback&) =
      delete;  ///< Non-copyable.
  ModelSelectionCallback(ModelSelectionCallback&&) =
      delete;  ///< Stable identity.
  ModelSelectionCallback& operator=(ModelSelectionCallback&&) =
      delete;  ///< Non-movable.
  /** @copydoc training::CallbackHandler::descriptor */
  const training::CallbackDescriptor& descriptor() const noexcept override;
  /** @brief Accepts one consecutive completed epoch, starting at epoch zero.
   * @param event An epoch_finished training event containing the monitor.
   * @return Checkpoint request on improvement or the configured interval.
   * @throws Error For a missing/nonfinite metric, replay, gap or counter bound.
   */
  training::CallbackCommandRequest handle(
      const training::CallbackEvent& event) override;
  /** @brief Returns the latest committed epoch selection state.
   * @return Detached best/latest counters and the improvement decision.
   */
  ModelSelectionState state() const noexcept;
  /** @copydoc CheckpointParticipant::participant_id */
  std::string participant_id() const override;
  /** @copydoc CheckpointParticipant::describe_state */
  StateManifest describe_state() const override;
  /** @copydoc CheckpointParticipant::save_state */
  void save_state(StateWriter& writer) const override;
  /** @copydoc CheckpointParticipant::load_state */
  void load_state(const StateReader& reader, RestorePolicy policy) override;

 private:
  ModelSelectionOptions options_;  ///< Frozen monitor and save semantics.
  training::CallbackDescriptor descriptor_;  ///< Dispatcher contract.
  std::string recipe_;         ///< Canonical immutable checkpoint recipe.
  ModelSelectionState state_;  ///< Complete recoverable selection history.
};

}  // namespace jyppx::liuding::vision::tasks
#endif  // JYPPX_LIUDING_VISION_TASKS_TRAINING_MODEL_SELECTION_HPP
