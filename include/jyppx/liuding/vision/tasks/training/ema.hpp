// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_VISION_TASKS_TRAINING_EMA_HPP
#define JYPPX_LIUDING_VISION_TASKS_TRAINING_EMA_HPP
/** @file ema.hpp @brief Successful-update model EMA with checkpointed weights.
 */

#include <jyppx/liuding/nn/model_checkpoint_participant.hpp>
#include <jyppx/liuding/training/optimizer_update_participant.hpp>

#include <cstdint>
#include <memory>
#include <string>

namespace jyppx::liuding::vision::tasks {

/** @brief Immutable exponential moving average recipe and storage bounds. */
struct ModelEmaOptions {
  std::string source_model_id =
      "model.main";       ///< Stable source/model identity.
  double decay = 0.9999;  ///< Maximum blend weight, finite in [0,1).
  double ramp_updates =
      2000;  ///< Positive ramp:
             ///< decay*(1-exp(-successful_updates/ramp_updates)).
  nn::ModelCheckpointParticipantOptions checkpoint{
      .participant_id =
          "model.ema"};  ///< Tensor and checkpoint resource limits.
};

/** @brief Tracks an independent, frozen evaluation copy of a training model.
 *
 * Construction copies source parameters/buffers into the supplied shadow.
 * Floating FP32/FP64 state uses the ramped average; integral/bool buffers are
 * copied exactly. Both modules must have identical named tensor topology and
 * fixed CPU/CUDA placement, without shared storage within or between modules.
 * Shadow construction/placement and any RNG isolation are the caller's job.
 * The shadow remains in eval mode with gradients disabled.
 *
 * Register only in FitRequest::optimizer_update_participants; Trainer also
 * registers its checkpoint state. Counts/recipe and all shadow tensors are
 * restored together, using the existing transactional model state codec.
 * Concurrent model access, forward mutation and device/dtype/topology changes
 * after construction are unsupported. A backend update failure fails the run;
 * already applied optimizer/EMA tensor writes are not rolled back.
 */
class ModelEma final : public training::OptimizerUpdateParticipant {
 public:
  /** @brief Binds source/shadow, freezes the shadow and copies initial state.
   * @param source Live model, already placed on its final device.
   * @param shadow Independent adapter of the same architecture and placement.
   * @param options Averaging identity, ramp and model checkpoint bounds.
   * @throws Error For topology, alias, dtype, resource or backend failures.
   */
  ModelEma(std::shared_ptr<nn::ModelAdapter> source,
           std::shared_ptr<nn::ModelAdapter> shadow,
           ModelEmaOptions options = {});
  /** @brief Releases retained model adapters and tensor bindings. */
  ~ModelEma() override;
  ModelEma(const ModelEma&) = delete;  ///< Unique mutable averaging state.
  ModelEma& operator=(const ModelEma&) = delete;  ///< Non-copyable state.
  ModelEma(ModelEma&&) = delete;  ///< Registered address remains stable.
  ModelEma& operator=(ModelEma&&) = delete;  ///< Non-movable state.

  /** @brief Returns the frozen shadow for validation, inference or export.
   * @return Adapter retained for this participant's lifetime.
   */
  nn::ModelAdapter& model() const noexcept;
  /** @brief Returns the immutable averaging and storage recipe.
   * @return Options retained by this instance.
   */
  const ModelEmaOptions& options() const noexcept;
  /** @brief Returns the blend weight for the last completed update.
   * @return Zero before the first update, otherwise the configured ramp value.
   */
  double current_decay() const noexcept;
  /** @copydoc training::OptimizerUpdateParticipant::completed_updates */
  std::uint64_t completed_updates() const noexcept override;
  /** @copydoc training::OptimizerUpdateParticipant::on_optimizer_update */
  void on_optimizer_update(std::uint64_t global_step) override;
  /** @copydoc CheckpointParticipant::participant_id */
  std::string participant_id() const override;
  /** @copydoc CheckpointParticipant::describe_state */
  StateManifest describe_state() const override;
  /** @copydoc CheckpointParticipant::save_state */
  void save_state(StateWriter& writer) const override;
  /** @copydoc CheckpointParticipant::load_state */
  void load_state(const StateReader& reader, RestorePolicy policy) override;

 private:
  struct Impl;
  std::unique_ptr<Impl>
      impl_;  ///< Models, immutable bindings and update count.
};

}  // namespace jyppx::liuding::vision::tasks
#endif  // JYPPX_LIUDING_VISION_TASKS_TRAINING_EMA_HPP
