// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_TRAINING_OPTIMIZER_UPDATE_PARTICIPANT_HPP
#define JYPPX_LIUDING_TRAINING_OPTIMIZER_UPDATE_PARTICIPANT_HPP
/** @file optimizer_update_participant.hpp
 * @brief Checkpointed task state advanced after a complete optimizer cycle.
 */

#include <jyppx/liuding/foundation/checkpoint.hpp>

#include <cstdint>

namespace jyppx::liuding::training {

/** @brief Advances model-derived state exactly once per successful update.
 *
 * Trainer invokes participants in ascending participant ID order, after every
 * optimizer has applied its update and before scheduler/callback publication.
 * Gradient accumulation, dropped tails and AMP overflow skips do not invoke
 * this interface. The counter must equal Trainer::global_step() at every safe
 * boundary and must be restored with the entire execution domain.
 *
 * An update exception fails the run; it does not roll back the already applied
 * optimizer cycle. No checkpoint may be published from that failed run.
 * Implementations and model access are serialized by the Trainer owner.
 */
class OptimizerUpdateParticipant : public CheckpointParticipant {
 public:
  /** @brief Returns the count of fully accepted optimizer update cycles.
   * @return Zero before training, or the last accepted global step.
   */
  virtual std::uint64_t completed_updates() const noexcept = 0;

  /** @brief Applies one consecutive, successful optimizer update observation.
   * @param global_step Exactly completed_updates()+1, never a replay or gap.
   * @throws Error For invalid ordering, topology changes or backend failure.
   */
  virtual void on_optimizer_update(std::uint64_t global_step) = 0;
};

}  // namespace jyppx::liuding::training

#endif  // JYPPX_LIUDING_TRAINING_OPTIMIZER_UPDATE_PARTICIPANT_HPP
