// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_TRAINING_SCHEDULER_ADAPTER_HPP
#define JYPPX_TRAINING_SCHEDULER_ADAPTER_HPP

/**
 * @file scheduler_adapter.hpp
 * @brief Binds descriptor-first schedulers to ordered training events.
 */

#include <jyppx/optim/lr_scheduler.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace jyppx::training {

/** @brief Carries one explicitly ordered scheduler lifecycle event. */
struct SchedulerInput {
  optim::SchedulerStepMode trigger =
      optim::SchedulerStepMode::manual;  ///< Lifecycle event being delivered.
  std::uint64_t event_index =
      0;  ///< One-based sequence number for this scheduler.
  std::optional<double>
      metric;  ///< Finite value required only for metric-driven schedulers.
};

/**
 * @brief Enforces scheduler event semantics at the training-loop boundary.
 *
 * The adapter rejects wrong lifecycle events, duplicates, gaps, unexpected
 * metrics, and missing metrics before the scheduler or optimizer can mutate.
 * It also delegates `CheckpointParticipant` directly to the scheduler's exact
 * descriptor-first participant, so training code registers one object for
 * both event delivery and checkpoint coordination.
 *
 * Instances are not internally synchronized. The training loop must serialize
 * event delivery and checkpoint I/O.
 */
class SchedulerAdapter final : public CheckpointParticipant {
 public:
  /**
   * @brief Creates an event and checkpoint adapter for one scheduler tree.
   * @param scheduler Non-null descriptor-first root scheduler.
   * @param checkpoint_options Positive checkpoint resource limits.
   * @throws InvalidArgument If the scheduler or limits are invalid.
   */
  explicit SchedulerAdapter(
      std::shared_ptr<optim::LrScheduler> scheduler,
      optim::LrSchedulerCheckpointOptions checkpoint_options = {});

  /**
   * @brief Returns the only event mode accepted by this adapter.
   * @return Immutable mode declared by the scheduler descriptor.
   */
  optim::SchedulerStepMode mode() const noexcept;

  /**
   * @brief Delivers one strictly ordered training event.
   * @param input Trigger, one-based sequence, and optional monitored metric.
   * @throws InvalidArgument If trigger, index, or metric semantics differ from
   * the scheduler descriptor.
   * @throws ResourceExhausted If the event sequence cannot advance.
   */
  void step(const SchedulerInput& input);

  /**
   * @brief Returns mutable access to the retained scheduler.
   * @return A reference valid for the adapter lifetime.
   */
  optim::LrScheduler& scheduler() noexcept;

  /**
   * @brief Returns read-only access to the retained scheduler.
   * @return A reference valid for the adapter lifetime.
   */
  const optim::LrScheduler& scheduler() const noexcept;

  /**
   * @brief Returns the scheduler's stable checkpoint identity.
   * @return An owned identifier matching `describe_state()`.
   */
  std::string participant_id() const override;

  /**
   * @brief Describes exact scheduler topology and mutable state fields.
   * @return The delegated descriptor-first scheduler manifest.
   */
  StateManifest describe_state() const override;

  /**
   * @brief Writes complete scheduler state through the delegated participant.
   * @param writer Storage-neutral checkpoint destination.
   */
  void save_state(StateWriter& writer) const override;

  /**
   * @brief Restores scheduler state transactionally.
   * @param reader Storage-neutral checkpoint source.
   * @param policy Weakest restore guarantee accepted by the caller.
   */
  void load_state(const StateReader& reader, RestorePolicy policy) override;

 private:
  std::shared_ptr<optim::LrScheduler>
      scheduler_;  ///< Scheduler retained for training event delivery.
  optim::LrSchedulerCheckpointParticipant
      participant_;  ///< Exact checkpoint delegate for the same scheduler.
};

}  // namespace jyppx::training

#endif  // JYPPX_TRAINING_SCHEDULER_ADAPTER_HPP
