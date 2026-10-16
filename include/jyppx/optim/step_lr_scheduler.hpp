// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_OPTIM_STEP_LR_SCHEDULER_HPP
#define JYPPX_OPTIM_STEP_LR_SCHEDULER_HPP

/**
 * @file step_lr_scheduler.hpp
 * @brief Defines the descriptor-first StepLR scheduler.
 */

#include <jyppx/optim/lr_scheduler.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace jyppx::optim {

/**
 * @brief Configures a descriptor-first StepLR scheduler.
 */
struct StepLrSchedulerOptions {
  std::string scheduler_id =
      "scheduler.main";  ///< Stable checkpoint participant identity.
  std::uint64_t step_size =
      1;  ///< Positive number of completed scheduler steps per decay.
  double gamma = 0.1;  ///< Finite non-negative decay multiplier.
};

/**
 * @brief Applies periodic multiplicative learning-rate decay.
 *
 * Calling `step()` increments the completed-step counter and multiplies all
 * current rates by `gamma` whenever the new count is divisible by
 * `step_size`. The generic base preflights the transition and commits all
 * optimizer groups atomically.
 */
class StepLrScheduler final : public LrScheduler {
 public:
  /**
   * @brief Creates a StepLR scheduler for a stable optimizer topology.
   * @param optimizer Non-null rate-control implementation retained by sharing.
   * @param options Stable scheduler identity and validated decay options.
   * @throws InvalidArgument If identity, topology, rates, or options are
   * invalid.
   */
  explicit StepLrScheduler(std::shared_ptr<OptimizerRateControl> optimizer,
                           StepLrSchedulerOptions options = {});

  /** @brief Releases StepLR-specific state. */
  ~StepLrScheduler() override;

 private:
  /**
   * @brief Computes the next StepLR counter and rates without mutation.
   * @param learning_rates Valid current or intermediate group rates.
   * @return Complete candidate state and candidate rates.
   */
  detail::SchedulerTransition preview_step_impl(
      std::span<const double> learning_rates) const override;

  /**
   * @brief Captures the StepLR counter as a leaf state node.
   * @return Fully owned leaf state containing the committed event count.
   */
  detail::SchedulerState capture_state_impl() const override;

  /**
   * @brief Returns the committed StepLR event count.
   * @return Number of successfully committed StepLR events.
   */
  std::uint64_t completed_steps_impl() const noexcept override;

  /**
   * @brief Requires a leaf state node before checkpoint commit.
   * @param state Candidate StepLR leaf state.
   */
  void validate_state_impl(const detail::SchedulerState& state) const override;

  /**
   * @brief Commits a prevalidated StepLR counter without throwing.
   * @param state Candidate accepted by `validate_state_impl()`.
   */
  void apply_state_impl(const detail::SchedulerState& state) noexcept override;

  StepLrSchedulerOptions options_;     ///< Immutable decay configuration.
  std::uint64_t completed_steps_ = 0;  ///< Successfully committed events.
};

}  // namespace jyppx::optim

#endif  // JYPPX_OPTIM_STEP_LR_SCHEDULER_HPP
