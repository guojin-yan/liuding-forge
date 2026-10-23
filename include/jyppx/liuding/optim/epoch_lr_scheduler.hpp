// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_OPTIM_EPOCH_LR_SCHEDULER_HPP
#define JYPPX_LIUDING_OPTIM_EPOCH_LR_SCHEDULER_HPP
/**
 * @file epoch_lr_scheduler.hpp
 * @brief Defines descriptor-first epoch learning-rate schedulers.
 */

#include <jyppx/liuding/optim/lr_scheduler.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace jyppx::liuding::optim {

/** @brief Configures multiplicative decay after every completed epoch. */
struct ExponentialLrSchedulerOptions {
  std::string scheduler_id =
      "scheduler.exponential";  ///< Stable checkpoint participant identity.
  double gamma = 0.1;  ///< Finite non-negative multiplier applied each epoch.
};

/**
 * @brief Multiplies every current optimizer-group rate after each epoch.
 *
 * The scheduler transforms the rates supplied by the transaction pipeline, so
 * it composes predictably inside a `ChainedLrScheduler`. Candidate rates and
 * the next counter are committed only after all groups pass validation.
 */
class ExponentialLrScheduler final : public LrScheduler {
 public:
  /**
   * @brief Creates an exponential scheduler for a stable optimizer topology.
   * @param optimizer Non-null atomic optimizer rate-control surface.
   * @param options Stable identity and finite non-negative multiplier.
   * @throws InvalidArgument If identity, topology, rates, or options are
   * invalid.
   */
  explicit ExponentialLrScheduler(
      std::shared_ptr<OptimizerRateControl> optimizer,
      ExponentialLrSchedulerOptions options = {});

  /** @brief Releases exponential scheduler state. */
  ~ExponentialLrScheduler() override;

 private:
  /**
   * @brief Previews one multiplicative epoch transition.
   * @param learning_rates Current or intermediate optimizer-group rates.
   * @return Candidate leaf state and complete replacement rates.
   */
  detail::SchedulerTransition preview_step_impl(
      std::span<const double> learning_rates) const override;

  /**
   * @brief Captures the committed exponential leaf state.
   * @return A leaf containing the completed-epoch counter.
   */
  detail::SchedulerState capture_state_impl() const override;

  /**
   * @brief Returns the number of committed exponential epochs.
   * @return Successfully committed epoch transitions.
   */
  std::uint64_t completed_steps_impl() const noexcept override;

  /**
   * @brief Validates that a restored state is an exponential leaf.
   * @param state Candidate state to validate without mutation.
   */
  void validate_state_impl(const detail::SchedulerState& state) const override;

  /**
   * @brief Commits a prevalidated exponential counter.
   * @param state Candidate accepted by `validate_state_impl()`.
   */
  void apply_state_impl(const detail::SchedulerState& state) noexcept override;

  ExponentialLrSchedulerOptions options_;  ///< Immutable decay configuration.
  std::uint64_t completed_steps_ = 0;      ///< Successfully committed epochs.
};

/** @brief Configures decay at explicit completed-epoch milestones. */
struct MultiStepLrSchedulerOptions {
  std::string scheduler_id =
      "scheduler.multi_step";  ///< Stable checkpoint participant identity.
  std::vector<std::uint64_t>
      milestones;      ///< Positive strictly increasing completed-epoch values.
  double gamma = 0.1;  ///< Finite non-negative multiplier at each milestone.
};

/**
 * @brief Multiplies current rates when a configured epoch is completed.
 *
 * Milestones are immutable, unique, and strictly increasing. The complete
 * list is stored as a typed descriptor value, so checkpoint restore cannot
 * silently continue with a different decay calendar.
 */
class MultiStepLrScheduler final : public LrScheduler {
 public:
  /**
   * @brief Creates a scheduler with explicit completed-epoch milestones.
   * @param optimizer Non-null atomic optimizer rate-control surface.
   * @param options Stable identity, milestones, and decay multiplier.
   * @throws InvalidArgument If identity, topology, rates, or options are
   * invalid.
   */
  MultiStepLrScheduler(std::shared_ptr<OptimizerRateControl> optimizer,
                       MultiStepLrSchedulerOptions options);

  /** @brief Releases multi-step scheduler state. */
  ~MultiStepLrScheduler() override;

 private:
  /**
   * @brief Previews one milestone-aware epoch transition.
   * @param learning_rates Current or intermediate optimizer-group rates.
   * @return Candidate leaf state and complete replacement rates.
   */
  detail::SchedulerTransition preview_step_impl(
      std::span<const double> learning_rates) const override;

  /**
   * @brief Captures the committed multi-step leaf state.
   * @return A leaf containing the completed-epoch counter.
   */
  detail::SchedulerState capture_state_impl() const override;

  /**
   * @brief Returns the number of committed multi-step epochs.
   * @return Successfully committed epoch transitions.
   */
  std::uint64_t completed_steps_impl() const noexcept override;

  /**
   * @brief Validates that a restored state is a multi-step leaf.
   * @param state Candidate state to validate without mutation.
   */
  void validate_state_impl(const detail::SchedulerState& state) const override;

  /**
   * @brief Commits a prevalidated multi-step counter.
   * @param state Candidate accepted by `validate_state_impl()`.
   */
  void apply_state_impl(const detail::SchedulerState& state) noexcept override;

  MultiStepLrSchedulerOptions options_;  ///< Immutable milestone calendar.
  std::uint64_t completed_steps_ = 0;    ///< Successfully committed epochs.
};

/** @brief Configures cosine annealing without warm restarts. */
struct CosineAnnealingLrSchedulerOptions {
  std::string scheduler_id =
      "scheduler.cosine_annealing";  ///< Stable participant identity.
  std::uint64_t t_max =
      1;  ///< Positive epochs from a maximum to the next minimum.
  double eta_min = 0.0;  ///< Finite non-negative minimum group rate.
};

/**
 * @brief Applies recursive cosine annealing to every optimizer group.
 *
 * The recurrence follows the chainable PyTorch cosine schedule: rates reach
 * `eta_min` at `t_max`, then follow the same cosine trajectory upward without
 * resetting mutable state. The implementation uses a numerically stable
 * half-angle weight and a dedicated minimum-boundary transition.
 */
class CosineAnnealingLrScheduler final : public LrScheduler {
 public:
  /**
   * @brief Creates a cosine scheduler for a stable optimizer topology.
   * @param optimizer Non-null atomic optimizer rate-control surface.
   * @param options Stable identity, positive period, and minimum rate.
   * @throws InvalidArgument If identity, topology, rates, or options are
   * invalid.
   */
  explicit CosineAnnealingLrScheduler(
      std::shared_ptr<OptimizerRateControl> optimizer,
      CosineAnnealingLrSchedulerOptions options = {});

  /** @brief Releases cosine scheduler state. */
  ~CosineAnnealingLrScheduler() override;

 private:
  /**
   * @brief Previews one recursive cosine epoch transition.
   * @param learning_rates Current or intermediate optimizer-group rates.
   * @return Candidate leaf state and complete replacement rates.
   */
  detail::SchedulerTransition preview_step_impl(
      std::span<const double> learning_rates) const override;

  /**
   * @brief Captures the committed cosine leaf state.
   * @return A leaf containing the completed-epoch counter.
   */
  detail::SchedulerState capture_state_impl() const override;

  /**
   * @brief Returns the number of committed cosine epochs.
   * @return Successfully committed epoch transitions.
   */
  std::uint64_t completed_steps_impl() const noexcept override;

  /**
   * @brief Validates that a restored state is a cosine leaf.
   * @param state Candidate state to validate without mutation.
   */
  void validate_state_impl(const detail::SchedulerState& state) const override;

  /**
   * @brief Commits a prevalidated cosine counter.
   * @param state Candidate accepted by `validate_state_impl()`.
   */
  void apply_state_impl(const detail::SchedulerState& state) noexcept override;

  CosineAnnealingLrSchedulerOptions
      options_;  ///< Immutable cosine period and minimum rate.
  std::uint64_t completed_steps_ = 0;  ///< Successfully committed epochs.
};

/** @brief Configures cosine annealing with geometrically growing restarts. */
struct CosineAnnealingWarmRestartsLrSchedulerOptions {
  std::string scheduler_id =
      "scheduler.cosine_annealing_warm_restarts";  ///< Stable identity.
  std::uint64_t t_0 = 1;     ///< Positive number of epochs in the first cycle.
  std::uint64_t t_mult = 1;  ///< Positive cycle-length growth multiplier.
  double eta_min = 0.0;      ///< Finite non-negative minimum group rate.
};

/**
 * @brief Applies cosine annealing and resets rates at cycle boundaries.
 *
 * Each completed epoch advances the position before calculating rates. At a
 * boundary, the next cycle starts at every construction-time base rate and
 * its length is multiplied by `t_mult`. The counter and complete cycle state
 * are validated and committed only after the optimizer accepts all rates.
 */
class CosineAnnealingWarmRestartsLrScheduler final : public LrScheduler {
 public:
  /**
   * @brief Creates a restart scheduler for a stable optimizer topology.
   * @param optimizer Non-null atomic optimizer rate-control surface.
   * @param options Stable identity, cycle geometry, and minimum rate.
   * @throws InvalidArgument If identity, topology, rates, or options are
   * invalid.
   */
  explicit CosineAnnealingWarmRestartsLrScheduler(
      std::shared_ptr<OptimizerRateControl> optimizer,
      CosineAnnealingWarmRestartsLrSchedulerOptions options = {});

  /** @brief Releases warm-restart scheduler state. */
  ~CosineAnnealingWarmRestartsLrScheduler() override;

  /**
   * @brief Returns the zero-based current cycle index.
   * @return Number of completed restart boundaries.
   */
  std::uint64_t cycle_index() const noexcept;

  /**
   * @brief Returns the current cycle length in epochs.
   * @return Positive length after applying all completed growth events.
   */
  std::uint64_t cycle_length() const noexcept;

  /**
   * @brief Returns completed epochs within the current cycle.
   * @return Value in `[0, cycle_length())`.
   */
  std::uint64_t cycle_position() const noexcept;

 private:
  /**
   * @brief Previews one restart-aware epoch transition.
   * @param learning_rates Current optimizer rates; topology is validated by
   * the base transaction before this hook runs.
   * @return Candidate cycle state and complete replacement rates.
   */
  detail::SchedulerTransition preview_step_impl(
      std::span<const double> learning_rates) const override;

  /**
   * @brief Captures the complete committed restart state.
   * @return Counter plus typed cycle index, length, and position.
   */
  detail::SchedulerState capture_state_impl() const override;

  /**
   * @brief Returns the number of committed restart events.
   * @return Successfully committed epoch transitions.
   */
  std::uint64_t completed_steps_impl() const noexcept override;

  /**
   * @brief Validates exact cycle-state shape and counter consistency.
   * @param state Candidate state to validate without mutation.
   */
  void validate_state_impl(const detail::SchedulerState& state) const override;

  /**
   * @brief Commits a prevalidated restart state without allocation.
   * @param state Candidate accepted by `validate_state_impl()`.
   */
  void apply_state_impl(const detail::SchedulerState& state) noexcept override;

  CosineAnnealingWarmRestartsLrSchedulerOptions
      options_;  ///< Immutable restart geometry and minimum rate.
  std::uint64_t completed_steps_ = 0;  ///< Successfully committed epochs.
  std::uint64_t cycle_index_ = 0;      ///< Completed restart boundaries.
  std::uint64_t cycle_length_ = 1;     ///< Epochs in the active cycle.
  std::uint64_t cycle_position_ = 0;   ///< Completed active-cycle epochs.
};

}  // namespace jyppx::liuding::optim

#endif  // JYPPX_LIUDING_OPTIM_EPOCH_LR_SCHEDULER_HPP