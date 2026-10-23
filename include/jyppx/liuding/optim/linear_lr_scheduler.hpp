// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_OPTIM_LINEAR_LR_SCHEDULER_HPP
#define JYPPX_LIUDING_OPTIM_LINEAR_LR_SCHEDULER_HPP
/**
 * @file linear_lr_scheduler.hpp
 * @brief Defines descriptor-first linear, warmup, constant, and polynomial
 * learning-rate schedulers.
 */

#include <jyppx/liuding/optim/lr_scheduler.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace jyppx::liuding::optim {

/** @brief Configures linear interpolation between two positive factors. */
struct LinearLrSchedulerOptions {
  std::string scheduler_id =
      "scheduler.linear";  ///< Stable checkpoint participant identity.
  double start_factor =
      1.0 / 3.0;            ///< Positive factor applied during construction.
  double end_factor = 1.0;  ///< Positive factor reached at total_iters.
  std::uint64_t total_iters =
      5;  ///< Number of events before the end factor is held.
};

/**
 * @brief Linearly interpolates group rates from start_factor to end_factor.
 *
 * Construction applies `start_factor` to the captured base rates. Each step
 * advances toward `end_factor`; after `total_iters` events the end factor is
 * held indefinitely. A zero total_iters reaches the end factor on the first
 * step, making the boundary explicit rather than dividing by zero.
 */
class LinearLrScheduler final : public LrScheduler {
 public:
  /**
   * @brief Creates a descriptor-first linear scheduler.
   * @param optimizer Non-null atomic optimizer rate-control surface.
   * @param options Stable identity and positive interpolation options.
   * @throws InvalidArgument If identity, topology, rates, or options are
   * invalid.
   */
  explicit LinearLrScheduler(std::shared_ptr<OptimizerRateControl> optimizer,
                             LinearLrSchedulerOptions options = {});

  /** @brief Releases linear scheduler state. */
  ~LinearLrScheduler() override;

 private:
  /**
   * @brief Previews one interpolation event without mutation.
   * @param learning_rates Current optimizer rates used as the transition base.
   * @return Candidate rates and leaf state for the next event.
   */
  detail::SchedulerTransition preview_step_impl(
      std::span<const double> learning_rates) const override;

  /**
   * @brief Captures the completed-event counter as a leaf state.
   * @return The serializable linear scheduler leaf state.
   */
  detail::SchedulerState capture_state_impl() const override;

  /**
   * @brief Returns the number of committed interpolation events.
   * @return Number of successfully committed interpolation events.
   */
  std::uint64_t completed_steps_impl() const noexcept override;

  /**
   * @brief Validates a restored linear leaf state without mutation.
   * @param state Candidate state decoded from checkpoint storage.
   */
  void validate_state_impl(const detail::SchedulerState& state) const override;

  /**
   * @brief Commits a prevalidated linear counter without throwing.
   * @param state State that passed descriptor and leaf validation.
   */
  void apply_state_impl(const detail::SchedulerState& state) noexcept override;

  LinearLrSchedulerOptions options_;   ///< Immutable interpolation options.
  std::uint64_t completed_steps_ = 0;  ///< Successfully committed events.
};

/** @brief Configures a constant-factor prefix followed by base rates. */
struct ConstantLrSchedulerOptions {
  std::string scheduler_id =
      "scheduler.constant";   ///< Stable checkpoint participant identity.
  double factor = 1.0 / 3.0;  ///< Positive factor used during the prefix.
  std::uint64_t total_iters =
      5;  ///< Number of events before restoring the base rates.
};

/**
 * @brief Holds rates at a factor for a bounded prefix of events.
 *
 * Construction applies `factor`. Steps numbered one through `total_iters`
 * retain that factor, and the following step reaches the captured base rates.
 * A zero total_iters therefore reaches base rates on the first step.
 */
class ConstantLrScheduler final : public LrScheduler {
 public:
  /**
   * @brief Creates a descriptor-first constant-prefix scheduler.
   * @param optimizer Non-null atomic optimizer rate-control surface.
   * @param options Stable identity, positive factor, and prefix length.
   * @throws InvalidArgument If identity, topology, rates, or options are
   * invalid.
   */
  explicit ConstantLrScheduler(std::shared_ptr<OptimizerRateControl> optimizer,
                               ConstantLrSchedulerOptions options = {});

  /** @brief Releases constant-prefix scheduler state. */
  ~ConstantLrScheduler() override;

 private:
  /**
   * @brief Previews one constant-prefix event without mutation.
   * @param learning_rates Current optimizer rates used as the transition base.
   * @return Candidate rates and leaf state for the next event.
   */
  detail::SchedulerTransition preview_step_impl(
      std::span<const double> learning_rates) const override;

  /**
   * @brief Captures the completed-event counter as a leaf state.
   * @return The serializable constant-prefix scheduler leaf state.
   */
  detail::SchedulerState capture_state_impl() const override;

  /**
   * @brief Returns the number of committed prefix events.
   * @return Number of successfully committed prefix events.
   */
  std::uint64_t completed_steps_impl() const noexcept override;

  /**
   * @brief Validates a restored constant-prefix leaf state.
   * @param state Candidate state decoded from checkpoint storage.
   */
  void validate_state_impl(const detail::SchedulerState& state) const override;

  /**
   * @brief Commits a prevalidated constant-prefix counter.
   * @param state State that passed descriptor and leaf validation.
   */
  void apply_state_impl(const detail::SchedulerState& state) noexcept override;

  ConstantLrSchedulerOptions options_;  ///< Immutable prefix options.
  std::uint64_t completed_steps_ = 0;   ///< Successfully committed events.
};

/** @brief Configures polynomial decay toward zero. */
struct PolynomialLrSchedulerOptions {
  std::string scheduler_id =
      "scheduler.polynomial";  ///< Stable checkpoint participant identity.
  std::uint64_t total_iters =
      5;               ///< Positive number of events in the decay interval.
  double power = 1.0;  ///< Finite non-negative polynomial exponent.
};

/**
 * @brief Applies closed-form polynomial decay from base rates toward zero.
 *
 * Construction keeps the captured base rates. At event `n`, the factor is
 * `pow(max(0, 1 - n / total_iters), power)` and remains at its terminal value
 * after the interval. A zero power is valid and intentionally produces a
 * constant factor of one, including at the terminal boundary.
 */
class PolynomialLrScheduler final : public LrScheduler {
 public:
  /**
   * @brief Creates a descriptor-first polynomial scheduler.
   * @param optimizer Non-null atomic optimizer rate-control surface.
   * @param options Stable identity, positive interval, and exponent.
   * @throws InvalidArgument If identity, topology, rates, or options are
   * invalid.
   */
  explicit PolynomialLrScheduler(
      std::shared_ptr<OptimizerRateControl> optimizer,
      PolynomialLrSchedulerOptions options = {});

  /** @brief Releases polynomial scheduler state. */
  ~PolynomialLrScheduler() override;

 private:
  /**
   * @brief Previews one polynomial decay event without mutation.
   * @param learning_rates Current optimizer rates used as the transition base.
   * @return Candidate rates and leaf state for the next event.
   */
  detail::SchedulerTransition preview_step_impl(
      std::span<const double> learning_rates) const override;

  /**
   * @brief Captures the completed-event counter as a leaf state.
   * @return The serializable polynomial scheduler leaf state.
   */
  detail::SchedulerState capture_state_impl() const override;

  /**
   * @brief Returns the number of committed polynomial events.
   * @return Number of successfully committed polynomial events.
   */
  std::uint64_t completed_steps_impl() const noexcept override;

  /**
   * @brief Validates a restored polynomial leaf state.
   * @param state Candidate state decoded from checkpoint storage.
   */
  void validate_state_impl(const detail::SchedulerState& state) const override;

  /**
   * @brief Commits a prevalidated polynomial counter.
   * @param state State that passed descriptor and leaf validation.
   */
  void apply_state_impl(const detail::SchedulerState& state) noexcept override;

  PolynomialLrSchedulerOptions options_;  ///< Immutable decay options.
  std::uint64_t completed_steps_ = 0;     ///< Successfully committed events.
};

/** @brief Configures standalone linear warmup from a positive start factor. */
struct WarmupLrSchedulerOptions {
  std::string scheduler_id =
      "scheduler.warmup";  ///< Stable checkpoint participant identity.
  std::uint64_t warmup_iters =
      1;  ///< Positive number of events reaching the base rates.
  double start_factor = 0.001;  ///< Positive construction-time warmup factor.
};

/**
 * @brief Linearly warms rates from start_factor to the captured base rates.
 *
 * Construction applies `start_factor`, events one through `warmup_iters`
 * interpolate to factor one, and later events hold the base rates. A follow-up
 * scheduler should be composed with `SequentialLrScheduler`; this class does
 * not retain a mutable child or a legacy map state.
 */
class WarmupLrScheduler final : public LrScheduler {
 public:
  /**
   * @brief Creates a descriptor-first standalone warmup scheduler.
   * @param optimizer Non-null atomic optimizer rate-control surface.
   * @param options Stable identity, positive interval, and start factor.
   * @throws InvalidArgument If identity, topology, rates, or options are
   * invalid.
   */
  explicit WarmupLrScheduler(std::shared_ptr<OptimizerRateControl> optimizer,
                             WarmupLrSchedulerOptions options = {});

  /** @brief Releases warmup scheduler state. */
  ~WarmupLrScheduler() override;

 private:
  /**
   * @brief Previews one warmup event without mutation.
   * @param learning_rates Current optimizer rates used as the transition base.
   * @return Candidate rates and leaf state for the next event.
   */
  detail::SchedulerTransition preview_step_impl(
      std::span<const double> learning_rates) const override;

  /**
   * @brief Captures the completed-event counter as a leaf state.
   * @return The serializable warmup scheduler leaf state.
   */
  detail::SchedulerState capture_state_impl() const override;

  /**
   * @brief Returns the number of committed warmup events.
   * @return Number of successfully committed warmup events.
   */
  std::uint64_t completed_steps_impl() const noexcept override;

  /**
   * @brief Validates a restored warmup leaf state.
   * @param state Candidate state decoded from checkpoint storage.
   */
  void validate_state_impl(const detail::SchedulerState& state) const override;

  /**
   * @brief Commits a prevalidated warmup counter.
   * @param state State that passed descriptor and leaf validation.
   */
  void apply_state_impl(const detail::SchedulerState& state) noexcept override;

  WarmupLrSchedulerOptions options_;   ///< Immutable warmup options.
  std::uint64_t completed_steps_ = 0;  ///< Successfully committed events.
};

}  // namespace jyppx::liuding::optim

#endif  // JYPPX_LIUDING_OPTIM_LINEAR_LR_SCHEDULER_HPP