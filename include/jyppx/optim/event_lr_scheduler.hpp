// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_OPTIM_EVENT_LR_SCHEDULER_HPP
#define JYPPX_OPTIM_EVENT_LR_SCHEDULER_HPP

/**
 * @file event_lr_scheduler.hpp
 * @brief Defines metric-, batch-, and callable-driven LR schedulers.
 */

#include <jyppx/optim/lr_scheduler.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace jyppx::optim {

/** @brief Selects whether lower or higher monitored metrics are better. */
enum class MetricDirection : std::uint8_t {
  minimize = 1,  ///< Treats a sufficiently lower metric as an improvement.
  maximize = 2,  ///< Treats a sufficiently higher metric as an improvement.
};

/** @brief Selects how plateau improvement thresholds are interpreted. */
enum class MetricThresholdMode : std::uint8_t {
  relative = 1,  ///< Scales the threshold by the best observed metric.
  absolute = 2,  ///< Uses the threshold as an absolute metric delta.
};

/** @brief Configures `ReduceLrOnPlateauScheduler`. */
struct ReduceLrOnPlateauSchedulerOptions {
  std::string scheduler_id =
      "scheduler.reduce_on_plateau";  ///< Stable checkpoint participant ID.
  MetricDirection direction =
      MetricDirection::minimize;  ///< Desired monitored metric direction.
  double factor = 0.1;  ///< Multiplicative decay in the open interval (0, 1).
  std::uint64_t patience =
      10;  ///< Non-improving observations tolerated before decay.
  double threshold = 1e-4;  ///< Non-negative improvement threshold.
  MetricThresholdMode threshold_mode =
      MetricThresholdMode::relative;  ///< Threshold interpretation.
  std::uint64_t cooldown = 0;  ///< Observations ignored after each decay event.
  std::vector<double>
      min_learning_rates;  ///< Empty, one broadcast value, or one per group.
  double eps = 1e-8;       ///< Minimum meaningful old-to-new rate difference.
};

/**
 * @brief Reduces rates after a monitored metric stops improving.
 *
 * `step(metric)` is transactional and persists best-value, patience, and
 * cooldown state through the generic scheduler checkpoint participant.
 */
class ReduceLrOnPlateauScheduler final : public LrScheduler {
 public:
  /**
   * @brief Creates a metric-driven plateau scheduler.
   * @param optimizer Non-null atomic optimizer rate-control surface.
   * @param options Immutable plateau and checkpoint identity options.
   * @throws InvalidArgument If any option or per-group minimum is invalid.
   */
  explicit ReduceLrOnPlateauScheduler(
      std::shared_ptr<OptimizerRateControl> optimizer,
      ReduceLrOnPlateauSchedulerOptions options = {});

  /** @brief Destroys the scheduler. */
  ~ReduceLrOnPlateauScheduler() override;

  /**
   * @brief Returns whether a best metric has been committed.
   * @return True after the first successful metric event.
   */
  bool has_best() const noexcept;

  /**
   * @brief Returns the best committed metric.
   * @return Best value, or zero before the first metric event.
   */
  double best() const noexcept;

  /**
   * @brief Returns consecutive non-improving observations outside cooldown.
   * @return Current bad-observation count.
   */
  std::uint64_t bad_observations() const noexcept;

  /**
   * @brief Returns remaining cooldown observations.
   * @return Current cooldown counter.
   */
  std::uint64_t cooldown_remaining() const noexcept;

 private:
  /**
   * @brief Rejects metric-less preview attempts.
   * @param learning_rates Valid current optimizer group rates.
   * @return This method never returns.
   */
  detail::SchedulerTransition preview_step_impl(
      std::span<const double> learning_rates) const override;

  /**
   * @brief Previews one metric observation without mutation.
   * @param learning_rates Valid current optimizer group rates.
   * @param metric Finite monitored value supplied by the caller.
   * @return Complete candidate plateau state and group rates.
   */
  detail::SchedulerTransition preview_metric_step_impl(
      std::span<const double> learning_rates, double metric) const override;

  /**
   * @brief Captures plateau counters and metric history.
   * @return Complete current plateau control state.
   */
  detail::SchedulerState capture_state_impl() const override;

  /**
   * @brief Returns the committed metric-event count.
   * @return Number of successful metric observations.
   */
  std::uint64_t completed_steps_impl() const noexcept override;

  /**
   * @brief Validates exact plateau control state.
   * @param state Candidate state decoded or previewed by the framework.
   */
  void validate_state_impl(const detail::SchedulerState& state) const override;

  /**
   * @brief Applies prevalidated plateau control state.
   * @param state State already accepted by `validate_state_impl()`.
   */
  void apply_state_impl(const detail::SchedulerState& state) noexcept override;

  ReduceLrOnPlateauSchedulerOptions options_;  ///< Immutable user options.
  std::vector<double>
      min_learning_rates_;  ///< Resolved minimum for every optimizer group.
  std::uint64_t completed_steps_ = 0;  ///< Committed metric observations.
  bool has_best_ = false;  ///< Whether `best_` contains an observation.
  double best_ = 0.0;      ///< Best committed finite monitored metric.
  std::uint64_t bad_observations_ = 0;    ///< Current patience counter.
  std::uint64_t cooldown_remaining_ = 0;  ///< Current cooldown counter.
};

/** @brief Selects interpolation within one-cycle schedule phases. */
enum class AnnealStrategy : std::uint8_t {
  cosine = 1,  ///< Uses half-cosine interpolation between phase endpoints.
  linear = 2,  ///< Uses linear interpolation between phase endpoints.
};

/** @brief Configures `OneCycleLrScheduler`. */
struct OneCycleLrSchedulerOptions {
  std::string scheduler_id =
      "scheduler.one_cycle";  ///< Stable checkpoint participant ID.
  std::vector<double>
      max_learning_rates;         ///< One broadcast maximum or one per group.
  std::uint64_t total_steps = 0;  ///< Positive optimizer-update event count.
  double pct_start = 0.3;  ///< Fraction assigned to the increasing phase.
  AnnealStrategy anneal_strategy =
      AnnealStrategy::cosine;     ///< Phase interpolation strategy.
  double div_factor = 25.0;       ///< Maximum-to-initial learning-rate divisor.
  double final_div_factor = 1e4;  ///< Initial-to-final learning-rate divisor.
  bool three_phase =
      false;  ///< Adds a symmetric return to initial before final decay.
  bool cycle_momentum =
      false;  ///< Reserved until atomic optimizer hyperparameter control.
};

/**
 * @brief Runs a finite batch-driven one-cycle learning-rate policy.
 *
 * Construction initializes every group to `max_lr / div_factor`. Exactly
 * `total_steps` successful events are accepted; the last event reaches the
 * configured final rate without negative extrapolation.
 */
class OneCycleLrScheduler final : public LrScheduler {
 public:
  /**
   * @brief Creates a finite one-cycle scheduler.
   * @param optimizer Non-null atomic optimizer rate-control surface.
   * @param options Immutable cycle shape and checkpoint identity options.
   * @throws InvalidArgument If cycle geometry or rates are invalid.
   * @throws NotImplemented If momentum cycling is requested.
   */
  OneCycleLrScheduler(std::shared_ptr<OptimizerRateControl> optimizer,
                      OneCycleLrSchedulerOptions options);

  /** @brief Destroys the scheduler. */
  ~OneCycleLrScheduler() override;

 private:
  /**
   * @brief Previews the next bounded batch event.
   * @param learning_rates Valid current optimizer group rates.
   * @return Complete candidate one-cycle state and group rates.
   */
  detail::SchedulerTransition preview_step_impl(
      std::span<const double> learning_rates) const override;

  /**
   * @brief Captures the committed batch-event counter.
   * @return Counter-only current state.
   */
  detail::SchedulerState capture_state_impl() const override;

  /**
   * @brief Returns the committed batch-event count.
   * @return Number of successful one-cycle events.
   */
  std::uint64_t completed_steps_impl() const noexcept override;

  /**
   * @brief Validates a counter-only leaf state within the finite cycle.
   * @param state Candidate state decoded or previewed by the framework.
   */
  void validate_state_impl(const detail::SchedulerState& state) const override;

  /**
   * @brief Applies a prevalidated batch-event counter.
   * @param state State already accepted by `validate_state_impl()`.
   */
  void apply_state_impl(const detail::SchedulerState& state) noexcept override;

  OneCycleLrSchedulerOptions options_;          ///< Immutable user options.
  std::vector<double> max_learning_rates_;      ///< Resolved per-group maxima.
  std::vector<double> initial_learning_rates_;  ///< Per-group initial rates.
  std::vector<double> final_learning_rates_;    ///< Per-group final rates.
  std::uint64_t warmup_steps_ = 0;     ///< Events in the increasing phase.
  std::uint64_t completed_steps_ = 0;  ///< Committed batch events.
};

/** @brief Selects a built-in cyclic learning-rate amplitude profile. */
enum class CyclicLrMode : std::uint8_t {
  triangular = 1,   ///< Keeps the same amplitude for every cycle.
  triangular2 = 2,  ///< Halves amplitude after every complete cycle.
  exp_range = 3,    ///< Multiplies amplitude by `gamma^iteration`.
};

/** @brief Selects the input supplied to a custom cyclic scale function. */
enum class CyclicScaleMode : std::uint8_t {
  cycle = 1,       ///< Supplies the one-based cycle number.
  iterations = 2,  ///< Supplies the one-based global batch-event number.
};

/** @brief Configures `CyclicLrScheduler`. */
struct CyclicLrSchedulerOptions {
  std::string scheduler_id =
      "scheduler.cyclic";  ///< Stable checkpoint participant ID.
  std::vector<double>
      base_learning_rates;  ///< One broadcast base or one per group.
  std::vector<double>
      max_learning_rates;          ///< One broadcast maximum or one per group.
  std::uint64_t step_size_up = 0;  ///< Positive rising half-cycle length.
  std::uint64_t step_size_down =
      0;  ///< Falling length, or zero to reuse `step_size_up`.
  CyclicLrMode mode = CyclicLrMode::triangular;  ///< Built-in scale profile.
  double gamma = 1.0;  ///< Positive base for `exp_range` scaling.
  bool cycle_momentum =
      false;  ///< Reserved until atomic optimizer hyperparameter control.
  std::function<double(double)>
      scale_function;             ///< Optional non-negative amplitude function.
  std::string scale_function_id;  ///< Stable identity required for a callback.
  CyclicScaleMode scale_mode =
      CyclicScaleMode::cycle;  ///< Input domain for a custom callback.
};

/** @brief Runs a repeating batch-driven triangular learning-rate cycle. */
class CyclicLrScheduler final : public LrScheduler {
 public:
  /**
   * @brief Creates a cyclic scheduler and initializes group base rates.
   * @param optimizer Non-null atomic optimizer rate-control surface.
   * @param options Immutable cycle, callable, and checkpoint options.
   * @throws InvalidArgument If rates, geometry, or callable identity is
   * invalid.
   * @throws NotImplemented If momentum cycling is requested.
   */
  CyclicLrScheduler(std::shared_ptr<OptimizerRateControl> optimizer,
                    CyclicLrSchedulerOptions options);

  /** @brief Destroys the scheduler. */
  ~CyclicLrScheduler() override;

 private:
  /**
   * @brief Previews the next triangular batch event.
   * @param learning_rates Valid current optimizer group rates.
   * @return Complete candidate cyclic state and group rates.
   */
  detail::SchedulerTransition preview_step_impl(
      std::span<const double> learning_rates) const override;

  /**
   * @brief Captures the committed batch-event counter.
   * @return Counter-only current state.
   */
  detail::SchedulerState capture_state_impl() const override;

  /**
   * @brief Returns the committed batch-event count.
   * @return Number of successful cyclic events.
   */
  std::uint64_t completed_steps_impl() const noexcept override;

  /**
   * @brief Validates a counter-only cyclic leaf state.
   * @param state Candidate state decoded or previewed by the framework.
   */
  void validate_state_impl(const detail::SchedulerState& state) const override;

  /**
   * @brief Applies a prevalidated batch-event counter.
   * @param state State already accepted by `validate_state_impl()`.
   */
  void apply_state_impl(const detail::SchedulerState& state) noexcept override;

  CyclicLrSchedulerOptions options_;  ///< Immutable user and callback options.
  std::vector<double> base_learning_rates_;  ///< Resolved per-group bases.
  std::vector<double> max_learning_rates_;   ///< Resolved per-group maxima.
  std::uint64_t step_size_down_ = 0;   ///< Resolved falling half-cycle length.
  std::uint64_t completed_steps_ = 0;  ///< Committed batch events.
};

/** @brief Configures `LambdaLrScheduler`. */
struct LambdaLrSchedulerOptions {
  std::string scheduler_id =
      "scheduler.lambda";   ///< Stable checkpoint participant ID.
  std::string callable_id;  ///< Stable identity for the callable bundle.
  std::vector<std::function<double(std::uint64_t)>>
      functions;  ///< One broadcast callback or one per optimizer group.
};

/**
 * @brief Computes epoch rates from registered, identity-bearing callables.
 *
 * A callable bundle must have a stable application-defined identity so exact
 * restore cannot accidentally continue with different executable semantics.
 */
class LambdaLrScheduler final : public LrScheduler {
 public:
  /**
   * @brief Creates a callable-driven scheduler and evaluates event zero.
   * @param optimizer Non-null atomic optimizer rate-control surface.
   * @param options Stable callable identity and one or per-group functions.
   * @throws InvalidArgument If identity, function count, or event-zero output
   * is invalid.
   */
  LambdaLrScheduler(std::shared_ptr<OptimizerRateControl> optimizer,
                    LambdaLrSchedulerOptions options);

  /** @brief Destroys the scheduler. */
  ~LambdaLrScheduler() override;

 private:
  /**
   * @brief Previews callable factors for the next epoch event.
   * @param learning_rates Valid current optimizer group rates.
   * @return Complete candidate callable state and group rates.
   */
  detail::SchedulerTransition preview_step_impl(
      std::span<const double> learning_rates) const override;

  /**
   * @brief Captures the committed epoch-event counter.
   * @return Counter-only current state.
   */
  detail::SchedulerState capture_state_impl() const override;

  /**
   * @brief Returns the committed epoch-event count.
   * @return Number of successful callable events.
   */
  std::uint64_t completed_steps_impl() const noexcept override;

  /**
   * @brief Validates a counter-only callable leaf state.
   * @param state Candidate state decoded or previewed by the framework.
   */
  void validate_state_impl(const detail::SchedulerState& state) const override;

  /**
   * @brief Applies a prevalidated epoch-event counter.
   * @param state State already accepted by `validate_state_impl()`.
   */
  void apply_state_impl(const detail::SchedulerState& state) noexcept override;

  LambdaLrSchedulerOptions options_;   ///< Immutable callable bundle options.
  std::uint64_t completed_steps_ = 0;  ///< Committed epoch events.
};

}  // namespace jyppx::optim

#endif  // JYPPX_OPTIM_EVENT_LR_SCHEDULER_HPP
