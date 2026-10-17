// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_TRAINING_CALLBACKS_EARLY_STOPPING_CALLBACK_HPP
#define JYPPX_TRAINING_CALLBACKS_EARLY_STOPPING_CALLBACK_HPP

/**
 * @file early_stopping_callback.hpp
 * @brief Defines descriptor-first early stopping with exact checkpoint state.
 */

#include <jyppx/training/callback_dispatcher.hpp>

#include <cstdint>
#include <string>
#include <string_view>

namespace jyppx::training::callbacks {

/**
 * @brief Selects the improvement direction for a monitored metric.
 *
 * Numeric values are persisted in callback topology and must remain stable.
 */
enum class EarlyStoppingMode : std::uint8_t {
  minimize = 1,  ///< A value below the prior best is an improvement.
  maximize = 2,  ///< A value above the prior best is an improvement.
};

/**
 * @brief Returns the stable lowercase name of an early-stopping mode.
 * @param mode Comparison direction to render.
 * @return A process-lifetime name, or `"unknown"` for an invalid value.
 */
std::string_view early_stopping_mode_name(EarlyStoppingMode mode) noexcept;

/** @brief Configures immutable early-stopping identity and semantics. */
struct EarlyStoppingOptions {
  std::string callback_id =
      "callback.early_stopping";  ///< Stable callback participant identity.
  std::string monitor =
      "validation.loss";  ///< Exact metric key required on `epoch_finished`.
  std::uint64_t patience =
      10;  ///< Allowed consecutive non-improving observations before stopping.
  double min_delta =
      0.0;  ///< Finite non-negative absolute improvement threshold.
  EarlyStoppingMode mode =
      EarlyStoppingMode::minimize;  ///< Improvement comparison direction.
  std::int32_t priority = 0;        ///< Dispatcher execution priority.
};

/** @brief Configures checkpoint resource limits for early-stopping state. */
struct EarlyStoppingCheckpointOptions {
  std::uint64_t max_topology_bytes =
      64ULL * 1024ULL;  ///< Maximum canonical immutable topology size.
  std::uint64_t max_state_bytes =
      64ULL * 1024ULL;  ///< Maximum mutable state encoding size.
};

/**
 * @brief Requests stop after a monitored epoch metric ceases to improve.
 *
 * The callback observes only `epoch_finished`, never receives a writable
 * Trainer, and may return only `stop`. Its immutable descriptor and complete
 * mutable state form a current-only exact checkpoint participant. A missing or
 * non-finite monitored value is rejected instead of silently skipping state.
 *
 * Instances are not internally synchronized. Dispatch and checkpoint I/O must
 * be serialized by the training loop.
 */
class EarlyStoppingCallback final : public CallbackHandler,
                                    public CheckpointParticipant {
 public:
  /**
   * @brief Creates a descriptor-first early-stopping callback.
   * @param options Stable identity, metric, patience, threshold, and order.
   * @param checkpoint_options Positive checkpoint resource limits.
   * @throws InvalidArgument If an identity, enum, threshold, or limit is
   * invalid.
   */
  explicit EarlyStoppingCallback(
      EarlyStoppingOptions options = {},
      EarlyStoppingCheckpointOptions checkpoint_options = {});

  /** @brief Prevents copying mutable callback state. */
  EarlyStoppingCallback(const EarlyStoppingCallback&) = delete;

  /** @brief Prevents assignment over mutable callback state. */
  EarlyStoppingCallback& operator=(const EarlyStoppingCallback&) = delete;

  /** @brief Prevents moving an object retained by callback registries. */
  EarlyStoppingCallback(EarlyStoppingCallback&&) = delete;

  /** @brief Prevents move assignment of registered checkpoint state. */
  EarlyStoppingCallback& operator=(EarlyStoppingCallback&&) = delete;

  /**
   * @brief Returns immutable callback identity and capabilities.
   * @return A descriptor reference valid for the callback lifetime.
   */
  const CallbackDescriptor& descriptor() const noexcept override;

  /**
   * @brief Updates patience from one completed epoch observation.
   * @param event Validated `epoch_finished` event containing `monitor`.
   * @return `stop` after patience is exceeded, otherwise `none`.
   * @throws InvalidArgument If the event or monitored metric is invalid.
   * @throws ResourceExhausted If the non-improvement counter would overflow.
   */
  CallbackCommandRequest handle(const CallbackEvent& event) override;

  /** @brief Clears the best value, patience counter, and stop decision. */
  void reset() noexcept;

  /**
   * @brief Returns whether a stop command has been reached.
   * @return `true` after consecutive non-improvements exceed patience.
   */
  bool should_stop() const noexcept;

  /**
   * @brief Returns the number of consecutive non-improving observations.
   * @return Current checked patience counter.
   */
  std::uint64_t non_improving_observations() const noexcept;

  /**
   * @brief Returns whether at least one monitored value was accepted.
   * @return `true` when `best_value()` is meaningful.
   */
  bool has_best_value() const noexcept;

  /**
   * @brief Returns the best accepted monitored value.
   * @return Best value, or zero before the first observation.
   */
  double best_value() const noexcept;

  /**
   * @brief Returns the stable callback checkpoint identity.
   * @return An owned identifier matching `describe_state()`.
   */
  std::string participant_id() const override;

  /**
   * @brief Describes canonical early-stopping topology and mutable state.
   * @return A validated current-only exact participant manifest.
   */
  StateManifest describe_state() const override;

  /**
   * @brief Writes immutable topology and complete mutable callback state.
   * @param writer Storage-neutral checkpoint destination.
   */
  void save_state(StateWriter& writer) const override;

  /**
   * @brief Restores compatible state after complete preflight validation.
   * @param reader Storage-neutral checkpoint source.
   * @param policy Restore policy that must accept exact state.
   * @throws CompatibilityError If manifest or topology differs.
   * @throws SerializationError If mutable state is malformed.
   * @throws ResourceExhausted If configured read limits are exceeded.
   */
  void load_state(const StateReader& reader, RestorePolicy policy) override;

 private:
  EarlyStoppingOptions options_;  ///< Immutable callback semantics.
  EarlyStoppingCheckpointOptions
      checkpoint_options_;         ///< Immutable checkpoint resource limits.
  CallbackDescriptor descriptor_;  ///< Frozen dispatcher contract.
  bool has_best_ = false;          ///< Whether a best value was observed.
  double best_value_ = 0.0;        ///< Best finite observed metric value.
  std::uint64_t non_improving_ =
      0;  ///< Consecutive observations that did not improve enough.
  bool should_stop_ = false;  ///< Whether patience has been exceeded.
};

}  // namespace jyppx::training::callbacks

#endif  // JYPPX_TRAINING_CALLBACKS_EARLY_STOPPING_CALLBACK_HPP
