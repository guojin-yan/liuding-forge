// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_TRAINING_AMP_GRAD_SCALER_HPP
#define JYPPX_TRAINING_AMP_GRAD_SCALER_HPP

/**
 * @file grad_scaler.hpp
 * @brief Defines multi-optimizer gradient scaling and its typed stable state.
 */

#include <jyppx/core/types.hpp>
#include <jyppx/optim/optimizer_adapter.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <string>

namespace jyppx::training::amp {

/** @brief Current typed GradScaler state format version. */
inline constexpr std::uint32_t kGradScalerStateFormatVersion = 1;

/**
 * @brief Configures dynamic loss scaling for float16 training.
 *
 * Every field contributes to continuation semantics and is checked during
 * state restore. `scaler_id` is also the default checkpoint participant ID.
 */
struct GradScalerOptions {
  double init_scale = 65536.0;  ///< Positive scale used at construction.
  double growth_factor = 2.0;   ///< Factor greater than one for scale growth.
  double backoff_factor = 0.5;  ///< Factor in `(0, 1)` for overflow recovery.
  int growth_interval = 2000;   ///< Consecutive finite cycles before growth.
  bool enabled = true;  ///< Whether scaling and overflow checks are active.
  std::string scaler_id =
      "grad_scaler.main";  ///< Stable nonempty checkpoint identity.
};

/**
 * @brief Owns a complete GradScaler state at a completed update boundary.
 *
 * Immutable configuration is included deliberately: applying mutable scale
 * state to a differently configured scaler would not reproduce continuation.
 */
struct GradScalerState {
  std::uint32_t format_version =
      kGradScalerStateFormatVersion;  ///< Stable state schema version.
  std::string scaler_id;  ///< Identity of the scaler that produced the state.
  double init_scale = 65536.0;  ///< Configured construction scale.
  double growth_factor = 2.0;   ///< Configured finite-cycle growth factor.
  double backoff_factor = 0.5;  ///< Configured overflow backoff factor.
  int growth_interval = 2000;   ///< Configured finite-cycle growth interval.
  bool enabled = true;          ///< Whether the producing scaler was enabled.
  double scale = 65536.0;       ///< Current dynamic loss scale.
  std::uint64_t growth_tracker = 0;  ///< Finite cycles since the last change.
};

/**
 * @brief Coordinates dynamic loss scaling across one or more optimizers.
 *
 * Each optimizer is unscaled and receives an explicit overflow decision. Its
 * owning strategy applies or skips the update and reports that result through
 * `mark_step()`. One call to `update()` then applies the global scale decision
 * for the complete iteration. Checkpoint state is available only after
 * construction or a completed multi-optimizer cycle.
 *
 * This class is not thread-safe. The caller must serialize access with
 * optimizer mutation and checkpoint operations.
 */
class GradScaler {
 public:
  /**
   * @brief Creates a scaler with validated immutable options.
   * @param options Dynamic scaling configuration and stable identity.
   * @throws std::invalid_argument If an option is invalid.
   */
  explicit GradScaler(GradScalerOptions options = {});

  /**
   * @brief Multiplies a loss by the current scale when enabled.
   * @param loss Loss tensor whose autograd graph must be preserved.
   * @return The scaled loss, or `loss` unchanged when disabled.
   */
  Tensor scale(const Tensor& loss) const;

  /**
   * @brief Divides gradients by the current scale and records non-finite data.
   * @param optimizer Non-null optimizer participating in the active cycle.
   * @throws std::invalid_argument If `optimizer` is null.
   * @throws std::logic_error If the optimizer was already marked this cycle.
   * @note Repeating `unscale()` before `mark_step()` is idempotent.
   */
  void unscale(const std::shared_ptr<optim::OptimizerAdapter>& optimizer);

  /**
   * @brief Reports whether a strategy-owned optimizer step may be applied.
   * @param optimizer Non-null optimizer already passed to `unscale()`.
   * @return `false` when this optimizer observed a non-finite gradient.
   * @throws std::invalid_argument If `optimizer` is null.
   * @throws std::logic_error If `unscale()` has not started its cycle.
   */
  bool step_allowed(
      const std::shared_ptr<optim::OptimizerAdapter>& optimizer) const;

  /**
   * @brief Commits the strategy-owned result for one optimizer.
   * @param optimizer Non-null optimizer passed to `unscale()`.
   * @param applied Whether the strategy actually applied the update.
   * @throws std::invalid_argument If `optimizer` is null.
   * @throws std::logic_error If the optimizer is not in the active cycle.
   */
  void mark_step(const std::shared_ptr<optim::OptimizerAdapter>& optimizer,
                 bool applied);

  /**
   * @brief Commits one global scale transition for the active optimizer set.
   * @throws std::logic_error If no optimizer was marked or one remains open.
   * Scale growth saturates at the largest finite `double` value.
   */
  void update();

  /**
   * @brief Returns the current dynamic loss scale.
   * @return `1.0` for a disabled scaler; otherwise the active positive scale.
   */
  double get_scale() const noexcept;

  /**
   * @brief Reports whether the current or most recently completed cycle
   * observed a non-finite gradient.
   * @return `true` after any participating optimizer reports inf or NaN.
   */
  bool found_inf() const noexcept;

  /**
   * @brief Reports whether exact state can be captured or restored now.
   * @return `true` when no optimizer cycle is in progress.
   */
  bool checkpoint_ready() const noexcept;

  /**
   * @brief Captures complete typed state at a consistency boundary.
   * @return An owned state including immutable configuration.
   * @throws std::logic_error If an optimizer cycle is in progress.
   */
  GradScalerState capture_state() const;

  /**
   * @brief Validates typed state without changing scaler state.
   * @param state Candidate state including immutable configuration.
   * @throws std::logic_error If an optimizer cycle is in progress.
   * @throws std::invalid_argument If schema, identity, configuration, or
   * mutable values are incompatible.
   */
  void validate_state(const GradScalerState& state) const;

  /**
   * @brief Atomically restores compatible typed state.
   * @param state Candidate state to validate before mutation.
   * @throws std::logic_error If an optimizer cycle is in progress.
   * @throws std::invalid_argument If schema, identity, configuration, or
   * mutable values are incompatible.
   */
  void restore_state(const GradScalerState& state);

  /**
   * @brief Returns immutable scaling configuration.
   * @return A reference valid for the scaler lifetime.
   */
  const GradScalerOptions& options() const noexcept;

 private:
  /** @brief Tracks one optimizer's transition within the active cycle. */
  enum class OptimizerStage {
    unscaled,  ///< Gradients were checked but no outcome was reported.
    stepped,   ///< The optimizer applied or skipped its strategy-owned update.
  };

  /** @brief Stores the active-cycle status for one optimizer identity. */
  struct OptimizerState {
    OptimizerStage stage =
        OptimizerStage::unscaled;  ///< Current per-optimizer cycle stage.
    bool found_inf = false;  ///< Whether this optimizer had inf or NaN data.
  };

  GradScalerOptions options_;  ///< Immutable behavior and checkpoint identity.
  double scale_;  ///< Positive dynamic scale used for the active cycle.
  std::uint64_t growth_tracker_;  ///< Consecutive finite completed cycles.
  bool found_inf_;  ///< Aggregate non-finite result for the latest cycle.
  std::map<const optim::OptimizerAdapter*, OptimizerState>
      optimizer_states_;  ///< Non-owning active-cycle optimizer states.
};

}  // namespace jyppx::training::amp

#endif  // JYPPX_TRAINING_AMP_GRAD_SCALER_HPP
