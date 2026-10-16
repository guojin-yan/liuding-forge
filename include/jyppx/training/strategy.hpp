// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_TRAINING_STRATEGY_HPP
#define JYPPX_TRAINING_STRATEGY_HPP

/**
 * @file strategy.hpp
 * @brief Defines the narrow execution strategy contract used by a Trainer.
 */

#include <jyppx/core/types.hpp>
#include <jyppx/foundation/checkpoint.hpp>

#include <cstdint>
#include <map>
#include <stop_token>
#include <string>
#include <string_view>

namespace jyppx::nn {
class ModelAdapter;
}  // namespace jyppx::nn

namespace jyppx::optim {
class OptimizerAdapter;
}  // namespace jyppx::optim

namespace jyppx::training {

/**
 * @brief Identifies a synchronization boundary requested by the training loop.
 *
 * Numeric values are part of the strategy protocol and must remain stable.
 */
enum class StrategyBarrierReason : std::uint8_t {
  event_commit = 1,  ///< Training and callback event state reached a boundary.
  checkpoint = 2,    ///< Participants are about to save or restore state.
  shutdown = 3,      ///< The run is entering explicit finalization.
};

/**
 * @brief Returns the stable lowercase barrier-reason name.
 * @param reason Reason to render.
 * @return A process-lifetime name, or `"unknown"` for an invalid value.
 */
std::string_view strategy_barrier_reason_name(
    StrategyBarrierReason reason) noexcept;

/**
 * @brief Identifies the result of one strategy-owned optimizer update.
 *
 * Numeric values are stable so future precision/distributed strategies can add
 * explicit skipped or coordinated outcomes without changing `applied`.
 */
enum class StrategyStepStatus : std::uint8_t {
  applied = 1,  ///< The optimizer completed exactly one parameter update.
};

/**
 * @brief Returns the stable lowercase strategy-step status name.
 * @param status Status to render.
 * @return A process-lifetime name, or `"unknown"` for an invalid value.
 */
std::string_view strategy_step_status_name(StrategyStepStatus status) noexcept;

/** @brief Freezes one strategy's topology and runtime placement. */
struct StrategyDescriptor {
  std::uint32_t schema_version = 1;  ///< Current descriptor schema version.
  std::string type_id;  ///< Stable implementation and semantic schema identity.
  std::string strategy_id;  ///< Stable checkpoint participant identity.
  std::string device;  ///< Canonical LibTorch device used for local execution.
  std::uint32_t world_size = 1;  ///< Fixed participating process count.
  std::uint32_t rank = 0;  ///< Fixed process rank within the strategy topology.
};

/** @brief Reports one successfully committed optimizer update. */
struct StrategyStepOutcome {
  StrategyStepStatus status =
      StrategyStepStatus::applied;  ///< Explicit update disposition.
  std::string optimizer_id;  ///< Stable identity of the updated optimizer.
  std::uint64_t optimizer_completed_steps =
      0;  ///< Optimizer's committed count after the update.
};

/**
 * @brief Maps finite named metric observations at a strategy boundary.
 *
 * The single-device strategy returns an identical map. Future distributed
 * implementations must publish explicit reduction semantics before changing
 * values.
 */
using StrategyMetricValues = std::map<std::string, double>;

/**
 * @brief Coordinates the execution stages owned by a training topology.
 *
 * The interface intentionally exposes no collective, process-group, native
 * optimizer, or writable Trainer object. Implementations are checkpoint
 * participants and are invoked serially by one training-loop owner.
 */
class TrainingStrategy : public CheckpointParticipant {
 public:
  /** @brief Enables destruction through the strategy interface. */
  ~TrainingStrategy() override = default;

  /**
   * @brief Returns immutable strategy topology and placement.
   * @return A descriptor reference valid for the strategy lifetime.
   */
  virtual const StrategyDescriptor& descriptor() const noexcept = 0;

  /**
   * @brief Places and validates a model/optimizer execution domain.
   * @param model Model state and invocation adapter retained by the caller.
   * @param optimizer Optimizer bound to exactly the model's trainable
   * parameters.
   * @throws Error If placement or topology validation fails.
   */
  virtual void prepare(nn::ModelAdapter& model,
                       optim::OptimizerAdapter& optimizer) = 0;

  /**
   * @brief Runs reverse-mode differentiation for one scalar loss.
   * @param loss Defined finite scalar loss on the strategy device.
   * @throws Error If the strategy is not ready or backward fails.
   */
  virtual void backward(const Tensor& loss) = 0;

  /**
   * @brief Clears gradients through the prepared optimizer.
   * @param optimizer Exact optimizer supplied to `prepare()`.
   * @param set_to_none Whether to release gradients instead of zeroing them.
   * @throws Error If identity, lifecycle, or backend clearing fails.
   */
  virtual void zero_grad(optim::OptimizerAdapter& optimizer,
                         bool set_to_none = true) = 0;

  /**
   * @brief Applies one update through the prepared optimizer.
   * @param optimizer Exact optimizer supplied to `prepare()`.
   * @return Explicit committed step outcome.
   * @throws Error If identity, lifecycle, or backend stepping fails.
   */
  virtual StrategyStepOutcome optimizer_step(
      optim::OptimizerAdapter& optimizer) = 0;

  /**
   * @brief Reduces finite named observations for the active topology.
   * @param values Deterministic metric map at one lifecycle boundary.
   * @return Reduced finite observations with stable keys.
   * @throws Error If a name or value violates strategy limits.
   */
  virtual StrategyMetricValues reduce_metrics(
      const StrategyMetricValues& values) const = 0;

  /**
   * @brief Reaches a topology synchronization or cancellation boundary.
   * @param reason Stable reason for the requested barrier.
   * @param stop Cooperative cancellation request.
   * @throws Cancelled If cancellation is requested.
   * @throws InvalidArgument If `reason` is invalid.
   */
  virtual void barrier(StrategyBarrierReason reason,
                       std::stop_token stop = {}) const = 0;

  /**
   * @brief Idempotently closes strategy-owned execution resources.
   * @throws Error If finalization cannot complete.
   */
  virtual void close() = 0;

  /**
   * @brief Reports whether model and optimizer topology was prepared.
   * @return `true` after a successful `prepare()`.
   */
  virtual bool prepared() const noexcept = 0;

  /**
   * @brief Reports whether explicit finalization completed.
   * @return `true` after `close()` succeeds.
   */
  virtual bool closed() const noexcept = 0;
};

}  // namespace jyppx::training

#endif  // JYPPX_TRAINING_STRATEGY_HPP
