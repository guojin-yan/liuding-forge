// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_TRAINING_SINGLE_DEVICE_STRATEGY_HPP
#define JYPPX_LIUDING_TRAINING_SINGLE_DEVICE_STRATEGY_HPP
/**
 * @file single_device_strategy.hpp
 * @brief Defines the validated v1.0 single-device execution strategy.
 */

#include <jyppx/liuding/training/strategy.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace jyppx::liuding::training {

/** @brief Configures immutable single-device placement and resource bounds. */
struct SingleDeviceStrategyOptions {
  std::string strategy_id =
      "strategy.single";  ///< Stable checkpoint participant identity.
  Device device = Device(torch::kCPU);  ///< Required model and loss device.
  bool non_blocking_transfer =
      false;  ///< Requests eligible asynchronous module transfers.
  std::size_t max_optimizers =
      256;  ///< Maximum disjoint optimizers in one execution domain.
  std::size_t max_parameters =
      100'000;  ///< Maximum bound trainable parameter count.
  std::size_t max_metrics =
      4096;  ///< Maximum observations accepted by one metric reduction.
  std::size_t max_string_bytes =
      4096;  ///< Maximum bytes in an identity or metric name.
  std::uint64_t max_topology_bytes =
      64ULL * 1024ULL * 1024ULL;  ///< Maximum prepared topology state bytes.
  std::uint64_t max_control_bytes =
      64ULL * 1024ULL;  ///< Maximum lifecycle counter state bytes.
};

/**
 * @brief Executes one model and optimizer on one validated LibTorch device.
 *
 * `prepare()` moves model parameters and buffers to the configured device, then
 * verifies that optimizer bindings are disjoint and together match every named
 * trainable parameter by stable ID and tensor identity. Backward accepts only
 * finite scalar losses on that device. Optimizer calls must pass an exact
 * prepared adapter; external updates are detected before the next strategy
 * step or checkpoint save.
 *
 * Current-only checkpoint schema 2 stores the prepared parameter topology,
 * backward counter, every optimizer baseline/step cursor, and closed state.
 * Saving is allowed only after `prepare()` and outside an in-flight operation.
 * Restoring validates all bytes before replacing counters and never mutates
 * model or optimizer tensors.
 *
 * Instances are single-owner and not internally synchronized. Model and
 * optimizer objects must outlive the prepared strategy.
 */
class SingleDeviceStrategy final : public TrainingStrategy {
 public:
  /**
   * @brief Creates an unprepared single-device strategy.
   * @param options Stable device, participant identity, and positive limits.
   * @throws InvalidArgument If options or device selection are invalid.
   */
  explicit SingleDeviceStrategy(SingleDeviceStrategyOptions options = {});

  /** @brief Releases private bookkeeping without invoking backend operations.
   */
  ~SingleDeviceStrategy() override;

  /** @brief Prevents copying borrowed execution ownership. */
  SingleDeviceStrategy(const SingleDeviceStrategy&) = delete;

  /** @brief Prevents assignment over prepared execution state. */
  SingleDeviceStrategy& operator=(const SingleDeviceStrategy&) = delete;

  /** @brief Prevents moving an object retained by a participant registry. */
  SingleDeviceStrategy(SingleDeviceStrategy&&) = delete;

  /** @brief Prevents move assignment of checkpointed strategy state. */
  SingleDeviceStrategy& operator=(SingleDeviceStrategy&&) = delete;

  /** @copydoc TrainingStrategy::descriptor */
  const StrategyDescriptor& descriptor() const noexcept override;

  /** @copydoc TrainingStrategy::prepare */
  void prepare(nn::ModelAdapter& model,
               std::span<optim::OptimizerAdapter* const> optimizers) override;

  /** @copydoc TrainingStrategy::backward */
  void backward(const Tensor& loss) override;

  /** @copydoc TrainingStrategy::zero_grad */
  void zero_grad(optim::OptimizerAdapter& optimizer,
                 bool set_to_none = true) override;

  /** @copydoc TrainingStrategy::optimizer_step */
  StrategyStepOutcome optimizer_step(
      optim::OptimizerAdapter& optimizer) override;

  /** @copydoc TrainingStrategy::reduce_metrics */
  StrategyMetricValues reduce_metrics(
      const StrategyMetricValues& values) const override;

  /** @copydoc TrainingStrategy::barrier */
  void barrier(StrategyBarrierReason reason,
               std::stop_token stop = {}) const override;

  /** @copydoc TrainingStrategy::close */
  void close() override;

  /** @copydoc TrainingStrategy::prepared */
  bool prepared() const noexcept override;

  /** @copydoc TrainingStrategy::closed */
  bool closed() const noexcept override;

  /**
   * @brief Returns successful backward call count.
   * @return Monotonic counter restored from checkpoint state.
   */
  std::uint64_t completed_backward_calls() const noexcept;

  /**
   * @brief Returns updates committed for one prepared optimizer.
   * @param optimizer_id Stable identity supplied during `prepare()`.
   * @return Monotonic counter restored from checkpoint state.
   * @throws InvalidArgument If the identity was not prepared.
   */
  std::uint64_t completed_optimizer_steps(std::string_view optimizer_id) const;

  /**
   * @brief Returns immutable strategy options.
   * @return A reference valid for the strategy lifetime.
   */
  const SingleDeviceStrategyOptions& options() const noexcept;

  /** @copydoc CheckpointParticipant::participant_id */
  std::string participant_id() const override;

  /** @copydoc CheckpointParticipant::describe_state */
  StateManifest describe_state() const override;

  /**
   * @brief Saves exact prepared topology and strategy-owned counters.
   * @param writer Storage-neutral destination supplied by the coordinator.
   * @throws NotReady If the strategy is unprepared or already closed.
   * @throws CompatibilityError If an optimizer changed outside this strategy.
   * @throws ResourceExhausted If encoded state exceeds configured limits.
   *
   * A saved snapshot always represents an open lifecycle boundary. Closing a
   * strategy is terminal and cannot be checkpointed for later continuation.
   */
  void save_state(StateWriter& writer) const override;

  /**
   * @brief Restores exact counters into a matching prepared open strategy.
   * @param reader Source containing schema-version-2 strategy state.
   * @param policy Restore policy that must accept exact state.
   * @throws NotReady If the strategy is unprepared or already closed.
   * @throws CompatibilityError If policy, manifest, topology, or optimizer
   * cursors are incompatible.
   * @throws SerializationError If control state is malformed or encodes a
   * closed lifecycle.
   * @throws ResourceExhausted If a field exceeds configured limits.
   *
   * All fields are validated before counters are committed. Failure leaves
   * lifecycle and strategy-owned counters unchanged.
   */
  void load_state(const StateReader& reader, RestorePolicy policy) override;

 private:
  /** @brief Hides borrowed adapters and checkpoint topology from public ABI. */
  struct Impl;

  std::unique_ptr<Impl> impl_;  ///< Owned descriptor and lifecycle state.
};

}  // namespace jyppx::liuding::training

#endif  // JYPPX_LIUDING_TRAINING_SINGLE_DEVICE_STRATEGY_HPP