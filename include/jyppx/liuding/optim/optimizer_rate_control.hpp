// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_OPTIM_OPTIMIZER_RATE_CONTROL_HPP
#define JYPPX_LIUDING_OPTIM_OPTIMIZER_RATE_CONTROL_HPP
/**
 * @file optimizer_rate_control.hpp
 * @brief Defines the narrow optimizer contract consumed by LR schedulers.
 */

#include <string>
#include <vector>

namespace jyppx::liuding::optim {

/**
 * @brief Exposes stable optimizer identity and atomic learning-rate control.
 *
 * Schedulers depend on this interface instead of an algorithm-specific native
 * optimizer or the legacy hand-written optimizer hierarchy. Implementations
 * retain ownership of algorithm state and parameter tensors. Returned identity
 * references remain valid for the lifetime of the implementation.
 *
 * Implementations are not required to provide internal synchronization.
 * Callers must serialize scheduler, optimizer-step, and checkpoint operations.
 */
class OptimizerRateControl {
 public:
  /** @brief Enables destruction through the rate-control interface. */
  virtual ~OptimizerRateControl() = default;

  /**
   * @brief Returns the stable optimizer identity used by checkpoint topology.
   * @return A nonempty logical identifier with process-lifetime stability.
   */
  virtual const std::string& optimizer_id() const noexcept = 0;

  /**
   * @brief Returns stable parameter-group identities in optimizer order.
   * @return A nonempty vector whose order and values never change.
   */
  virtual const std::vector<std::string>& parameter_group_ids()
      const noexcept = 0;

  /**
   * @brief Returns one current learning rate per stable parameter group.
   * @return Finite non-negative rates in `parameter_group_ids()` order.
   * @throws Error If the optimizer no longer satisfies its stable descriptor.
   */
  virtual std::vector<double> learning_rates() const = 0;

  /**
   * @brief Atomically validates and replaces every group learning rate.
   * @param learning_rates Finite non-negative rates matching group count.
   * @throws InvalidArgument If a count or value violates the contract.
   * @throws Error If the optimizer cannot apply the complete update.
   */
  virtual void set_learning_rates(
      const std::vector<double>& learning_rates) = 0;
};

}  // namespace jyppx::liuding::optim

#endif  // JYPPX_LIUDING_OPTIM_OPTIMIZER_RATE_CONTROL_HPP