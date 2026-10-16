// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_OPTIM_OPTIMIZER_ADAPTER_HPP
#define JYPPX_OPTIM_OPTIMIZER_ADAPTER_HPP

/**
 * @file optimizer_adapter.hpp
 * @brief Defines the narrow optimizer contract consumed by training strategies.
 */

#include <jyppx/core/types.hpp>
#include <jyppx/optim/optimizer_rate_control.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace jyppx::optim {

/**
 * @brief Binds one model parameter to its stable optimizer identity.
 *
 * The tensor remains model-owned. Adapters retain a handle and must not replace
 * it with an unrelated tensor after construction.
 */
struct OptimizerParameterBinding {
  std::string parameter_id;  ///< Stable nonempty model parameter identity.
  Tensor parameter;  ///< Defined trainable tensor retained by shared handle.
};

/**
 * @brief Exposes only the optimizer operations required by a training loop.
 *
 * Algorithm configuration and checkpoint encoding remain implementation-owned.
 * The adapter does not expose a writable native optimizer, mutable parameter
 * groups, or pointer-keyed state. Instances are not internally synchronized;
 * callers must serialize gradient, step, scheduler, and checkpoint operations.
 */
class OptimizerAdapter : public OptimizerRateControl {
 public:
  /** @brief Callable loss closure accepted by delegated optimizers. */
  using LossClosure = std::function<Tensor()>;

  /** @brief Enables destruction through the optimizer adapter interface. */
  ~OptimizerAdapter() override = default;

  /**
   * @brief Applies one optimizer update.
   * @param closure Optional backend closure evaluated during the update.
   * @return Closure result, or an undefined tensor when no closure is used.
   * @throws Error If validation, bookkeeping, or the backend update fails.
   */
  virtual Tensor step(LossClosure closure = {}) = 0;

  /**
   * @brief Clears gradients for every bound parameter.
   * @param set_to_none Whether to release gradients instead of zeroing them.
   * @throws Error If the backend cannot clear a gradient.
   */
  virtual void zero_grad(bool set_to_none = true) = 0;

  /**
   * @brief Returns stable parameter identities and model-owned tensor handles.
   * @return Bindings in deterministic optimizer group and parameter order.
   * @throws Error If current native topology violates the frozen descriptor.
   */
  virtual std::vector<OptimizerParameterBinding> parameter_bindings() const = 0;

  /**
   * @brief Returns the total number of bound parameters.
   * @return Positive parameter count frozen at adapter construction.
   */
  virtual std::size_t parameter_count() const noexcept = 0;

  /**
   * @brief Returns the number of successfully completed optimizer updates.
   * @return Monotonic count restored by the algorithm participant.
   */
  virtual std::uint64_t completed_steps() const noexcept = 0;
};

}  // namespace jyppx::optim

#endif  // JYPPX_OPTIM_OPTIMIZER_ADAPTER_HPP
