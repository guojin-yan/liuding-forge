// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_DISTRIBUTED_DDP_HPP
#define JYPPX_DISTRIBUTED_DDP_HPP

#include <jyppx/distributed/distributed.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace jyppx::distributed {

/**
 * @brief Data-parallel wrapper for a concrete `torch::nn::Module` subtype.
 *
 * `forward()` delegates to the wrapped module. Each trainable parameter
 * gradient is passed through `average_gradients()` by an autograd hook. The
 * single-process backend returns gradients unchanged; a future c10d backend
 * can provide cross-process reduction without changing this wrapper contract.
 *
 * @tparam ModuleT Concrete module type that exposes `forward()` and
 * `parameters()`.
 */
template <typename ModuleT>
class DistributedDataParallel {
 public:
  /**
   * @brief Wraps a non-null module and installs its gradient hooks.
   *
   * @param module Module whose trainable parameters participate in gradient
   * averaging.
   * @throws InvalidArgument If `module` is null.
   */
  explicit DistributedDataParallel(std::shared_ptr<ModuleT> module)
      : module_(std::move(module)) {
    if (!module_) {
      throw InvalidArgument("DistributedDataParallel module cannot be null");
    }
    install_hooks();
  }

  /**
   * @brief Forwards arguments to the wrapped module.
   *
   * @tparam Args Argument types accepted by `ModuleT::forward()`.
   * @param args Arguments forwarded with their value categories preserved.
   * @return The result returned by `ModuleT::forward()`.
   */
  template <typename... Args>
  decltype(auto) forward(Args&&... args) {
    return module_->forward(std::forward<Args>(args)...);
  }

  /**
   * @brief Returns the wrapped module.
   *
   * @return Shared ownership of the same module supplied to the constructor.
   */
  std::shared_ptr<ModuleT> module() const { return module_; }

 private:
  /** Installs one averaging hook for every defined trainable parameter. */
  void install_hooks() {
    for (const auto& parameter : module_->parameters()) {
      if (!parameter.defined() || !parameter.requires_grad()) {
        continue;
      }
      hooks_.push_back(parameter.register_hook([](Tensor gradient) {
        return average_gradients(std::move(gradient));
      }));
    }
  }

  /** Module whose forward pass and gradients are wrapped. */
  std::shared_ptr<ModuleT> module_;

  /** Autograd hook handles retained for the lifetime of the wrapper. */
  std::vector<unsigned> hooks_;
};

}  // namespace jyppx::distributed

#endif  // JYPPX_DISTRIBUTED_DDP_HPP
