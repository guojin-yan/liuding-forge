// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_TRAINING_AMP_AUTOCAST_HPP
#define JYPPX_LIUDING_TRAINING_AMP_AUTOCAST_HPP
/**
 * @file autocast.hpp
 * @brief Defines scoped CPU and CUDA autocast state ownership.
 */

#include <jyppx/liuding/foundation/error.hpp>

#include <ATen/autocast_mode.h>
#include <c10/core/DeviceType.h>

namespace jyppx::liuding::training::amp {

/**
 * @class AutocastGuard
 * @brief RAII guard enabling autocast for the current scope.
 *
 * Wraps the ATen autocast thread-local state: while the guard is alive, ops
 * dispatched on `device_type` run in `dtype` (e.g. float16 on CUDA). The
 * previous state is restored on destruction.
 *
 * @example
 * @code
 * {
 *     amp::AutocastGuard guard{c10::DeviceType::CUDA, at::kHalf};
 *     Tensor loss = loss_fn(model->forward(input), target);
 * }  // autocast disabled again here
 * loss.backward();
 * @endcode
 */
class AutocastGuard {
 public:
  /**
   * @brief Enables autocast for one CPU or CUDA scope.
   *
   * CPU autocast accepts only BFloat16. CUDA autocast accepts Half and
   * BFloat16. Construction is transactional: a failure restores the
   * thread-local autocast state observed on entry.
   *
   * @param device_type Device whose autocast dispatcher is configured.
   * @param dtype Lower-precision dtype used by eligible operations.
   * @throws jyppx::liuding::InvalidArgument if the device/dtype pair is unsupported.
   */
  AutocastGuard(c10::DeviceType device_type, at::ScalarType dtype)
      : device_type_(device_type) {
    const bool cpu_dtype_supported =
        device_type == c10::DeviceType::CPU && dtype == at::kBFloat16;
    const bool cuda_dtype_supported =
        device_type == c10::DeviceType::CUDA &&
        (dtype == at::kHalf || dtype == at::kBFloat16);
    if (!cpu_dtype_supported && !cuda_dtype_supported) {
      throw ::jyppx::liuding::InvalidArgument(
          "Autocast rejected unsupported device/dtype pair",
          ::jyppx::liuding::ErrorDetails{
              .component = "training.amp.autocast",
              .operation = "device_dtype",
              .expected = "CPU/BFloat16 or CUDA/Half|BFloat16",
              .actual = "unsupported device/dtype pair",
              .hint = "Use CPU/BFloat16 or CUDA/Half|BFloat16"});
    }

    previous_dtype_ = at::autocast::get_autocast_dtype(device_type_);
    was_enabled_ = at::autocast::is_autocast_enabled(device_type_);
    try {
      at::autocast::set_autocast_dtype(device_type_, dtype);
      at::autocast::set_autocast_enabled(device_type_, true);
      at::autocast::increment_nesting();
      active_ = true;
    } catch (...) {
      at::autocast::set_autocast_enabled(device_type_, was_enabled_);
      at::autocast::set_autocast_dtype(device_type_, previous_dtype_);
      throw;
    }
  }

  /** @brief Restores the complete thread-local autocast state. */
  ~AutocastGuard() {
    if (!active_) {
      return;
    }
    const int nesting = at::autocast::decrement_nesting();
    at::autocast::set_autocast_enabled(device_type_, was_enabled_);
    at::autocast::set_autocast_dtype(device_type_, previous_dtype_);
    if (nesting == 0) {
      at::autocast::clear_cache();
    }
  }

  /** @brief Copying is disabled because one scope owns one state restore. */
  AutocastGuard(const AutocastGuard&) = delete;

  /** @brief Copy assignment is disabled for scoped state ownership. */
  AutocastGuard& operator=(const AutocastGuard&) = delete;

  /** @brief Moving is disabled so destruction order remains explicit. */
  AutocastGuard(AutocastGuard&&) = delete;

  /** @brief Move assignment is disabled for scoped state ownership. */
  AutocastGuard& operator=(AutocastGuard&&) = delete;

 private:
  c10::DeviceType device_type_;  ///< Device whose thread-local state is owned.
  at::ScalarType previous_dtype_ =
      at::kFloat;  ///< Autocast dtype observed on entry.
  bool was_enabled_ =
      false;             ///< Whether autocast was enabled before construction.
  bool active_ = false;  ///< Whether nesting was successfully incremented.
};

}  // namespace jyppx::liuding::training::amp

#endif  // JYPPX_LIUDING_TRAINING_AMP_AUTOCAST_HPP