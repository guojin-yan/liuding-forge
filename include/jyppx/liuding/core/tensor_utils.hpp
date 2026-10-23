// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_CORE_TENSOR_UTILS_HPP
#define JYPPX_LIUDING_CORE_TENSOR_UTILS_HPP
/**
 * @file tensor_utils.hpp
 * @brief Provides common tensor validation, cloning, and placement helpers.
 */

#include "jyppx/liuding/core/types.hpp"

namespace jyppx::liuding {

/**
 * @brief Tests whether a defined tensor contains at least one NaN value.
 * @param tensor Tensor to inspect.
 * @return `true` when `tensor` is defined and contains NaN; otherwise `false`.
 */
bool has_nan(const Tensor& tensor);

/**
 * @brief Tests whether a defined tensor contains positive or negative infinity.
 * @param tensor Tensor to inspect.
 * @return `true` when `tensor` is defined and contains infinity; otherwise
 * `false`.
 */
bool has_inf(const Tensor& tensor);

/**
 * @brief Tests whether every value in a defined tensor is finite.
 * @param tensor Tensor to inspect.
 * @return `true` only when `tensor` is defined and all values are finite.
 */
bool is_finite(const Tensor& tensor);

/**
 * @brief Clones every defined tensor while preserving undefined entries.
 * @param tensors Tensor handles to clone.
 * @return A list with the same size and independent storage for each defined
 * tensor.
 */
TensorList clone_tensors(const TensorList& tensors);

/**
 * @brief Moves every defined tensor to a compute device.
 * @param tensors Tensor handles to move.
 * @param device Destination device.
 * @return A list with the same size; undefined entries remain undefined.
 * @throws c10::Error If LibTorch cannot perform a requested transfer.
 */
TensorList to_device(const TensorList& tensors, const Device& device);

}  // namespace jyppx::liuding

#endif  // JYPPX_LIUDING_CORE_TENSOR_UTILS_HPP