#ifndef JYPPX_CORE_TENSOR_UTILS_HPP
#define JYPPX_CORE_TENSOR_UTILS_HPP

#include <jyppx/core/types.hpp>

namespace jyppx {

/**
 * @brief Check if tensor contains NaN values
 */
bool has_nan(const Tensor& tensor);

/**
 * @brief Check if tensor contains Inf values
 */
bool has_inf(const Tensor& tensor);

/**
 * @brief Check if tensor is finite (no NaN or Inf)
 */
bool is_finite(const Tensor& tensor);

/**
 * @brief Clone a tensor list
 */
TensorList clone_tensors(const TensorList& tensors);

/**
 * @brief Move tensors to device
 */
TensorList to_device(const TensorList& tensors, const Device& device);

} // namespace jyppx

#endif // JYPPX_CORE_TENSOR_UTILS_HPP
