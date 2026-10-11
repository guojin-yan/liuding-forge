#ifndef JYPPX_NN_INIT_HPP
#define JYPPX_NN_INIT_HPP

#include <jyppx/core/types.hpp>

namespace jyppx {
namespace nn {
namespace init {

/**
 * @brief Initialize tensor with uniform distribution
 */
void uniform_(Tensor& tensor, double a = 0.0, double b = 1.0);

/**
 * @brief Initialize tensor with normal distribution
 */
void normal_(Tensor& tensor, double mean = 0.0, double std = 1.0);

/**
 * @brief Initialize tensor with constant value
 */
void constant_(Tensor& tensor, double val);

/**
 * @brief Initialize tensor with zeros
 */
void zeros_(Tensor& tensor);

/**
 * @brief Initialize tensor with ones
 */
void ones_(Tensor& tensor);

/**
 * @brief Xavier uniform initialization
 */
void xavier_uniform_(Tensor& tensor, double gain = 1.0);

/**
 * @brief Xavier normal initialization
 */
void xavier_normal_(Tensor& tensor, double gain = 1.0);

/**
 * @brief Kaiming uniform initialization (He initialization)
 */
void kaiming_uniform_(Tensor& tensor, double a = 0.0, const std::string& mode = "fan_in");

/**
 * @brief Kaiming normal initialization (He initialization)
 */
void kaiming_normal_(Tensor& tensor, double a = 0.0, const std::string& mode = "fan_in");

} // namespace init
} // namespace nn
} // namespace jyppx

#endif // JYPPX_NN_INIT_HPP
