#ifndef JYPPX_CORE_DEVICE_HPP
#define JYPPX_CORE_DEVICE_HPP

#include <jyppx/core/types.hpp>
#include <string>

namespace jyppx {

/**
 * @brief Get default device (CUDA if available, otherwise CPU)
 */
Device get_default_device();

/**
 * @brief Check if CUDA is available
 */
bool cuda_is_available();

/**
 * @brief Get number of CUDA devices
 */
int cuda_device_count();

/**
 * @brief Parse device string (e.g., "cpu", "cuda", "cuda:0")
 */
Device parse_device(const std::string& device_str);

} // namespace jyppx

#endif // JYPPX_CORE_DEVICE_HPP
