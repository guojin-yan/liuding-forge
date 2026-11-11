// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_EXPERIMENT_LOCAL_EXECUTION_DEVICES_HPP
#define JYPPX_LIUDING_EXPERIMENT_LOCAL_EXECUTION_DEVICES_HPP
/** @file devices.hpp
 * @brief Physical GPU observations used by native local admission.
 */
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
namespace jyppx::liuding::experiment {
/** @brief A physical NVIDIA GPU observation; no CUDA context is created. */
struct LocalCudaDevice {
  std::string uuid;  ///< Full lowercase physical GPU UUID; MIG excluded.
  std::uint64_t total_memory_bytes = 0;  ///< Driver-observed total bytes.
  std::uint64_t free_memory_bytes = 0;   ///< Driver-observed free bytes now.
  std::uint32_t compute_major = 0;       ///< CUDA compute capability major.
  std::uint32_t compute_minor = 0;       ///< CUDA compute capability minor.
};
/** @brief Checks the full lowercase physical UUID syntax without loading NVML.
 * @param uuid Proposed physical GPU identity.
 * @return True only for the supported complete UUID form.
 */
bool local_cuda_uuid_is_valid(std::string_view uuid) noexcept;
/** @brief Reads physical GPU identities and capabilities using driver NVML.
 * CPU-only builds need no CUDA SDK; no CUDA context or reservation is created.
 * @return At most 64 physical GPU observations, each read completely.
 * @throws Error If NVML is unavailable or an observation is incomplete.
 */
std::vector<LocalCudaDevice> local_cuda_devices();
}  // namespace jyppx::liuding::experiment
#endif  // JYPPX_LIUDING_EXPERIMENT_LOCAL_EXECUTION_DEVICES_HPP
