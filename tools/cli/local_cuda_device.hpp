// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_TOOLS_CLI_LOCAL_CUDA_DEVICE_HPP
#define JYPPX_LIUDING_TOOLS_CLI_LOCAL_CUDA_DEVICE_HPP

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::liuding::cli {

/** @brief A physical NVIDIA GPU observation; no CUDA context is created. */
struct LocalCudaDevice {
  std::string uuid;
  std::uint64_t total_memory_bytes = 0;
  std::uint64_t free_memory_bytes = 0;
  std::uint32_t compute_major = 0;
  std::uint32_t compute_minor = 0;
};

/** @brief Accepts only a full, lowercase physical GPU UUID, excluding MIG. */
bool local_cuda_uuid_is_valid(std::string_view uuid) noexcept;

/** @brief Reads physical GPU identities and capabilities using driver NVML.
 * Loads the installed driver at runtime; CPU-only builds need no CUDA SDK.
 * Throws if NVML is unavailable or an observation cannot be read completely.
 */
std::vector<LocalCudaDevice> local_cuda_devices();

}  // namespace jyppx::liuding::cli

#endif  // JYPPX_LIUDING_TOOLS_CLI_LOCAL_CUDA_DEVICE_HPP
