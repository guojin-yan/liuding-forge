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

/** @brief A ROCm device bound to both HIP UUID bytes and PCI location.
 * Windows HIP visibility uses ordinals. The frozen ordinal is checked against
 * this identity before launch; enumeration drift is rejected, never remapped.
 */
struct LocalRocmDevice {
  std::string id;  ///< AMD-<PCI components separated by hyphens>-<32 hex UUID>.
  std::string pci_bus_id;  ///< Canonical lowercase dddd:bb:dd.f PCI location.
  std::string uuid;  ///< 32 lowercase hex digits encoding HIP's 16 UUID bytes.
  std::string
      architecture;  ///< Exact HIP gcnArchName, including feature suffixes.
  std::uint32_t ordinal = 0;  ///< Ordinal in the unmasked HIP inventory.
  std::uint64_t total_memory_bytes = 0;  ///< HIP-observed usable total bytes.
  std::uint64_t free_memory_bytes =
      0;  ///< Conservative observed free bytes now.
  std::string
      memory_source;  ///< hip, or hip_wddm for global Windows accounting.
};
/** @brief Validates the complete AMD PCI/UUID identity without loading HIP.
 * @param id Proposed identity from the ROCm inventory.
 * @return True only for canonical full PCI/UUID syntax.
 */
bool local_rocm_id_is_valid(std::string_view id) noexcept;
/** @brief Queries unmasked HIP devices, preserving the caller's current device.
 * Requires a ROCm build and an unmasked launcher. HIP_VISIBLE_DEVICES,
 * CUDA_VISIBLE_DEVICES, ROCR_VISIBLE_DEVICES and GPU_DEVICE_ORDINAL must be
 * absent throughout launcher lifetime, including before HIP initialization.
 * Never modify process visibility after initializing HIP. Querying memory may
 * initialize HIP device contexts; this is not a context-free driver query.
 * Windows free memory is the minimum of HIP and adapter-wide WDDM dedicated
 * segment availability (using the larger committed/resident occupancy).
 * It is a sample, not an allocation guarantee or enforced memory quota.
 * @return At most 64 complete device observations, sorted by stable identity.
 * @throws Error On unsupported build, masks, invalid identity or HIP failure.
 */
std::vector<LocalRocmDevice> local_rocm_devices();
}  // namespace jyppx::liuding::experiment
#endif  // JYPPX_LIUDING_EXPERIMENT_LOCAL_EXECUTION_DEVICES_HPP
