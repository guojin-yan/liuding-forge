// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_BENCHMARK_ACCELERATOR_SUPPORT_HPP
#define JYPPX_BENCHMARK_ACCELERATOR_SUPPORT_HPP

#include <jyppx/liuding/foundation/version.hpp>

#include <torch/cuda.h>
#include <torch/utils.h>

#ifdef JYPPX_BENCH_ROCM
#include <ATen/hip/HIPContextLight.h>
#include <c10/hip/HIPCachingAllocator.h>
#include <hip/hip_runtime_api.h>
#define JYPPX_BENCH_BACKEND "rocm"
#else
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDACachingAllocator.h>
#define JYPPX_BENCH_BACKEND "cuda"
#endif

#include <algorithm>
#include <array>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace jyppx::liuding::benchmarks {

/** @brief Returns the native identity of visible accelerator zero or throws. */
inline nlohmann::json accelerator_identity() {
  const auto* properties = at::cuda::getDeviceProperties(0);
  if (properties == nullptr)
    throw std::runtime_error("missing device properties");
#ifdef JYPPX_BENCH_ROCM
  const auto check = [](hipError_t status) {
    if (status != hipSuccess)
      throw std::runtime_error(std::string("HIP identity query failed: ") +
                               hipGetErrorString(status));
  };
  const auto bounded_text = [](const auto& buffer) {
    const auto end = std::find(std::begin(buffer), std::end(buffer), '\0');
    if (end == std::begin(buffer) || end == std::end(buffer))
      throw std::runtime_error("empty or unterminated HIP identity field");
    return std::string(std::begin(buffer), end);
  };
  hipUUID native_uuid{};
  check(hipDeviceGetUuid(&native_uuid, 0));
  std::string uuid;
  constexpr char digits[] = "0123456789abcdef";
  bool nonzero = false;
  for (const auto byte : native_uuid.bytes) {
    const auto value = static_cast<unsigned char>(byte);
    nonzero = nonzero || value != 0;
    uuid += digits[value >> 4];
    uuid += digits[value & 15];
  }
  if (!nonzero) throw std::runtime_error("HIP returned an empty device UUID");
  std::array<char, 32> pci{};
  check(hipDeviceGetPCIBusId(pci.data(), static_cast<int>(pci.size()), 0));
  int driver = 0, runtime = 0;
  check(hipDriverGetVersion(&driver));
  check(hipRuntimeGetVersion(&runtime));
  if (driver <= 0 || runtime <= 0 || properties->totalGlobalMem == 0)
    throw std::runtime_error("invalid HIP version or memory observation");
  return {{"device_index", 0},
          {"device_name", bounded_text(properties->name)},
          {"hip_uuid", uuid},
          {"pci_bus_id", bounded_text(pci)},
          {"architecture", bounded_text(properties->gcnArchName)},
          {"hip_driver_api_version", driver},
          {"hip_runtime_version", runtime},
          {"compile_hip_version", HIP_VERSION},
          {"total_memory_bytes", properties->totalGlobalMem}};
#else
  // Preserve the existing NVIDIA sample identity schema.
  return {{"device_index", 0},
          {"device_name", properties->name},
          {"compute_capability", std::to_string(properties->major) + "." +
                                     std::to_string(properties->minor)},
          {"compile_toolkit_version", CUDART_VERSION}};
#endif
}

/** @brief Reports the exact binary, native device and fixed thread settings. */
inline nlohmann::json benchmark_identity() {
  return {
      {"schema_version", 1},
      {"kind", "jyppx-" JYPPX_BENCH_BACKEND "-benchmark-identity"},
      {"build_fingerprint", nlohmann::json::parse(build_fingerprint_json())},
      {JYPPX_BENCH_BACKEND, accelerator_identity()},
      {"runtime",
       {{"intraop_threads", torch::get_num_threads()},
        {"interop_threads", torch::get_num_interop_threads()}}}};
}

}  // namespace jyppx::liuding::benchmarks

#endif  // JYPPX_BENCHMARK_ACCELERATOR_SUPPORT_HPP
