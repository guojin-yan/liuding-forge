// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_BENCHMARK_RESOURCE_TORCH_SUPPORT_HPP
#define JYPPX_BENCHMARK_RESOURCE_TORCH_SUPPORT_HPP
#include "resource_support.hpp"

#include <torch/torch.h>

#ifdef JYPPX_RESOURCE_CUDA
#include <torch/cuda.h>

#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace jyppx::benchmarks {
inline torch::Device configure_device(const Arguments& args, Json& sample) {
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  torch::manual_seed(20261021);
  sample["execution"]["intraop_threads"] = torch::get_num_threads();
  sample["execution"]["interop_threads"] = torch::get_num_interop_threads();
  if (args.device == "cpu") return torch::Device(torch::kCPU);
#ifdef JYPPX_RESOURCE_CUDA
  require(torch::cuda::is_available(),
          "CUDA requested but unavailable at runtime");
  const auto* properties = at::cuda::getDeviceProperties(0);
  std::string uuid = "GPU-";
  constexpr char hex[] = "0123456789abcdef";
  for (std::size_t index = 0; index < 16; ++index) {
    if (index == 4 || index == 6 || index == 8 || index == 10) uuid += '-';
    const auto byte = static_cast<unsigned char>(properties->uuid.bytes[index]);
    uuid += hex[byte >> 4];
    uuid += hex[byte & 15];
  }
  int driver_version = 0, runtime_version = 0;
  require(cudaDriverGetVersion(&driver_version) == cudaSuccess &&
              cudaRuntimeGetVersion(&runtime_version) == cudaSuccess,
          "cannot query actual CUDA runtime/driver versions");
  sample["execution"]["cuda"] = {
      {"device_index", 0},
      {"device_name", properties->name},
      {"device_uuid", uuid},
      {"driver_api_version", driver_version},
      {"runtime_version", runtime_version},
      {"compute_capability", std::to_string(properties->major) + "." +
                                 std::to_string(properties->minor)},
      {"compile_toolkit_version", CUDART_VERSION}};
  sample["execution"]["timing"] =
      "steady_clock with cuda:0 synchronize before and after each stage";
  torch::cuda::synchronize(0);
  c10::cuda::CUDACachingAllocator::resetPeakStats(0);
  return torch::Device(torch::kCUDA, 0);
#else
  throw std::runtime_error("CUDA requested from a CPU-only benchmark binary");
#endif
}

inline void synchronize(const torch::Device& device) {
#ifdef JYPPX_RESOURCE_CUDA
  if (device.is_cuda()) torch::cuda::synchronize(0);
#else
  require(!device.is_cuda(), "CUDA synchronization is unavailable");
#endif
}

template <class Function>
double device_seconds(const torch::Device& device, Function&& operation) {
  synchronize(device);
  return seconds([&] {
    std::forward<Function>(operation)();
    synchronize(device);
  });
}

inline void verify_tensor(const torch::Tensor& tensor,
                          const torch::Device& device, std::int64_t elements) {
  require(tensor.defined() && tensor.device() == device &&
              tensor.scalar_type() == torch::kFloat32 &&
              tensor.numel() == elements &&
              torch::isfinite(tensor).all().item<bool>(),
          "output dtype/device/extent/finiteness mismatch");
}

inline void cuda_metrics(const torch::Device& device, Json& sample) {
#ifdef JYPPX_RESOURCE_CUDA
  if (device.is_cuda()) {
    synchronize(device);
    const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
    sample["metrics"]["peak_cuda_allocated_bytes"] =
        metric(static_cast<double>(stats.allocated_bytes[0].peak), "bytes",
               "lower", true,
               "LibTorch caching allocator device 0 peak since before load; "
               "excludes driver/context");
    sample["metrics"]["peak_cuda_reserved_bytes"] =
        metric(static_cast<double>(stats.reserved_bytes[0].peak), "bytes",
               "lower", true,
               "LibTorch caching allocator device 0 peak since before load; "
               "excludes driver/context");
  }
#else
  static_cast<void>(device);
  static_cast<void>(sample);
#endif
}
}  // namespace jyppx::benchmarks
#endif
