// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "local_cuda_device.hpp"

#include <jyppx/liuding/foundation/error.hpp>

#include <algorithm>
#include <array>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <dlfcn.h>
#endif

namespace jyppx::liuding::cli {
namespace {

// Stable NVML C ABI. No CUDA/NVML headers, link-time SDK or CUDA context is
// required. These are the documented v1 memory structure and opaque handle.
struct NvmlDeviceOpaque;
using NvmlDevice = NvmlDeviceOpaque*;
struct NvmlMemory {
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
};

class Nvml final {
 public:
  Nvml() {
#if defined(_WIN32)
    library_ =
        LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
#elif defined(__linux__)
    library_ = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
    if (!library_)
      throw NotImplemented("NVIDIA driver NVML is unavailable on this host");
    try {
      initialize_ = symbol<int (*)()>("nvmlInit_v2");
      shutdown_ = symbol<int (*)()>("nvmlShutdown");
      count_ = symbol<int (*)(unsigned int*)>("nvmlDeviceGetCount_v2");
      device_ = symbol<int (*)(unsigned int, NvmlDevice*)>(
          "nvmlDeviceGetHandleByIndex_v2");
      uuid_ =
          symbol<int (*)(NvmlDevice, char*, unsigned int)>("nvmlDeviceGetUUID");
      memory_ =
          symbol<int (*)(NvmlDevice, NvmlMemory*)>("nvmlDeviceGetMemoryInfo");
      compute_ = symbol<int (*)(NvmlDevice, int*, int*)>(
          "nvmlDeviceGetCudaComputeCapability");
      check(initialize_());
      initialized_ = true;
    } catch (...) {
      close();
      throw;
    }
  }

  ~Nvml() { close(); }
  Nvml(const Nvml&) = delete;
  Nvml& operator=(const Nvml&) = delete;

  std::vector<LocalCudaDevice> devices() const {
    unsigned int count = 0;
    check(count_(&count));
    if (count > 64)
      throw ResourceExhausted("Local CUDA inspection allows at most 64 GPUs");
    std::vector<LocalCudaDevice> result;
    for (unsigned int index = 0; index < count; ++index) {
      NvmlDevice device = nullptr;
      check(device_(index, &device));
      std::array<char, 96> uuid{};
      check(uuid_(device, uuid.data(), static_cast<unsigned int>(uuid.size())));
      const auto end = std::find(uuid.begin(), uuid.end(), '\0');
      LocalCudaDevice observation;
      observation.uuid.assign(uuid.begin(), end);
      if (end == uuid.end() || !local_cuda_uuid_is_valid(observation.uuid))
        throw CompatibilityError(
            "NVML returned an unsupported physical GPU UUID");
      NvmlMemory memory{};
      check(memory_(device, &memory));
      int major = 0;
      int minor = 0;
      check(compute_(device, &major, &minor));
      if (memory.total == 0 || memory.free > memory.total || major <= 0 ||
          major > 99 || minor < 0 || minor > 99)
        throw CompatibilityError("NVML returned invalid GPU capabilities");
      observation.total_memory_bytes = memory.total;
      observation.free_memory_bytes = memory.free;
      observation.compute_major = static_cast<std::uint32_t>(major);
      observation.compute_minor = static_cast<std::uint32_t>(minor);
      result.push_back(std::move(observation));
    }
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) { return a.uuid < b.uuid; });
    for (std::size_t index = 1; index < result.size(); ++index)
      if (result[index - 1].uuid == result[index].uuid)
        throw CompatibilityError("NVML returned duplicate physical GPU UUIDs");
    return result;
  }

 private:
  static void check(int status) {
    if (status != 0)
      throw Error(ErrorCode::backend, "NVIDIA driver NVML query failed",
                  ErrorDetails{.component = "cli.local_cuda",
                               .actual = std::to_string(status)});
  }

  template <typename T>
  T symbol(const char* name) const {
#if defined(_WIN32)
    const auto address = GetProcAddress(library_, name);
#elif defined(__linux__)
    const auto address = dlsym(library_, name);
#else
    const void* address = nullptr;
    static_cast<void>(name);
#endif
    if (!address)
      throw NotImplemented("NVIDIA driver lacks a required NVML entry point");
    return reinterpret_cast<T>(address);
  }

  void close() noexcept {
    if (initialized_) static_cast<void>(shutdown_());
    initialized_ = false;
    if (library_) {
#if defined(_WIN32)
      FreeLibrary(library_);
#elif defined(__linux__)
      dlclose(library_);
#endif
      library_ = nullptr;
    }
  }

#if defined(_WIN32)
  HMODULE library_ = nullptr;
#else
  void* library_ = nullptr;
#endif
  bool initialized_ = false;
  int (*initialize_)() = nullptr;
  int (*shutdown_)() = nullptr;
  int (*count_)(unsigned int*) = nullptr;
  int (*device_)(unsigned int, NvmlDevice*) = nullptr;
  int (*uuid_)(NvmlDevice, char*, unsigned int) = nullptr;
  int (*memory_)(NvmlDevice, NvmlMemory*) = nullptr;
  int (*compute_)(NvmlDevice, int*, int*) = nullptr;
};

}  // namespace

bool local_cuda_uuid_is_valid(std::string_view uuid) noexcept {
  if (uuid.size() != 40 || !uuid.starts_with("GPU-")) return false;
  for (std::size_t index = 4; index < uuid.size(); ++index) {
    if (index == 12 || index == 17 || index == 22 || index == 27) {
      if (uuid[index] != '-') return false;
    } else if (!((uuid[index] >= '0' && uuid[index] <= '9') ||
                 (uuid[index] >= 'a' && uuid[index] <= 'f'))) {
      return false;
    }
  }
  return true;
}

std::vector<LocalCudaDevice> local_cuda_devices() { return Nvml{}.devices(); }

}  // namespace jyppx::liuding::cli
