// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/experiment/local_execution/devices.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>

#if defined(JYPPX_LOCAL_WITH_HIP)
#include <hip/hip_runtime_api.h>
#endif

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#if defined(JYPPX_LOCAL_WITH_HIP)
#include <d3dkmthk.h>
#endif
#elif defined(__linux__)
#include <dlfcn.h>
#endif

namespace jyppx::liuding::experiment {
namespace {

#if defined(_WIN32) && defined(JYPPX_LOCAL_WITH_HIP)
// hipMemGetInfo on Windows can ignore allocations in other processes. Bind
// WDDM's adapter-wide dedicated segments to HIP's LUID and use the smaller
// free observation. Missing global accounting must fail closed.
std::uint64_t windows_rocm_free_memory(int ordinal) {
  std::array<char, 8> luid{};
  unsigned int node_mask = 0;
  if (hipDeviceGetLuid(luid.data(), &node_mask, ordinal) != hipSuccess ||
      node_mask == 0)
    throw CompatibilityError("HIP did not provide a Windows GPU LUID");
  D3DKMT_QUERYSTATISTICS query{};
  static_assert(sizeof(query.AdapterLuid) == luid.size());
  std::memcpy(&query.AdapterLuid, luid.data(), luid.size());
  const auto adapter = query.AdapterLuid;
  query.Type = D3DKMT_QUERYSTATISTICS_ADAPTER;
  if (D3DKMTQueryStatistics(&query) != 0)
    throw CompatibilityError("Cannot inspect Windows GPU memory segments");
  const auto count = query.QueryResult.AdapterInformation.NbSegments;
  if (count == 0 || count > 64)
    throw CompatibilityError("Unsupported Windows GPU memory segment count");
  std::uint64_t free = 0;
  bool dedicated = false;
  for (ULONG index = 0; index < count; ++index) {
    query = {};
    query.AdapterLuid = adapter;
    query.Type = D3DKMT_QUERYSTATISTICS_SEGMENT;
    query.QuerySegment.SegmentId = index;
    if (D3DKMTQueryStatistics(&query) != 0)
      throw CompatibilityError("Cannot inspect global Windows GPU memory");
    const auto& segment = query.QueryResult.SegmentInformation;
    if (segment.Aperture) continue;
    dedicated = true;
    const auto limit = segment.CommitLimit;
    if (limit == std::numeric_limits<std::uint64_t>::max())
      throw CompatibilityError("Unbounded Windows dedicated GPU segment");
    const auto used = std::max(segment.BytesCommitted, segment.BytesResident);
    const auto available = limit - std::min(limit, used);
    if (available > std::numeric_limits<std::uint64_t>::max() - free)
      throw CompatibilityError("Invalid Windows GPU memory counters");
    free += available;
  }
  if (!dedicated)
    throw CompatibilityError("Windows GPU has no dedicated memory segments");
  return free;
}
#endif

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

bool local_rocm_id_is_valid(std::string_view id) noexcept {
  if (id.size() != 49 || !id.starts_with("AMD-")) return false;
  bool nonzero = false;
  for (std::size_t index = 4; index < id.size(); ++index) {
    if (index == 8 || index == 11 || index == 14 || index == 16) {
      if (id[index] != '-') return false;
    } else {
      const auto ch = id[index];
      if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) return false;
      if ((index == 12 && ch > '1') || (index == 15 && ch > '7')) return false;
      if (index > 16 && ch != '0') nonzero = true;
    }
  }
  return nonzero;
}

std::vector<LocalRocmDevice> local_rocm_devices() {
#if defined(JYPPX_LOCAL_WITH_HIP)
  for (const auto* name : {"HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                           "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"}) {
#if defined(_WIN32)
    std::size_t size = 0;
    if (getenv_s(&size, nullptr, 0, name) != 0 || size != 0)
#else
    if (std::getenv(name) != nullptr)
#endif
      throw CompatibilityError(
          "ROCm physical placement requires an unmasked launcher environment");
  }
  const auto check = [](hipError_t status) {
    if (status != hipSuccess)
      throw Error(ErrorCode::backend, "Local ROCm device query failed",
                  ErrorDetails{.component = "experiment.local_rocm",
                               .actual = hipGetErrorString(status)});
  };
  int count = 0;
  const auto status = hipGetDeviceCount(&count);
  if (status == hipErrorNoDevice) return {};
  check(status);
  if (count < 0 || count > 64)
    throw ResourceExhausted("Local ROCm inspection allows at most 64 GPUs");
  if (count == 0) return {};
  int previous = 0;
  check(hipGetDevice(&previous));
  struct Restore {
    int device;
    ~Restore() { static_cast<void>(hipSetDevice(device)); }
  } restore{previous};
  std::vector<LocalRocmDevice> result;
  for (int index = 0; index < count; ++index) {
    hipDeviceProp_t properties{};
    check(hipGetDeviceProperties(&properties, index));
    hipUUID uuid{};
    check(hipDeviceGetUuid(&uuid, index));
    std::array<char, 32> pci{};
    check(
        hipDeviceGetPCIBusId(pci.data(), static_cast<int>(pci.size()), index));
    const auto end = std::find(pci.begin(), pci.end(), '\0');
    if (end == pci.end())
      throw CompatibilityError("HIP returned an invalid PCI location");
    LocalRocmDevice observation;
    observation.pci_bus_id.assign(pci.begin(), end);
    if (observation.pci_bus_id.size() != 12 ||
        observation.pci_bus_id[4] != ':' || observation.pci_bus_id[7] != ':' ||
        observation.pci_bus_id[10] != '.')
      throw CompatibilityError("HIP returned a noncanonical PCI location");
    constexpr char hex[] = "0123456789abcdef";
    for (const auto byte : uuid.bytes) {
      const auto value = static_cast<unsigned char>(byte);
      observation.uuid += hex[value >> 4];
      observation.uuid += hex[value & 15];
    }
    auto location = observation.pci_bus_id;
    std::replace(location.begin(), location.end(), ':', '-');
    std::replace(location.begin(), location.end(), '.', '-');
    observation.id = "AMD-" + location + "-" + observation.uuid;
    if (!local_rocm_id_is_valid(observation.id))
      throw CompatibilityError("HIP returned an invalid physical GPU identity");
    const auto arch_end = std::find(std::begin(properties.gcnArchName),
                                    std::end(properties.gcnArchName), '\0');
    if (arch_end == std::end(properties.gcnArchName))
      throw CompatibilityError("HIP returned an invalid GPU architecture");
    observation.architecture.assign(properties.gcnArchName, arch_end);
    if (!observation.architecture.starts_with("gfx"))
      throw CompatibilityError("HIP returned an unsupported GPU architecture");
    observation.ordinal = static_cast<std::uint32_t>(index);
    check(hipSetDevice(index));
    std::size_t free = 0;
    std::size_t total = 0;
    check(hipMemGetInfo(&free, &total));
    if (total == 0 || free > total)
      throw CompatibilityError("HIP returned invalid GPU memory counters");
    observation.total_memory_bytes = total;
    observation.free_memory_bytes = free;
    observation.memory_source = "hip";
#if defined(_WIN32)
    observation.free_memory_bytes = std::min(observation.free_memory_bytes,
                                             windows_rocm_free_memory(index));
    observation.memory_source = "hip_wddm";
#endif
    result.push_back(std::move(observation));
  }
  check(hipSetDevice(previous));
  std::sort(result.begin(), result.end(),
            [](const auto& a, const auto& b) { return a.id < b.id; });
  for (std::size_t index = 1; index < result.size(); ++index)
    if (result[index - 1].pci_bus_id == result[index].pci_bus_id)
      throw CompatibilityError("HIP returned duplicate physical PCI locations");
  return result;
#else
  throw NotImplemented("Local ROCm device inspection requires a ROCm build");
#endif
}

}  // namespace jyppx::liuding::experiment
