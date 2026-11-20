// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

// Independent test observer: does not use the product's inventory or formatter.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <hip/hip_runtime_api.h>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
void check(hipError_t status) {
  if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
}  // namespace

int main(int argc, char** argv) {
  try {
    int count = 0;
    check(hipGetDeviceCount(&count));
    if (count != 1) throw std::runtime_error("Expected one visible HIP device");
    check(hipSetDevice(0));
    hipDeviceProp_t properties{};
    check(hipGetDeviceProperties(&properties, 0));
    hipUUID uuid{};
    check(hipDeviceGetUuid(&uuid, 0));
    std::ostringstream encoded;
    encoded << std::hex << std::setfill('0');
    for (const auto byte : uuid.bytes)
      encoded << std::setw(2)
              << static_cast<unsigned int>(static_cast<unsigned char>(byte));
    std::array<char, 32> pci{};
    check(hipDeviceGetPCIBusId(pci.data(), static_cast<int>(pci.size()), 0));
    const bool hold = argc == 3 && std::string_view(argv[1]) == "--hold";
    const auto size = hold ? std::size_t{2} << 30U : std::size_t{64} << 10U;
    void* memory = nullptr;
    check(hipMalloc(&memory, size));
    check(hipMemset(memory, 0x3a, size));
    std::vector<unsigned char> host(64U << 10U);
    check(hipMemcpy(host.data(), memory, host.size(), hipMemcpyDeviceToHost));
    if (!std::all_of(host.begin(), host.end(),
                     [](auto v) { return v == 0x3a; }))
      throw std::runtime_error("GPU allocation contents differ from reference");
    check(hipDeviceSynchronize());
    auto masks = nlohmann::json::object();
    for (const auto* name : {"HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                             "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"}) {
      const auto* value = std::getenv(name);
      masks[name] = value ? value : "";
    }
    const auto data =
        nlohmann::json{{"pci_bus_id", pci.data()},
                       {"uuid", encoded.str()},
                       {"architecture", properties.gcnArchName},
                       {"device_count", count},
                       {"logical_device", 0},
                       {"visibility", masks},
                       {"total_memory_bytes", properties.totalGlobalMem},
                       {"allocation_verified", true}};
    if (hold) {
      const auto ready = std::filesystem::u8path(argv[2]);
      std::ofstream(ready) << data.dump();
      const auto until =
          std::chrono::steady_clock::now() + std::chrono::seconds(30);
      while (std::filesystem::exists(ready) &&
             std::chrono::steady_clock::now() < until)
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    } else {
      std::ofstream("launches.txt", std::ios::app) << "rocm\n";
      std::cout << data.dump() << '\n';
    }
    check(hipFree(memory));
    check(hipDeviceReset());
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
