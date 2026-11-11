// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

// Test-only CUDA runtime observer. Never installed or used by product code.
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace {
void check(cudaError_t result) {
  if (result != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(result));
}

std::string uuid_text(const cudaUUID_t& uuid) {
  std::ostringstream result;
  result << "GPU-" << std::hex << std::setfill('0');
  for (std::size_t index = 0; index < sizeof(uuid.bytes); ++index) {
    if (index == 4 || index == 6 || index == 8 || index == 10) result << '-';
    result << std::setw(2)
           << static_cast<unsigned int>(
                  static_cast<unsigned char>(uuid.bytes[index]));
  }
  return result.str();
}
}  // namespace

int main(int argc, char** argv) {
  try {
    int count = 0;
    check(cudaGetDeviceCount(&count));
    if (count != 1)
      throw std::runtime_error("Expected exactly one visible CUDA device");
    check(cudaSetDevice(0));
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, 0));
    const bool hold = argc == 3 && std::string_view(argv[1]) == "--hold";
    void* allocation = nullptr;
    check(cudaMalloc(&allocation, hold ? 256U << 20U : 64U << 10U));
    check(cudaMemset(allocation, 0, hold ? 256U << 20U : 64U << 10U));
    check(cudaDeviceSynchronize());
    const auto* visible = std::getenv("CUDA_VISIBLE_DEVICES");
    const auto data = nlohmann::json{
        {"uuid", uuid_text(properties.uuid)},
        {"device_count", count},
        {"logical_device", 0},
        {"visible_devices", visible ? visible : ""},
        {"compute_capability", {properties.major, properties.minor}},
        {"total_memory_bytes", properties.totalGlobalMem}};
    if (hold) {
      const auto ready = std::filesystem::u8path(argv[2]);
      std::ofstream(ready) << data.dump();
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(30);
      while (std::filesystem::exists(ready) &&
             std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    } else {
      std::ofstream("launches.txt", std::ios::app) << "cuda\n";
      std::cout << data.dump() << '\n';
    }
    check(cudaFree(allocation));
    check(cudaDeviceReset());
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
