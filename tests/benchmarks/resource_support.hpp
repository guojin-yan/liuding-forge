// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_BENCHMARK_RESOURCE_SUPPORT_HPP
#define JYPPX_BENCHMARK_RESOURCE_SUPPORT_HPP

#include <jyppx/foundation/sha256.hpp>
#include <jyppx/foundation/version.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
// clang-format off
#include <Windows.h>
#include <Psapi.h>
// clang-format on
#endif

namespace jyppx::benchmarks {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

inline void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

struct Arguments {
  std::string profile;
  std::string device = "cpu";
  std::filesystem::path directory;
};

inline Arguments arguments(int argc, char** argv) {
  Arguments result;
  for (int index = 1; index < argc; index += 2) {
    require(index + 1 < argc, "each argument requires a value");
    const std::string_view option(argv[index]);
    if (option == "--profile")
      result.profile = argv[index + 1];
    else if (option == "--device")
      result.device = argv[index + 1];
    else if (option == "--work-dir")
      result.directory = argv[index + 1];
    else
      throw std::runtime_error("unknown argument: " + std::string(option));
  }
  require(!result.profile.empty(), "--profile is required");
  require(!result.directory.empty(), "--work-dir is required");
  require(result.device == "cpu" || result.device == "cuda:0",
          "device must be cpu or cuda:0");
  require(std::filesystem::create_directory(result.directory),
          "--work-dir must be a new directory with an existing parent");
  return result;
}

template <class Function>
double seconds(Function&& operation) {
  const auto start = Clock::now();
  std::forward<Function>(operation)();
  const auto elapsed =
      std::chrono::duration<double>(Clock::now() - start).count();
  require(std::isfinite(elapsed) && elapsed > 0, "invalid measured duration");
  return elapsed;
}

inline Json unavailable(std::string reason, std::string unit,
                        std::string direction = "lower") {
  return {{"status", "unavailable"},
          {"reason", std::move(reason)},
          {"unit", std::move(unit)},
          {"direction", std::move(direction)},
          {"gate", false}};
}

inline Json metric(double value, std::string unit,
                   std::string direction = "lower", bool gate = true,
                   std::string scope = {}) {
  require(std::isfinite(value) && value > 0,
          "metric must be finite and positive");
  return {{"status", "available"},
          {"value", value},
          {"unit", std::move(unit)},
          {"direction", std::move(direction)},
          {"gate", gate},
          {"scope", std::move(scope)}};
}

inline Json peak_rss() {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS memory{};
  if (GetProcessMemoryInfo(GetCurrentProcess(), &memory, sizeof(memory)) &&
      memory.PeakWorkingSetSize > 0) {
    return metric(static_cast<double>(memory.PeakWorkingSetSize), "bytes",
                  "lower", true,
                  "process lifetime PeakWorkingSetSize; includes runtime");
  }
  return unavailable("GetProcessMemoryInfo failed", "bytes");
#elif defined(__linux__)
  std::ifstream status("/proc/self/status");
  std::string field;
  while (status >> field) {
    if (field == "VmHWM:") {
      std::uint64_t kib = 0;
      std::string unit;
      if (status >> kib >> unit && unit == "kB" && kib > 0) {
        return metric(
            static_cast<double>(kib) * 1024.0, "bytes", "lower", true,
            "process lifetime /proc/self/status VmHWM; includes runtime");
      }
      break;
    }
    std::string rest;
    std::getline(status, rest);
  }
  return unavailable("Linux VmHWM could not be read", "bytes");
#else
  return unavailable("process RSS measurement unsupported on this platform",
                     "bytes");
#endif
}

inline std::string digest(std::string_view text) {
  return "sha256:" +
         sha256_hex(std::as_bytes(std::span(text.data(), text.size())));
}

inline Json report(const Arguments& args, Json workload,
                   std::uint64_t operations) {
  const auto descriptor = workload.dump();
  return {{"schema_version", 1},
          {"kind", "jyppx-resource-benchmark-sample"},
          {"profile", args.profile},
          {"workload",
           {{"id", args.profile + "/v1"},
            {"descriptor", std::move(workload)},
            {"digest", digest(descriptor)}}},
          {"build_fingerprint", Json::parse(build_fingerprint_json())},
          {"execution", {{"device", args.device}, {"timing", "steady_clock"}}},
          {"validation",
           {{"expected_operations", operations},
            {"completed_operations", 0},
            {"finite_output", false},
            {"content_verified", false}}},
          {"metrics", Json::object()}};
}

inline void finish(Json& sample, std::uint64_t completed) {
  require(sample["validation"]["expected_operations"].get<std::uint64_t>() ==
              completed,
          "frozen workload did not complete");
  sample["validation"]["completed_operations"] = completed;
  sample["validation"]["finite_output"] = true;
  sample["validation"]["content_verified"] = true;
  sample["metrics"]["peak_rss_bytes"] = peak_rss();
  if (!sample["metrics"].contains("peak_cuda_allocated_bytes")) {
    sample["metrics"]["peak_cuda_allocated_bytes"] =
        unavailable("CPU execution", "bytes");
    sample["metrics"]["peak_cuda_reserved_bytes"] =
        unavailable("CPU execution", "bytes");
  }
  std::cout << sample.dump() << '\n';
}

inline std::uint64_t directory_bytes(const std::filesystem::path& root) {
  std::uint64_t size = 0;
  for (const auto& entry : std::filesystem::recursive_directory_iterator(root))
    if (entry.is_regular_file()) size += entry.file_size();
  return size;
}
}  // namespace jyppx::benchmarks
#endif
