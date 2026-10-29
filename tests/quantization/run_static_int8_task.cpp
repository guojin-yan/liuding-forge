// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "../../tools/quantization/exclusive_file.hpp"

#include <jyppx/liuding/deployment/onnx_runtime/session.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::byte> ReadBytes(const std::filesystem::path& path,
                                 std::uintmax_t maximum) {
  const auto count = std::filesystem::file_size(path);
  if (count == 0U || count > maximum)
    throw std::runtime_error("Task input violates its byte bound");
  std::vector<std::byte> bytes(static_cast<std::size_t>(count));
  std::ifstream input(path, std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size())))
    throw std::runtime_error("Cannot read task bytes");
  return bytes;
}

void WriteBytes(const std::filesystem::path& path,
                std::span<const std::byte> bytes) {
  jyppx::liuding::quantization::tools_detail::PublishExclusive(path, bytes);
}

double Milliseconds(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
  if (argc != 3) {
    std::cerr << "Usage: liuding-forge-int8-task MODEL.onnx TASK_DIRECTORY\n";
    return 2;
  }
  try {
    static_assert(std::endian::native == std::endian::little);
    namespace rt = jyppx::liuding::onnx_runtime;
    const std::filesystem::path model_path(argv[1]);
    const std::filesystem::path root(argv[2]);
    const auto model = ReadBytes(model_path, 16U << 20U);
    const auto digest = jyppx::liuding::sha256_hex(model);
    constexpr std::size_t kSamples = 10000U;
    constexpr std::size_t kInputFeatures = 784U;
    constexpr std::size_t kOutputFeatures = 10U;
    const auto raw_input = ReadBytes(root / "input.f32",
                                     kSamples * kInputFeatures * sizeof(float));
    const auto raw_labels = ReadBytes(root / "labels.u8", kSamples);
    if (raw_input.size() != kSamples * kInputFeatures * sizeof(float) ||
        raw_labels.size() != kSamples)
      throw std::runtime_error("Incorrect exact task fixture byte count");
    std::vector<float> input(kSamples * kInputFeatures);
    // Explicit copy avoids interpreting byte vector storage as aligned floats.
    auto destination = std::as_writable_bytes(std::span(input));
    std::copy(raw_input.begin(), raw_input.end(), destination.begin());
    rt::SessionOptions options;
    options.provider.name = "CPUExecutionProvider";
    options.allow_cpu_fallback = false;
    options.cpu_threads = 1U;
    options.disable_graph_optimizations = true;
    options.profile_prefix =
        root / (model_path.stem().string() + "-native-trace");
    const auto load_start = std::chrono::steady_clock::now();
    rt::Session session(model, {.hex = digest}, options);
    const auto load_ms = Milliseconds(load_start);
    if (session.runtime_version() != "1.23.2" ||
        session.execution_provider().name != "CPUExecutionProvider")
      throw std::runtime_error("Unexpected actual runtime version or provider");
    std::vector<float> output;
    output.reserve(kSamples * kOutputFeatures);
    constexpr std::size_t kBatchSize = 64U;
    double first_ms = 0.0;
    const auto all_start = std::chrono::steady_clock::now();
    for (std::size_t offset = 0U; offset < kSamples; offset += kBatchSize) {
      const auto count = std::min(kBatchSize, kSamples - offset);
      const std::array<std::int64_t, 2U> shape{
          static_cast<std::int64_t>(count),
          static_cast<std::int64_t>(kInputFeatures)};
      const auto run_start = std::chrono::steady_clock::now();
      const auto result =
          session.run(std::span(input).subspan(offset * kInputFeatures,
                                               count * kInputFeatures),
                      shape);
      if (offset == 0U) first_ms = Milliseconds(run_start);
      if (result.shape !=
              std::vector<std::int64_t>{
                  static_cast<std::int64_t>(count),
                  static_cast<std::int64_t>(kOutputFeatures)} ||
          result.values.size() != count * kOutputFeatures)
        throw std::runtime_error("Wrong task output shape");
      output.insert(output.end(), result.values.begin(), result.values.end());
    }
    const auto inference_ms = Milliseconds(all_start);
    const auto profile = session.end_profiling();
    std::size_t correct = 0U;
    for (std::size_t i = 0U; i < kSamples; ++i) {
      const auto row =
          std::span(output).subspan(i * kOutputFeatures, kOutputFeatures);
      const auto prediction = static_cast<std::size_t>(
          std::max_element(row.begin(), row.end()) - row.begin());
      const auto label = std::to_integer<std::uint8_t>(raw_labels[i]);
      if (label >= kOutputFeatures)
        throw std::runtime_error("Invalid task label");
      if (prediction == label) ++correct;
    }
    WriteBytes(root / (model_path.stem().string() + "-native-output.f32"),
               std::as_bytes(std::span(output)));
    std::cout << "{\"schema_version\":1,\"model_sha256\":\"" << digest
              << "\",\"runtime_version\":\"" << session.runtime_version()
              << "\",\"provider\":\"CPUExecutionProvider\",\"samples\":"
              << kSamples << ",\"correct\":" << correct << ",\"accuracy\":"
              << static_cast<double>(correct) / static_cast<double>(kSamples)
              << ",\"load_ms\":" << load_ms << ",\"first_run_ms\":" << first_ms
              << ",\"profiled_inference_ms\":" << inference_ms
              << ",\"trace_written\":"
              << (std::filesystem::is_regular_file(profile) ? "true" : "false")
              << ",\"compute_quantized\":false,\"execution_claim\":\"runtime "
                 "trace requires independent review\"}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "INT8 task runtime failed: " << error.what() << '\n';
    return 1;
  }
}
