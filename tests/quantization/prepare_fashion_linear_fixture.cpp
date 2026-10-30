// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "../../tools/quantization/exclusive_file.hpp"

#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/quantization/quantization.hpp>

#include <bit>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> ReadFloats(const std::filesystem::path& path,
                              std::size_t count) {
  static_assert(std::endian::native == std::endian::little);
  if (std::filesystem::file_size(path) != count * sizeof(float))
    throw std::runtime_error(
        "Task FP32 fixture has the wrong exact byte count");
  std::vector<float> result(count);
  std::ifstream input(path, std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(result.data()),
                  static_cast<std::streamsize>(count * sizeof(float))))
    throw std::runtime_error("Cannot read bounded task FP32 fixture");
  return result;
}

// Normalize only the line terminator. The frozen content and all other
// characters remain unchanged so calibration lineage is platform independent.
bool ReadPortableLine(std::istream& input, std::string& line) {
  if (!std::getline(input, line)) return false;
  if (!line.empty() && line.back() == '\r') line.pop_back();
  return true;
}

void WriteBytes(const std::filesystem::path& path,
                std::span<const std::byte> bytes) {
  jyppx::liuding::quantization::tools_detail::PublishExclusive(path, bytes);
}

}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
  const bool per_output_channel =
      argc == 3 && std::filesystem::path(argv[1]) == "--per-output-channel";
  if (argc != 2 && !per_output_channel) {
    std::cerr << "Usage: liuding-forge-int8-fashion-fixture "
                 "[--per-output-channel] TASK_DIRECTORY\n";
    return 2;
  }
  try {
    namespace q = jyppx::liuding::quantization;
    const std::filesystem::path root(argv[per_output_channel ? 2 : 1]);
    constexpr std::size_t kInputFeatures = 784U;
    constexpr std::size_t kOutputFeatures = 10U;
    constexpr std::size_t kCalibrationSamples = 160U;
    const auto weights =
        ReadFloats(root / "weights.f32", kInputFeatures * kOutputFeatures);
    const auto bias = ReadFloats(root / "bias.f32", kOutputFeatures);
    const auto calibration = ReadFloats(root / "calibration.f32",
                                        kCalibrationSamples * kInputFeatures);
    std::ifstream digest_input(root / "dataset.sha256", std::ios::binary);
    std::string dataset_digest;
    if (!ReadPortableLine(digest_input, dataset_digest) ||
        dataset_digest.size() != 64U)
      throw std::runtime_error("Missing calibration dataset content identity");
    q::CalibrationSession session(
        {.dataset_version = "fashion-mnist-official-train/v1",
         .dataset_digest = {.hex = dataset_digest},
         .preprocessing_fingerprint =
             "uint8-grayscale-to-fp32-div255-row-major/v1",
         .seed = 20261029U});
    std::ifstream identities(root / "calibration.ids", std::ios::binary);
    std::string identity;
    for (std::size_t i = 0U; i < kCalibrationSamples; ++i) {
      if (!ReadPortableLine(identities, identity) || identity.empty() ||
          identity.size() > 256U)
        throw std::runtime_error("Missing bounded calibration sample identity");
      session.observe(
          identity,
          std::span(calibration).subspan(i * kInputFeatures, kInputFeatures));
    }
    if (ReadPortableLine(identities, identity))
      throw std::runtime_error(
          "Unexpected trailing calibration sample identity");
    if (per_output_channel) {
      const auto artifact = q::quantize_linear_per_output_channel(
          {.in_features = kInputFeatures,
           .out_features = kOutputFeatures,
           .weights = weights,
           .bias = bias},
          session.finalize());
      const auto bytes =
          q::serialize_per_output_channel_linear_artifact(artifact);
      WriteBytes(root / "source.per-output-channel.qint8", bytes);
      std::cout << "{\"source_artifact_sha256\":\""
                << jyppx::liuding::sha256_hex(bytes)
                << "\",\"calibration_samples\":160,\"in_features\":784,\"out_"
                   "features\":10,"
                   "\"weight_granularity\":\"per-output-channel\"}\n";
      return 0;
    }
    const auto artifact = q::quantize_linear({.in_features = kInputFeatures,
                                              .out_features = kOutputFeatures,
                                              .weights = weights,
                                              .bias = bias},
                                             session.finalize());
    const auto bytes = q::serialize_linear_artifact(artifact);
    const auto preparation = q::static_int8_linear_export_json(artifact);
    WriteBytes(root / "source.qint8", bytes);
    WriteBytes(
        root / "preparation.json",
        std::as_bytes(std::span(preparation.data(), preparation.size())));
    std::cout << "{\"source_artifact_sha256\":\""
              << jyppx::liuding::sha256_hex(bytes)
              << "\",\"calibration_samples\":160,\"in_features\":784,\"out_"
                 "features\":10}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "INT8 task fixture failed: " << error.what() << '\n';
    return 1;
  }
}
