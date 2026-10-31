// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "exclusive_file.hpp"

#include <jyppx/liuding/deployment/onnx_export/static_int8_linear.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int Usage() {
  std::cerr << "Usage: liuding-forge-int8-export [--per-output-channel] "
               "[--graph portable|ort-matmul-integer-to-float] "
               "SOURCE.qint8 OUTPUT.onnx\n";
  return 2;
}

std::vector<std::byte> ReadSource(const std::filesystem::path& path) {
  constexpr std::uintmax_t kMaximum = 16U << 20U;
  const auto count = std::filesystem::file_size(path);
  if (count == 0U || count > kMaximum)
    throw std::runtime_error("Q0 source violates the 16 MiB artifact bound");
  std::vector<std::byte> bytes(static_cast<std::size_t>(count));
  std::ifstream input(path, std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size())))
    throw std::runtime_error("Cannot read Q0 source artifact");
  if (input.peek() != std::char_traits<char>::eof() || input.bad() ||
      std::filesystem::file_size(path) != count)
    throw std::runtime_error("Q0 source byte count changed during read");
  return bytes;
}

}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
  if (argc < 3) return Usage();
  try {
    namespace ex = jyppx::liuding::onnx_export;
    namespace q = jyppx::liuding::quantization;
    bool per_output_channel = false;
    bool graph_selected = false;
    auto graph = ex::StaticInt8LinearGraph::portable;
    int source_index = 1;
    while (source_index < argc - 2) {
      const std::filesystem::path option(argv[source_index]);
      if (option == "--") {
        ++source_index;
        break;
      }
      if (option == "--per-output-channel" && !per_output_channel) {
        per_output_channel = true;
        ++source_index;
      } else if (option == "--graph" && !graph_selected &&
                 source_index + 1 < argc - 2) {
        const std::filesystem::path value(argv[source_index + 1]);
        if (value == "portable") {
          graph = ex::StaticInt8LinearGraph::portable;
        } else if (value == "ort-matmul-integer-to-float") {
          graph = ex::StaticInt8LinearGraph::ort_matmul_integer_to_float;
        } else {
          return Usage();
        }
        graph_selected = true;
        source_index += 2;
      } else {
        return Usage();
      }
    }
    if (argc - source_index != 2) return Usage();
    const std::filesystem::path source(argv[source_index]);
    const std::filesystem::path destination(argv[source_index + 1]);
    const auto bytes = ReadSource(source);
    const auto in_features =
        per_output_channel
            ? q::deserialize_per_output_channel_linear_artifact(bytes)
                  .in_features
            : q::deserialize_linear_artifact(bytes).in_features;
    const std::array<std::int64_t, 2U> shape{
        1, static_cast<std::int64_t>(in_features)};
    const jyppx::liuding::ArtifactDigest digest{
        .hex = jyppx::liuding::sha256_hex(bytes)};
    const auto exported =
        per_output_channel ? ex::export_static_int8_linear_per_output_channel(
                                 bytes, digest, shape, {}, {}, graph)
                           : ex::export_static_int8_linear(bytes, digest, shape,
                                                           {}, {}, graph);
    q::tools_detail::PublishExclusive(destination, exported.artifact.onnx);
    std::cout
        << "{\"schema_version\":1,\"source_artifact_sha256\":\""
        << exported.source_artifact_digest.hex << "\",\"onnx_sha256\":\""
        << exported.artifact.digest.hex << "\",\"profile_id\":\""
        << exported.profile_id
        << "\",\"onnx_bytes\":" << exported.artifact.onnx.size()
        << ",\"compute_quantized\":false,\"qat\":false,\"execution_claim\":"
           "\"native exported INT8 profile; runtime trace required\"}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Native INT8 export failed: " << error.what() << '\n';
    return 1;
  }
}
