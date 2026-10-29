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
#include <vector>

namespace {

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
  return bytes;
}

}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
  if (argc != 3) {
    std::cerr << "Usage: liuding-forge-int8-export SOURCE.qint8 OUTPUT.onnx\n";
    return 2;
  }
  try {
    namespace ex = jyppx::liuding::onnx_export;
    namespace q = jyppx::liuding::quantization;
    const std::filesystem::path source(argv[1]);
    const std::filesystem::path destination(argv[2]);
    const auto bytes = ReadSource(source);
    const auto canonical = q::deserialize_linear_artifact(bytes);
    const std::array<std::int64_t, 2U> shape{
        1, static_cast<std::int64_t>(canonical.in_features)};
    const auto exported = ex::export_static_int8_linear(
        bytes, {.hex = jyppx::liuding::sha256_hex(bytes)}, shape);
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
