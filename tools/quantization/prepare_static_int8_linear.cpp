// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/quantization/quantization.hpp>

#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
  if (argc != 2) {
    std::cerr << "Usage: liuding-forge-int8-prepare SOURCE.qint8 > "
                 "preparation.json\n";
    return 2;
  }
  try {
    namespace q = jyppx::liuding::quantization;
    const q::QuantizationLimits limits;
    std::ifstream input(std::filesystem::path(argv[1]), std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open the Q0 source artifact");
    std::vector<std::byte> bytes;
    std::array<char, 8192U> chunk{};
    while (input) {
      input.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
      const auto count = static_cast<std::size_t>(input.gcount());
      if (count > limits.max_artifact_bytes - bytes.size())
        throw std::runtime_error("Q0 source exceeds the 16 MiB artifact limit");
      const auto data = std::as_bytes(std::span(chunk.data(), count));
      bytes.insert(bytes.end(), data.begin(), data.end());
    }
    if (!input.eof())
      throw std::runtime_error("Failed to read the Q0 source artifact");
    const auto source = q::deserialize_linear_artifact(bytes, limits);
    std::cout << q::static_int8_linear_export_json(source, limits) << '\n';
    if (!std::cout)
      throw std::runtime_error("Failed to write preparation JSON");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Static INT8 preparation failed: " << error.what() << '\n';
    return 1;
  }
}
