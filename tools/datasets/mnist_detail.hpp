// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_TOOLS_DATASETS_MNIST_DETAIL_HPP
#define JYPPX_TOOLS_DATASETS_MNIST_DETAIL_HPP
#include <array>
#include <cstdint>
#include <filesystem>
#include <string>

namespace jyppx::liuding::datasets::detail {
struct MnistFile {
  std::string name;
  std::uint64_t compressed_bytes;
  std::string compressed_sha256;
  std::uint64_t decoded_bytes;
  std::string decoded_sha256;
};
struct MnistContract {
  std::uint32_t train_samples;
  std::uint32_t test_samples;
  std::uint64_t max_output_bytes;
  std::array<MnistFile, 4> files;
};
const MnistContract& official_mnist_contract();
// Shared preflight; performs no I/O and creates no output.
void validate_mnist_contract(const MnistContract& contract);
// Synthetic-test seam only; not installed or accepted by the CLI/example.
std::filesystem::path prepare_mnist_contract(
    const std::filesystem::path& source_dir,
    const std::filesystem::path& data_root, const MnistContract& contract);
}  // namespace jyppx::liuding::datasets::detail
#endif
