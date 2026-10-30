// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_TOOLS_DATASETS_FASHION_MNIST_DETAIL_HPP
#define JYPPX_TOOLS_DATASETS_FASHION_MNIST_DETAIL_HPP
#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace jyppx::liuding::datasets::detail {
// Private C++ contract seam for bounded synthetic fixtures only. Not installed,
// exported, CLI-configurable, or accepted from a runtime contract JSON file.
struct FashionArchive {
  std::string name;
  std::uint64_t bytes;
  std::string sha256;
  std::string upstream_md5;
};
struct FashionSplit {
  std::string name;
  std::uint32_t samples;
  std::uint32_t samples_per_class;
  FashionArchive images;
  FashionArchive labels;
};
using FashionContract = std::array<FashionSplit, 2>;
const FashionContract& official_fashion_contract();
std::filesystem::path prepare_fashion_mnist_contract(
    const std::filesystem::path& source_dir,
    const std::filesystem::path& data_root, std::string_view selection,
    const FashionContract& contract);
}  // namespace jyppx::liuding::datasets::detail
#endif  // JYPPX_TOOLS_DATASETS_FASHION_MNIST_DETAIL_HPP
