// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_TOOLS_DATASETS_CIFAR10_DETAIL_HPP
#define JYPPX_TOOLS_DATASETS_CIFAR10_DETAIL_HPP

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace jyppx::liuding::datasets::detail {
// Private synthetic-fixture seam only. Not installed, exported, accepted from
// user contract JSON, or configurable through the product CLI/example.
struct CifarBatch {
  std::string name;
  std::string split;
  std::uint32_t samples;
};
struct CifarContract {
  std::uint64_t archive_bytes;
  std::string sha256;
  std::string official_md5;
  std::string source;
  std::string dataset_license;
  std::uint64_t max_decoded_bytes;
  std::uint32_t train_samples_per_class;
  std::uint32_t validation_samples_per_class;
  std::vector<CifarBatch> batches;
};
const CifarContract& official_cifar10_contract();
std::filesystem::path prepare_cifar10_contract(
    const std::filesystem::path& archive,
    const std::filesystem::path& data_root, const CifarContract& contract);
// MD5 is retained solely for the upstream auxiliary checksum; SHA-256 is the
// mandatory primary identity. Exposed here for independent known-vector tests.
std::string cifar10_md5_hex(std::span<const unsigned char> bytes);
}  // namespace jyppx::liuding::datasets::detail

#endif  // JYPPX_TOOLS_DATASETS_CIFAR10_DETAIL_HPP
