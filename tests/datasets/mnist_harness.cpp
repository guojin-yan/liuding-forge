// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "../../tools/datasets/mnist_detail.hpp"

#include <jyppx/liuding/foundation/error.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

namespace {
int run(const std::filesystem::path& contract_path,
        const std::filesystem::path& source,
        const std::filesystem::path& root) {
  try {
    if (!std::filesystem::is_regular_file(contract_path) ||
        std::filesystem::file_size(contract_path) > 65536U)
      throw jyppx::liuding::InvalidArgument(
          "Invalid private MNIST fixture contract");
    std::ifstream input(contract_path);
    const auto json = nlohmann::json::parse(input);
    jyppx::liuding::datasets::detail::MnistContract contract{
        json.at("train_samples").get<std::uint32_t>(),
        json.at("test_samples").get<std::uint32_t>(),
        json.at("max_output_bytes").get<std::uint64_t>(),
        {}};
    const auto& files = json.at("files");
    if (!files.is_array() || files.size() != contract.files.size())
      throw jyppx::liuding::InvalidArgument(
          "Four MNIST fixture files required");
    for (std::size_t i = 0U; i < files.size(); ++i) {
      const auto& file = files[i];
      contract.files[i] = {file.at("name").get<std::string>(),
                           file.at("compressed_bytes").get<std::uint64_t>(),
                           file.at("compressed_sha256").get<std::string>(),
                           file.at("decoded_bytes").get<std::uint64_t>(),
                           file.at("decoded_sha256").get<std::string>()};
    }
    jyppx::liuding::datasets::detail::prepare_mnist_contract(source, root,
                                                             contract);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
}  // namespace
#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
  if (argc != 4) return 2;
#ifdef _WIN32
  return run(std::filesystem::path(argv[1]), std::filesystem::path(argv[2]),
             std::filesystem::path(argv[3]));
#else
  return run(std::filesystem::u8path(argv[1]), std::filesystem::u8path(argv[2]),
             std::filesystem::u8path(argv[3]));
#endif
}
