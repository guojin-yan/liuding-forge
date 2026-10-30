// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "../../tools/datasets/cifar10_detail.hpp"

#include <jyppx/liuding/foundation/error.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

namespace {
int run(const std::filesystem::path& contract_path,
        const std::filesystem::path& archive,
        const std::filesystem::path& data_root) {
  try {
    if (!std::filesystem::is_regular_file(contract_path) ||
        std::filesystem::file_size(contract_path) > 65536U)
      throw jyppx::liuding::InvalidArgument("Invalid fixture contract file");
    std::ifstream stream(contract_path);
    const auto json = nlohmann::ordered_json::parse(stream);
    jyppx::liuding::datasets::detail::CifarContract contract{
        json.at("archive_bytes").get<std::uint64_t>(),
        json.at("sha256").get<std::string>(),
        json.at("official_md5").get<std::string>(),
        json.at("source").get<std::string>(),
        json.at("dataset_license").get<std::string>(),
        json.at("max_decoded_bytes").get<std::uint64_t>(),
        json.at("samples_per_class").at("train").get<std::uint32_t>(),
        json.at("samples_per_class").at("validation").get<std::uint32_t>(),
        {}};
    for (const auto& [name, value] : json.at("batches").items())
      contract.batches.push_back({name, value.at("split").get<std::string>(),
                                  value.at("samples").get<std::uint32_t>()});
    const auto destination =
        jyppx::liuding::datasets::detail::prepare_cifar10_contract(
            archive, data_root, contract);
    const auto text = destination.generic_u8string();
    std::cout << std::string(reinterpret_cast<const char*>(text.data()),
                             text.size())
              << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
  if (argc != 4) return 2;
  return run(std::filesystem::path(argv[1]), std::filesystem::path(argv[2]),
             std::filesystem::path(argv[3]));
}
#else
int main(int argc, char* argv[]) {
  if (argc != 4) return 2;
  return run(std::filesystem::u8path(argv[1]), std::filesystem::u8path(argv[2]),
             std::filesystem::u8path(argv[3]));
}
#endif
