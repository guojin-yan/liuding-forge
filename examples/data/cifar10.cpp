// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "../../tools/datasets/cifar10.hpp"

#include <jyppx/liuding/foundation/error.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

namespace {
int run(const std::filesystem::path& archive,
        const std::filesystem::path& data_root) {
  try {
    // The archive is an explicit local binary-version file. Preparation checks
    // its complete SHA-256/MD5, streams all records, and publishes one new
    // directory only after validating every batch, label, and class count.
    const auto directory =
        jyppx::liuding::datasets::prepare_cifar10(archive, data_root);
    std::ifstream input(directory / "preparation.json");
    if (!input)
      throw jyppx::liuding::IoError(
          "Could not read CIFAR-10 preparation report");
    const auto report = nlohmann::json::parse(input);
    const auto path = directory.generic_u8string();
    std::cout << "verified dataset: "
              << std::string(reinterpret_cast<const char*>(path.data()),
                             path.size())
              << "\nclasses in label order: " << report.at("classes").dump()
              << "\ntrain counts: "
              << report.at("class_counts").at("train").dump()
              << "\nvalidation counts: "
              << report.at("class_counts").at("validation").dump()
              << "\npixels: " << report.at("pixels").get<std::string>() << '\n';
    // ImageFolder consumers use directory/train and directory/validation.
    // Validation is the complete official test split, not an unseen holdout.
    // Keep the recipe's original seed, split sizes, and task threshold.
    return 0;
  } catch (const jyppx::liuding::Error& error) {
    std::cerr << error.message() << '\n';
    return 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: liuding-forge-example-cifar10-data "
                 "LOCAL_BINARY_TAR_GZ EXTERNAL_DATA_ROOT\n";
    return 2;
  }
  return run(std::filesystem::path(argv[1]), std::filesystem::path(argv[2]));
}
#else
int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: liuding-forge-example-cifar10-data "
                 "LOCAL_BINARY_TAR_GZ EXTERNAL_DATA_ROOT\n";
    return 2;
  }
  return run(std::filesystem::u8path(argv[1]),
             std::filesystem::u8path(argv[2]));
}
#endif
