// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "../../tools/datasets/paired_affine.hpp"

#include <jyppx/liuding/foundation/error.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

namespace {
int run(const std::filesystem::path& root) {
  try {
    // The native preparation helper streams and verifies all 1,280 records.
    // It publishes a new external directory only after both hashes match.
    const auto directory =
        jyppx::liuding::datasets::prepare_paired_affine(root);
    std::ifstream input(directory / "train.jsonl", std::ios::binary);
    std::string line;
    if (!std::getline(input, line))
      throw jyppx::liuding::IoError("Could not read the first prepared record");
    const auto sample = nlohmann::json::parse(line);

    // Keep positional inputs separate from supervision. Training adds the
    // leading batch dimension: left/right [batch,2], class_id [batch],
    // regression [batch,1]. See the paired-affine recipe for Tensor assembly.
    const nlohmann::json model_inputs = {sample.at("left"), sample.at("right")};
    const nlohmann::json targets = {{"class_id", sample.at("class_id")},
                                    {"regression", sample.at("regression")}};
    const auto path = directory.generic_u8string();
    std::cout << "verified dataset: "
              << std::string(reinterpret_cast<const char*>(path.data()),
                             path.size())
              << "\nsample id: " << sample.at("id").get<std::string>()
              << "\npositional model inputs: " << model_inputs.dump()
              << "\nsupervision targets: " << targets.dump() << '\n';
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
  if (argc != 2) {
    std::cerr << "usage: liuding-forge-example-paired-affine-data "
                 "EXTERNAL_DATA_ROOT\n";
    return 2;
  }
  return run(std::filesystem::path(argv[1]));
}
#else
int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: liuding-forge-example-paired-affine-data "
                 "EXTERNAL_DATA_ROOT\n";
    return 2;
  }
  return run(std::filesystem::u8path(argv[1]));
}
#endif
