// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
// Test-only synthetic-contract adapter. Never installed or used by product CLI.
#include "../../tools/datasets/fashion_mnist_detail.hpp"
#include "../../tools/datasets/preparation_support.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;
namespace ds = jyppx::liuding::datasets::detail;
ds::FashionArchive archive(const Json& json) {
  return {json.at("name"), json.at("bytes"), json.at("sha256"),
          json.at("official_md5")};
}
int run(const std::vector<fs::path>& args) {
  try {
    if (args.size() == 2U && args[1] == "--publication-collision") {
      const auto root = ds::external_root(args[0]);
      ds::StagingDirectory staging(root, "collision",
                                   ".fashion-mnist-preparing-");
      fs::create_directory(root / "collision");
      // An empty racing directory must survive no-replacement publication.
      try {
        staging.publish();
      } catch (const std::exception&) {
        return fs::is_directory(root / "collision") &&
                       fs::is_empty(root / "collision")
                   ? 0
                   : 1;
      }
      return 1;
    }
    if (args.size() != 4U) return 2;
    std::ifstream stream(args[2], std::ios::binary);
    if (!stream || fs::file_size(args[2]) > 65536U) return 2;
    const auto json = Json::parse(stream);
    ds::FashionContract contract;
    std::size_t i = 0U;
    for (const auto name : {"train", "validation"}) {
      const auto& split = json.at("files").at(name);
      contract[i++] = {name, split.at("samples"), split.at("samples_per_class"),
                       archive(split.at("images")),
                       archive(split.at("labels"))};
    }
    const auto selection = args[3].generic_u8string();
    const auto result = ds::prepare_fashion_mnist_contract(
        args[0], args[1],
        std::string(reinterpret_cast<const char*>(selection.data()),
                    selection.size()),
        contract);
    const auto encoded = result.generic_u8string();
    std::cout << Json{{"destination", std::string(reinterpret_cast<const char*>(
                                                      encoded.data()),
                                                  encoded.size())}}
                     .dump()
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
  std::vector<fs::path> args;
  for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
  return run(args);
}
#else
int main(int argc, char* argv[]) {
  std::vector<fs::path> args;
  for (int i = 1; i < argc; ++i) args.emplace_back(fs::u8path(argv[i]));
  return run(args);
}
#endif
