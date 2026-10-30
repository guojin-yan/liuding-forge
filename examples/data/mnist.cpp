// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "../../tools/datasets/mnist.hpp"

#include "../../tools/datasets/mnist_download.hpp"

#include <jyppx/liuding/foundation/error.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>

namespace {
int run(const std::filesystem::path& source,
        const std::filesystem::path& data_root, bool download) {
  try {
    const auto directory =
        download ? jyppx::liuding::datasets::download_prepare_mnist(data_root)
                 : jyppx::liuding::datasets::prepare_mnist(source, data_root);
    std::ifstream report_input(directory / "preparation.json",
                               std::ios::binary);
    const auto report = nlohmann::json::parse(report_input);
    // Demonstrate the unchanged IDX layout without allocating the dataset:
    // images have a 16-byte header, labels an 8-byte header, then raw records.
    std::ifstream images(directory / "train-images-idx3-ubyte",
                         std::ios::binary);
    std::ifstream labels(directory / "train-labels-idx1-ubyte",
                         std::ios::binary);
    images.seekg(16);
    labels.seekg(8);
    std::array<unsigned char, 784> first_image{};
    unsigned char first_label = 0U;
    images.read(reinterpret_cast<char*>(first_image.data()),
                static_cast<std::streamsize>(first_image.size()));
    labels.read(reinterpret_cast<char*>(&first_label), 1);
    if (!images || !labels)
      throw jyppx::liuding::IoError(
          "Could not read first prepared MNIST sample");
    const auto [minimum, maximum] =
        std::minmax_element(first_image.begin(), first_image.end());
    const auto path = directory.generic_u8string();
    const auto root = directory.parent_path().generic_u8string();
    std::cout << "verified IDX: "
              << std::string(reinterpret_cast<const char*>(path.data()),
                             path.size())
              << "\nsamples: " << report.at("samples").dump()
              << "\nfirst train label: " << static_cast<unsigned>(first_label)
              << "\nfirst grayscale image: [28,28], uint8 range "
              << static_cast<unsigned>(*minimum) << ".."
              << static_cast<unsigned>(*maximum) << "\nSet JYPPX_DATA_ROOT to: "
              << std::string(reinterpret_cast<const char*>(root.data()),
                             root.size())
              << "\nThen run the MNIST training example.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
}  // namespace
#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
  if (argc != 3) {
#else
int main(int argc, char* argv[]) {
  if (argc != 3) {
#endif
    std::cerr << "usage: liuding-forge-example-mnist-data "
                 "(LOCAL_GZIP_DIRECTORY | --download) "
                 "NEW_EXTERNAL_DATA_ROOT\n";
    return 2;
  }
#ifdef _WIN32
  return run(
      std::filesystem::path(argv[1]), std::filesystem::path(argv[2]),
      std::filesystem::path(argv[1]) == std::filesystem::path("--download"));
#else
  return run(std::filesystem::u8path(argv[1]), std::filesystem::u8path(argv[2]),
             std::string_view(argv[1]) == "--download");
#endif
}
