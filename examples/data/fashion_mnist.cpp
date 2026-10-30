// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "../../tools/datasets/fashion_mnist.hpp"

#include <jyppx/liuding/data_image/image_decoder.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

namespace {
int run(const std::filesystem::path& source,
        const std::filesystem::path& root) {
  try {
    // Offline preparation verifies all four compressed archives and every
    // original sample. Default selection retains first100 per class per split.
    const auto directory =
        jyppx::liuding::datasets::prepare_fashion_mnist(source, root);
    std::ifstream index(directory / "index/train.jsonl", std::ios::binary);
    std::string line;
    if (!std::getline(index, line))
      throw jyppx::liuding::IoError(
          "Could not read the first prepared image index");
    const auto row = nlohmann::json::parse(line);
    namespace image = jyppx::liuding::data::image;
    image::ImageDecodeOptions options;
    options.format = image::ImageFormat::png;
    options.channels = image::ImageChannels::rgb;
    options.layout = image::ImageLayout::hwc;
    options.dtype = image::ImageDtype::preserve;
    options.limits.max_input_bytes = 8192U;
    options.limits.max_dimension = 28U;
    options.limits.max_pixels = 784U;
    options.limits.max_output_bytes = 2352U;
    const auto decoded = image::read_image_file(
        directory / row.at("path").get<std::string>(), options);
    if (decoded.width != 28U || decoded.height != 28U || decoded.channels != 3U)
      throw jyppx::liuding::CompatibilityError(
          "Unexpected prepared image geometry");
    // tensor is one owning CPU uint8 HWC [28,28,3] image. Keep class_id
    // separate as supervision. A model pipeline explicitly adds resize,
    // normalization, layout conversion and the batch dimension; preparation
    // changes none.
    const auto pixel = decoded.tensor[0][0][0].item<unsigned char>();
    std::cout << "source index: " << row.at("source_index")
              << "\nclass target: " << row.at("class_id")
              << "\nRGB uint8 HWC shape: [28,28,3]"
              << "\nfirst red byte: " << static_cast<unsigned>(pixel) << '\n';
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
    std::cerr << "usage: liuding-forge-example-fashion-mnist-data "
                 "SOURCE_DIRECTORY EXTERNAL_DATA_ROOT\n";
    return 2;
  }
  return run(std::filesystem::path(argv[1]), std::filesystem::path(argv[2]));
}
#else
int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: liuding-forge-example-fashion-mnist-data "
                 "SOURCE_DIRECTORY EXTERNAL_DATA_ROOT\n";
    return 2;
  }
  return run(std::filesystem::u8path(argv[1]),
             std::filesystem::u8path(argv[2]));
}
#endif
