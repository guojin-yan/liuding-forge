// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
/** @file image_training_output.cpp
 * @brief Trains an ImageFolder classifier with bounded preview evidence. */
#include "image_output_support.hpp"

#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <iostream>

namespace {
template <class Character>
int Main(int argc, Character* argv[]) {
  using namespace image_output_example;
  constexpr auto usage =
      "usage: liuding-forge-example-image-training-output "
      "IMAGE_ROOT NEW_OUTPUT_DIRECTORY cpu|cuda:0 current|copy\n";
  try {
    if (argc == 2 && fs::path(argv[1]) == fs::path("--help")) {
      std::cout << usage;
      return 0;
    }
    if (argc != 5) {
      std::cerr << usage;
      return 2;
    }
    const auto device = fs::path(argv[3]).string();
    const auto mode = fs::path(argv[4]).string();
    Require(device == "cpu" || device == "cuda:0", "Choose cpu or cuda:0");
    Require(mode == "current" || mode == "copy", "Choose current or copy");
    at::set_num_threads(1);
    at::set_num_interop_threads(1);
    at::globalContext().setDeterministicAlgorithms(true, false);
    at::globalContext().setFloat32MatmulPrecision("highest");
    const auto report = Run({.images = fs::path(argv[1]),
                             .output = fs::path(argv[2]),
                             .device = forge::Device(device),
                             .explicit_copy = mode == "copy"});
    std::ofstream saved(fs::path(argv[2]) / "report.json");
    saved << report.dump(2) << '\n';
    saved.close();
    Require(saved.good(), "Cannot save the training report");
    std::cout << report.dump() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) { return Main(argc, argv); }
#else
int main(int argc, char* argv[]) { return Main(argc, argv); }
#endif
