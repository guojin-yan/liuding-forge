// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_TOOLS_DATASETS_FASHION_MNIST_HPP
#define JYPPX_TOOLS_DATASETS_FASHION_MNIST_HPP
#include <filesystem>
#include <string_view>

namespace jyppx::liuding::datasets {
// Offline tool-private preparation entry. Accepts only four exact official
// compressed archives embedded in the installed tool. Streams every source
// sample, including unselected samples. Output must be external, absent, and
// reached through stable caller-owned non-link ancestors. No Python or
// download. Throws foundation Error; removes only its own staging directory on
// failure.
std::filesystem::path prepare_fashion_mnist(
    const std::filesystem::path& source_dir,
    const std::filesystem::path& data_root,
    std::string_view selection = "first100-per-class");
}  // namespace jyppx::liuding::datasets
#endif  // JYPPX_TOOLS_DATASETS_FASHION_MNIST_HPP
