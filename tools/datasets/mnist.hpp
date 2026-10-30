// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_TOOLS_DATASETS_MNIST_HPP
#define JYPPX_TOOLS_DATASETS_MNIST_HPP
#include <filesystem>

namespace jyppx::liuding::datasets {
// Tool-private offline preparation; not an installed library API. The source
// directory contains the four pinned official gzip files. Publishes a new
// data_root/mnist directory only; never downloads or replaces an old cache.
std::filesystem::path prepare_mnist(const std::filesystem::path& source_dir,
                                    const std::filesystem::path& data_root);
}  // namespace jyppx::liuding::datasets
#endif
