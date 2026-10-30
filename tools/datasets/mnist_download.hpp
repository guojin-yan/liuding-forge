// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_TOOLS_DATASETS_MNIST_DOWNLOAD_HPP
#define JYPPX_TOOLS_DATASETS_MNIST_DOWNLOAD_HPP
#include <filesystem>

namespace jyppx::liuding::datasets {
// Explicit network operation; never called by the offline preparation API.
// Fetches the four pinned HTTPS sources into a disposable external stage,
// then delegates IDX validation/publication to prepare_mnist. No replacement,
// redirects, persistent gzip cache, implicit dependency fetch, or Python.
std::filesystem::path download_prepare_mnist(
    const std::filesystem::path& data_root);
}  // namespace jyppx::liuding::datasets
#endif
