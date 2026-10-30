// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_TOOLS_DATASETS_CIFAR10_HPP
#define JYPPX_TOOLS_DATASETS_CIFAR10_HPP

#include <filesystem>

namespace jyppx::liuding::datasets {
// Offline preparation of the exact pinned binary archive into all 60,000 RGB
// PNGs and preparation.json. Does not load pickle, download, or select a
// subset. Paths may not traverse links or point into the code repository. Their
// trusted ancestors must remain stable; only this invocation's stage is rolled
// back. Publication never replaces an existing cifar10-imagefolder-v1
// directory. Requires the explicitly enabled and independently qualified native
// codec SDK.
std::filesystem::path prepare_cifar10(const std::filesystem::path& archive,
                                      const std::filesystem::path& data_root);
}  // namespace jyppx::liuding::datasets

#endif  // JYPPX_TOOLS_DATASETS_CIFAR10_HPP
