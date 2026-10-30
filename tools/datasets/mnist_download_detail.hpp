// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_TOOLS_DATASETS_MNIST_DOWNLOAD_DETAIL_HPP
#define JYPPX_TOOLS_DATASETS_MNIST_DOWNLOAD_DETAIL_HPP
#include "mnist_detail.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string_view>

namespace jyppx::liuding::datasets::detail {
inline constexpr std::string_view kMnistHttpsSource =
    "https://storage.googleapis.com/cvdf-datasets/mnist/";
inline constexpr std::size_t kMnistDownloadChunkBytes = 64U * 1024U;
// A successful transfer delivers exactly one final response metadata callback
// and zero or more body blocks before returning. Metadata may precede or follow
// body blocks; private staged bytes cannot be published until both are checked.
// Both callbacks may throw. Native C callbacks must catch exceptions.
struct MnistHttpsCallbacks {
  std::function<void(std::uint32_t, std::optional<std::uint64_t>)> headers;
  std::function<void(std::span<const std::byte>)> bytes;
};
using MnistHttpsTransfer = std::function<void(std::string_view, std::uint64_t,
                                              const MnistHttpsCallbacks&)>;
void require_mnist_https_url(std::string_view url);
void transfer_mnist_https(std::string_view url, std::uint64_t expected_bytes,
                          const MnistHttpsCallbacks& callbacks);
// Synthetic fixture/fake transport seam. Never installed or CLI-selectable;
// production always uses the immutable official contract and OS TLS backend.
std::filesystem::path download_prepare_mnist_contract(
    const std::filesystem::path& data_root, const MnistContract& contract,
    const MnistHttpsTransfer& transfer);
}  // namespace jyppx::liuding::datasets::detail
#endif
