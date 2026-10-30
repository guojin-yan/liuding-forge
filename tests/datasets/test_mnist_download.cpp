// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "../../tools/datasets/mnist_download_detail.hpp"

#include <jyppx/liuding/foundation/error.hpp>

#include <array>
#include <chrono>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <string_view>

namespace {
namespace data = jyppx::liuding::datasets::detail;
TEST(MnistHttpsContract, AllowsOnlyFrozenFullHttpsUrls) {
  const std::array<std::string_view, 4> names{
      "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
      "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"};
  for (const auto name : names) {
    const auto url =
        std::string("https://storage.googleapis.com/cvdf-datasets/mnist/") +
        std::string(name);
    EXPECT_NO_THROW(data::require_mnist_https_url(url));
  }
  const std::array<std::string_view, 11> rejected{
      "http://storage.googleapis.com/cvdf-datasets/mnist/"
      "train-images-idx3-ubyte.gz",
      "https://evil.example/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
      "https://storage.googleapis.com.evil.example/cvdf-datasets/mnist/"
      "train-images-idx3-ubyte.gz",
      "https://storage.googleapis.com@evil.example/cvdf-datasets/mnist/"
      "train-images-idx3-ubyte.gz",
      "https://user:secret@storage.googleapis.com/cvdf-datasets/mnist/"
      "train-images-idx3-ubyte.gz",
      "https://storage.googleapis.com/cvdf-datasets/mnist/../"
      "train-images-idx3-ubyte.gz",
      "https://storage.googleapis.com/cvdf-datasets/mnist/"
      "train-images-idx3-ubyte.gz?mirror=1",
      "https://storage.googleapis.com/cvdf-datasets/mnist/"
      "train-images-idx3-ubyte.gz#fragment",
      "https://storage.googleapis.com:443/cvdf-datasets/mnist/"
      "train-images-idx3-ubyte.gz",
      "file:///tmp/train-images-idx3-ubyte.gz",
      ""};
  for (const auto url : rejected) {
    EXPECT_THROW(data::require_mnist_https_url(url),
                 jyppx::liuding::InvalidArgument);
    // The concrete OS backend must reject these before opening any request.
    EXPECT_THROW(data::transfer_mnist_https(url, 1U, {}),
                 jyppx::liuding::InvalidArgument);
  }
}

TEST(MnistHttpsContract, RejectsInvalidContractBeforeTransportOrFilesystem) {
  auto contract = data::official_mnist_contract();
  contract.train_samples = 0U;
  const auto root =
      std::filesystem::temp_directory_path() /
      ("liuding-mnist-invalid-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  ASSERT_FALSE(std::filesystem::exists(root));
  unsigned transfers = 0U;
  const data::MnistHttpsTransfer fake = [&](std::string_view, std::uint64_t,
                                            const data::MnistHttpsCallbacks&) {
    ++transfers;
  };
  EXPECT_THROW(data::download_prepare_mnist_contract(root, contract, fake),
               jyppx::liuding::InvalidArgument);
  EXPECT_EQ(transfers, 0U);
  EXPECT_FALSE(std::filesystem::exists(root));
}
}  // namespace
