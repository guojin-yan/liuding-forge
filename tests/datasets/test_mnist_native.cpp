// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "../../tools/datasets/mnist_detail.hpp"

#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <string>

TEST(MnistNativeContract, MatchesFrozenManifestAndCompleteIdxSplits) {
  std::ifstream input(JYPPX_LIUDING_MNIST_MANIFEST);
  ASSERT_TRUE(input);
  const auto expected = nlohmann::json::parse(input);
  const auto& actual =
      jyppx::liuding::datasets::detail::official_mnist_contract();
  EXPECT_EQ(actual.train_samples, 60000U);
  EXPECT_EQ(actual.test_samples, 10000U);
  ASSERT_EQ(actual.files.size(), expected.at("files").size());
  std::uint64_t total = 0U;
  for (std::size_t i = 0U; i < actual.files.size(); ++i) {
    const auto& source = expected.at("files").at(i);
    const auto& file = actual.files[i];
    EXPECT_EQ(file.name, source.at("name").get<std::string>());
    EXPECT_EQ(file.compressed_bytes,
              source.at("size_bytes").get<std::uint64_t>());
    EXPECT_EQ(file.compressed_sha256, source.at("sha256").get<std::string>());
    EXPECT_EQ(file.decoded_bytes,
              source.at("decoded_size_bytes").get<std::uint64_t>());
    EXPECT_EQ(file.decoded_sha256.size(), 64U);
    total += file.decoded_bytes;
  }
  EXPECT_EQ(actual.max_output_bytes, total);
  EXPECT_EQ(total, 54950048U);
}
