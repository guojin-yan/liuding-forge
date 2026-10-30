// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "../../tools/datasets/cifar10_detail.hpp"

#include <array>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace {
namespace detail = jyppx::liuding::datasets::detail;

TEST(Cifar10NativeChecksum, MatchesRfc1321IncludingMultipleBlocksAndPadding) {
  constexpr std::array<std::pair<std::string_view, std::string_view>, 7> cases{
      {{"", "d41d8cd98f00b204e9800998ecf8427e"},
       {"a", "0cc175b9c0f1b6a831c399e269772661"},
       {"abc", "900150983cd24fb0d6963f7d28e17f72"},
       {"message digest", "f96b697d7cb7938d525a2f31aaf161d0"},
       {"abcdefghijklmnopqrstuvwxyz", "c3fcd3d76192e4007dfb496cca67e13b"},
       {"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        "d174ab98d277d9f5a5611c2c9f419d9f"},
       {"1234567890123456789012345678901234567890123456789012345678901234567890"
        "1234567890",
        "57edf4a22be3c955ac49da2e2107b67a"}}};
  for (const auto& [text, expected] : cases)
    EXPECT_EQ(
        detail::cifar10_md5_hex(
            {reinterpret_cast<const unsigned char*>(text.data()), text.size()}),
        expected);
  const std::string million(1000000U, 'a');
  EXPECT_EQ(detail::cifar10_md5_hex(
                {reinterpret_cast<const unsigned char*>(million.data()),
                 million.size()}),
            "7707d6ae4e027c70eea2a935c2296f21");
}

TEST(Cifar10NativeContract, MatchesCheckedInFrozenSourceAndCompleteSplits) {
  std::ifstream input(JYPPX_LIUDING_CIFAR10_CONTRACT_FILE);
  ASSERT_TRUE(input);
  const auto expected = nlohmann::json::parse(input);
  const auto& actual = detail::official_cifar10_contract();
  EXPECT_EQ(actual.archive_bytes,
            expected.at("archive_bytes").get<std::uint64_t>());
  EXPECT_EQ(actual.sha256, expected.at("sha256").get<std::string>());
  EXPECT_EQ(actual.official_md5,
            expected.at("official_md5").get<std::string>());
  EXPECT_EQ(actual.source, expected.at("source").get<std::string>());
  EXPECT_EQ(actual.dataset_license,
            expected.at("dataset_license").get<std::string>());
  EXPECT_EQ(actual.max_decoded_bytes,
            expected.at("max_decoded_bytes").get<std::uint64_t>());
  EXPECT_EQ(actual.train_samples_per_class,
            expected.at("samples_per_class").at("train").get<std::uint32_t>());
  EXPECT_EQ(
      actual.validation_samples_per_class,
      expected.at("samples_per_class").at("validation").get<std::uint32_t>());
  ASSERT_EQ(actual.batches.size(), expected.at("batches").size());
  std::uint64_t samples = 0U;
  for (const auto& batch : actual.batches) {
    ASSERT_TRUE(expected.at("batches").contains(batch.name));
    const auto& source = expected.at("batches").at(batch.name);
    EXPECT_EQ(batch.split, source.at("split").get<std::string>());
    EXPECT_EQ(batch.samples, source.at("samples").get<std::uint32_t>());
    samples += batch.samples;
  }
  EXPECT_EQ(samples, 60000U);
}
}  // namespace
